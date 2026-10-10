"""
Ring try-on prototype: detect the hand and fingers with MediaPipe and draw a ring with cv2.line.

The ring is modelled as a 3D band around the finger: only the half facing the camera is drawn
(the other half is behind the finger), and any part covered by another finger or the palm is
hidden, so the ring disappears when the finger it is on can't be seen.

Edge snapping (finger_fit.py) measures the finger's real outline at the ring in every frame, so
the ring takes the finger's actual width, centre line and direction instead of trusting the
landmarks alone; it also learns the user's finger size, so the ring keeps fitting while the hand
moves toward or away from the camera.

Controls (keys, and sliders in the "Ring Controls" window):
    1..5   select finger (thumb, index, middle, ring, pinky) - default: ring
    [ ]    move the ring towards the knuckle / towards the next joint
    - =    make the ring smaller / bigger
    e      edge snapping on/off
    f      fill the window (crop) / fit the whole frame (black bars)
    r      rotate the camera image by 90 degrees (phone streams often arrive sideways)
    h      swap Left/Right (only needed if the camera app already mirrors its image)
    d      debug overlay: hand skeleton, finger names, measured finger edges, hidden part of ring
    s      save a snapshot (clean frame, frame with ring, hand data) to ./captures
    q/ESC  quit (or close the window)

Usage:
    python main.py                                         # auto-detect camera (asks if several)
    python main.py --list-cameras                          # show which camera indices work
    python main.py --source 1                              # camera index 1
    python main.py --source http://192.168.1.5:8080/video --rotate 90   # phone camera over Wi-Fi
    python main.py --source a.mp4 --no-mirror              # video file

Run `python main.py -h` for all options.
"""
import argparse
import contextlib
import itertools
import json
import math
import os
import sys
import threading
import time
import urllib.request
from types import SimpleNamespace

import cv2 as cv
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

import finger_fit

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")
CAPTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures")
WINDOW = "Ring Try-On"
CONTROLS = "Ring Controls"

RING_COLOR = (0, 0, 255)  # red (BGR)

# MediaPipe hand landmark indices: (base joint, next joint, tip) per finger. The ring sits on the
# bone between the first two joints (for the thumb that is MCP(2)->IP(3)).
FINGERS = {
    "Thumb":  (2, 3, 4),
    "Index":  (5, 6, 8),
    "Middle": (9, 10, 12),
    "Ring":   (13, 14, 16),
    "Pinky":  (17, 18, 20),
}
FINGER_KEYS = {ord("1"): "Thumb", ord("2"): "Index", ord("3"): "Middle",
               ord("4"): "Ring", ord("5"): "Pinky"}
SWAP_HANDS = {"Left": "Right", "Right": "Left"}
ROTATIONS = {90: cv.ROTATE_90_CLOCKWISE, 180: cv.ROTATE_180, 270: cv.ROTATE_90_COUNTERCLOCKWISE}

# The hand as capsules around its bones: (joint, joint, radius as a fraction of the spacing
# between neighbouring knuckles). Gives the ring its size and hides whatever is behind a bone.
BONES = [
    (1, 2, 0.55), (2, 3, 0.50), (3, 4, 0.45),          # thumb
    (5, 6, 0.44), (6, 7, 0.40), (7, 8, 0.36),          # index
    (9, 10, 0.45), (10, 11, 0.41), (11, 12, 0.37),     # middle
    (13, 14, 0.43), (14, 15, 0.39), (15, 16, 0.35),    # ring
    (17, 18, 0.38), (18, 19, 0.34), (19, 20, 0.31),    # pinky
    (0, 1, 0.60), (0, 5, 0.60), (0, 9, 0.60), (0, 13, 0.60), (0, 17, 0.60),  # palm
    (1, 5, 0.55), (5, 9, 0.50), (9, 13, 0.50), (13, 17, 0.50),
]
BONE_RADIUS = {(i, j): r for i, j, r in BONES}
BONE_A = np.array([i for i, _, _ in BONES])
BONE_B = np.array([j for _, j, _ in BONES])
BONE_R = np.array([r for _, _, r in BONES])

# Where the ring sits on its bone (0 = knuckle, 1 = next joint). 0.69 is where a real ring's
# centre measured against MediaPipe's tracked landmarks; tune live with [ and ].
RING_POSITION = 0.69
RING_SIZE = 1.0         # multiplier on the ring's inner diameter; tune live with - and =
BAND_WIDTH = 0.25       # ring band width along the finger, as a fraction of the finger width
CLEARANCE = 0.03        # ring inner diameter = finger width x (1 + CLEARANCE) ("Loose fit" slider)
BAND_THICKNESS = 0.08   # metal thickness, as a fraction of finger width: the band shows past the edges
SNAP_MIN_CONF = 0.35    # finger edge measurements below this confidence are ignored...
SNAP_MIN_CONF_CUT = 0.14  # ...unless the hand is cut off by the frame edge: then the landmark size is
                          # unreliable and weaker measurements still beat it (weighted by confidence)
VIEW_MARGIN = 0.03      # the whole hand is "in view" when all landmarks are this far inside the frame
OCCLUSION_MARGIN = 0.2  # depth tolerance (fraction of knuckle spacing) before a bone hides the ring
# The ring fades out once less than HIDE_BELOW of it is visible and back in above SHOW_ABOVE (the
# gap stops it blinking when a finger hovers at the edge of covering it).
HIDE_BELOW, SHOW_ABOVE = 0.4, 0.7
TRACK_GRACE_S = 0.5     # keep a hand's smoothing/label/fade through detection dropouts this long

MAX_CAMERA_INDEX = 6           # camera indices probed by auto-detect and --list-cameras
STREAM_TIMEOUT_MS = 5000       # give up on an unreachable or stalled phone stream after this
MAX_FRAME_SIZE = (1280, 960)   # larger frames (e.g. 1080p phone streams) are scaled down
HOLD = object()  # "edge measurement skipped this frame, keep the current fit"
KEYS_HELP = "1-5 finger  [ ] move  - = size  e snap  f fill  r rotate  h L/R  d debug  s save  q quit"
VIEW_HINT = "Move your hand back - keep your whole hand in view"


class OneEuroFilter:
    """Per-landmark speed-adaptive low-pass filter: strong smoothing while a landmark is still,
    little lag while it moves. Depth (z) is smoothed more because it is noisier."""

    def __init__(self, min_cutoff=0.5, beta=0.05, d_cutoff=1.0, z_factor=0.5):
        self.min_cutoff, self.beta, self.d_cutoff, self.z_factor = min_cutoff, beta, d_cutoff, z_factor
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        """x: (n, 3) landmarks in pixels, t: time in seconds."""
        if self.x_prev is None:
            self.x_prev, self.dx_prev, self.t_prev = x, np.zeros_like(x), t
            return x
        dt = max(t - self.t_prev, 1e-3)
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * (x - self.x_prev) / dt + (1 - a_d) * self.dx_prev
        speed = np.linalg.norm(dx_hat[:, :2], axis=1, keepdims=True)  # px/s per landmark
        cutoff = self.min_cutoff + self.beta * speed
        a = self._alpha(np.hstack([cutoff, cutoff, cutoff * self.z_factor]), dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmark model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH + ".part")
        os.replace(MODEL_PATH + ".part", MODEL_PATH)  # an interrupted download never becomes the model


def create_detector(num_hands=1):
    ensure_model()
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=num_hands,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    return vision.HandLandmarker.create_from_options(options)


def hand_points(hand_landmarks, world_landmarks, w, h):
    """(21, 3) hand in pixels: x, y from the image landmarks, z (away from the camera) from
    MediaPipe's metric world landmarks seen along the viewing direction that best explains the
    2D shape. World landmarks can be tilted ~30 degrees relative to the image, and their depth is
    far more reliable than the image landmarks' z."""
    p2 = np.array([[lm.x * w, lm.y * h] for lm in hand_landmarks])
    wl = np.array([[lm.x, lm.y, lm.z] for lm in world_landmarks])
    pc, wc = p2 - p2.mean(0), wl - wl.mean(0)
    m = np.linalg.lstsq(wc, pc, rcond=None)[0]   # 3x2 linear map: world -> image
    u, s, vt = np.linalg.svd(m, full_matrices=False)
    axes = u @ vt                                 # closest orthographic camera
    view = np.cross(axes[:, 0], axes[:, 1])       # world direction pointing away from the camera
    return np.column_stack([p2, s.mean() * (wc @ view)])


def knuckle_spacing(pts):
    """Mean 3D distance between neighbouring knuckles (MCP joints): the hand's size unit."""
    knuckles = pts[[5, 9, 13, 17]]
    return float(np.linalg.norm(np.diff(knuckles, axis=0), axis=1).mean())


def in_plane(v):
    """Unit vector across direction v in the image plane (v x z), or x if v points at the camera."""
    u = np.cross(v, (0.0, 0.0, 1.0))
    return u / np.linalg.norm(u) if np.linalg.norm(u) > 1e-6 else np.array([1.0, 0.0, 0.0])


def ring_pose(pts, finger, position, radius, half_band, offset=0.0, angle=0.0, unit=None):
    """3D frame of the ring on the finger's base bone (pixels, z away from the camera), or None.
    `offset` moves the centre across the finger (px) and `angle` turns the finger axis in the image
    plane (radians), both from the measured finger edges; `unit` sizes the other fingers' capsules
    that can hide the ring (default: the knuckle spacing). d runs along the finger, u across it in
    the image plane and v = d x u leans toward the camera, so ring points at angles 0..pi are the
    half facing the camera."""
    base, nxt, _ = FINGERS[finger]
    unit = knuckle_spacing(pts) if unit is None else unit
    bone = pts[nxt] - pts[base]
    length = np.linalg.norm(bone)
    if length < 1e-6 or unit < 1e-6:
        return None
    d = bone / length
    centre = pts[base] + position * bone + offset * in_plane(d)
    if angle:
        c, s = math.cos(angle), math.sin(angle)
        d = np.array([c * d[0] - s * d[1], s * d[0] + c * d[1], d[2]])
    u = in_plane(d)
    return {"centre": centre, "d": d, "u": u, "v": np.cross(d, u), "radius": radius,
            "half_band": half_band, "unit": unit, "bone": (base, nxt)}


def ring_points(pose, theta):
    """Points on the ring's centre line at the given angles, and their outward normals."""
    normals = np.outer(np.cos(theta), pose["u"]) + np.outer(np.sin(theta), pose["v"])
    return pose["centre"] + pose["radius"] * normals, normals


def hidden_by_bones(points, pts, unit, exclude):
    """For each 3D point: does a bone capsule (other than `exclude`) cover it from the camera's
    point of view? Uses an orthographic view along +z."""
    keep = np.array([(i, j) != exclude for i, j, _ in BONES])
    a, b, radius = pts[BONE_A[keep]], pts[BONE_B[keep]], BONE_R[keep] * unit
    ab = b - a
    rel = points[:, None, :2] - a[None, :, :2]
    t = np.clip((rel * ab[None, :, :2]).sum(-1) / np.maximum((ab[:, :2] ** 2).sum(-1), 1e-9), 0.0, 1.0)
    closest = a[None] + t[..., None] * ab[None]            # bone point under each ring point
    dist = np.linalg.norm(points[:, None, :2] - closest[..., :2], axis=-1)
    front = closest[..., 2] - np.sqrt(np.maximum(radius ** 2 - dist ** 2, 0.0))  # capsule surface
    covered = (dist < radius) & (front < points[:, None, 2] - OCCLUSION_MARGIN * unit)
    return covered.any(axis=1)


def ring_band(pose, pts):
    """Samples around the half of the band facing the camera: its two edges, how much each
    sample faces the camera (0..1) and whether another bone covers it."""
    n = int(np.clip(pose["radius"] * math.pi / 2.5, 12, 96))
    surface, normals = ring_points(pose, np.linspace(0.0, math.pi, n))
    offset = pose["half_band"] * pose["d"]
    # When the finger points at the camera we see the ring's flat side, which faces it as well.
    facing = np.maximum(-normals[:, 2], abs(pose["d"][2]))
    return {"edge_a": surface + offset, "edge_b": surface - offset, "facing": facing,
            "visible": ~hidden_by_bones(surface, pts, pose["unit"], exclude=pose["bone"])}


def ring_visibility(band):
    """Share of the ring that can be seen, weighting each sample by how much it faces the
    camera: slivers seen edge-on past a finger in front barely count."""
    weight = np.maximum(band["facing"], 0.05)
    return float((band["visible"] * weight).sum() / weight.sum())


def fixed(p):
    """Pixel position -> cv.line point with 3 fractional bits (shift=3), for smooth motion."""
    return int(round(p[0] * 8)), int(round(p[1] * 8))


def colour(c):
    return tuple(int(v) for v in np.clip(c, 0, 255))


def draw_ring(frame, band, alpha, color=RING_COLOR):
    """Draw the visible part of the band with cv.line: lines across the band fill it, shaded by
    how much each part faces the camera, with darker edges and a highlight. `alpha` fades the
    ring in and out instead of letting it pop."""
    edge_a, edge_b, facing, visible = band["edge_a"], band["edge_b"], band["facing"], band["visible"]
    if alpha < 0.02 or not visible.any():
        return
    h, w = frame.shape[:2]
    corners = np.vstack([edge_a[:, :2], edge_b[:, :2]])
    x0, y0 = np.clip(np.floor(corners.min(0)).astype(int) - 4, 0, (w, h))
    x1, y1 = np.clip(np.ceil(corners.max(0)).astype(int) + 5, 0, (w, h))
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    layer = roi.copy()
    ea, eb = edge_a[:, :2] - (x0, y0), edge_b[:, :2] - (x0, y0)
    base = np.array(color, float)  # lit from the camera: the part facing it is brightest
    step = np.linalg.norm(np.diff(ea, axis=0), axis=1).max(initial=1.0)
    fill = max(3, math.ceil(step) + 1)  # thick enough that neighbouring lines leave no gaps
    for k in np.flatnonzero(visible):
        cv.line(layer, fixed(ea[k]), fixed(eb[k]), colour(base * (0.5 + 0.5 * facing[k])),
                fill, cv.LINE_AA, 3)
    edge, highlight = colour(base * 0.4), colour(base * 0.6 + 100)
    for k in np.flatnonzero(visible[:-1] & visible[1:]):
        for e in (ea, eb):
            cv.line(layer, fixed(e[k]), fixed(e[k + 1]), edge, 2, cv.LINE_AA, 3)
        if facing[k] > 0.85:
            cv.line(layer, fixed((ea[k] + eb[k]) / 2), fixed((ea[k + 1] + eb[k + 1]) / 2),
                    highlight, 1, cv.LINE_AA, 3)
    cv.addWeighted(layer, alpha, roi, 1 - alpha, 0, dst=roi)


def draw_debug(frame, pts, pose, fit):
    for c in vision.HandLandmarksConnections.HAND_CONNECTIONS:
        cv.line(frame, fixed(pts[c.start]), fixed(pts[c.end]), (200, 200, 200), 1, cv.LINE_AA, 3)
    for p in pts:
        cv.circle(frame, (int(p[0]), int(p[1])), 3, (0, 255, 0), -1, cv.LINE_AA)
    for name, (_, _, tip) in FINGERS.items():
        x, y = int(pts[tip, 0]), int(pts[tip, 1])
        cv.putText(frame, name, (x + 6, y - 6), cv.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv.LINE_AA)
    if pose is not None:  # the whole ring, including the half behind the finger
        ring, _ = ring_points(pose, np.linspace(0.0, 2 * math.pi, 49))
        for p, q in zip(ring[:-1], ring[1:]):
            cv.line(frame, fixed(p), fixed(q), (160, 160, 160), 1, cv.LINE_AA, 3)
    if fit is not None and fit["ok"]:  # the finger edges found in the image (yellow)
        half = 0.4 * fit["width_px"]
        for e in fit["edges_img"]:
            p, d = np.array(e["point"]), np.array(e["dir"])
            cv.line(frame, fixed(p - half * d), fixed(p + half * d), (0, 255, 255), 1, cv.LINE_AA, 3)


def put_label(frame, text, org, scale=0.6):
    cv.putText(frame, text, org, cv.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv.LINE_AA)
    cv.putText(frame, text, org, cv.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv.LINE_AA)


def label_position(pts, w, h, text, scale=0.6):
    """Just past the wrist, on the side away from the fingers, kept inside the frame and below the
    two top bars (status line and the "move your hand back" hint)."""
    wrist = pts[0, :2]
    away = wrist - pts[9, :2]
    x, y = wrist + away / max(np.linalg.norm(away), 1e-6) * 30
    width = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, scale, 3)[0][0]
    top = 2 * int(60 * np.clip(w / 1500, 0.35, 0.5)) + 18  # see draw_bar
    return int(np.clip(x - 80, 5, max(5, w - width - 5))), int(np.clip(y + 10, top, h - 35))


def draw_bar(frame, text, top=True, row=0, color=(255, 255, 255)):
    """One line of text on a darkened strip at the top or bottom of the frame (`row` counts away
    from that edge); the font shrinks when needed so the line always fits."""
    h, w = frame.shape[:2]
    scale = float(np.clip(w / 1500, 0.35, 0.5))
    height = int(60 * scale)
    text_w = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 1.0, 1)[0][0]
    font = min(scale, (w - 16) / max(text_w, 1))
    y0 = row * height if top else h - (row + 1) * height
    strip = frame[max(y0, 0):max(y0 + height, 0)]
    cv.convertScaleAbs(strip, dst=strip, alpha=0.35)  # darken in place, no float copy
    cv.putText(frame, text, (8, y0 + int(height * 0.68)), cv.FONT_HERSHEY_SIMPLEX, font, color, 1, cv.LINE_AA)


def hand_in_view(pts, w, h):
    """Is the whole hand inside the frame? With the wrist or palm cut off, MediaPipe's hand scale,
    and with it the landmark-based ring size, becomes unreliable (off by -39%/+21% in tests)."""
    mx, my = VIEW_MARGIN * w, VIEW_MARGIN * h
    x, y = pts[:, 0], pts[:, 1]
    return bool(((x >= mx) & (x <= w - mx) & (y >= my) & (y <= h - my)).all())


def compose_view(frame, size, fill):
    """The frame scaled to the window's image area: the whole frame with black bars ("fit"), or
    cropped to cover the area ("fill"). The ring is already drawn on the frame, so it stays exactly
    on the finger in both modes."""
    ww, wh = int(size[0]), int(size[1])
    fh, fw = frame.shape[:2]
    if ww < 16 or wh < 16 or (ww, wh) == (fw, fh):
        return frame
    scale = (max if fill else min)(ww / fw, wh / fh)
    sw, sh = round(fw * scale), round(fh * scale)
    sw, sh = (max(sw, ww), max(sh, wh)) if fill else (min(sw, ww), min(sh, wh))
    img = cv.resize(frame, (sw, sh), interpolation=cv.INTER_AREA if scale < 1 else cv.INTER_LINEAR)
    if fill:
        x0, y0 = (sw - ww) // 2, (sh - wh) // 2
        return img[y0:y0 + wh, x0:x0 + ww]
    view = np.zeros((wh, ww, 3), np.uint8)
    x0, y0 = (ww - sw) // 2, (wh - sh) // 2
    view[y0:y0 + sh, x0:x0 + sw] = img
    return view


class HandTrack:
    """State kept per hand between frames: landmark smoothing, a steady Left/Right label, the
    ring's fade level and its fit to the measured finger."""

    def __init__(self, wrist, t):
        self.wrist = wrist
        self.last_seen = t
        self.filter = OneEuroFilter()
        self.p_right = None
        self.label = None
        self.seen = 0.0     # smoothed share of the ring that is visible (starts as hidden)
        self.shown = False  # ring currently shown (vs hidden because it is covered)
        self.alpha = 0.0    # ring opacity, follows `shown` with a short fade
        self.finger = None  # finger the fit below belongs to
        self.size_corr = 0.0  # log correction of the finger width from the measured edges
        self.offset = 0.0   # ring centre offset across the finger, in finger widths
        self.angle = 0.0    # finger axis correction in the image plane (radians)
        self.t_fit = None
        self.frames = 0     # frames this hand has been processed in
        self.last_fit = None  # latest edge measurement (for debug and snapshots)
        self.fit_used = False  # whether that measurement was confident enough to use

    def vote(self, p_right):
        """Average MediaPipe's per-frame handedness and only switch on a clear majority, so the
        label doesn't flicker on ambiguous poses."""
        self.p_right = p_right if self.p_right is None else 0.75 * self.p_right + 0.25 * p_right
        if self.label is None or abs(self.p_right - 0.5) > 0.1:
            self.label = "Right" if self.p_right >= 0.5 else "Left"
        return self.label


class RingTracker:
    def __init__(self, num_hands=1, mirrored=True, swap_hands=False):
        self.num_hands = num_hands
        self.detector = create_detector(num_hands)
        self.tracks = []
        self.finger = "Ring"
        self.position = RING_POSITION
        self.size = RING_SIZE
        self.clearance = CLEARANCE
        self.band_width = BAND_WIDTH
        self.snap = True          # measure the finger's edges (edge snapping)
        self.k = {}               # per finger: learned finger width / knuckle spacing
        self._k_jump_since = None
        self.debug = False
        self.mirrored = mirrored  # whether frames passed to process() are flipped horizontally
        self.swap_hands = swap_hands
        self._t0 = None
        self._last_ts = -1

    def reset(self):
        """Start over with a fresh detector and no tracked hands (e.g. after rotating)."""
        self.detector.close()
        self.detector = create_detector(self.num_hands)
        self.tracks = []

    def toggle_swap(self):
        self.swap_hands = not self.swap_hands
        for track in self.tracks:
            if track.label is not None:
                track.label, track.p_right = SWAP_HANDS[track.label], 1.0 - track.p_right

    def _match_tracks(self, hands, w, h, t):
        """Pair each detected hand with the nearest recently seen hand (by wrist), so smoothing,
        the Left/Right vote and the ring's fade stay with the same physical hand, also when
        MediaPipe loses it for a few frames."""
        limit = 0.25 * max(w, h)  # a wrist further away than this is a new hand
        dist = [[np.linalg.norm(tr.wrist - pts[0, :2]) for tr in self.tracks] for pts in hands]
        # Try every pairing (at most 2 hands, a few tracks) and keep the one that moves the wrists
        # least, so a hand moving fast toward the other one can't take over its track.
        pairings = [p for p in itertools.product([None, *range(len(self.tracks))], repeat=len(hands))
                    if all(k is None or dist[i][k] <= limit for i, k in enumerate(p))
                    and len({k for k in p if k is not None}) == sum(k is not None for k in p)]
        best = min(pairings, key=lambda p: sum(limit if k is None else dist[i][k] for i, k in enumerate(p)))
        tracks = []
        for pts, k in zip(hands, best):
            if k is None:
                track = HandTrack(pts[0, :2], t)
            else:
                track = self.tracks[k]
                if t - track.last_seen > 0.2:  # back after a dropout: don't smooth across the gap
                    track.filter = OneEuroFilter()
                track.wrist, track.last_seen = pts[0, :2], t
            tracks.append(track)
        self.tracks = tracks + [tr for k, tr in enumerate(self.tracks)
                                if k not in best and t - tr.last_seen < TRACK_GRACE_S]
        return tracks

    def _measure(self, frame, pts, track):
        """The selected finger's outline at the ring, measured in the image (edge snapping); None
        when snapping is off or the ring was mostly covered in the previous frame, HOLD on frames
        that skip the measurement (every 2nd frame once the fit has settled: it is smoothed anyway,
        and this halves the ~4 ms it costs)."""
        track.frames += 1
        if not self.snap or track.seen < 0.6:
            return None
        if track.frames > 10 and track.frames % 2:
            return HOLD
        base, nxt, _ = FINGERS[self.finger]
        unit = knuckle_spacing(pts)
        prior = 2 * BONE_RADIUS[(base, nxt)] * unit
        prev = None
        if track.finger == self.finger and track.fit_used:  # where the ring's edges are now (px along u)
            width = self.k.get(self.finger, 2 * BONE_RADIUS[(base, nxt)]) * unit * math.exp(track.size_corr)
            prev = ((track.offset - 0.5) * width, (track.offset + 0.5) * width)
        track.last_fit = finger_fit.measure_finger(frame, pts, base, nxt, self.position, prior, prev)
        return track.last_fit

    def _fit_ring(self, track, fit, unit, in_view, facing, t):
        """Finger width (px), ring centre offset across the finger (in finger widths) and finger
        axis correction (radians) for this frame.

        Width = K x knuckle spacing x correction. The knuckle spacing follows the hand toward and
        away from the camera in every frame; K (finger width / knuckle spacing) is learned slowly
        from confident measurements while the whole hand is in view; the correction, offset and
        angle follow the current measurement and relax back when there is none. With snapping off
        the ring uses the landmarks only (default K)."""
        base, nxt, _ = FINGERS[self.finger]
        default = 2 * BONE_RADIUS[(base, nxt)]
        k = self.k.get(self.finger, default) if self.snap else default
        predicted = k * unit
        if track.finger != self.finger:  # the fit state belongs to one finger
            track.finger, track.size_corr, track.offset, track.angle = self.finger, 0.0, 0.0, 0.0
        dt = t - track.t_fit if track.t_fit is not None else 0.0
        track.t_fit = t
        min_conf = SNAP_MIN_CONF if in_view else SNAP_MIN_CONF_CUT
        if fit is HOLD:
            pass  # measurement skipped this frame: keep the fit as it is
        elif self.snap and fit is not None and fit["ok"] and fit["confidence"] >= min_conf:
            track.fit_used = True
            conf = min(1.0, fit["confidence"])
            err = math.log(fit["width_px"] / predicted)
            gain = 0.6 * conf
            track.size_corr += (err - track.size_corr) * gain
            track.offset += (fit["centre_offset_px"] / fit["width_px"] - track.offset) * gain
            # finger_fit turns the axis toward u for a positive angle; ring_pose turns the other way
            track.angle += (-math.radians(fit["axis_angle_deg"]) - track.angle) * gain
            # Learn this user's finger (big jumps must persist for 0.5 s), but not from views far off
            # normal proportions: close up, with the finger pointing at the camera, it looks ~35%
            # wider than its knuckle spacing suggests.
            if in_view and facing and abs(math.log(fit["width_px"] / (default * unit))) < math.log(1.25):
                if abs(err) < math.log(1.25):
                    self._k_jump_since = None
                elif self._k_jump_since is None:
                    self._k_jump_since = t
                if self._k_jump_since is None or t - self._k_jump_since > 0.5:
                    self.k[self.finger] = k * math.exp(0.05 * conf * err)
        else:
            track.fit_used = False
            relax = math.exp(-max(dt, 0.0) / 0.5)
            track.offset *= relax
            track.angle *= relax
            if in_view:  # with the hand cut off the knuckle spacing is unreliable: keep the fit
                track.size_corr *= relax
        if not self.snap:
            return predicted, 0.0, 0.0
        return predicted * math.exp(track.size_corr), track.offset, track.angle

    def process(self, frame, t=None):
        """Detect hands in `frame` (BGR), draw the ring on it in place and return per-hand info.
        t: frame time in seconds (default: now); fixed steps make offline replays repeatable."""
        h, w = frame.shape[:2]
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        t = time.monotonic() if t is None else t
        if self._t0 is None:
            self._t0 = t
        ts = max(int((t - self._t0) * 1000), self._last_ts + 1)  # must strictly increase
        self._last_ts = ts
        result = self.detector.detect_for_video(mp_image, ts)

        hands = [(hand_points(lms, world, w, h), handed[0]) for lms, world, handed in
                 zip(result.hand_landmarks, result.hand_world_landmarks, result.handedness)]
        tracks = self._match_tracks([p for p, _ in hands], w, h, t)
        # Measure the fingers before anything is drawn: rings and labels would add edges.
        fits = [self._measure(frame, raw_pts, track) for (raw_pts, _), track in zip(hands, tracks)]
        info = []
        for (raw_pts, category), track, fit in zip(hands, tracks, fits):
            pts = track.filter(raw_pts, t)
            # MediaPipe names the hand as it appears in the image it was given, so a mirrored
            # (selfie) view shows the user's right hand as a left one: swap the name back.
            raw = category.category_name
            name = SWAP_HANDS.get(raw, raw) if self.mirrored != self.swap_hands else raw
            label = track.vote(category.score if name == "Right" else 1.0 - category.score)
            # With the hand as it appears in the image, the palm normal tells palm from back.
            normal = np.cross(pts[5] - pts[0], pts[17] - pts[0])
            palm = (normal[2] < 0) == (raw == "Right")
            facing = abs(normal[2]) >= 0.6 * np.linalg.norm(normal)  # palm or back toward the camera
            in_view = hand_in_view(raw_pts, w, h)

            width, offset, angle = self._fit_ring(track, fit, knuckle_spacing(pts), in_view, facing, t)
            # The inner surface sits on the skin (plus clearance); the band shows its thickness
            # past the finger's edges, like a real ring (less the ~1.5 px its strokes add).
            radius = 0.5 * width * (1 + self.clearance) * self.size + BAND_THICKNESS * width - 1.5
            # Size the hand's capsules from the fitted finger too: with the hand cut off by the frame
            # edge the knuckle spacing is off, and too-fat neighbours would hide a visible ring.
            unit = width / (2 * BONE_RADIUS[FINGERS[self.finger][:2]])
            pose = ring_pose(pts, self.finger, self.position, radius, 0.5 * self.band_width * width,
                             offset * width, angle, unit)
            band = ring_band(pose, pts) if pose is not None else None
            seen = ring_visibility(band) if band is not None else 0.0
            # Hide the ring when its finger is covered (e.g. hand turned sideways), with a fade.
            # Averaged over a few frames: the first frame after the hand is found is less accurate.
            track.seen += (seen - track.seen) * 0.5
            if track.shown and track.seen < HIDE_BELOW or not track.shown and track.seen > SHOW_ABOVE:
                track.shown = not track.shown
            track.alpha += (float(track.shown) - track.alpha) * 0.4

            fit = track.last_fit if fit is HOLD else fit  # latest measurement, for display
            if self.debug:
                draw_debug(frame, pts, pose, fit)
            if band is not None:
                draw_ring(frame, band, track.alpha)
            text = f"{label} hand - {self.finger} finger"
            if self.debug:
                text += f" ({'palm' if palm else 'back'}, ring {seen:.0%} visible, finger {width:.0f}px"
                text += f", fit {fit['confidence']:.2f})" if fit is not None and fit["ok"] else ")"
            put_label(frame, text, label_position(pts, w, h, text))
            edges = None
            if fit is not None:
                edges = {"ok": fit["ok"], "confidence": round(fit["confidence"], 3), "used": track.fit_used}
                if fit["ok"]:
                    edges.update(width_px=round(fit["width_px"], 1), edge_kinds=fit["edge_kinds"],
                                 centre_offset_px=round(fit["centre_offset_px"], 1),
                                 axis_angle_deg=round(fit["axis_angle_deg"], 2))
                else:
                    edges["reason"] = fit.get("reason")
            info.append({"hand": label, "p_right": round(float(track.p_right), 3),
                         "mediapipe_label": raw, "palm_facing_camera": bool(palm),
                         "finger": self.finger, "ring_visible": round(seen, 3),
                         "hand_in_view": in_view, "finger_width_px": round(width, 1),
                         "ring_centre_px": np.round(pose["centre"][:2], 1).tolist() if pose else None,
                         "edge_fit": edges, "landmarks_px": np.round(pts, 1).tolist(),
                         "landmarks_raw_px": np.round(raw_pts, 1).tolist()})
        return info

    def close(self):
        self.detector.close()


@contextlib.contextmanager
def quiet_opencv():
    """Silence OpenCV's warning for every camera index that doesn't exist."""
    prev = cv.utils.logging.setLogLevel(cv.utils.logging.LOG_LEVEL_ERROR)
    try:
        yield
    finally:
        cv.utils.logging.setLogLevel(prev)


def camera_apis():
    # On Windows try DirectShow first: phone webcam apps such as DroidCam register DirectShow
    # virtual cameras that Media Foundation doesn't list.
    if sys.platform == "win32":
        return [cv.CAP_DSHOW, cv.CAP_MSMF]
    return [cv.CAP_ANY]


def open_camera(index, width=1280, height=720):
    """Open camera `index` with the first backend that actually delivers frames, else None."""
    with quiet_opencv():
        for api in camera_apis():
            cap = cv.VideoCapture(index, api)
            if cap.isOpened():
                cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
                for _ in range(30):  # some cameras need a few reads before the first frame
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        return cap
                    time.sleep(0.05)
            cap.release()
    return None


def describe_capture(cap):
    w, h = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    return f"{w}x{h} via {cap.getBackendName()}"


def list_cameras():
    """[(index, description)] for every camera index that delivers frames."""
    cams = []
    for i in range(MAX_CAMERA_INDEX):
        cap = open_camera(i)
        if cap is not None:
            cams.append((i, describe_capture(cap)))
            cap.release()
    return cams


def choose_camera():
    """Index of the working camera (asks which one if there are several), or None."""
    print("Searching for cameras...")
    cams = list_cameras()
    if len(cams) > 1 and sys.stdin.isatty():
        for i, desc in cams:
            print(f"  [{i}] {desc}")
        choice = input(f"Which camera? [{cams[0][0]}]: ").strip()
        for i, _ in cams:
            if choice == str(i):
                return i
    return cams[0][0] if cams else None


class LatestFrameReader:
    """Reads a network stream on a background thread and keeps only the newest frame, so the
    ring never lags behind the phone feed when detection is slower than the stream."""

    def __init__(self, cap):
        self.cap = cap
        self._frame = None
        self._alive = True
        self._cond = threading.Condition()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while self._alive:
            ok, frame = self.cap.read()
            with self._cond:
                if ok:
                    self._frame = frame
                else:
                    self._alive = False
                self._cond.notify_all()

    def read(self):
        with self._cond:
            self._cond.wait_for(lambda: self._frame is not None or not self._alive,
                                timeout=STREAM_TIMEOUT_MS / 1000)
            frame, self._frame = self._frame, None
        return frame is not None, frame

    def release(self):
        self._alive = False
        self._thread.join(timeout=STREAM_TIMEOUT_MS / 1000 + 1)
        if not self._thread.is_alive():  # never release while the thread is inside read()
            self.cap.release()


def open_source(source):
    """'auto', camera index, stream URL or video file -> object with read() and release()."""
    if "://" in source:
        cap = cv.VideoCapture(source, cv.CAP_FFMPEG, [
            cv.CAP_PROP_OPEN_TIMEOUT_MSEC, STREAM_TIMEOUT_MS,
            cv.CAP_PROP_READ_TIMEOUT_MSEC, STREAM_TIMEOUT_MS])
        if not cap.isOpened():
            raise SystemExit(f"Cannot open stream {source}\nCheck that the phone app's server is "
                             "running, the IP address and port, and that phone and PC are on "
                             "the same Wi-Fi.")
        print(f"Streaming from {source} ({describe_capture(cap)})")
        return LatestFrameReader(cap)
    if source == "auto" or source.isdigit():
        index = choose_camera() if source == "auto" else int(source)
        cap = open_camera(index) if index is not None else None
        if cap is None:
            problem = "No working camera found." if index is None else f"Camera {index} is not available."
            raise SystemExit(f"{problem}")
        print(f"Using camera {index} ({describe_capture(cap)})")
        return cap
    if not os.path.isfile(source):
        raise SystemExit(f"Video file not found: {source}")
    cap = cv.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video file {source}")
    return cap


def save_snapshot(clean, rendered, info, settings):
    """Save the frame without and with the ring plus the hand data, for offline analysis."""
    os.makedirs(CAPTURE_DIR, exist_ok=True)
    now = time.time()
    stem = os.path.join(CAPTURE_DIR, time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
                        + f"-{int(now * 1000) % 1000:03d}")
    cv.imwrite(stem + "_clean.png", clean)
    cv.imwrite(stem + "_ring.png", rendered)
    with open(stem + ".json", "w") as f:
        json.dump({"settings": settings, "hands": info}, f, indent=1)
    print(f"Saved {stem}_clean.png / _ring.png / .json")


class Controls:
    """The "Ring Controls" side panel: sliders (OpenCV trackbars in their own window, so the video
    keeps its full size) and a live readout of the fit. Keys change the same settings and the
    sliders follow them."""

    def __init__(self, bars):
        # bars: (label, object, attribute, minimum, maximum, scale); slider position = value x scale.
        # Labels: at most 10 characters, Windows shortens longer ones to "Loose ... %".
        self.bars = bars
        self.pos = {}
        cv.namedWindow(CONTROLS, cv.WINDOW_AUTOSIZE)
        for label, obj, attr, lo, hi, scale in bars:
            p = int(np.clip(round(float(getattr(obj, attr)) * scale), lo, hi))
            cv.createTrackbar(label, CONTROLS, p, hi, lambda v: None)
            cv.setTrackbarMin(label, CONTROLS, lo)
            self.pos[label] = p

    def alive(self):
        return cv.getWindowProperty(CONTROLS, cv.WND_PROP_VISIBLE) >= 1

    def sync(self):
        """Apply slider moves to the settings, and move a slider when a key changed its setting."""
        for label, obj, attr, lo, hi, scale in self.bars:
            p = cv.getTrackbarPos(label, CONTROLS)
            value = getattr(obj, attr)
            wanted = int(np.clip(round(float(value) * scale), lo, hi))
            if p != self.pos[label]:
                setattr(obj, attr, bool(p) if isinstance(value, bool) else p / scale)
            elif p != wanted:
                cv.setTrackbarPos(label, CONTROLS, wanted)
                p = wanted
            self.pos[label] = p

    def show(self, lines):
        panel = np.full((24 * len(lines) + 14, 500, 3), 30, np.uint8)
        for i, line in enumerate(lines):
            cv.putText(panel, line, (10, 26 + 24 * i), cv.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1,
                       cv.LINE_AA)
        cv.imshow(CONTROLS, panel)


def panel_lines(tracker, info, fps):
    """Readout for the controls panel: the finger width the ring uses, whether the measured edges
    were trusted, and what the app has learned. Always the same number of lines, so the panel
    window doesn't change size when a hand comes and goes."""
    hand = info[0] if info else None
    if hand is None:
        lines = ["No hand detected"]
    elif not tracker.snap:
        lines = [f"Finger {hand['finger_width_px']:.0f} px - edge snap OFF, sized from the landmarks"]
    else:
        fit = hand["edge_fit"]
        if fit and fit["ok"] and fit["used"]:
            how = f"from its edges, confidence {fit['confidence']:.2f}"
        elif fit and fit["ok"]:
            how = f"edges ({fit['width_px']:.0f} px) too uncertain ({fit['confidence']:.2f}), not used"
        else:
            how = "edges not measured"
        lines = [f"Finger {hand['finger_width_px']:.0f} px - {how}"]
    base, nxt, _ = FINGERS[tracker.finger]
    k = tracker.k.get(tracker.finger) if tracker.snap else None  # snap off uses the default
    lines.append(f"{tracker.finger} finger width: {k or 2 * BONE_RADIUS[(base, nxt)]:.2f} x knuckle spacing"
                 + (" (learned)" if k else " (default, learning...)"))
    lines.append("Whole hand in view: " + ("-" if not info else "yes" if all(i["hand_in_view"] for i in info)
                                           else "NO - move your hand back"))
    lines.append(f"Speed: {fps:.0f} FPS")
    lines += ["Keys: e snap  f fill  [ ] position  - = size  1-5 finger",
              "      r rotate  h swap L/R  d debug  s save  q quit"]
    return lines


def prepare(frame, rotation, mirror):
    """Camera frame -> the frame the app works on: scaled down to MAX_FRAME_SIZE first (cheaper to
    rotate and flip), rotated, mirrored. Rotating 90 degrees and mirroring together is a single
    transpose."""
    h, w = frame.shape[:2]
    if rotation in (90, 270):
        h, w = w, h
    scale = min(1.0, MAX_FRAME_SIZE[0] / w, MAX_FRAME_SIZE[1] / h)
    if scale < 1.0:  # bilinear is 3x faster than area averaging and looks the same down to half size
        frame = cv.resize(frame, None, fx=scale, fy=scale,
                          interpolation=cv.INTER_LINEAR if scale >= 0.5 else cv.INTER_AREA)
    if mirror and rotation == 90:
        return cv.transpose(frame)
    if mirror and rotation == 270:
        return cv.flip(cv.transpose(frame), -1)
    if rotation:
        frame = cv.rotate(frame, ROTATIONS[rotation])
    return cv.flip(frame, 1) if mirror else frame


def main():
    ap = argparse.ArgumentParser(description="Ring try-on: draws a ring on the selected finger.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", nargs="?", const="auto", default="auto",
                    help="'auto' (default), camera index, phone stream URL or video file")
    ap.add_argument("--list-cameras", action="store_true", help="list working cameras and exit")
    ap.add_argument("--no-mirror", action="store_true",
                    help="don't mirror the view (e.g. rear camera looking at the hand, video file)")
    ap.add_argument("--rotate", type=int, choices=(0, 90, 180, 270), default=0,
                    help="rotate the camera image clockwise; phone streams are often sideways (key: r)")
    ap.add_argument("--swap-hands", action="store_true",
                    help="swap Left/Right, for camera apps that already mirror their image (key: h)")
    ap.add_argument("--hands", type=int, choices=(1, 2), default=1,
                    help="hands tracked at once (default 1); 2 makes detection about half as fast")
    ap.add_argument("--finger", type=str.capitalize, choices=list(FINGERS), default="Ring",
                    help="finger that wears the ring (default Ring, keys: 1-5)")
    ap.add_argument("--ring-pos", type=float, default=RING_POSITION,
                    help=f"ring position on the finger, 0 = knuckle .. 1 = next joint "
                         f"(default {RING_POSITION}, keys: [ ])")
    ap.add_argument("--ring-size", type=float, default=RING_SIZE,
                    help=f"ring size multiplier (default {RING_SIZE}, keys: - =)")
    ap.add_argument("--no-snap", action="store_true",
                    help="size the ring from the landmarks only, without measuring the finger's edges (key: e)")
    ap.add_argument("--loose-fit", type=float, default=CLEARANCE,
                    help=f"gap between finger and ring, as a fraction of finger width (default {CLEARANCE})")
    ap.add_argument("--band-width", type=float, default=BAND_WIDTH,
                    help=f"ring band width along the finger, as a fraction of finger width (default {BAND_WIDTH})")
    ap.add_argument("--fill", action="store_true",
                    help="fill the window and crop the edges, instead of showing the whole frame (key: f)")
    ap.add_argument("--no-controls", action="store_true", help="don't open the Ring Controls slider panel")
    args = ap.parse_args()

    if args.list_cameras:
        cams = list_cameras()
        for i, desc in cams:
            print(f"  camera {i}: {desc}")
        if not cams:
            print("No working camera found.")
        return

    src = open_source(args.source)
    tracker = RingTracker(num_hands=args.hands, mirrored=not args.no_mirror,
                          swap_hands=args.swap_hands)
    tracker.finger, tracker.position, tracker.size = args.finger, args.ring_pos, args.ring_size
    tracker.snap, tracker.clearance, tracker.band_width = not args.no_snap, args.loose_fit, args.band_width
    view = SimpleNamespace(fill=args.fill)
    cv.namedWindow(WINDOW, cv.WINDOW_NORMAL | cv.WINDOW_FREERATIO)  # resizable; compose_view keeps the aspect
    controls = None if args.no_controls else Controls([
        ("Edge snap", tracker, "snap", 0, 1, 1), ("Position %", tracker, "position", 10, 95, 100),
        ("Size %", tracker, "size", 50, 200, 100), ("Loose %", tracker, "clearance", 0, 15, 100),
        ("Band %", tracker, "band_width", 10, 50, 100), ("Fill view", view, "fill", 0, 1, 1)])
    window_size = None  # size the video window was last given automatically
    rotation, save_next = args.rotate, False
    fps, prev = 0.0, time.monotonic()

    try:
        while True:
            ok, frame = src.read()
            if not ok:
                print("Video source ended or stopped sending frames.")
                break
            frame = prepare(frame, rotation, not args.no_mirror)
            if controls is not None:
                if controls.alive():
                    controls.sync()
                else:
                    controls = None  # panel closed: carry on with the keys
            clean = frame.copy() if save_next else None
            info = tracker.process(frame)
            if clean is not None:
                save_snapshot(clean, frame, info, {
                    "source": args.source, "rotate": rotation, "mirrored": not args.no_mirror,
                    "swap_hands": tracker.swap_hands, "finger": tracker.finger,
                    "ring_pos": tracker.position, "ring_size": tracker.size, "snap": tracker.snap,
                    "loose_fit": tracker.clearance, "band_width": tracker.band_width,
                    "learned_finger_width_ratio": tracker.k.get(tracker.finger)})
                save_next = False

            now = time.monotonic()
            fps = 0.95 * fps + 0.05 / max(now - prev, 1e-6) if fps else 1.0 / max(now - prev, 1e-6)
            prev = now
            # The video window starts at the frame size; once the user resizes it, its size is kept
            # and the frame is fitted (or cropped, in fill mode) into it.
            fh, fw = frame.shape[:2]
            try:
                area = tuple(cv.getWindowImageRect(WINDOW)[2:])
            except cv.error:
                area = (fw, fh)
            if window_size is None or (window_size != (fw, fh) and area == window_size):
                if window_size is None:
                    cv.moveWindow(WINDOW, 10, 10)
                cv.resizeWindow(WINDOW, fw, fh)
                if controls is not None:  # keep the panel beside the (re)sized video
                    cv.moveWindow(CONTROLS, fw + 40, 10)
                window_size = area = (fw, fh)
            display = compose_view(frame, area, view.fill)
            hands = ", ".join(f"{i['hand']} hand" for i in info) or "no hand detected"
            draw_bar(display, f"FPS {fps:4.1f} | {hands} | {tracker.finger} finger | snap "
                              f"{'on' if tracker.snap else 'off'} | ring at {tracker.position:.2f} size "
                              f"{tracker.size:.2f} | rotate {rotation}")
            if any(not i["hand_in_view"] for i in info):
                draw_bar(display, VIEW_HINT, row=1, color=(0, 215, 255))
            draw_bar(display, KEYS_HELP, top=False)
            cv.imshow(WINDOW, display)
            if controls is not None:
                controls.show(panel_lines(tracker, info, fps))
            key = cv.waitKey(1) & 0xFF
            if key in (ord("q"), 27) or cv.getWindowProperty(WINDOW, cv.WND_PROP_VISIBLE) < 1:
                break  # q/ESC, or the window was closed with its X button
            if key in FINGER_KEYS:
                tracker.finger = FINGER_KEYS[key]
            elif key == ord("d"):
                tracker.debug = not tracker.debug
            elif key == ord("e"):
                tracker.snap = not tracker.snap
            elif key == ord("f"):
                view.fill = not view.fill
            elif key == ord("h"):
                tracker.toggle_swap()
            elif key == ord("r"):
                rotation = (rotation + 90) % 360
                tracker.reset()  # the hand jumps to a new place in the rotated image
            elif key in (ord("["), ord("]")):
                tracker.position = float(np.clip(tracker.position + (0.02 if key == ord("]") else -0.02), 0.1, 0.95))
            elif key in (ord("-"), ord("=")):
                tracker.size = float(np.clip(tracker.size + (0.05 if key == ord("=") else -0.05), 0.5, 2.0))
            elif key == ord("s"):
                save_next = True
    finally:
        tracker.close()
        src.release()
        cv.destroyAllWindows()
        source = "" if args.source == "auto" else f' --source "{args.source}"'
        print(f"To start with the same settings next time:\n  python main.py{source} --rotate {rotation} "
              f"--finger {tracker.finger} --ring-pos {tracker.position:.2f} --ring-size {tracker.size:.2f}"
              + (f" --hands {args.hands}" if args.hands != 1 else "")
              + (" --swap-hands" if tracker.swap_hands else "") + (" --no-mirror" if args.no_mirror else "")
              + ("" if tracker.snap else " --no-snap") + (" --fill" if view.fill else "")
              + (f" --loose-fit {tracker.clearance:.2f}" if round(tracker.clearance, 2) != CLEARANCE else "")
              + (f" --band-width {tracker.band_width:.2f}" if round(tracker.band_width, 2) != BAND_WIDTH else "")
              + (" --no-controls" if args.no_controls else ""))


if __name__ == "__main__":
    main()
