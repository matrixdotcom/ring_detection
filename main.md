"""
Ring try-on prototype: detect hand + fingers with MediaPipe, draw a ring with cv2.line.

Controls:
    1..5  select finger (thumb, index, middle, ring, pinky)  - default: ring
    d     toggle debug overlay (landmarks + finger labels)
    q/ESC quit

Usage:
    python main.py                                         # auto-detect camera (asks if several)
    python main.py --list-cameras                          # show which camera indices work
    python main.py --source 1                              # camera index 1
    python main.py --source http://192.168.1.5:8080/video  # phone camera streamed over Wi-Fi
    python main.py --source a.mp4 --no-mirror              # video file

Run `python main.py -h` for how to use a phone as the camera.
"""
import argparse
import contextlib
import math
import os
import sys
import threading
import time
import urllib.request

import cv2 as cv
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")

RING_COLOR = (0, 0, 255)  # red (BGR)

# MediaPipe hand landmark indices: (MCP/base knuckle, PIP/next joint, tip) per finger.
# For the thumb the "base" segment is MCP(2)->IP(3).
FINGERS = {
    "Thumb":  (2, 3, 4),
    "Index":  (5, 6, 8),
    "Middle": (9, 10, 12),
    "Ring":   (13, 14, 16),
    "Pinky":  (17, 18, 20),
}
FINGER_KEYS = {ord("1"): "Thumb", ord("2"): "Index", ord("3"): "Middle",
               ord("4"): "Ring", ord("5"): "Pinky"}
KNUCKLES = [5, 9, 13, 17]  # MCP joints of index..pinky, used to estimate finger width
SWAP_HANDS = {"Left": "Right", "Right": "Left"}

# Where the ring sits on the base phalanx (0 = knuckle, 1 = next joint).
RING_POSITION = 0.40
# Finger width as a fraction of the spacing between neighbouring knuckles.
WIDTH_FACTOR = 0.85

MAX_CAMERA_INDEX = 6      # camera indices probed by auto-detect and --list-cameras
STREAM_TIMEOUT_MS = 5000  # give up on an unreachable or stalled phone stream after this
MAX_FRAME_WIDTH = 1280    # larger frames (e.g. 1080p phone streams) are scaled down

PHONE_HELP = """\
Using a phone as the camera (plugged in by USB it is only a file-transfer device):
  * Wi-Fi stream: install "IP Webcam" on the phone, tap "Start server", then run
        python main.py --source http://<phone-ip>:8080/video
    (DroidCam phone app: http://<phone-ip>:4747/video). Phone and PC must be on the same Wi-Fi.
  * Webcam app: install DroidCam or Iriun on the phone AND the PC and connect them; the phone
    then shows up as a normal camera:  python main.py --list-cameras
  * Android 14+: tap the phone's USB notification and choose "Webcam", if it is offered.
If a camera exists but won't open: close other apps using it and check Windows Settings >
Privacy > Camera > "Allow desktop apps to access your camera"."""


class OneEuroFilter:
    """Speed-adaptive low-pass filter: strong smoothing when still, low lag when moving."""

    def __init__(self, min_cutoff=1.2, beta=0.02, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.x_prev is None:
            self.x_prev, self.dx_prev, self.t_prev = x, np.zeros_like(x), t
            return x
        dt = max(t - self.t_prev, 1e-3)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.linalg.norm(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmark model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def create_detector(num_hands=2):
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


def landmarks_to_px(hand_landmarks, w, h):
    """(21, 3) array: x, y in pixels, z scaled like x (relative depth)."""
    return np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in hand_landmarks], dtype=np.float64)


def finger_width(pts, finger):
    """Estimate finger width in pixels from neighbouring-knuckle spacing."""
    base = FINGERS[finger][0]
    if finger == "Thumb":
        # thumb is roughly as wide as the index-middle knuckle spacing
        spacing = np.linalg.norm(pts[5, :2] - pts[9, :2])
        return spacing * 1.0
    i = KNUCKLES.index(base)
    gaps = [np.linalg.norm(pts[KNUCKLES[j], :2] - pts[base, :2])
            for j in (i - 1, i + 1) if 0 <= j < len(KNUCKLES)]
    return float(np.mean(gaps)) * WIDTH_FACTOR


def ring_geometry(pts, finger):
    """Return ring centre, unit finger direction, unit perpendicular, half width, and
    the foreshortening ratio for the band's ellipse (0 = edge-on, 1 = face-on)."""
    mcp, pip_, _ = FINGERS[finger]
    a, b = pts[mcp], pts[pip_]
    seg2d = b[:2] - a[:2]
    seg_len2d = np.linalg.norm(seg2d)
    if seg_len2d < 1e-6:
        return None
    direction = seg2d / seg_len2d
    perp = np.array([-direction[1], direction[0]])
    centre = a[:2] + RING_POSITION * seg2d
    half_w = finger_width(pts, finger) / 2.0
    # When the finger points toward the camera the 2D segment shrinks relative to its 3D
    # length, so the ring opening becomes more visible (rounder ellipse).
    seg_len3d = np.linalg.norm(b - a)
    tilt = math.sqrt(max(0.0, 1.0 - (seg_len2d / max(seg_len3d, 1e-6)) ** 2))
    minor_ratio = float(np.clip(0.25 + 0.75 * tilt, 0.25, 1.0))
    return centre, direction, perp, half_w, minor_ratio


def draw_ring(frame, geom, color=RING_COLOR, segments=32):
    """Draw the ring purely with cv.line: an elliptical band around the finger plus the
    cross line across the finger."""
    centre, direction, perp, half_w, minor_ratio = geom
    thickness = max(2, int(half_w * 0.18))
    band = []
    for k in range(segments + 1):
        t = 2 * math.pi * k / segments
        p = centre + perp * (half_w * math.cos(t)) + direction * (half_w * minor_ratio * math.sin(t))
        band.append(tuple(int(round(v)) for v in p))
    for p1, p2 in zip(band[:-1], band[1:]):
        cv.line(frame, p1, p2, color, thickness, cv.LINE_AA)
    # straight line across the finger = the visible front of the ring
    left = tuple(int(round(v)) for v in centre - perp * half_w)
    right = tuple(int(round(v)) for v in centre + perp * half_w)
    cv.line(frame, left, right, color, thickness, cv.LINE_AA)


def draw_debug(frame, pts):
    for c in vision.HandLandmarksConnections.HAND_CONNECTIONS:
        p1 = tuple(int(v) for v in pts[c.start, :2])
        p2 = tuple(int(v) for v in pts[c.end, :2])
        cv.line(frame, p1, p2, (200, 200, 200), 1, cv.LINE_AA)
    for p in pts:
        cv.circle(frame, (int(p[0]), int(p[1])), 3, (0, 255, 0), -1, cv.LINE_AA)
    for name, (_, _, tip) in FINGERS.items():
        x, y = int(pts[tip, 0]), int(pts[tip, 1])
        cv.putText(frame, name, (x + 6, y - 6), cv.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv.LINE_AA)


def put_label(frame, text, org, scale=0.6):
    cv.putText(frame, text, org, cv.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv.LINE_AA)
    cv.putText(frame, text, org, cv.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv.LINE_AA)


class RingTracker:
    def __init__(self, num_hands=2, mirrored=True):
        self.detector = create_detector(num_hands)
        self.filters = {}  # handedness label -> OneEuroFilter over all 21 landmarks
        self.finger = "Ring"
        self.debug = True
        self.mirrored = mirrored  # whether frames passed to process() are flipped horizontally
        self._t0 = time.monotonic()
        self._last_ts = -1

    def process(self, frame):
        """frame: BGR image (mirrored for selfie view unless mirrored=False). Draws in place,
        returns info."""
        h, w = frame.shape[:2]
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        t = time.monotonic()
        ts = max(int((t - self._t0) * 1000), self._last_ts + 1)  # must strictly increase
        self._last_ts = ts
        result = self.detector.detect_for_video(mp_image, ts)

        seen, info = set(), []
        for lms, handed in zip(result.hand_landmarks, result.handedness):
            # MediaPipe labels hands assuming a mirrored (selfie) image, so the label only
            # matches the user's real hand for mirrored frames; swap it for raw frames.
            label, score = handed[0].category_name, handed[0].score
            if not self.mirrored:
                label = SWAP_HANDS.get(label, label)
            if label in seen:  # two hands with same label: keep filters stable, use raw key
                label_key = label + "_2"
            else:
                label_key = label
            seen.add(label_key)

            pts = landmarks_to_px(lms, w, h)
            f = self.filters.setdefault(label_key, OneEuroFilter())
            pts = f(pts, t)

            if self.debug:
                draw_debug(frame, pts)
            geom = ring_geometry(pts, self.finger)
            if geom is not None:
                draw_ring(frame, geom)

            wx, wy = int(pts[0, 0]), int(pts[0, 1])
            put_label(frame, f"{label} hand ({score:.2f}) - {self.finger} finger",
                      (max(5, wx - 120), min(h - 10, wy + 30)))
            info.append({"hand": label, "score": score, "finger": self.finger,
                         "ring_center": tuple(geom[0]) if geom else None})

        # Drop filters for hands that left the frame so they don't snap from old positions.
        for k in list(self.filters):
            if k not in seen:
                del self.filters[k]
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
            raise SystemExit(f"{problem}\n\n{PHONE_HELP}")
        print(f"Using camera {index} ({describe_capture(cap)})")
        return cap
    if not os.path.isfile(source):
        raise SystemExit(f"Video file not found: {source}")
    cap = cv.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video file {source}")
    return cap


def main():
    ap = argparse.ArgumentParser(description="Ring try-on: draws a ring on the selected finger.",
                                 epilog=PHONE_HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", nargs="?", const="auto", default="auto",
                    help="'auto' (default), camera index, phone stream URL or video file")
    ap.add_argument("--list-cameras", action="store_true", help="list working cameras and exit")
    ap.add_argument("--no-mirror", action="store_true",
                    help="don't flip the view (e.g. rear camera looking at the hand, video file)")
    args = ap.parse_args()

    if args.list_cameras:
        cams = list_cameras()
        for i, desc in cams:
            print(f"  camera {i}: {desc}")
        if not cams:
            print(f"No working camera found.\n\n{PHONE_HELP}")
        return

    src = open_source(args.source)
    tracker = RingTracker(mirrored=not args.no_mirror)
    fps, prev = 0.0, time.monotonic()
    try:
        while True:
            ok, frame = src.read()
            if not ok:
                print("Video source ended or stopped sending frames.")
                break
            if frame.shape[1] > MAX_FRAME_WIDTH:
                scale = MAX_FRAME_WIDTH / frame.shape[1]
                frame = cv.resize(frame, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
            if not args.no_mirror:
                frame = cv.flip(frame, 1)
            info = tracker.process(frame)

            now = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(now - prev, 1e-6)
            prev = now
            status = f"FPS {fps:4.1f} | hands {len(info)} | finger [{tracker.finger}] (1-5) | debug [d] | quit [q]"
            put_label(frame, status, (10, 25))
            if not info:
                put_label(frame, "No hand detected", (10, 55))

            cv.imshow("Ring Try-On", frame)
            key = cv.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in FINGER_KEYS:
                tracker.finger = FINGER_KEYS[key]
            elif key == ord("d"):
                tracker.debug = not tracker.debug
    finally:
        tracker.close()
        src.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
