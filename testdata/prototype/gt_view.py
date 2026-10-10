"""Straightened, ruler-annotated views of the ring finger for manual ground-truth annotation.
x = offset across the landmark axis (px of the original frame, + = n direction),
y = offset along the axis from the ring centre (px, + = toward PIP).
Also prints luma / chroma profiles along a few rows."""
import json
import os
import sys

import cv2 as cv
import numpy as np

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # testdata/
S = os.path.join(D, "frames")      # images
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")   # outputs


def load_cases():
    lm = json.load(open(os.path.join(D, "landmarks.json")))
    cases = {}
    for cid, v in lm.items():
        pts = np.array(v["pts"], float)
        cases[cid] = (v["image"], pts)
    return cases


def frame_geom(pts, base=13, nxt=14, position=0.69):
    knuckles = pts[[5, 9, 13, 17]]
    unit = float(np.linalg.norm(np.diff(knuckles, axis=0), axis=1).mean())
    b2 = pts[nxt, :2] - pts[base, :2]
    L2 = float(np.linalg.norm(b2))
    a = b2 / L2
    n = np.array([a[1], -a[0]])
    c = pts[base, :2] + position * b2
    return c, a, n, L2, unit


def straighten(img, c, a, n, x_range, y_range, scale):
    """patch[j, i] = img(c + x_i n + y_j a), x_i = x_range[0] + i/scale, y_j likewise."""
    (x0, x1), (y0, y1) = x_range, y_range
    W, H = int(round((x1 - x0) * scale)), int(round((y1 - y0) * scale))
    # dst (i, j) -> src = c + (x0 + i/scale) n + (y0 + j/scale) a  (WARP_INVERSE_MAP)
    M = np.array([[n[0] / scale, a[0] / scale, c[0] + x0 * n[0] + y0 * a[0]],
                  [n[1] / scale, a[1] / scale, c[1] + x0 * n[1] + y0 * a[1]]])
    return cv.warpAffine(img, M, (W, H), flags=cv.INTER_LINEAR | cv.WARP_INVERSE_MAP,
                         borderMode=cv.BORDER_REPLICATE)


if __name__ == "__main__":
    cases = load_cases()
    which = sys.argv[1:] or list(cases)
    for cid in which:
        path, pts = cases[cid]
        img = cv.imread(f"{S}/{path}")
        c, a, n, L2, unit = frame_geom(pts)
        W0 = 2 * 0.43 * unit
        half = max(1.6 * W0, 45)
        scale = float(np.clip(560 / (2 * half), 2, 8))
        yr = (-0.45 * L2, 0.35 * L2)
        patch = straighten(img, c, a, n, (-half, half), yr, scale)
        # Contrast stretch for viewing
        lab = cv.cvtColor(patch, cv.COLOR_BGR2LAB)
        lab[..., 0] = cv.createCLAHE(2.0, (4, 4)).apply(lab[..., 0])
        view = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
        H, W = view.shape[:2]
        top = 30
        canvas = cv.copyMakeBorder(view, top, 30, 0, 0, cv.BORDER_CONSTANT, value=(255, 255, 255))
        for xv in range(int(-half) // 5 * 5, int(half) + 1, 5):
            xi = int(round((xv + half) * scale))
            ln = 12 if xv % 10 == 0 else 6
            col = (0, 0, 255) if xv == 0 else (0, 0, 0)
            cv.line(canvas, (xi, top - ln), (xi, top), col, 1)
            cv.line(canvas, (xi, top + H), (xi, top + H + ln), col, 1)
            if xv % 20 == 0:
                cv.putText(canvas, str(xv), (xi - 10, 12), cv.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        for yv in (-0.15 * L2, 0.0, 0.15 * L2):
            yi = top + int(round((yv - yr[0]) * scale))
            cv.line(canvas, (0, yi), (8, yi), (0, 0, 255), 2)
            cv.line(canvas, (W - 8, yi), (W, yi), (0, 0, 255), 2)
        for xv in (-W0 / 2, W0 / 2):  # model prior width
            xi = int(round((xv + half) * scale))
            cv.line(canvas, (xi, top + H), (xi, top + H + 25), (255, 0, 0), 2)
        cv.imwrite(f"{P}/gt_{cid}.png", canvas)
        print(f"{cid}: L2={L2:.1f} unit={unit:.1f} W0={W0:.1f} half={half:.1f} scale={scale:.2f} "
              f"a={a.round(3)} n={n.round(3)} c={c.round(1)}")
