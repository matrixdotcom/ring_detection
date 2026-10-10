"""Simulated 'hand moves toward / away from the camera': zoom a frame about the hand centre
(0.6x -> 1.2x -> 0.6x, 40 frames), run the real RingTracker frame by frame (VIDEO mode, One Euro
smoothing, like the live app), then measure the finger. Compares measured width and the
landmark prior with the true width (GT x zoom)."""
import json
import os
import sys
import time

import cv2 as cv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # ring_detection/ (main.py)
import main  # noqa: E402
from finger_measure import measure_finger  # noqa: E402

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # testdata/
S = os.path.join(D, "frames")      # images
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")   # outputs
gt = json.load(open(os.path.join(D, "gt.json")))
lm = json.load(open(os.path.join(D, "landmarks.json")))
zs = np.concatenate([np.linspace(0.6, 1.2, 20), np.linspace(1.2, 0.6, 20)])
report = {}
for cid, img_name in (("c2", "clean2.png"), ("c7", "clean7.png"), ("rh0", "hands/right_hands.jpg")):
    img = cv.imread(f"{S}/{img_name}")
    h, w = img.shape[:2]
    pts0 = np.array(lm[cid]["pts"])
    centre = pts0[:, :2].mean(0)
    tr = main.RingTracker(num_hands=2 if cid == "rh0" else 1, mirrored=cid != "rh0")
    rows = []
    for z in zs:
        M = np.array([[z, 0, (1 - z) * centre[0]], [0, z, (1 - z) * centre[1]]])
        fr = cv.warpAffine(img, M, (w, h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT,
                           borderValue=(128, 128, 128))
        info = tr.process(fr.copy())
        time.sleep(0.01)
        if not info:
            rows.append((z, None, None, None))
            continue
        # the tracked hand nearest to the zoomed wrist of the reference hand
        ref = M[:, :2] @ pts0[0, :2] + M[:, 2]
        hand = min(info, key=lambda i: np.linalg.norm(np.array(i["landmarks_px"][0][:2]) - ref))
        pts = np.array(hand["landmarks_px"])
        r = measure_finger(fr, pts)
        rows.append((z, r["width_px"] if r["ok"] else None, r["prior_width_px"], r["confidence"] if r["ok"] else 0.0))
    tr.close()
    G = gt[cid]["width"]
    meas = np.array([(r[1] / (G * r[0])) if r[1] else np.nan for r in rows])
    prior = np.array([(r[2] / (G * r[0])) if r[2] else np.nan for r in rows])
    conf = np.array([r[3] if r[3] is not None else np.nan for r in rows])
    report[cid] = dict(z=zs.round(3).tolist(), measured_over_true=np.round(meas, 3).tolist(),
                       prior_over_true=np.round(prior, 3).tolist(), conf=np.round(conf, 2).tolist())
    ok = np.isfinite(meas)
    print(f"{cid}: frames {len(rows)}, measured on {ok.sum()}; measured/true: median {np.nanmedian(meas):.3f}, "
          f"p10-p90 {np.nanpercentile(meas, 10):.3f}-{np.nanpercentile(meas, 90):.3f}, "
          f"|err|<=10% on {np.mean(np.abs(meas[ok] - 1) <= 0.10) * 100:.0f}% | prior/true median {np.nanmedian(prior):.3f} "
          f"p10-p90 {np.nanpercentile(prior, 10):.3f}-{np.nanpercentile(prior, 90):.3f}")
    print("   z      meas/true  prior/true  conf")
    for z, mm, pp, cc in zip(zs[::4], meas[::4], prior[::4], conf[::4]):
        print(f"   {z:.2f}   {mm:8.3f}   {pp:8.3f}   {cc:.2f}")
json.dump(report, open(f"{P}/zoom_sequence.json", "w"))
