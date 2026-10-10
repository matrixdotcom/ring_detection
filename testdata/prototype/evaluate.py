"""Run measure_finger on every test case, compare with ground truth, time it, and draw the
detected edges. Usage: python evaluate.py [tag] [param=value ...]"""
import json
import os
import sys
import time

import cv2 as cv
import numpy as np

from finger_measure import measure_finger
from gt_view import D, P, S, load_cases

tag = sys.argv[1] if len(sys.argv) > 1 and "=" not in sys.argv[1] else "eval"
params = {}
for arg in sys.argv[1:]:
    if "=" in arg:
        k, v = arg.split("=")
        params[k] = int(v) if k in ("n_slopes", "n_candidates") else float(v)
cases = load_cases()
gt = json.load(open(os.path.join(D, "gt.json")))
ORDER = ["c5", "c6", "c7", "c1", "c2", "c3", "rh0", "rh1", "lh0", "lh1"]
cv.setNumThreads(1)  # be conservative: the live app also runs MediaPipe


def draw_case(img, res, g, cid):
    d = res["debug"]
    c, a, n, W0, s, cx, cy = d["c"], d["a"], d["n"], d["W0"], d["s"], d["cx"], d["cy"]
    # (A) original-orientation crop
    R = 1.5 * W0
    x0, y0 = int(c[0] - R), int(c[1] - R)
    size = int(2 * R)
    big = 380 / size
    crop = cv.copyMakeBorder(img, size, size, size, size, cv.BORDER_CONSTANT, value=(40, 40, 40))
    crop = crop[y0 + size:y0 + 2 * size, x0 + size:x0 + 2 * size]
    crop = cv.resize(crop, (380, 380), interpolation=cv.INTER_LINEAR)

    def P2(q):
        return (int(round((q[0] - x0) * big * 8)), int(round((q[1] - y0) * big * 8)))

    L2 = d["L2"]
    for sgn in (-1, 1):  # prior edges (blue)
        q0 = c + sgn * W0 / 2 * n - 0.2 * L2 * a
        q1 = c + sgn * W0 / 2 * n + 0.2 * L2 * a
        cv.line(crop, P2(q0), P2(q1), (255, 120, 0), 1, cv.LINE_AA, 3)
    cv.line(crop, P2(c - 0.45 * L2 * a), P2(c + 0.35 * L2 * a), (0, 200, 0), 1, cv.LINE_AA, 3)
    for e in (d["eL"], d["eR"]):
        xs, ys = e["pts"]
        inl = e.get("inliers", np.ones(len(xs), bool))
        for xv, yv, ok in zip(xs, ys, inl):
            q = c + (xv - cx) / s * n + yv / s * a
            cv.circle(crop, P2(q), 8 * 2, (0, 255, 255) if ok else (0, 120, 255), -1, cv.LINE_AA, 3)
        yy = np.array([-1.0, 1.0]) * 0.17 * L2 * s
        q0 = c + (e["j0"] + e["k"] * yy[0] - cx) / s * n + yy[0] / s * a
        q1 = c + (e["j0"] + e["k"] * yy[1] - cx) / s * n + yy[1] / s * a
        cv.line(crop, P2(q0), P2(q1), (0, 0, 255), 2, cv.LINE_AA, 3)
    cm = np.array(res["centre_px"])
    ax = np.array(res["axis_dir"])
    cv.line(crop, P2(cm - 0.2 * L2 * ax), P2(cm + 0.2 * L2 * ax), (255, 0, 255), 1, cv.LINE_AA, 3)
    for side in ("left", "right"):  # GT at ring row (cyan crosses)
        q = c + g[side] * n
        pp = P2(q)
        pp = (pp[0] // 8, pp[1] // 8)
        cv.drawMarker(crop, pp, (255, 255, 0), cv.MARKER_CROSS, 14, 2)
    # (B) straightened patch
    patch = d["patch"]
    up = 380 / patch.shape[1]
    pv = cv.resize(patch, (380, int(patch.shape[0] * up)), interpolation=cv.INTER_NEAREST)
    hpv = pv.shape[0]
    for j in d["cand"]:
        cv.line(pv, (int(j * up), 0), (int(j * up), 6), (200, 200, 200), 1)
    for e in (d["eL"], d["eR"]):
        xs, ys = e["pts"]
        for xv, yv in zip(xs, ys):
            cv.circle(pv, (int((xv + 0.5) * up), int((yv + cy + 0.5) * up)), 2, (0, 255, 255), -1)
        for yv in (-cy, cy):
            pass
        cv.line(pv, (int((e["j0"] - e["k"] * cy + 0.5) * up), 0),
                (int((e["j0"] + e["k"] * cy + 0.5) * up), hpv - 1), (0, 0, 255), 1, cv.LINE_AA)
    for side in ("left", "right"):
        xg = int((g[side] * s + cx + 0.5) * up)
        cv.line(pv, (xg, hpv - 8), (xg, hpv - 1), (255, 255, 0), 2)
    # (C) evidence profile
    ph = 90
    plot = np.full((ph, 380, 3), 255, np.uint8)
    snr = d["snr"]
    mx = max(snr.max(), 1e-6)
    pts_ = np.stack([(np.arange(len(snr)) + 0.5) * up, ph - 5 - snr / mx * (ph - 15)], 1).astype(np.int32)
    cv.polylines(plot, [pts_], False, (60, 60, 60), 1, cv.LINE_AA)
    for side in ("left", "right"):
        xg = int((g[side] * s + cx + 0.5) * up)
        cv.line(plot, (xg, 0), (xg, ph), (255, 200, 0), 1)
    for j in (d["jl"], d["jr"]):
        cv.line(plot, (int((j + 0.5) * up), 0), (int((j + 0.5) * up), ph), (0, 0, 255), 1)
    cv.line(plot, (int((cx + 0.5) * up), 0), (int((cx + 0.5) * up), ph), (0, 160, 0), 1)
    cv.putText(plot, f"max SNR {mx:.1f}", (4, 12), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    right = np.vstack([pv, plot])
    hh = max(right.shape[0], 380)
    right = cv.copyMakeBorder(right, 0, hh - right.shape[0], 0, 0, cv.BORDER_CONSTANT, value=(255, 255, 255))
    left = cv.copyMakeBorder(crop, 0, hh - 380, 0, 0, cv.BORDER_CONSTANT, value=(255, 255, 255))
    tile = np.hstack([left, np.full((hh, 6, 3), 255, np.uint8), right])
    werr = (res["width_px"] - g["width"]) / g["width"] * 100
    perr = (W0 - g["width"]) / g["width"] * 100
    txt = (f"{cid}: W {res['width_px']:.1f} (GT {g['width']:.1f}+-{g['unc_width']:.0f}, err {werr:+.0f}%; prior "
           f"{W0:.1f} {perr:+.0f}%) ctr {res['centre_offset_px']:+.1f} (GT {g['centre']:+.1f}) "
           f"ang {res['axis_angle_deg']:+.1f} conf {res['confidence']:.2f} {res['edge_kinds']}")
    bar = np.full((22, tile.shape[1], 3), 255, np.uint8)
    cv.putText(bar, txt, (4, 15), cv.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv.LINE_AA)
    return np.vstack([bar, tile])


rows, tiles, times = [], [], {}
for cid in ORDER:
    path, pts = cases[cid]
    img = cv.imread(f"{S}/{path}")
    res = measure_finger(img, pts, params=params, debug=True)
    # timing (no debug), 30 reps
    ts = []
    for _ in range(30):
        t0 = time.perf_counter()
        measure_finger(img, pts, params=params)
        ts.append((time.perf_counter() - t0) * 1000)
    times[cid] = float(np.median(ts))
    g = gt[cid]
    if not res["ok"]:
        print(cid, "FAILED:", res.get("reason"))
        continue
    werr = res["width_px"] - g["width"]
    cerr = res["centre_offset_px"] - g["centre"]
    lerr = res["left_offset_px"] - g["left"]
    rerr = res["right_offset_px"] - g["right"]
    pe = res["prior_width_px"] - g["width"]
    ok_w = abs(werr) <= max(0.10 * g["width"], g["unc_width"])
    ok_e = abs(lerr) <= max(0.06 * g["width"], g["unc_left"] + 1) and abs(rerr) <= max(0.06 * g["width"], g["unc_right"] + 1)
    rows.append(dict(case=cid, width=round(res["width_px"], 1), gt_width=g["width"], gt_unc=g["unc_width"],
                     width_err_pct=round(100 * werr / g["width"], 1), prior_err_pct=round(100 * pe / g["width"], 1),
                     left=round(res["left_offset_px"], 1), gt_left=g["left"], right=round(res["right_offset_px"], 1),
                     gt_right=g["right"], centre=round(res["centre_offset_px"], 1), gt_centre=g["centre"],
                     centre_err=round(cerr, 1), angle=round(res["axis_angle_deg"], 1),
                     conf=round(res["confidence"], 2), kinds=res["edge_kinds"], correct=bool(ok_w and ok_e),
                     ms=round(times[cid], 2),
                     parts={k: round(v, 2) for k, v in res["conf_parts"].items()}))
    r = rows[-1]
    print(f"{cid:4s} W {r['width']:6.1f} GT {r['gt_width']:5.1f}+-{r['gt_unc']:3.1f} err {r['width_err_pct']:+6.1f}% "
          f"(prior {r['prior_err_pct']:+6.1f}%) L {r['left']:+6.1f}/{r['gt_left']:+6.1f} R {r['right']:+6.1f}/{r['gt_right']:+6.1f} "
          f"ang {r['angle']:+5.1f} conf {r['conf']:.2f} {'OK ' if r['correct'] else 'BAD'} {r['kinds']} {r['ms']:.2f}ms {r['parts']}")
    tiles.append(draw_case(img, res, g, cid))
W = max(t.shape[1] for t in tiles)
tiles = [cv.copyMakeBorder(t, 0, 8, 0, W - t.shape[1], cv.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles]
half = (len(tiles) + 1) // 2
col1, col2 = np.vstack(tiles[:half]), np.vstack(tiles[half:])
col2 = cv.copyMakeBorder(col2, 0, col1.shape[0] - col2.shape[0], 0, 0, cv.BORDER_CONSTANT, value=(255, 255, 255))
cv.imwrite(f"{P}/{tag}_sheet.jpg", np.hstack([col1, np.full((col1.shape[0], 10, 3), 255, np.uint8), col2]),
           [cv.IMWRITE_JPEG_QUALITY, 90])
for i, t in enumerate(tiles):
    cv.imwrite(f"{P}/{tag}_{ORDER[i]}.png", t)
json.dump(rows, open(f"{P}/{tag}_results.json", "w"), indent=1)
n_ok = sum(r["correct"] for r in rows)
print(f"correct {n_ok}/{len(rows)}; median runtime {np.median(list(times.values())):.2f} ms, max {max(times.values()):.2f} ms")
