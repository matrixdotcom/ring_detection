"""
Measure a finger's real outline at the ring position from the image (edge snapping).

    fit = measure_finger(frame_bgr, pts, base, nxt, position, prior_width)
    fit["ok"], fit["width_px"], fit["centre_offset_px"], fit["axis_angle_deg"], fit["confidence"]

pts are the 21 hand landmarks in pixels (x, y and, if available, z away from the camera); base/nxt
are the finger's first two joints (e.g. 13, 14 for the ring finger); prior_width is the width the
landmarks predict, used for scale and as a weak prior. numpy + OpenCV only, ~3 ms per call.

  1. Straighten a patch around the ring position: columns across the landmark axis (+-1.15 W0),
     rows along it (+-0.15 of the 2D bone length), NS samples per W0.
  2. Features in CIE Lab, Gaussian derivatives across the finger: signed first derivative of
     L, a*, b* (step edges: finger vs shirt / wall / gap) and the luma second derivative
     (valley = dark crease between touching fingers, ridge = bright background gap).
  3. Line evidence for a common slope k of both edges (x = x0 + k*y, |k| <= 0.2): features are
     sheared with cv.remap so lines become columns, then a trimmed mean along each column of the
     SIGNED features: background texture (random signs) cancels, edges with a consistent
     polarity along the finger survive. Normalised by the patch's median gradient (SNR units).
  4. Candidate edges = peaks of the evidence per slope; every pair (left < right) is scored:
     evidence(L) + evidence(R) - heavy-tailed width prior (log width/W0) - centre prior
     - slope prior - penalty for a strong edge between them (pair spans two fingers)
     - penalty when the inner side of a step edge looks less like the finger's centre than its
       outer side does (edge of something else, e.g. the far side of a gap).
  5. Per-row refinement near each chosen line (extremum of the same feature, sub-pixel),
     robust line fit per edge (own slope) -> width perpendicular to the measured axis, centre
     offset, axis angle and a confidence.

Offsets are measured along n = (a_y, -a_x), where a is the 2D direction base -> nxt; this is the
same direction as `u` in main.ring_pose. A positive axis angle turns the axis from a toward n.
"""
import math

import cv2 as cv
import numpy as np

KINDS = ("step", "valley", "ridge")

DEFAULTS = dict(
    ns=40.0,              # samples per predicted finger width W0
    half_across=1.15,     # patch half-width across the finger, in W0
    half_along=0.15,      # patch half-length along the bone, in 2D bone lengths
    sigma_edge=0.05,      # first-derivative scale, in W0
    sigma_valley=0.04,    # second-derivative scale (creases / gaps), in W0
    sigma_along=1.0,      # smoothing along the finger, samples
    chroma_weight=1.5,    # weight of a*/b* steps relative to L
    valley_weight=1.3,    # weight of crease/gap evidence vs steps (crease centre beats its flanks)
    trim=0.2,             # trimmed mean along a line: drop this share at each end
    max_slope=0.2,        # common edge slope vs the landmark axis: atan(0.2) = 11.3 deg
    n_slopes=9,
    w_range=(0.45, 1.9),  # allowed width / W0
    sigma_logw=0.25,      # width prior scale (Student-t, nu=2, on log(width/W0))
    sigma_centre=0.25,    # centre prior: offset/W0 ~ N(0, sigma)
    sigma_slope=0.12,     # slope prior
    lam_prior=2.0,        # weight of the priors, in SNR units
    lam_inside=1.5,       # penalty per SNR unit of an edge inside the pair (above inside_free)
    inside_free=1.0,
    lam_side=2.0,
    lam_prev=3.0,         # bonus for the edge pair the ring is already on (hysteresis against flip-flop)
    crease_guard=0.18,    # a step this close (in W0) inside a crease edge is the crease's own shading
    n_candidates=8,
    max_bend_deg=45.0,    # gate: next joint bent more than this -> finger folded, not a straight strip
    max_tilt_deg=45.0,    # gate: bone tilted toward/away from the camera more than this
)

_KCACHE = {}


def _kernels(sigma):
    key = round(sigma, 3)
    if key in _KCACHE:
        return _KCACHE[key]
    r = max(2, int(math.ceil(3.5 * sigma)))
    t = np.arange(-r, r + 1, dtype=np.float64)
    g = np.exp(-0.5 * (t / sigma) ** 2)
    g /= g.sum()
    d1 = t * g                       # correlation kernel -> +1 response to a unit ramp
    d1 /= (t * d1).sum()
    d2 = (t ** 2 / sigma ** 4 - 1 / sigma ** 2) * g
    d2 -= d2.mean()
    d2 /= (t ** 2 * d2).sum() / 2    # -> +1 response to x^2/2
    _KCACHE[key] = tuple(k.astype(np.float32) for k in (g, d1, d2))
    return _KCACHE[key]


def _straighten(img, c, a, n, half_x, half_y, s):
    """Patch with rows along a, columns along n, s samples per image pixel (pre-blurred when
    downsampling). Returns patch, column and row index of the ring centre, share inside frame."""
    W = int(round(2 * half_x * s)) | 1
    H = int(round(2 * half_y * s)) | 1
    cx, cy = (W - 1) / 2, (H - 1) / 2
    corners = np.array([c + sx * half_x * n + sy * half_y * a for sx in (-1, 1) for sy in (-1, 1)])
    blur = 0.5 / s if s < 0.9 else 0.0
    m = int(math.ceil(3 * blur)) + 3
    x0, y0 = np.floor(corners.min(0)).astype(int) - m
    x1, y1 = np.ceil(corners.max(0)).astype(int) + m
    h, w = img.shape[:2]
    X0, Y0, X1, Y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    if X1 - X0 < 4 or Y1 - Y0 < 4:
        return None, cx, cy, 0.0
    inside = (X1 - X0) * (Y1 - Y0) / max((x1 - x0) * (y1 - y0), 1)
    roi = img[Y0:Y1, X0:X1]
    if blur > 0:
        roi = cv.GaussianBlur(roi, (0, 0), blur)
    o = c - (X0, Y0)
    M = np.array([[n[0] / s, a[0] / s, o[0] - cx * n[0] / s - cy * a[0] / s],
                  [n[1] / s, a[1] / s, o[1] - cx * n[1] / s - cy * a[1] / s]])
    patch = cv.warpAffine(roi, M, (W, H), flags=cv.INTER_LINEAR | cv.WARP_INVERSE_MAP,
                          borderMode=cv.BORDER_REPLICATE)
    return patch, cx, cy, inside


_MCACHE = {}


def _shear_maps(H, W, slopes, rows_y):
    key = (H, W, slopes.tobytes())
    if key not in _MCACHE:
        K = len(slopes)
        mapx = (np.arange(W, dtype=np.float32)[None, None, :]
                + slopes[:, None, None] * rows_y[None, :, None]).reshape(K * H, W).astype(np.float32)
        mapy = np.repeat(np.tile(np.arange(H, dtype=np.float32), K)[:, None], W, 1)
        if len(_MCACHE) > 64:
            _MCACHE.clear()
        _MCACHE[key] = cv.convertMaps(mapx, mapy, cv.CV_16SC2)   # fixed-point maps: faster remap
    return _MCACHE[key]


def _shear(F, maps, K):
    """F (H, W, C) float32 -> (K, H, W, C): line x = x0 + k*y of slope k becomes column x0."""
    H, W = F.shape[:2]
    out = cv.remap(F, maps[0], maps[1], cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)
    return out.reshape(K, H, W, -1)


def _trimmed_mean(X, trim, blocks=7):
    """Robust mean along axis 1: average rows in ~`blocks` consecutive blocks, sort the block
    means and drop `trim` of them at each end (cheap stand-in for a trimmed mean)."""
    n = X.shape[1]
    nb = min(blocks, n)
    edges = np.linspace(0, n, nb + 1).astype(int)
    B = np.add.reduceat(X, edges[:-1], axis=1) / np.diff(edges).reshape(1, -1, *([1] * (X.ndim - 2)))
    cut = int(nb * trim + 0.5)
    if cut == 0 or nb - 2 * cut < 2:
        return B.mean(1)
    B.sort(axis=1)
    return B[:, cut:nb - cut].mean(1)


def measure_finger(frame, pts, base, nxt, position, prior_width, prev_edges=None, params=None, debug=False):
    """frame: BGR uint8 (not drawn on yet); pts: (21, 2+) landmarks in pixels; prev_edges: the
    (left, right) edge offsets in px along n where the ring currently sits, if any - near-ties then go
    to that pair, so the fit doesn't flip between similar edges on ambiguous frames. Returns a dict
    with "ok" and, when ok: width_px, centre_offset_px, axis_angle_deg, confidence, conf_left/right,
    edge_kinds, edges_img (both edges as image lines) and left/right_offset_px."""
    p = DEFAULTS if not params else dict(DEFAULTS, **params)
    pts = np.asarray(pts, np.float64)
    bone = pts[nxt, :2] - pts[base, :2]
    L2 = float(np.linalg.norm(bone))
    W0 = float(prior_width)
    out = {"ok": False, "prior_width_px": W0, "bone_len_px": L2, "confidence": 0.0}
    if W0 < 8 or L2 < 0.4 * W0:
        out["reason"] = "finger too small or pointing at the camera"
        return out
    if pts.shape[1] >= 3:   # 3D gates (z from main.hand_points, pixels away from the camera)
        b3, d3 = pts[nxt] - pts[base], pts[nxt + 1] - pts[nxt]
        bend = math.degrees(math.acos(np.clip(b3 @ d3 / (np.linalg.norm(b3) * np.linalg.norm(d3) + 1e-9), -1, 1)))
        tilt = math.degrees(math.asin(min(1.0, abs(b3[2]) / (np.linalg.norm(b3) + 1e-9))))
        out.update(bend_deg=bend, tilt_deg=tilt)
        if bend > p["max_bend_deg"] or tilt > p["max_tilt_deg"]:
            out["reason"] = f"finger bent ({bend:.0f} deg) or tilted ({tilt:.0f} deg): not a straight strip"
            return out
    a = bone / L2
    n = np.array([a[1], -a[0]])
    c = pts[base, :2] + position * bone
    ns = p["ns"]
    s = ns / W0                                    # samples per image px
    patch, cx, cy, inside_frac = _straighten(frame, c, a, n, p["half_across"] * W0, p["half_along"] * L2, s)
    if patch is None or patch.shape[0] < 5 or inside_frac < 0.7:
        out["reason"] = "ring region outside the frame"
        return out
    H, W = patch.shape[:2]
    lab = cv.cvtColor(patch.astype(np.float32) * np.float32(1 / 255.0), cv.COLOR_BGR2Lab)
    g_y, _, _ = _kernels(p["sigma_along"])
    _, d1, _ = _kernels(p["sigma_edge"] * ns)
    _, _, d2 = _kernels(p["sigma_valley"] * ns)
    se, sv, cw = p["sigma_edge"] * ns, p["sigma_valley"] * ns, p["chroma_weight"]
    G = cv.sepFilter2D(lab, cv.CV_32F, d1 * np.float32(se), g_y, borderType=cv.BORDER_REPLICATE)
    G[..., 1:] *= np.float32(cw)
    V = cv.sepFilter2D(np.ascontiguousarray(lab[..., 0]), cv.CV_32F, d2 * np.float32(sv * sv), g_y,
                       borderType=cv.BORDER_REPLICATE)
    noise = float(np.median(np.sqrt((G * G).sum(-1)))) + 0.15   # texture/noise level (+ floor)

    rows_y = (np.arange(H) - cy).astype(np.float32)
    slopes = np.linspace(-p["max_slope"], p["max_slope"], p["n_slopes"]).astype(np.float32)
    K = len(slopes)
    maps = _shear_maps(H, W, slopes, rows_y)
    Fk = _shear(np.dstack([G, V[..., None]]), maps, K)               # (K, H, W, 4)
    m = _trimmed_mean(Fk, p["trim"])                                 # (K, W, 4)
    colour = lab.mean(0) * np.float32([0.5, 1, 1])                   # (W, 3) column colour; luma half weight
    S_all = np.stack([np.sqrt((m[..., :3] ** 2).sum(-1)),
                      p["valley_weight"] * np.maximum(m[..., 3], 0),
                      p["valley_weight"] * np.maximum(-m[..., 3], 0)])  # (3, K, W)
    kind = S_all.argmax(0)
    snr = S_all.max(0) / noise                                       # (K, W)
    any_slope = snr.max(0)                                           # (W,)

    # ---- candidate peaks per slope (NMS), all pairs, vectorised scoring ----
    nms = max(1, int(round(0.06 * ns)))
    is_peak = np.zeros_like(snr, bool)
    is_peak[:, 1:-1] = (snr[:, 1:-1] >= snr[:, :-2]) & (snr[:, 1:-1] > snr[:, 2:])
    # NMS: a peak survives if no stronger peak of the same slope lies within nms samples
    pk_k, pk_j = np.nonzero(is_peak)
    val = snr[pk_k, pk_j]
    same = (pk_k[:, None] == pk_k[None, :]) & (np.abs(pk_j[:, None] - pk_j[None, :]) <= nms)
    stronger = same & ((val[None, :] > val[:, None]) | ((val[None, :] == val[:, None]) & (pk_j[None, :] < pk_j[:, None])))
    alive = ~stronger.any(1)
    pk_k, pk_j, val = pk_k[alive], pk_j[alive], val[alive]
    # keep the n_candidates strongest per slope
    order = np.lexsort((-val, pk_k))
    pk_k, pk_j, val = pk_k[order], pk_j[order], val[order]
    first = np.searchsorted(pk_k, pk_k)
    rank = np.arange(len(pk_k)) - first
    sel = rank < p["n_candidates"]
    cand_k, cand_j = pk_k[sel], pk_j[sel]
    if len(cand_j) < 2:
        out["reason"] = "fewer than two edge candidates"
        return out
    ii, jj = np.nonzero((cand_k[:, None] == cand_k[None, :]) & (cand_j[:, None] < cand_j[None, :]))
    kp, jl, jr = cand_k[ii], cand_j[ii], cand_j[jj]
    w = (jr - jl) / ns
    keep = (w >= p["w_range"][0]) & (w <= p["w_range"][1])
    kp, jl, jr, w = kp[keep], jl[keep], jr[keep], w[keep]
    if len(w) == 0:
        out["reason"] = "no edge pair with a plausible width"
        return out
    mo = ((jl + jr) / 2 - cx) / ns
    lw = p["sigma_logw"]
    prior = (1.5 * np.log1p(np.log(w) ** 2 / (2 * lw * lw))          # Student-t (nu=2) on log width
             + 0.5 * (mo / p["sigma_centre"]) ** 2 + 0.5 * (slopes[kp] / p["sigma_slope"]) ** 2)
    ev = snr[kp, jl] + snr[kp, jr]
    # strongest interior edge (any slope), away from the two edges. A crease's shaded flank (a step
    # just inside a valley edge) belongs to the crease, not to a second finger between the edges.
    pk = np.flatnonzero(is_peak.any(0))
    guard = max(nms, int(round(p["crease_guard"] * ns)))
    gl, gr = np.where(kind[kp, jl] == 1, guard, nms), np.where(kind[kp, jr] == 1, guard, nms)
    inner_mask = (pk[None, :] > (jl + gl)[:, None]) & (pk[None, :] < (jr - gr)[:, None])
    inner = np.where(inner_mask, any_slope[pk][None, :], 0.0).max(1) if len(pk) else np.zeros(len(w))
    inside = np.maximum(inner - p["inside_free"], 0.0)
    # side consistency of step edges vs the colour of the central 30% of the pair
    cum = np.concatenate([np.zeros((1, 3), np.float32), np.cumsum(colour, 0)], 0)

    def span(a0, a1):   # mean column colour over [a0, a1) (sampled at slope 0: cheap approximation)
        a0 = np.clip(a0, 0, W - 1)
        a1 = np.clip(a1, a0 + 1, W)
        return (cum[a1] - cum[a0]) / (a1 - a0)[:, None]

    half_c = np.maximum(1, ((jr - jl) * 0.15).astype(int))
    mid = (jl + jr) // 2
    centre_col = span(mid - half_c, mid + half_c + 1)
    sw, sg = max(2, int(round(0.10 * ns))), max(1, int(round(0.04 * ns)))
    side = np.zeros(len(w))
    for j, sgn in ((jl, 1), (jr, -1)):
        inner_c = span(j + sg, j + sg + sw) if sgn > 0 else span(j - sg - sw + 1, j - sg + 1)
        outer_c = span(j - sg - sw + 1, j - sg + 1) if sgn > 0 else span(j + sg, j + sg + sw)
        di = np.linalg.norm(inner_c - centre_col, axis=1)
        do = np.linalg.norm(outer_c - centre_col, axis=1)
        bad = (kind[kp, j] == 0) & (di > do)
        side += np.where(bad, np.minimum(1.0, 4 * (di - do) / np.maximum(di + do, 1e-6)), 0.0)
    score = ev - p["lam_prior"] * prior - p["lam_inside"] * inside - p["lam_side"] * side
    if prev_edges is not None:   # near-ties keep the pair the ring is already on: no flip-flop
        pl, pr = cx + prev_edges[0] * s, cx + prev_edges[1] * s
        tp = 0.08 * ns
        score = score + p["lam_prev"] * np.exp(-0.5 * (((jl - pl) / tp) ** 2 + ((jr - pr) / tp) ** 2))
    b = int(np.argmax(score))
    kb, jlb, jrb = int(kp[b]), int(jl[b]), int(jr[b])
    k0 = float(slopes[kb])

    # ---- per-row refinement of each chosen line (vectorised), robust line fit ----
    rad = max(2, int(round(0.06 * ns)))
    offs = np.arange(-rad, rad + 1)
    rows = np.arange(H)

    def refine(j):
        t = int(kind[kb, j])
        if t == 0:   # step: project the colour gradient on the line's mean gradient direction
            dv = m[kb, j, :3] / (np.linalg.norm(m[kb, j, :3]) + 1e-9)
            Fm = G @ dv.astype(np.float32)
        else:
            Fm = V if t == 1 else -V
        cols = np.clip(np.rint(j + k0 * rows_y).astype(int)[:, None] + offs[None, :], 1, W - 2)
        vals = Fm[rows[:, None], cols]
        q = vals.argmax(1)
        jc = cols[rows, q]
        y0, y1, y2 = Fm[rows, jc - 1], Fm[rows, jc], Fm[rows, jc + 1]
        den = y0 - 2 * y1 + y2
        off = np.where(den < 0, 0.5 * (y0 - y2) / np.where(den < 0, den, -1.0), 0.0)
        xs = jc + np.clip(off, -0.5, 0.5)
        good = (y1 > 0) & (q > 0) & (q < 2 * rad)                    # a real peak inside the window
        ys = rows_y.astype(np.float64)
        res = dict(j0=float(j), k=k0, rms=np.inf, inl=0.0, pts=(xs, ys), inliers=good, kind=t)
        if good.sum() < 4:
            return res
        wts = np.sqrt(np.maximum(y1, 1e-3))
        inl = good
        for _ in range(3):
            wq, yq, xq = wts[inl] ** 2, ys[inl], xs[inl]
            sw_ = wq.sum(); ym = (wq * yq).sum() / sw_; xm = (wq * xq).sum() / sw_
            vy = (wq * (yq - ym) ** 2).sum()
            k1 = (wq * (yq - ym) * (xq - xm)).sum() / vy if vy > 1e-9 else k0
            k1 = min(max(k1, k0 - 0.12), k0 + 0.12)
            coef = (xm - k1 * ym, k1)
            r = xs - (coef[0] + coef[1] * ys)
            mad = float(np.median(np.abs(r[inl]))) + 0.25
            new = good & (np.abs(r) < max(3.0 * 1.4826 * mad, 0.04 * ns))
            if new.sum() < 4 or (new == inl).all():
                break
            inl = new
        res.update(j0=float(coef[0]), k=float(coef[1]), rms=float(np.sqrt(np.mean(r[inl] ** 2))),
                   inl=float(inl.mean()), inliers=inl)
        return res

    eL, eR = refine(jlb), refine(jrb)
    if not eL["j0"] < eR["j0"]:
        out["reason"] = "refinement crossed the edges"
        return out
    xl, xr = (eL["j0"] - cx) / s, (eR["j0"] - cx) / s                 # image px across the axis
    k_mid = 0.5 * (eL["k"] + eR["k"])
    theta = math.atan(k_mid)
    width = (xr - xl) * math.cos(theta)
    centre_off = 0.5 * (xl + xr)

    # ---- confidence ----
    # Per-edge reliability from the line evidence (logistic in SNR, calibrated on test frames:
    # SNR<1 -> ~15% of edges right, 1.5-3 -> ~85%, >=3 -> ~98%), times the ambiguity margin to the
    # best clearly different pair.
    snr_l, snr_r = float(snr[kb, jlb]), float(snr[kb, jrb])
    edge_conf = lambda v: 1.0 / (1.0 + math.exp(-2.2 * (v - 1.2)))
    conf_l, conf_r = edge_conf(snr_l), edge_conf(snr_r)
    alt = (np.abs(np.log(w / w[b])) > 0.12) | (np.abs(mo - mo[b]) > 0.10)
    margin = float(score[b] - score[alt].max()) if alt.any() else 10.0
    conf = conf_l * conf_r * math.sqrt(1 - math.exp(-max(margin, 0.0)))
    edge_pts = []
    for e in (eL, eR):   # each edge as an image-space line: point at the ring row + direction
        q = c + (e["j0"] - cx) / s * n
        dirv = a + e["k"] * n
        edge_pts.append(dict(point=q.tolist(), dir=(dirv / np.linalg.norm(dirv)).tolist()))

    out.update(
        ok=True, width_px=float(width), centre_offset_px=float(centre_off),
        axis_angle_deg=math.degrees(theta), left_offset_px=float(xl), right_offset_px=float(xr),
        confidence=float(conf), conf_left=conf_l, conf_right=conf_r, edges_img=edge_pts,
        edge_kinds=[KINDS[eL["kind"]], KINDS[eR["kind"]]], snr=(snr_l, snr_r), margin=margin,
    )
    if debug:
        out["debug"] = dict(patch=patch, cx=cx, cy=cy, s=s, c=c, a=a, n=n, W0=W0, L2=L2, eL=eL, eR=eR)
    return out
