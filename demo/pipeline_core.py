"""Pure numpy/OpenCV geometry for the seal-inspection demo (no PyTorch).

Extracted verbatim from seal_inspection/core.py: pack cropping, mask->ring
contour extraction and the perpendicular unrolling that flattens the seal ring
into a canonical strip. Only numpy + opencv are required.
"""
import numpy as np
import cv2

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def normalize(gray):
    """Percentile contrast-stretch a grayscale image to 0-255 (robust to exposure)."""
    lo, hi = np.percentile(gray, [1, 99.5])
    hi = max(hi, lo + 1)
    return np.clip((gray.astype(np.float32) - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def conveyor_columns(norm_img):
    """Left/right edges of the bright conveyor band."""
    col_med = np.median(norm_img, 0).astype(np.float32)
    on = np.where(col_med > col_med.max() * 0.5)[0]
    left, right = on.min(), on.max()
    grad = np.gradient(col_med)
    l = int(np.argmax(grad[max(0, left - 60):left + 60]) + max(0, left - 60))
    r = int(np.argmin(grad[right - 60:right + 60]) + (right - 60))
    return l, r


def pack_bbox(gray, margin=40):
    """Locate the tray via classical background subtraction; return (x0,y0,x1,y1)+margin."""
    n = normalize(gray)
    h, w = n.shape
    try:
        cl, cr = conveyor_columns(n)
    except Exception:
        cl, cr = 0, w
    top = np.median(n[20:240, :], 0)
    bot = np.median(n[h - 240:h - 20, :], 0)
    ref = np.maximum(top, bot)
    diff = np.clip(np.tile(ref, (h, 1)) - n.astype(np.float32), 0, 255)
    diff[:, :cl] = 0
    diff[:, cr:] = 0
    mask = cv2.morphologyEx((diff > 20).astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((11, 11), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((41, 41), np.uint8))
    cnts = [c for c in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            if cv2.contourArea(c) > h * w * 0.02]
    if not cnts:
        return 0, 0, w, h
    x, y, bw, bh = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    return max(0, x - margin), max(0, y - margin), min(w, x + bw + margin), min(h, y + bh + margin)


def _clean_contour(contour, n=360, k=7):
    """Resample a contour to n arc-length points and circularly smooth it."""
    P = contour.reshape(-1, 2).astype(np.float32)
    if len(P) < 8:
        return P
    loop = np.r_[P, P[:1]]
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(loop, axis=0).T))]
    if d[-1] < 1:
        return P
    t = np.linspace(0, d[-1], n, endpoint=False)
    r = np.stack([np.interp(t, d, loop[:, 0]), np.interp(t, d, loop[:, 1])], 1)
    ker = np.ones(k) / k
    for ax in (0, 1):
        a = r[:, ax]
        r[:, ax] = np.convolve(np.r_[a[-k:], a, a[:k]], ker, "same")[k:-k]
    return r


def _resample_closed(poly, n):
    """Resample a closed polygon to n points evenly spaced by arc length."""
    p = np.r_[poly, poly[:1]]
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    t = np.linspace(0, d[-1], n, endpoint=False)
    return np.stack([np.interp(t, d, p[:, 0]), np.interp(t, d, p[:, 1])], 1)


def _smooth_closed(a, k=15):
    """Circular moving-average smoothing (the ring wraps around)."""
    ker = np.ones(k) / k
    return np.convolve(np.r_[a[-k:], a, a[:k]], ker, "same")[k:-k]


def mask_to_ring(mask, band_px=90):
    """Extract (outer, inner) ring contours from a predicted binary mask.

    OUTER = largest raw contour (follows wavy pack edges). INNER = the hole inside
    the ring, or a fallback erosion of the filled outer by band_px.
    Returns (outer Nx2, inner Nx2) float arrays, or (None, None) if no ring.
    """
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((35, 35), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None, None
    outer = max(cnts, key=cv2.contourArea)
    fill = np.zeros_like(m)
    cv2.drawContours(fill, [outer], -1, 255, -1)
    hole = cv2.subtract(fill, m)
    holes = [c for c in cv2.findContours(hole, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
             if cv2.contourArea(c) > 0.2 * cv2.contourArea(outer)]
    if holes:
        inner = max(holes, key=cv2.contourArea)
    else:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band_px + 1, 2 * band_px + 1))
        ic, _ = cv2.findContours(cv2.erode(fill, k), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not ic:
            return None, None
        inner = max(ic, key=cv2.contourArea)
    return _clean_contour(outer), _clean_contour(inner)


def _anchor_origin(O):
    """Rotate/flip a resampled closed contour so the strip origin (column 0) is the
    top of the ring and the winding is consistent, independent of the mask
    resolution or where cv2.findContours started the contour."""
    c = O.mean(0)
    area = float(np.sum(O[:, 0] * np.roll(O[:, 1], -1) - np.roll(O[:, 0], -1) * O[:, 1]))
    if area < 0:
        O = O[::-1].copy()
    ang = np.arctan2(O[:, 1] - c[1], O[:, 0] - c[0])
    d = np.angle(np.exp(1j * (ang + np.pi / 2.0)))   # wrapped angular distance to top (-pi/2)
    i0 = int(np.argmin(np.abs(d)))
    return np.roll(O, -i0, 0)



def unroll_maps(outer, inner, strip_h, strip_w):
    """Build (map_x, map_y) that flatten the ring into a strip (perpendicular-to-outer)."""
    O = _resample_closed(outer, strip_w)
    O = _anchor_origin(O)  # resolution-independent origin/winding
    O[:, 0] = _smooth_closed(O[:, 0]); O[:, 1] = _smooth_closed(O[:, 1])
    T = np.roll(O, -1, 0) - np.roll(O, 1, 0)
    Tn = np.maximum(np.hypot(T[:, 0], T[:, 1]), 1e-6)
    Nrm = np.stack([-T[:, 1] / Tn, T[:, 0] / Tn], 1)
    cw = int(max(outer[:, 0].max(), inner[:, 0].max())) + 10
    ch = int(max(outer[:, 1].max(), inner[:, 1].max())) + 10
    fill_out = np.zeros((ch, cw), np.uint8); cv2.drawContours(fill_out, [outer.astype(np.int32)], -1, 255, -1)
    probe = (O + 4 * Nrm).astype(int)
    if (fill_out[np.clip(probe[:, 1], 0, ch - 1), np.clip(probe[:, 0], 0, cw - 1)] > 0).mean() < 0.5:
        Nrm = -Nrm
    fill_in = np.zeros((ch, cw), np.uint8); cv2.drawContours(fill_in, [inner.astype(np.int32)], -1, 255, -1)
    dt_in = cv2.distanceTransform(255 - fill_in, cv2.DIST_L2, 5)
    L = _smooth_closed(dt_in[np.clip(O[:, 1].astype(int), 0, ch - 1), np.clip(O[:, 0].astype(int), 0, cw - 1)])
    a = np.linspace(-0.15, 1.15, strip_h)[:, None]
    map_x = (O[:, 0][None, :] + a * (L[None, :] * Nrm[:, 0][None, :])).astype(np.float32)
    map_y = (O[:, 1][None, :] + a * (L[None, :] * Nrm[:, 1][None, :])).astype(np.float32)
    return map_x, map_y
