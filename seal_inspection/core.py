"""
core.py — shared building blocks for the NIR seal-inspection pipeline.

Background
----------
A near-infrared (NIR) line-scan camera images food trays as they cross a gap
between two conveyors. We inspect the heat-sealed flange (the "seal") for
defects. Two facts shape the whole design:

  1. NIR *transmission*: denser / wetter material absorbs more IR, so product,
     contamination and liquid appear DARK; thin air gaps appear bright.
  2. The camera is FREE-RUNNING (not encoder-triggered), so every scan carries
     its own non-rigid, wavy distortion. A single rotation / affine / homography
     cannot straighten it — so we never globally rectify. Instead we FOLLOW the
     seal's real edges and unwrap the seal band into a flat strip. Working on the
     strip makes downstream defect detection position-independent.

The system is two chained models:

    raw image -> [seal U-Net] -> seal ring -> unroll -> [defect U-Net] -> verdict

This module holds the imaging/geometry helpers shared by every script:
  * image normalization
  * pack detection (crop to the tray)
  * segmentation-model inference
  * mask <-> polygon conversion (CVAT polygons <-> pixel masks)
  * unrolling the seal ring into a strip
"""
from __future__ import annotations
import numpy as np
import cv2

# ImageNet statistics — the encoders are ImageNet-pretrained, so inputs are
# normalized with these before being fed to the network.
IMAGENET_MEAN = np.array((0.485, 0.456, 0.406), np.float32)
IMAGENET_STD = np.array((0.229, 0.224, 0.225), np.float32)


# --------------------------------------------------------------------------- #
#  Image normalization
# --------------------------------------------------------------------------- #
def normalize(gray: np.ndarray) -> np.ndarray:
    """Percentile contrast-stretch a grayscale image to the full 0-255 range.

    Robust min/max (1st / 99.5th percentiles) avoid letting a few hot/dead
    pixels squash the contrast. This keeps the network's input consistent
    across scans with slightly different exposure.
    """
    lo, hi = np.percentile(gray, [1, 99.5])
    hi = max(hi, lo + 1)  # guard against a flat image (hi == lo)
    return np.clip((gray.astype(np.float32) - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
#  Pack detection — crop the full frame down to just the tray
# --------------------------------------------------------------------------- #
def conveyor_columns(norm_img: np.ndarray) -> tuple[int, int]:
    """Find the left/right edges of the bright conveyor band.

    The frame has dark side-bands outside the conveyor. We look at the
    column-wise median brightness, take the columns above half-max as "on the
    conveyor", and refine the edges with the brightness gradient.
    """
    col_med = np.median(norm_img, 0).astype(np.float32)
    on = np.where(col_med > col_med.max() * 0.5)[0]
    left, right = on.min(), on.max()
    grad = np.gradient(col_med)
    # left edge = strongest rising gradient near `left`; right edge = strongest fall near `right`
    l = int(np.argmax(grad[max(0, left - 60):left + 60]) + max(0, left - 60))
    r = int(np.argmin(grad[right - 60:right + 60]) + (right - 60))
    return l, r


def pack_bbox(gray: np.ndarray, margin: int = 40) -> tuple[int, int, int, int]:
    """Locate the tray and return its bounding box (x0, y0, x1, y1) + a margin.

    Method: build a per-column "conveyor reference" from the top/bottom strips
    (which are conveyor, not pack), subtract it from the image, threshold the
    difference, clean it morphologically, and take the largest blob. This is a
    classical background-subtraction pack detector — no learning needed.
    """
    n = normalize(gray)
    h, w = n.shape
    try:
        cl, cr = conveyor_columns(n)
    except Exception:
        cl, cr = 0, w  # fall back to the whole width if column detection fails
    top = np.median(n[20:240, :], 0)          # rows near the top edge = conveyor
    bot = np.median(n[h - 240:h - 20, :], 0)  # rows near the bottom edge = conveyor
    ref = np.maximum(top, bot)                # per-column conveyor brightness
    diff = np.clip(np.tile(ref, (h, 1)) - n.astype(np.float32), 0, 255)
    diff[:, :cl] = 0                          # ignore the dark side-bands
    diff[:, cr:] = 0
    mask = cv2.morphologyEx((diff > 20).astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((11, 11), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((41, 41), np.uint8))
    cnts = [c for c in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            if cv2.contourArea(c) > h * w * 0.02]
    if not cnts:
        return 0, 0, w, h
    x, y, bw, bh = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    return max(0, x - margin), max(0, y - margin), min(w, x + bw + margin), min(h, y + bh + margin)


# --------------------------------------------------------------------------- #
#  Segmentation-model inference
# --------------------------------------------------------------------------- #
def load_unet(ckpt_path: str, device: str = "cpu"):
    """Load a saved U-Net checkpoint and return (model, checkpoint_dict).

    Checkpoints store the encoder name, the input resolution used in training
    (`img`), the decision threshold and the normalization stats, so downstream
    code is self-configuring.
    """
    import torch
    import segmentation_models_pytorch as smp
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = smp.Unet(ck["encoder"], encoder_weights=None, in_channels=3, classes=1)
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck


def predict_probability(model, image_gray: np.ndarray, img_size: int, device: str = "cpu") -> np.ndarray:
    """Run the seg model on a (already-cropped) grayscale image.

    The grayscale crop is normalized, resized to the square `img_size` the model
    was trained at, replicated to 3 channels, ImageNet-normalized, and pushed
    through the network. Returns a probability map at `img_size`x`img_size`.
    """
    import torch
    rgb = cv2.resize(np.stack([normalize(image_gray)] * 3, -1), (img_size, img_size)).astype(np.float32) / 255.0
    x = ((rgb - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)[None]
    with torch.no_grad():
        prob = torch.sigmoid(model(torch.from_numpy(x).to(device)))[0, 0].cpu().numpy()
    return prob


# --------------------------------------------------------------------------- #
#  Mask  <->  polygon   (pixels <-> CVAT annotations)
# --------------------------------------------------------------------------- #
def polygons_to_band_mask(outer: np.ndarray, inner: np.ndarray, height: int, width: int) -> np.ndarray:
    """Rasterize the seal: fill the OUTER polygon, then punch out the INNER hole.

    A seal is a ring (rounded-rectangle band) = outer flange edge minus inner
    well edge. CVAT stores it as two polygons; this turns them into the 0/1
    training mask.
    """
    m = np.zeros((height, width), np.uint8)
    cv2.fillPoly(m, [outer.astype(np.int32)], 1)
    cv2.fillPoly(m, [inner.astype(np.int32)], 0)
    return m


def _clean_contour(contour: np.ndarray, n: int = 360, k: int = 7) -> np.ndarray:
    """Resample a contour to `n` arc-length points and circularly smooth it.

    A predicted mask's boundary — especially the inner edge around product
    overflow — can have thousands of ragged pixel-level points. Unrolling that
    directly distorts the strip and breaks defect detection. Smoothing it to a
    clean ring (like a hand-drawn polygon) fixes that.
    """
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


def mask_to_ring(mask: np.ndarray, band_px: int = 90):
    """Extract (outer, inner) ring contours from a predicted binary mask.

    * OUTER = convex hull of the foreground. The flange outline is convex, so the
      hull removes spurious inward "diverts" the network sometimes makes at
      corners or around printed labels.
    * INNER = the hole inside the ring if one is found; otherwise we fall back to
      eroding the filled outer by `band_px` (a typical band width) so we still
      return a closed ring when the model failed to open the centre.

    Returns (outer Nx2, inner Nx2) as float arrays, or (None, None) if no ring.
    """
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((35, 35), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None, None
    outer = cv2.convexHull(np.vstack([c.reshape(-1, 2) for c in cnts]))
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
    # smooth the ragged predicted contours into clean rings before they're unrolled
    return _clean_contour(outer), _clean_contour(inner)


def simplify_contour(contour: np.ndarray, ow: int, oh: int,
                     step: float = 9.0, corner_deg: float = 7.0,
                     straight_every: int = 16, win: int = 2) -> np.ndarray:
    """Curvature-adaptive polygon simplification for CVAT pre-annotations.

    A raw contour has thousands of points; CVAT wants ~tens. We resample by
    arc-length, then KEEP a point if either:
      * it is a corner (the turning angle over +/-`win` samples exceeds
        `corner_deg`), so rounded corners stay dense, or
      * it is every `straight_every`-th sample, so long straight edges stay
        sparse.
    `step` (px between resampled points) is the main density knob.
    """
    P = contour.reshape(-1, 2).astype(np.float32)
    loop = np.r_[P, P[:1]]
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(loop, axis=0).T))]
    L = d[-1]
    if L < 10:
        return P
    n = max(40, int(L / step))
    t = np.linspace(0, L, n, endpoint=False)
    Pr = np.stack([np.interp(t, d, loop[:, 0]), np.interp(t, d, loop[:, 1])], 1)
    v1 = Pr - np.roll(Pr, win, 0)
    v2 = np.roll(Pr, -win, 0) - Pr
    dth = np.abs((np.arctan2(v2[:, 1], v2[:, 0]) - np.arctan2(v1[:, 1], v1[:, 0]) + np.pi) % (2 * np.pi) - np.pi)
    thr = np.deg2rad(corner_deg)
    keep = [i for i in range(n) if dth[i] > thr or i % straight_every == 0]
    out = Pr[keep]
    out[:, 0] = np.clip(out[:, 0], 0, ow - 1)
    out[:, 1] = np.clip(out[:, 1], 0, oh - 1)
    return out


def visvalingam(points: np.ndarray, target: int) -> np.ndarray:
    """Reduce a closed polygon to ~`target` vertices, keeping the important ones.

    Repeatedly removes the vertex whose removal changes the polygon area least
    (Visvalingam-Whyatt). Corners (large triangle area) survive; redundant points
    on straight runs go first. Use this when you need an EXACT vertex budget — the
    step-based simplifier above only controls density approximately.
    """
    P = [tuple(p) for p in points]

    def tri_area(a, b, c):
        return abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])) / 2.0

    while len(P) > target:
        n = len(P)
        areas = [tri_area(P[(i - 1) % n], P[i], P[(i + 1) % n]) for i in range(n)]
        P.pop(int(np.argmin(areas)))
    return np.array(P, np.float32)


# --------------------------------------------------------------------------- #
#  Unroll the seal ring into a flat strip
# --------------------------------------------------------------------------- #
def _resample_closed(poly: np.ndarray, n: int) -> np.ndarray:
    """Resample a closed polygon to `n` points evenly spaced by arc length."""
    p = np.r_[poly, poly[:1]]
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    t = np.linspace(0, d[-1], n, endpoint=False)
    return np.stack([np.interp(t, d, p[:, 0]), np.interp(t, d, p[:, 1])], 1)


def _smooth_closed(a: np.ndarray, k: int = 15) -> np.ndarray:
    """Circular moving-average smoothing (the ring wraps around)."""
    ker = np.ones(k) / k
    return np.convolve(np.r_[a[-k:], a, a[:k]], ker, "same")[k:-k]


def _is_ccw(poly: np.ndarray) -> bool:
    return cv2.contourArea(poly.astype(np.float32), oriented=True) > 0


def unroll_maps(outer: np.ndarray, inner: np.ndarray, strip_h: int, strip_w: int):
    """Build the (map_x, map_y) sampling grids that flatten the ring into a strip.

    For each of `strip_w` positions around the perimeter we sample `strip_h`
    points marching inward along the OUTER contour's local normal (top of the
    strip = outer edge, bottom = inner edge). The same maps can be applied to the
    image AND to a defect mask, so the defect deforms identically to the pixels it
    sits on (no train/inference mismatch).

    This is the "perpendicular-to-outer" method (correspondence-free). The earlier
    outer<->inner interpolation lives in `unroll_maps_legacy`; the production
    detector ensembles the two (see pipeline.py) because each catches defects the
    other smears.
    """
    # Perpendicular-to-outer sampling: walk inward along the SMOOTHED OUTER contour's
    # local normal to a depth = band width (distance to the inner edge). This is
    # correspondence-free (no fragile outer<->inner point pairing), so the strip stays
    # stable for thin defects even when the predicted contour differs slightly from GT.
    O = _resample_closed(outer, strip_w)
    O[:, 0] = _smooth_closed(O[:, 0]); O[:, 1] = _smooth_closed(O[:, 1])
    T = np.roll(O, -1, 0) - np.roll(O, 1, 0)
    Tn = np.maximum(np.hypot(T[:, 0], T[:, 1]), 1e-6)
    Nrm = np.stack([-T[:, 1] / Tn, T[:, 0] / Tn], 1)            # contour normal
    cw = int(max(outer[:, 0].max(), inner[:, 0].max())) + 10
    ch = int(max(outer[:, 1].max(), inner[:, 1].max())) + 10
    fill_out = np.zeros((ch, cw), np.uint8); cv2.drawContours(fill_out, [outer.astype(np.int32)], -1, 255, -1)
    probe = (O + 4 * Nrm).astype(int)                          # orient the normal INWARD
    if (fill_out[np.clip(probe[:, 1], 0, ch - 1), np.clip(probe[:, 0], 0, cw - 1)] > 0).mean() < 0.5:
        Nrm = -Nrm
    fill_in = np.zeros((ch, cw), np.uint8); cv2.drawContours(fill_in, [inner.astype(np.int32)], -1, 255, -1)
    dt_in = cv2.distanceTransform(255 - fill_in, cv2.DIST_L2, 5)  # distance to inner edge
    L = _smooth_closed(dt_in[np.clip(O[:, 1].astype(int), 0, ch - 1), np.clip(O[:, 0].astype(int), 0, cw - 1)])
    a = np.linspace(-0.15, 1.15, strip_h)[:, None]             # 0=outer edge, 1=inner edge, +/-15% margin
    map_x = (O[:, 0][None, :] + a * (L[None, :] * Nrm[:, 0][None, :])).astype(np.float32)
    map_y = (O[:, 1][None, :] + a * (L[None, :] * Nrm[:, 1][None, :])).astype(np.float32)
    return map_x, map_y


def unroll_maps_legacy(outer: np.ndarray, inner: np.ndarray, strip_h: int, strip_w: int):
    """Legacy unroll: linearly interpolate each column from the OUTER edge to the
    paired point on the INNER edge (the two contours are resampled, wound the same
    way and start-aligned). Kept as the second branch of the production ensemble —
    it catches defects the perpendicular-to-outer method smears, and vice-versa.
    The defect model `defect_strip.prev.pt` was trained on strips from THIS unroll,
    so the two must stay paired.
    """
    O = _resample_closed(outer, strip_w)
    I = _resample_closed(inner, strip_w)
    if _is_ccw(O) != _is_ccw(I):                      # wind both the same way
        I = I[::-1]
    j = int(np.argmin(np.hypot(I[:, 0] - O[0, 0], I[:, 1] - O[0, 1])))  # align start points
    I = np.roll(I, -j, axis=0)
    for arr in (O, I):
        arr[:, 0] = _smooth_closed(arr[:, 0]); arr[:, 1] = _smooth_closed(arr[:, 1])
    a = np.linspace(-0.15, 1.15, strip_h)[:, None]    # +/-15% margin past both edges
    map_x = (O[:, 0][None, :] * (1 - a) + I[:, 0][None, :] * a).astype(np.float32)
    map_y = (O[:, 1][None, :] * (1 - a) + I[:, 1][None, :] * a).astype(np.float32)
    return map_x, map_y


def unroll(image: np.ndarray, outer: np.ndarray, inner: np.ndarray,
           strip_h: int, strip_w: int, nearest: bool = False) -> np.ndarray:
    """Unwrap the seal band of `image` into a (strip_h x strip_w) strip."""
    map_x, map_y = unroll_maps(outer, inner, strip_h, strip_w)
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.remap(image, map_x, map_y, interp, borderValue=0)
