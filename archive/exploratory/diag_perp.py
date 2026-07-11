#!/usr/bin/env python3
"""Compare OLD (outer<->inner correspondence) vs NEW (perpendicular-to-outer) unroll on seal_2260.
NEW unroll: sample along the inward normal of the SMOOTHED OUTER contour, to a local depth = band width
(from the inner edge via distance transform) + margin. No inner-point correspondence -> stable for thin defects.
Shows GT-seal and PRED-seal strips for both methods, with the defect overlaid, to check consistency."""
import numpy as np, cv2, torch
from seal_inspection import core, cvat
from seal_inspection.paths import ROOT as R; NAME = "seal_2260_1780692167999_raw.png"; HS, WS, MARG = 128, 1536, 0.15
dev = "cuda" if torch.cuda.is_available() else "cpu"
seal, sk = core.load_unet(f"{R}/models/best_lite_reviewed_1280.pt", dev)
g = cv2.imread(f"{R}/data/images/prod2/{NAME}", cv2.IMREAD_GRAYSCALE); H, W = g.shape

def perp_unroll_maps(outer, inner):
    """Perpendicular-to-outer sampling maps."""
    O = core._resample_closed(outer, WS)
    O[:, 0] = core._smooth_closed(O[:, 0]); O[:, 1] = core._smooth_closed(O[:, 1])
    T = np.roll(O, -1, 0) - np.roll(O, 1, 0)
    Tn = np.maximum(np.hypot(T[:, 0], T[:, 1]), 1e-6)
    N = np.stack([-T[:, 1] / Tn, T[:, 0] / Tn], 1)               # one perpendicular
    fill_out = np.zeros((H, W), np.uint8); cv2.drawContours(fill_out, [outer.astype(np.int32)], -1, 255, -1)
    # make N point INWARD (toward the pack interior)
    probe = (O + 4 * N).astype(int)
    inside = fill_out[np.clip(probe[:, 1], 0, H - 1), np.clip(probe[:, 0], 0, W - 1)] > 0
    if inside.mean() < 0.5:
        N = -N
    fill_in = np.zeros((H, W), np.uint8); cv2.drawContours(fill_in, [inner.astype(np.int32)], -1, 255, -1)
    dt_in = cv2.distanceTransform(255 - fill_in, cv2.DIST_L2, 5)  # distance to inner edge
    L = dt_in[np.clip(O[:, 1].astype(int), 0, H - 1), np.clip(O[:, 0].astype(int), 0, W - 1)]  # local band width
    L = core._smooth_closed(L)
    a = np.linspace(-MARG, 1 + MARG, HS)[:, None]
    mx = (O[:, 0][None, :] + a * (L[None, :] * N[:, 0][None, :])).astype(np.float32)
    my = (O[:, 1][None, :] + a * (L[None, :] * N[:, 1][None, :])).astype(np.float32)
    return mx, my

# GT contours + 2D defect mask
node = [im for im in cvat.iter_images(f"{R}/data/annotations/prod2_reviewed.xml") if im.get("name") == NAME][0]
og, ig = cvat.seal_outer_inner(node)
defect2d = np.zeros((H, W), np.uint8)
for d in cvat.polygons(node, "defect") + cvat.polygons(node, "liquid"):
    cv2.fillPoly(defect2d, [d.astype(np.int32)], 255)
# predicted contours
x0, y0, x1, y1 = core.pack_bbox(g); prob = core.predict_probability(seal, g[y0:y1, x0:x1], sk["img"], dev)
full = np.zeros((H, W), np.float32); full[y0:y1, x0:x1] = cv2.resize(prob, (x1 - x0, y1 - y0))
op, ip = core.mask_to_ring((full > sk.get("thresh", .5)).astype(np.uint8) * 255)

def strip_with_defect(mx, my):
    s = cv2.remap(core.normalize(g), mx, my, cv2.INTER_LINEAR, borderValue=0)
    dm = cv2.remap(defect2d, mx, my, cv2.INTER_NEAREST, borderValue=0)
    v = cv2.cvtColor(s, cv2.COLOR_GRAY2BGR); v[dm > 127] = (0, 230, 0)
    return v

rows = []
for tag, fn in [("OLD GT", lambda: core.unroll_maps(og, ig, HS, WS)),
                ("OLD PRED", lambda: core.unroll_maps(op, ip, HS, WS)),
                ("NEW GT", lambda: perp_unroll_maps(og, ig)),
                ("NEW PRED", lambda: perp_unroll_maps(op, ip))]:
    mx, my = fn(); v = strip_with_defect(mx, my)
    b = np.full((22, WS, 3), 30, np.uint8); cv2.putText(b, tag, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    rows.append(np.vstack([b, v]))
cv2.imwrite(f"{R}/outputs/diag_perp.png", np.vstack(rows)); print("wrote outputs/diag_perp.png")
