#!/usr/bin/env python3
"""Is seal_2260 in the seal TRAIN or VAL split, and does the predicted seal include the defect in the band?"""
import random, cv2, numpy as np, torch
from seal_inspection import core, cvat
from seal_inspection.paths import ROOT as R; NAME = "seal_2260_1780692167999_raw.png"; SEED = 42; VAL_PER = 2
dev = "cuda" if torch.cuda.is_available() else "cpu"

# replicate the seal trainer's per-product split for prod2
names = []
for node in cvat.iter_images(f"{R}/data/annotations/prod2_reviewed.xml"):
    tg = cvat.tags(node)
    if "exclude" in tg or "reviewed" not in tg:
        continue
    if cvat.seal_outer_inner(node) is None:
        continue
    names.append(node.get("name"))
random.Random(SEED).shuffle(names)
val = names[:VAL_PER]
print(f"prod2 reviewed packs: {len(names)}  val(held-out)={val}")
print(f"seal_2260 is in {'VAL (held-out, NOT trained on)' if NAME in val else 'TRAIN (trained on)'}")

# predicted seal mask vs GT band; do the defect pixels land in the predicted band or the hole?
seal, sk = core.load_unet(f"{R}/models/best_lite_reviewed_1280.pt", dev)
g = cv2.imread(f"{R}/data/images/prod2/{NAME}", cv2.IMREAD_GRAYSCALE); H, W = g.shape
node = [im for im in cvat.iter_images(f"{R}/data/annotations/prod2_reviewed.xml") if im.get("name") == NAME][0]
og, ig = cvat.seal_outer_inner(node)
gt_band = core.polygons_to_band_mask(og, ig, H, W)
defect = np.zeros((H, W), np.uint8)
for d in cvat.polygons(node, "defect") + cvat.polygons(node, "liquid"):
    cv2.fillPoly(defect, [d.astype(np.int32)], 1)
x0, y0, x1, y1 = core.pack_bbox(g)
prob = core.predict_probability(seal, g[y0:y1, x0:x1], sk["img"], dev)
full = np.zeros((H, W), np.float32); full[y0:y1, x0:x1] = cv2.resize(prob, (x1 - x0, y1 - y0))
pred_band = (full > sk.get("thresh", .5)).astype(np.uint8)
inter = (gt_band & pred_band).sum(); dice = 2 * inter / (gt_band.sum() + pred_band.sum())
dpx = int(defect.sum())
in_pred_band = int((defect & pred_band).sum())
in_gt_band = int((defect & gt_band).sum())
print(f"\nseal Dice (pred vs GT band): {dice:.3f}")
print(f"defect pixels: {dpx}")
print(f"  inside GT band:   {in_gt_band} ({100*in_gt_band/dpx:.0f}%)")
print(f"  inside PRED band: {in_pred_band} ({100*in_pred_band/dpx:.0f}%)  <- if low, the model excluded the defect from the band")
