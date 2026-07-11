#!/usr/bin/env python3
"""For the 8 hold-out false positives, render the unrolled strip with the defect
probability heatmap overlaid and the max-score location marked, to diagnose the cause."""
import glob, os, numpy as np, cv2, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from seal_inspection import core
from seal_inspection.paths import ROOT as R; dev = "cuda" if torch.cuda.is_available() else "cpu"
MEAN, STD = core.IMAGENET_MEAN, core.IMAGENET_STD
seal, sk = core.load_unet(f"{R}/models/best_lite_reviewed_1280.pt", dev)
defm, dk = core.load_unet(f"{R}/models/defect_strip.pt", dev); HS, WS = dk["HS"], dk["WS"]

FP = ["seal_3108_1780734392895_raw", "seal_2604_1780700521459_raw", "seal_1524_1780672777658_raw",
      "seal_2076_1780689633560_raw", "seal_538_1780628998873_raw", "seal_2256_1780692100213_raw",
      "seal_1345_1780667241449_raw", "seal_1371_1780667534707_raw"]

def strip_and_prob(nm):
    h = glob.glob(f"{R}/data/images/*/{nm}.png") + glob.glob(f"{R}/data/images/*/{nm}.jpg")
    if not h: return None
    prod = os.path.basename(os.path.dirname(h[0]))
    g = cv2.imread(h[0], 0); H, W = g.shape; x0, y0, x1, y1 = core.pack_bbox(g)
    prob = core.predict_probability(seal, g[y0:y1, x0:x1], sk["img"], dev)
    full = np.zeros((H, W), np.float32); full[y0:y1, x0:x1] = cv2.resize(prob, (x1-x0, y1-y0))
    O, I = core.mask_to_ring((full > sk.get("thresh", .5)).astype(np.uint8)*255)
    if O is None: return None
    mx, my = core.unroll_maps(O, I, HS, WS)
    strip = cv2.remap(core.normalize(g), mx, my, cv2.INTER_LINEAR, borderValue=0)
    x = ((np.stack([strip]*3, -1)/255.0 - MEAN)/STD).transpose(2, 0, 1)[None].astype(np.float32)
    with torch.no_grad(): p = torch.sigmoid(defm(torch.from_numpy(x).to(dev)))[0, 0].cpu().numpy()
    ps = cv2.GaussianBlur(p, (0, 0), 2)
    return prod, strip, ps

fig, axes = plt.subplots(len(FP), 1, figsize=(13, 2.1*len(FP)))
for ax, nm in zip(axes, FP):
    r = strip_and_prob(nm)
    if r is None: ax.set_title(nm+" (sin anillo)"); ax.axis("off"); continue
    prod, strip, ps = r
    yy, xx = np.unravel_index(ps.argmax(), ps.shape)
    ax.imshow(strip, cmap="gray", aspect="auto")
    ax.imshow(ps, cmap="jet", alpha=0.35, aspect="auto")
    ax.plot(xx, yy, "o", ms=14, mfc="none", mec="lime", mew=2)
    ax.set_title(f"{prod}  {nm.split('_')[1]}  (score={ps.max():.2f})", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.savefig(f"{R}/fp_diagnosis.png", dpi=120)
print("wrote fp_diagnosis.png")
