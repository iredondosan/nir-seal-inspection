#!/usr/bin/env python3
"""End-to-end eval on the preserved hold-out for a given defect model.
predicted seal -> perpendicular unroll -> defect dscore. Reports AUROC + recall/FP."""
import glob, os, sys, argparse, numpy as np, cv2, torch
from seal_inspection import core

R = "/home/ubuntu/TFM/seal-inspection"
dev = "cuda" if torch.cuda.is_available() else "cpu"

ap = argparse.ArgumentParser()
ap.add_argument("--defect", required=True)
ap.add_argument("--seal", default=f"{R}/models/best_lite_reviewed_1280.pt")
a = ap.parse_args()

seal, sk = core.load_unet(a.seal, dev)
defm, dk = core.load_unet(a.defect, dev)
HS, WS = dk["HS"], dk["WS"]; MEAN, STD = core.IMAGENET_MEAN, core.IMAGENET_STD

def dscore(strip):
    x = ((np.stack([strip] * 3, -1) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
    with torch.no_grad():
        p = torch.sigmoid(defm(torch.from_numpy(x).to(dev)))[0, 0].cpu().numpy()
    return float(cv2.GaussianBlur(p, (0, 0), 2).max())

# authoritative labels from the preserved hold-out
lab = {}
for ln in open(f"{R}/data/holdout_labels.csv").read().splitlines()[1:]:
    nm, l = ln.split(","); lab[nm] = int(l)

scores, labels = [], []
for nm, l in lab.items():
    hits = glob.glob(f"{R}/data/images/*/{nm}.png") + glob.glob(f"{R}/data/images/*/{nm}.jpg")
    if not hits: continue
    g = cv2.imread(hits[0], cv2.IMREAD_GRAYSCALE); H, W = g.shape
    x0, y0, x1, y1 = core.pack_bbox(g)
    prob = core.predict_probability(seal, g[y0:y1, x0:x1], sk["img"], dev)
    full = np.zeros((H, W), np.float32); full[y0:y1, x0:x1] = cv2.resize(prob, (x1 - x0, y1 - y0))
    O, I = core.mask_to_ring((full > sk.get("thresh", .5)).astype(np.uint8) * 255)
    if O is None: continue
    mx, my = core.unroll_maps(O, I, HS, WS)
    strip = cv2.remap(core.normalize(g), mx, my, cv2.INTER_LINEAR, borderValue=0)
    scores.append(dscore(strip)); labels.append(l)

scores, labels = np.array(scores), np.array(labels)
pos, neg = scores[labels == 1], scores[labels == 0]
au = float(np.mean([(x > y) + 0.5 * (x == y) for x in pos for y in neg]))
print(f"model={os.path.basename(a.defect)}  n={len(scores)}  ({int((labels==1).sum())} defect / {int((labels==0).sum())} good)")
print(f"  END-TO-END AUROC = {au:.3f}")
for thr in [0.30,0.43,0.5,0.7,0.85]:
    tp = int(((scores >= thr) & (labels == 1)).sum()); fp = int(((scores >= thr) & (labels == 0)).sum())
    nd = int((labels == 1).sum()); ng = int((labels == 0).sum())
    print(f"  @{thr:.2f}: recall {tp}/{nd} ({tp/nd:.0%})  FP {fp}/{ng} ({fp/ng:.1%})")
print(f"  min defect score = {pos.min():.3f}  (max thr for 100% recall)")
