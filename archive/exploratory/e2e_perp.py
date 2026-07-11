#!/usr/bin/env python3
"""End-to-end eval with the NEW perpendicular-to-outer unroll + retrained defect model.
For every test strip: predicted seal -> core.unroll_maps (new) -> defect score. Reports
AUROC + recall/FP at a few thresholds, and the seal_2260 score specifically."""
import glob, os, numpy as np, cv2, torch
from seal_inspection import core
from seal_inspection.paths import ROOT as R
dev = "cuda" if torch.cuda.is_available() else "cpu"
seal, sk = core.load_unet(f"{R}/models/best_lite_reviewed_1280.pt", dev)
defm, dk = core.load_unet(f"{R}/models/defect_strip.pt", dev)
HS, WS = dk["HS"], dk["WS"]; MEAN, STD = core.IMAGENET_MEAN, core.IMAGENET_STD

def dscore(strip):
    x = ((np.stack([strip] * 3, -1) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
    with torch.no_grad():
        p = torch.sigmoid(defm(torch.from_numpy(x).to(dev)))[0, 0].cpu().numpy()
    return float(cv2.GaussianBlur(p, (0, 0), 2).max())

scores, labels, names = [], [], []
for ip in sorted(glob.glob(f"{R}/data/strips/test/img/*.png")):
    nm = os.path.splitext(os.path.basename(ip))[0]
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
    s = dscore(strip)
    gt = cv2.imread(ip.replace("/img/", "/mask/"), 0)
    scores.append(s); labels.append(1 if gt.sum() > 0 else 0); names.append(nm)
scores, labels = np.array(scores), np.array(labels)
pos, neg = scores[labels == 1], scores[labels == 0]
au = float(np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg]))
print(f"END-TO-END (predicted seal -> NEW unroll -> new model)  n={len(scores)}  AUROC {au:.3f}")
for thr in [0.43, 0.5, 0.7, 0.85, 0.92]:
    tp = int(((scores >= thr) & (labels == 1)).sum()); fp = int(((scores >= thr) & (labels == 0)).sum())
    print(f"  @thr {thr:.2f}: recall {tp}/{int((labels==1).sum())}  FP {fp}/{int((labels==0).sum())}")
# max threshold for 100% recall
order = np.sort(pos)
print(f"  min defect score = {order[0]:.3f}  (max thr for 100% recall)")
i = names.index("seal_2260_1780692167999_raw")
print(f"\nseal_2260 score: {scores[i]:.3f}  -> {'DETECTED' if scores[i] >= 0.43 else 'MISSED'} @0.43")
# list the 3 lowest-scoring defects
idx = np.where(labels == 1)[0]
for k in idx[np.argsort(scores[idx])][:4]:
    print(f"  weakest defect: {names[k][:40]:40s} {scores[k]:.3f}")
