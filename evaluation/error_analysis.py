#!/usr/bin/env python3
"""Dump the misclassified hold-out pieces (FN/FP) with product + score at OP=0.5,
using the deployed end-to-end pipeline. Grounds the thesis error-analysis section."""
import glob, os, numpy as np, cv2, torch
from seal_inspection import core
from seal_inspection.paths import ROOT as R; dev = "cuda" if torch.cuda.is_available() else "cpu"
MEAN, STD = core.IMAGENET_MEAN, core.IMAGENET_STD; OP = 0.5
seal, sk = core.load_unet(f"{R}/models/best_lite_reviewed_1280.pt", dev)
defm, dk = core.load_unet(f"{R}/models/defect_strip.pt", dev); HS, WS = dk["HS"], dk["WS"]
lab = {ln.split(",")[0]: int(ln.split(",")[1]) for ln in open(f"{R}/data/holdout_labels.csv").read().splitlines()[1:]}
def dsc(strip):
    x = ((np.stack([strip]*3, -1)/255.0 - MEAN)/STD).transpose(2, 0, 1)[None].astype(np.float32)
    with torch.no_grad(): p = torch.sigmoid(defm(torch.from_numpy(x).to(dev)))[0, 0].cpu().numpy()
    return float(cv2.GaussianBlur(p, (0, 0), 2).max())
rows = []
for nm, l in lab.items():
    h = glob.glob(f"{R}/data/images/*/{nm}.png") + glob.glob(f"{R}/data/images/*/{nm}.jpg")
    if not h: continue
    prod = os.path.basename(os.path.dirname(h[0]))
    g = cv2.imread(h[0], 0); H, W = g.shape; x0, y0, x1, y1 = core.pack_bbox(g)
    prob = core.predict_probability(seal, g[y0:y1, x0:x1], sk["img"], dev)
    full = np.zeros((H, W), np.float32); full[y0:y1, x0:x1] = cv2.resize(prob, (x1-x0, y1-y0))
    O, I = core.mask_to_ring((full > sk.get("thresh", .5)).astype(np.uint8)*255)
    if O is None: continue
    mx, my = core.unroll_maps(O, I, HS, WS)
    s = dsc(cv2.remap(core.normalize(g), mx, my, cv2.INTER_LINEAR, borderValue=0))
    rows.append((nm, prod, l, s))
S = np.array([r[3] for r in rows]); L = np.array([r[2] for r in rows])
print("N=%d  good=%d  defect=%d" % (len(rows), int((L==0).sum()), int(L.sum())))
fn = [r for r in rows if r[2]==1 and r[3] < OP]
fp = [r for r in rows if r[2]==0 and r[3] >= OP]
print("\n== FALSOS NEGATIVOS (defecto no detectado, score < %.2f) ==" % OP)
for nm, prod, l, s in sorted(fn, key=lambda r: r[3]): print("  %-28s prod=%-8s score=%.3f" % (nm, prod, s))
print("\n== FALSOS POSITIVOS (correcto marcado defecto, score >= %.2f) ==" % OP)
for nm, prod, l, s in sorted(fp, key=lambda r: -r[3]): print("  %-28s prod=%-8s score=%.3f" % (nm, prod, s))
# margins: how close are the FN to the threshold, and the good-score distribution
posd = S[L==1]; negd = S[L==0]
print("\ndefect scores: min=%.3f  median=%.3f  max=%.3f" % (posd.min(), np.median(posd), posd.max()))
print("good  scores: median=%.3f  p90=%.3f  max=%.3f" % (np.median(negd), np.percentile(negd,90), negd.max()))
print("FP products:", {p: sum(1 for r in fp if r[1]==p) for p in sorted(set(r[1] for r in fp))})
