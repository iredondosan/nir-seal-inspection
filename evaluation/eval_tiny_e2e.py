#!/usr/bin/env python3
"""End-to-end eval for the TinyUNet (1-channel) defect model on the 179-pack hold-out:
predicted seal (ImageNet) -> perpendicular unroll -> TinyUNet dscore."""
import glob, os, numpy as np, cv2, torch
from seal_inspection import core
from seal_inspection.tiny_unet import TinyUNet
from seal_inspection.paths import ROOT as R; dev = "cuda" if torch.cuda.is_available() else "cpu"

seal, sk = core.load_unet(f"{R}/models/best_lite_reviewed_1280.pt", dev)
ck = torch.load(f"{R}/models/tiny_defect.pt", map_location="cpu", weights_only=False)
HS, WS = ck["HS"], ck["WS"]
tiny = TinyUNet(base=ck.get("base", 16), in_ch=ck.get("in_ch", 1)).to(dev).eval()
tiny.load_state_dict(ck["state_dict"])
print("TinyUNet loaded OK")

def dscore(strip):
    x = torch.from_numpy(((strip.astype(np.float32)/255.0 - 0.5)/0.5))[None, None].to(dev)
    with torch.no_grad():
        p = torch.sigmoid(tiny(x))[0, 0].cpu().numpy()
    return float(cv2.GaussianBlur(p, (0, 0), 2).max())

lab = {}
for ln in open(f"{R}/data/holdout_labels.csv").read().splitlines()[1:]:
    nm, l = ln.split(","); lab[nm] = int(l)
scores, labels = [], []
for nm, l in lab.items():
    hits = glob.glob(f"{R}/data/images/*/{nm}.png") + glob.glob(f"{R}/data/images/*/{nm}.jpg")
    if not hits: continue
    g = cv2.imread(hits[0], 0); H, W = g.shape; x0, y0, x1, y1 = core.pack_bbox(g)
    prob = core.predict_probability(seal, g[y0:y1, x0:x1], sk["img"], dev)
    full = np.zeros((H, W), np.float32); full[y0:y1, x0:x1] = cv2.resize(prob, (x1-x0, y1-y0))
    O, I = core.mask_to_ring((full > sk.get("thresh", .5)).astype(np.uint8)*255)
    if O is None: continue
    mx, my = core.unroll_maps(O, I, HS, WS)
    strip = cv2.remap(core.normalize(g), mx, my, cv2.INTER_LINEAR, borderValue=0)
    scores.append(dscore(strip)); labels.append(l)
scores, labels = np.array(scores), np.array(labels)
pos, neg = scores[labels == 1], scores[labels == 0]
au = float(np.mean([(a > b)+0.5*(a == b) for a in pos for b in neg]))
print(f"TinyUNet END-TO-END  n={len(scores)} ({int(labels.sum())} defect / {int((labels==0).sum())} good)  AUROC={au:.3f}")
for thr in [0.30, 0.50, 0.70]:
    tp = int(((scores >= thr) & (labels == 1)).sum()); fp = int(((scores >= thr) & (labels == 0)).sum())
    print(f"  @{thr:.2f}: recall {tp}/{int(labels.sum())} ({tp/labels.sum():.0%})  FP {fp}/{int((labels==0).sum())} ({fp/(labels==0).sum():.1%})")
print(f"  min defect score = {pos.min():.3f}")

try:
    from seal_inspection.results import save_results
    save_results("eval_tiny_e2e", {
        "auroc": float(au), "n_def": int(labels.sum()), "n_good": int((labels == 0).sum()),
        "min_defect_score": float(pos.min()),
        "operating_points": {str(thr): {"recall": int(((scores >= thr) & (labels == 1)).sum()),
                                        "fp": int(((scores >= thr) & (labels == 0)).sum())} for thr in [0.30,0.50,0.70]},
    })
except Exception as _e:
    print('[results] skip:', _e)
