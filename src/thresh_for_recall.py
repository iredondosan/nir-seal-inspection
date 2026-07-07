#!/usr/bin/env python3
"""For a defect model, find the threshold needed for each recall level and its false-alarm cost."""
import glob, os, argparse, numpy as np, cv2, torch
from seal_inspection import core
R = "/home/ubuntu/TFM/seal-inspection"
dev = "cuda" if torch.cuda.is_available() else "cpu"
ap = argparse.ArgumentParser(); ap.add_argument("--defect", required=True)
ap.add_argument("--seal", default=f"{R}/models/best_lite_reviewed_1280.pt"); a = ap.parse_args()
seal, sk = core.load_unet(a.seal, dev); defm, dk = core.load_unet(a.defect, dev)
HS, WS = dk["HS"], dk["WS"]; MEAN, STD = core.IMAGENET_MEAN, core.IMAGENET_STD

def dscore(strip):
    x = ((np.stack([strip]*3, -1)/255.0 - MEAN)/STD).transpose(2, 0, 1)[None].astype(np.float32)
    with torch.no_grad():
        p = torch.sigmoid(defm(torch.from_numpy(x).to(dev)))[0, 0].cpu().numpy()
    return float(cv2.GaussianBlur(p, (0, 0), 2).max())

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
    full = np.zeros((H, W), np.float32); full[y0:y1, x0:x1] = cv2.resize(prob, (x1-x0, y1-y0))
    O, I = core.mask_to_ring((full > sk.get("thresh", .5)).astype(np.uint8)*255)
    if O is None: continue
    mx, my = core.unroll_maps(O, I, HS, WS)
    strip = cv2.remap(core.normalize(g), mx, my, cv2.INTER_LINEAR, borderValue=0)
    scores.append(dscore(strip)); labels.append(l)
scores, labels = np.array(scores), np.array(labels)
pos = np.sort(scores[labels == 1]); neg = scores[labels == 0]
nd, ng = len(pos), len(neg)
print(f"model={os.path.basename(a.defect)}  defects={nd} goods={ng}")
print(f"sorted defect scores (ascending): {[round(float(x),3) for x in pos]}")
print(f"{'recall':>8} {'threshold':>10} {'false alarms':>16}")
for k in range(nd, nd-5, -1):          # nd/nd, (nd-1)/nd, ...
    thr = float(pos[nd-k])             # exactly k defects have score >= thr
    fp = int((neg >= thr).sum())
    print(f"  {k}/{nd}   {thr:10.3f}   {fp}/{ng} ({fp/ng:5.1%})")
