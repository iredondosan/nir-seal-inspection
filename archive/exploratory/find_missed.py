#!/usr/bin/env python3
"""List the lowest-scoring (missed) defect packs end-to-end for a defect model, with product + defect area."""
import glob, os, argparse, numpy as np, cv2, torch
from seal_inspection import core
from seal_inspection.paths import ROOT as R
dev = "cuda" if torch.cuda.is_available() else "cpu"
ap = argparse.ArgumentParser(); ap.add_argument("--defect", required=True); a = ap.parse_args()
seal, sk = core.load_unet(f"{R}/models/best_lite_reviewed_1280.pt", dev)
defm, dk = core.load_unet(a.defect, dev)
HS, WS = dk["HS"], dk["WS"]; MEAN, STD = core.IMAGENET_MEAN, core.IMAGENET_STD

def dscore(strip):
    x = ((np.stack([strip]*3, -1)/255.0 - MEAN)/STD).transpose(2, 0, 1)[None].astype(np.float32)
    with torch.no_grad():
        p = torch.sigmoid(defm(torch.from_numpy(x).to(dev)))[0, 0].cpu().numpy()
    return float(cv2.GaussianBlur(p, (0, 0), 2).max())

lab = {}
for ln in open(f"{R}/data/holdout_labels.csv").read().splitlines()[1:]:
    nm, l = ln.split(","); lab[nm] = int(l)

rows = []
for nm, l in lab.items():
    if l != 1: continue
    hits = glob.glob(f"{R}/data/images/*/{nm}.png") + glob.glob(f"{R}/data/images/*/{nm}.jpg")
    if not hits: continue
    prod = hits[0].split("/images/")[1].split("/")[0]
    g = cv2.imread(hits[0], cv2.IMREAD_GRAYSCALE); H, W = g.shape
    x0, y0, x1, y1 = core.pack_bbox(g)
    prob = core.predict_probability(seal, g[y0:y1, x0:x1], sk["img"], dev)
    full = np.zeros((H, W), np.float32); full[y0:y1, x0:x1] = cv2.resize(prob, (x1-x0, y1-y0))
    O, I = core.mask_to_ring((full > sk.get("thresh", .5)).astype(np.uint8)*255)
    if O is None:
        rows.append((0.0, nm, prod, -1, "SEAL-FAIL")); continue
    mx, my = core.unroll_maps(O, I, HS, WS)
    strip = cv2.remap(core.normalize(g), mx, my, cv2.INTER_LINEAR, borderValue=0)
    s = dscore(strip)
    # GT defect area from annotation (unrolled), to gauge defect size
    rows.append((s, nm, prod, 0, ""))

rows.sort()
print(f"{'score':>7}  {'product':8}  name")
for s, nm, prod, _, note in rows[:6]:
    print(f"{s:7.4f}  {prod:8}  {nm}  {note}")
print("...")
print(f"(total {len(rows)} defect packs; lowest {sum(1 for r in rows if r[0]<0.05)} are < 0.05)")
