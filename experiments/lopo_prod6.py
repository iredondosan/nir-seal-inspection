#!/usr/bin/env python3
"""Leave-one-pack-out CV for prod6 defects. For each of the 3 physical defect packs:
hold out its captures (fold-specific holdout), rebuild strips, retrain +sealjit, score
the held-out captures end-to-end (predicted seal -> unroll -> dscore). No leakage: the
model never sees the held-out physical pack."""
import subprocess, os, glob, numpy as np, cv2, torch
from seal_inspection import core
from seal_inspection.paths import ROOT as R; PY = f"{R}/.venv/bin/python"
dev = "cuda" if torch.cuda.is_available() else "cpu"
MEAN, STD = core.IMAGENET_MEAN, core.IMAGENET_STD
THR = 0.5

FOLDS = {"A": ["prod6_bad_001", "prod6_bad_004", "prod6_bad_007"],
         "B": ["prod6_bad_002", "prod6_bad_005", "prod6_bad_008"],
         "C": ["prod6_bad_003", "prod6_bad_006", "prod6_bad_009"]}
base = [l.strip() for l in open(f"{R}/data/holdout.txt") if l.strip()]
seal, sk = core.load_unet(f"{R}/models/best_lite_reviewed_1280.pt", dev)

def score_pack(defm, dk, nm):
    HS, WS = dk["HS"], dk["WS"]
    hits = glob.glob(f"{R}/data/images/*/{nm}.png") + glob.glob(f"{R}/data/images/*/{nm}.jpg")
    if not hits: return None
    g = cv2.imread(hits[0], cv2.IMREAD_GRAYSCALE); H, W = g.shape
    x0, y0, x1, y1 = core.pack_bbox(g)
    prob = core.predict_probability(seal, g[y0:y1, x0:x1], sk["img"], dev)
    full = np.zeros((H, W), np.float32); full[y0:y1, x0:x1] = cv2.resize(prob, (x1-x0, y1-y0))
    O, I = core.mask_to_ring((full > sk.get("thresh", .5)).astype(np.uint8)*255)
    if O is None: return None
    mx, my = core.unroll_maps(O, I, HS, WS)
    strip = cv2.remap(core.normalize(g), mx, my, cv2.INTER_LINEAR, borderValue=0)
    x = ((np.stack([strip]*3, -1)/255.0 - MEAN)/STD).transpose(2, 0, 1)[None].astype(np.float32)
    with torch.no_grad():
        p = torch.sigmoid(defm(torch.from_numpy(x).to(dev)))[0, 0].cpu().numpy()
    return float(cv2.GaussianBlur(p, (0, 0), 2).max())

results = {}
for F, caps in FOLDS.items():
    hf = f"{R}/data/holdout_fold{F}.txt"
    open(hf, "w").write("\n".join(base + caps) + "\n")
    subprocess.run([PY, "data_prep/make_strips.py", "--holdout", hf, "--out", f"data/strips_fold{F}"],
                   cwd=R, check=True, capture_output=True)
    subprocess.run([PY, "training/train_defect.py", "--strips", f"data/strips_fold{F}", "--sealjit",
                    "--out", f"models/defect_fold{F}.pt", "--epochs", "60"],
                   cwd=R, check=True, capture_output=True)
    defm, dk = core.load_unet(f"{R}/models/defect_fold{F}.pt", dev)
    results[F] = [score_pack(defm, dk, c) for c in caps]
    print(f"fold {F} done: held-out capture scores = {[round(s,3) if s is not None else None for s in results[F]]}", flush=True)

print("\n===== LEAVE-ONE-PACK-OUT CV — prod6 defect detection (threshold 0.5) =====")
cap_hits = cap_tot = pack_hits = 0
for F, caps in FOLDS.items():
    sc = results[F]; valid = [s for s in sc if s is not None]
    packmax = max(valid) if valid else 0.0
    ch = sum(1 for s in valid if s >= THR); cap_hits += ch; cap_tot += len(valid)
    ph = packmax >= THR; pack_hits += ph
    print(f"  pack {F}: captures={[round(s,3) for s in valid]}  pack-max={packmax:.3f}  "
          f"captures_detected={ch}/{len(valid)}  pack={'DETECTED' if ph else 'MISSED'}")
print(f"\n  per-capture recall: {cap_hits}/{cap_tot}")
print(f"  per-pack recall:    {pack_hits}/3")
open(f"{R}/outputs/lopo_done.flag", "w").write("done")
