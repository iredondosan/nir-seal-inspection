#!/usr/bin/env python3
"""Regenerate end-to-end QC overlays with the FINAL models, across all products.
Runs the full pipeline (predict seal -> unroll -> defect -> verdict) on every held-out
DEFECT pack + a sample of good packs, saving composites for the thesis."""
import os, glob, random, cv2, torch
from seal_inspection import core
from seal_inspection.pipeline import process_pack
R = "/home/ubuntu/TFM/seal-inspection"
random.seed(1)
device = "cuda" if torch.cuda.is_available() else "cpu"
seal, sk = core.load_unet(f"{R}/models/best_lite_reviewed_1280.pt", device)
defm, dk = core.load_unet(f"{R}/models/defect_strip.pt", device)
import argparse
ap = argparse.ArgumentParser(); ap.add_argument("--dthr", type=float, default=float(dk.get("thr", .5)))
ap.add_argument("--out", default="outputs/final_overlays"); a = ap.parse_args()
out = f"{R}/{a.out}"; os.makedirs(out, exist_ok=True)

# which held-out test packs are defects vs good
defect, good = [], []
for ip in sorted(glob.glob(f"{R}/data/strips/test/img/*.png")):
    base = os.path.splitext(os.path.basename(ip))[0]
    m = cv2.imread(ip.replace("/img/", "/mask/"), cv2.IMREAD_GRAYSCALE)
    (defect if (m is not None and m.sum() > 0) else good).append(base)
sample = defect + good                                      # ALL test packs (191)
print(f"overlays: {len(defect)} defect + {len(good)} good", flush=True)

def find_raw(base):
    for ext in (".png", ".jpg"):
        hits = glob.glob(f"{R}/data/images/*/{base}{ext}")
        if hits:
            return hits[0]
    return None

n = 0
for base in sample:
    p = find_raw(base)
    if p is None:
        continue
    g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    comp, ndet = process_pack(g, seal, sk["img"], sk.get("thresh", .5),
                              defm, dk["HS"], dk["WS"], a.dthr, device)
    if comp is None:
        continue
    label = "DEFECT" if base in defect else "good"
    cv2.imwrite(f"{out}/{label}_{base}.png", comp); n += 1
    print(f"{base} [{label}] -> {'DEFECT' if ndet else 'OK'} ({ndet})", flush=True)
print(f"wrote {n} composites to {out}", flush=True)
