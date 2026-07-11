#!/usr/bin/env python3
"""Define the GLOBAL hold-out for clean end-to-end evaluation.

Drawn ONLY from NON-reviewed packs (per-product stratified), so:
  * the seal model (trained on reviewed-only packs) never sees the hold-out, and
  * all reviewed ground-truth packs stay in training for BOTH models.
The defect model excludes the hold-out from its training strips (make_strips reads holdout.txt).

Mirrors make_strips.py pack-eligibility exactly so the splits are consistent.
Writes data/holdout.txt (one basename per line) and prints the composition + a name,label CSV."""
import os, random
import xml.etree.ElementTree as ET
from seal_inspection.paths import ROOT
SEED = 7; F_DEF = 0.30; F_GOOD = 0.20      # fraction of NON-reviewed defects / goods held out per product
SOURCES = [("data/annotations/prod1_reviewed.xml", "prod1", "all"),
           ("data/annotations/prod2_reviewed.xml", "prod2", "all"),
           ("data/annotations/prod3_reviewed.xml", "prod3", "all"),
           ("data/annotations/prod4_reviewed.xml", "prod4", "all"),
           ("data/annotations/prod5_reviewed.xml", "prod5", "all")]  # prod6 (prod6) is train-only, never in the hold-out

def tagset(im): return {t.get("label") for t in im.findall("tag")}

# collect packs exactly as make_strips would, tagging reviewed-status + kind
packs = []   # (basename, prod, kind, reviewed)
for xmlrel, prod, mode in SOURCES:
    p = f"{ROOT}/{xmlrel}"
    if not os.path.exists(p): continue
    for im in ET.parse(p).getroot().findall("image"):
        tg = tagset(im)
        if "exclude" in tg: continue
        if len([q for q in im.findall("polygon") if q.get("label") == "sellado"]) < 2: continue
        defs = any(q.get("label") in ("defect", "liquid") for q in im.findall("polygon"))
        if mode == "good":
            if not (tg & {"good", "reviewed"}): continue
            kind = "good"
        elif mode == "defect_reviewed":
            if "reviewed" not in tg or not defs: continue
            kind = "defect"
        else:
            if not (tg & {"good", "defect", "reviewed"}): continue
            kind = "defect" if defs else "good"
        base = os.path.splitext(im.get("name"))[0]
        packs.append((base, prod, kind, "reviewed" in tg))

rng = random.Random(SEED)
hold = []; labels = {}
print(f"{'product':9} {'kind':6} {'eligible':>8} {'non-rev':>8} {'held-out':>9}")
print("-" * 46)
for prod in ["prod1", "prod2", "prod3", "prod4", "prod5", "prod6", "prod6_bad"]:
    for kind in ["defect", "good"]:
        pool = [b for b, pr, k, rev in packs if pr == prod and k == kind]
        nonrev = [b for b, pr, k, rev in packs if pr == prod and k == kind and not rev]
        if not pool: continue
        rng.shuffle(nonrev)
        f = F_DEF if kind == "defect" else F_GOOD
        n = int(round(len(nonrev) * f))
        picked = nonrev[:n]
        for b in picked: hold.append(b); labels[b] = 1 if kind == "defect" else 0
        print(f"{prod:9} {kind:6} {len(pool):8} {len(nonrev):8} {len(picked):9}")

with open(f"{ROOT}/data/holdout.txt", "w") as f:
    f.write("\n".join(sorted(hold)) + "\n")
with open(f"{ROOT}/data/holdout_labels.csv", "w") as f:
    f.write("name,label\n" + "\n".join(f"{b},{labels[b]}" for b in sorted(hold)) + "\n")
nd = sum(labels.values()); ng = len(hold) - nd
print("-" * 46)
print(f"HOLD-OUT: {len(hold)} packs = {nd} defect + {ng} good  ->  data/holdout.txt")
print("All reviewed packs remain in training for both models.")
