"""
make_strips.py — build the DEFECT dataset (Stage 2).

For every LABELED pack we unroll the seal into a strip and unroll its defect
polygons into a matching mask (same mapping, so the defect deforms with the
pixels). Output is a set of (strip, mask) PNG pairs split into train/test.

Key choices:
  * PACK-LEVEL, PER-PRODUCT split — every product contributes defects to the test
    set, and no scan of a test pack ever appears in train (no leakage).
  * Tiny products (e.g. prod6's 3 distinct defects) use FORCE_TRAIN / FORCE_TEST to
    pin specific packs.

Usage:
    python -m seal_inspection.make_strips --root /path/to/project --out data/strips
"""
from __future__ import annotations
import os
import argparse
import random
from collections import defaultdict
import numpy as np
import cv2
from . import core, cvat

STRIP_H, STRIP_W = 128, 1536       # strip size fed to the defect model
TEST_FRAC = 0.2
SEED = 42

# (annotation xml, image folder, mode) per source.
#   'all'             -> good + defect labeled packs (defect = has a defect polygon)
#   'good'            -> negatives only
#   'defect_reviewed' -> only reviewed packs that have a defect polygon
SOURCES = [
    ("data/annotations/prod1_reviewed.xml", "data/images/prod1", "all"),
    ("data/annotations/prod2_reviewed.xml", "data/images/prod2", "all"),
    ("data/annotations/prod3_reviewed.xml", "data/images/prod3", "all"),
    ("data/annotations/prod6_reviewed.xml", "data/images/prod6", "all"),
]
FORCE_TEST = set()      # prod6 is train-only: never in test
FORCE_TRAIN = set()


def collect_packs(root: str):
    """Return a list of pack dicts {name, w, h, outer, inner, defects, kind, folder}."""
    packs = []
    for xml_rel, img_rel, mode in SOURCES:
        path = os.path.join(root, xml_rel)
        if not os.path.exists(path):
            print("missing:", xml_rel)
            continue
        n = 0
        for node in cvat.iter_images(path):
            tg = cvat.tags(node)
            if "exclude" in tg:
                continue
            seal = cvat.seal_outer_inner(node)
            if seal is None:                  # need both seal polygons to unroll
                continue
            defects = cvat.polygons(node, "defect")
            if mode == "good":
                if not (tg & {"good", "reviewed"}):
                    continue
                defects, kind = [], "good"
            elif mode == "defect_reviewed":
                if "reviewed" not in tg or not defects:
                    continue
                kind = "defect"
            else:  # 'all'
                if not (tg & {"good", "defect", "reviewed"}):
                    continue
                kind = "defect" if defects else "good"
            packs.append(dict(name=node.get("name"), w=int(node.get("width")), h=int(node.get("height")),
                              outer=seal[0], inner=seal[1], defects=defects, kind=kind, folder=img_rel))
            n += 1
        print(f"{xml_rel} [{mode}] -> {n} packs")
    return packs


def split_per_product(packs):
    """Assign each pack to 'train' or 'test', stratified per (folder, kind)."""
    rng = random.Random(SEED)
    buckets = defaultdict(list)
    for p in packs:
        buckets[(p["folder"], p["kind"])].append(p)
    test_ids = set()
    for (folder, kind), lst in sorted(buckets.items()):
        rng.shuffle(lst)
        for p in lst:
            if p["name"] in FORCE_TEST:
                test_ids.add(id(p))
        rest = [p for p in lst if p["name"] not in FORCE_TEST and p["name"] not in FORCE_TRAIN]
        k = int(len(rest) * TEST_FRAC)
        if kind == "defect" and len(rest) >= 2 and k == 0:
            k = 1                              # guarantee a defect in test if the product has >=2
        for p in rest[:k]:
            test_ids.add(id(p))
    return test_ids


def main():
    ap = argparse.ArgumentParser(description="Build the defect (strip, mask) dataset.")
    ap.add_argument("--root", default="/home/ubuntu/TFM/seal-inspection")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(a.root, "data/strips")
    for sp in ("train", "test"):
        for k in ("img", "mask"):
            os.makedirs(f"{out}/{sp}/{k}", exist_ok=True)

    packs = collect_packs(a.root)
    test_ids = split_per_product(packs)
    counts = {"train": [0, 0], "test": [0, 0]}
    for p in packs:
        gray = cv2.imread(os.path.join(a.root, p["folder"], p["name"]), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        sp = "test" if id(p) in test_ids else "train"
        map_x, map_y = core.unroll_maps(p["outer"], p["inner"], STRIP_H, STRIP_W)
        strip = cv2.remap(core.normalize(gray), map_x, map_y, cv2.INTER_LINEAR, borderValue=0)
        # rasterize defects on the full frame, then unroll with the SAME mapping
        dm = np.zeros((p["h"], p["w"]), np.uint8)
        for d in p["defects"]:
            cv2.fillPoly(dm, [d.astype(np.int32)], 255)
        smask = (cv2.remap(dm, map_x, map_y, cv2.INTER_LINEAR, borderValue=0) > 127).astype(np.uint8) * 255
        base = os.path.splitext(p["name"])[0]
        cv2.imwrite(f"{out}/{sp}/img/{base}.png", strip)
        cv2.imwrite(f"{out}/{sp}/mask/{base}.png", smask)
        counts[sp][0 if p["kind"] == "good" else 1] += 1

    print(f"train: {counts['train'][1]} defect / {counts['train'][0]} good")
    print(f"test:  {counts['test'][1]} defect / {counts['test'][0]} good")


if __name__ == "__main__":
    main()
