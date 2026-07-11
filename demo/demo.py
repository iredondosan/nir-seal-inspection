#!/usr/bin/env python3
"""End-to-end demo of the two-stage seal-inspection pipeline.

Runs the *real* deployment pipeline (`seal_inspection.pipeline`) on a folder of sample
packs: seal segmentation -> ring extraction -> unroll -> defect segmentation -> verdict,
writing a QC composite per pack (predicted seal in cyan, detected defects in red, plus the
unrolled strip) to demo/out/.

The NIR images and the 2 GB of weights are NOT in the repository. Download the demo bundle
from the GitHub Release and unpack it into demo/ (weights -> demo/weights/, a few sample
images -> demo/samples/). Then:

    python demo/demo.py                       # default: demo/samples + demo/weights
    python demo/demo.py --input path/to/dir   # your own NIR packs
    make demo

Weights fall back to models/ if present (author / thesis box).
"""
from __future__ import annotations
import argparse
import runpy
import sys
from pathlib import Path

from seal_inspection.paths import REPO_ROOT

HERE = Path(__file__).resolve().parent


def _first(*cands: Path) -> str | None:
    for c in cands:
        if c and Path(c).exists():
            return str(c)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Two-stage seal-inspection demo")
    ap.add_argument("--input", default=str(HERE / "samples"),
                    help="folder with NIR pack images (default: demo/samples)")
    ap.add_argument("--seal", default=_first(HERE / "weights" / "seal.pt",
                                             REPO_ROOT / "models" / "best_lite_reviewed_1280.pt"))
    ap.add_argument("--defect", default=_first(HERE / "weights" / "defect.pt",
                                               REPO_ROOT / "models" / "defect_strip.pt"))
    ap.add_argument("--out", default=str(HERE / "out"))
    a = ap.parse_args()

    if not a.seal or not a.defect:
        sys.exit("Weights not found. Download the demo bundle from the GitHub Release "
                 "into demo/weights/ (seal.pt, defect.pt), or pass --seal/--defect.")
    if not Path(a.input).exists():
        sys.exit(f"Input folder '{a.input}' not found. Put sample packs in demo/samples/ "
                 "or pass --input.")

    # Reuse the real deployment entry point unchanged.
    sys.argv = ["seal_inspection.pipeline",
                "--seal", a.seal, "--defect", a.defect,
                "--input", a.input, "--out", a.out]
    print(f"seal   = {a.seal}\ndefect = {a.defect}\ninput  = {a.input}\nout    = {a.out}\n")
    runpy.run_module("seal_inspection.pipeline", run_name="__main__")


if __name__ == "__main__":
    main()
