# seal_inspection

Clean, documented implementation of the NIR line-scan **seal-inspection pipeline**:

```
raw NIR image → seal segmentation → unroll seal into a strip → defect segmentation → pack verdict
```

Two lightweight U-Nets are chained; the unroll between them is a pure geometric
transform, which makes defect detection position-independent.

## Why this design
- **NIR transmission:** dense/wet material (product, contamination, liquid) absorbs
  IR and looks **dark**; thin air gaps look **bright**.
- **Free-running camera:** every scan has its own wavy distortion, so we never
  globally rectify — we **follow the seal's real edges** and unwrap the band.

## Layout
| File | Role |
|---|---|
| `core.py` | Shared imaging/geometry: normalization, pack detection, model inference, **mask↔polygon**, **unroll**. Fully documented. |
| `cvat.py` | Read/write CVAT-1.1 annotations (seal = `sellado` outer+inner; `defect`; image tags). |
| `train_seal.py` | Stage 1 — train the seal segmenter (MobileNetV3-small U-Net) across all products, with copy-paste contaminant augmentation. |
| `make_strips.py` | Build the Stage-2 dataset: unroll each labeled pack's seal + defect mask; per-product, no-leakage train/test split. |
| `train_defect.py` | Stage 2 — train the defect segmenter (ResNet-18 U-Net) with oversampling + copy-paste. |
| `pipeline.py` | End-to-end inference on **unlabeled** packs → QC composites + DEFECT/OK verdict. |

## Annotation conventions (CVAT)
- Seal = **two `sellado` polygons** per pack (outer flange edge + inner well edge).
- `defect` polygons mark defects (drawn on the raw image; clipped to the band on unroll).
- Image tags: `reviewed` (human-verified GT — what training uses), `good`/`defect`
  (pack class), `exclude` (e.g. sticker over the seal → dropped).

## Usage
```bash
# Stage 1 — seal model (native resolution, mixed precision)
python -m seal_inspection.train_seal  --root $ROOT --base models/best_lite.pt --img 1280 --batch 2 --epochs 40

# Stage 2 — defect dataset, then defect model
python -m seal_inspection.make_strips --root $ROOT
python -m seal_inspection.train_defect --root $ROOT

# End-to-end inference on a folder of packs
python -m seal_inspection.pipeline --seal models/seal_1280.pt --defect models/defect_strip.pt \
    --input data/images/prod2 --out outputs/pipeline --limit 20
```

## Mask ⇄ polygon (the round-trip that powers the labeling loop)
- **Polygon → mask** (`core.polygons_to_band_mask`): fill outer, punch out inner → seal-band GT.
- **Mask → polygon** (`core.mask_to_ring` + `core.simplify_contour` / `core.visvalingam`):
  threshold → contours (convex-hull outer, hole/erode inner) → simplified polygon for CVAT.

## Notes
- Not tracked in git: `data/` (images + XML), `models/` (`*.pt`/`*.onnx`), `outputs/`.
- Evaluation uses **boundary-aware** metrics for the seal (Boundary-IoU/HD95/ASSD — Dice
  hides thin-ring errors) and **AUROC + pixel-Dice** for defects; defect splits are
  **pack-level** (no scan of a test pack appears in train).
```
