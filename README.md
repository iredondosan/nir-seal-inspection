# nir-seal-inspection

NIR line-scan seal inspection of food trays — deep-learning segmentation of the heat-sealed flange (and defect detection) optimized for CPU / Rust edge deployment.

A NIR line-scan camera images trays crossing a gap between two conveyors. The goal is to inspect the heat-sealed flange (the "seal") of each tray for defects. Because the camera is **free-running** (no encoder trigger), packs appear with non-rigid wavy distortion, so the pipeline **follows the real seal edges** rather than trying to globally rectify the image.

## Pipeline
1. Remove black side bands and detect the pack (per-column conveyor reference, background subtraction).
2. Segment the **seal ring** (outer flange edge − inner well edge) with a lightweight U-Net.
3. Unwrap the seal into a flattened strip for inspection.
4. (In progress) classify / segment **defects** on the strip.

## Model
- U-Net with a **MobileNetV3-small** encoder, pack-cropped 384×384 input (~3.6M params).
- Trained across multiple products via an active-learning loop: auto-generate CVAT pre-annotations → human-correct a small set → fine-tune.
- Deployment artifact: **INT8 ONNX (~4.2 MB)** run via Rust `ort` on low-power x86 (no GPU).

## Evaluation
Region overlap (Dice) hides thin-ring edge errors, so evaluation also reports **Boundary-IoU, HD95, ASSD**. An inference-time **quality score** (geometry + probability-map confidence, no ground truth needed) flags low-confidence predictions for review.

## Repository layout
- `src/` — training, prediction, evaluation, quality-scoring, pre-annotation, unrolling, quantization, and Rust-inference helper scripts.
- `docs/` — project log (`TFM_seal_inspection_LOG.md`) and `thesis_figures/`.
- `notebooks/` — exploratory notebooks.
- `rust_infer/` — Rust ONNX inference app.

**Not tracked in git** (large / IP — kept on the workstation): `data/` (images + annotation XMLs), `models/` (`*.pt` / `*.onnx` weights), `outputs/`, `.venv/`.

## Context
Master's thesis (TFM) project. Seal defects are ultimately about hermetic seal integrity; the visual labels are a documented proxy.
