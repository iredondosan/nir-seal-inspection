# Architecture

How the NIR seal-inspection system is put together, and why. This complements the
per-file docstrings and the result-to-code map in [`REPRODUCE.md`](REPRODUCE.md).

## 1. Imaging and the core constraint

A near-infrared (NIR) **line-scan** camera images food trays crossing the gap between two
conveyors, backlit so the sensor sees IR *transmission*: dense/wet material (product,
contamination, liquid) absorbs IR and appears **dark**; thin air gaps appear **bright**.

Two facts shape the whole design:

1. **Free-running camera** (no encoder trigger) → every scan carries its own non-rigid,
   wavy distortion. No single rotation/affine/homography can straighten it, so the system
   **never globally rectifies**. Instead it *follows the seal's real edges* and unwraps the
   seal band into a flat strip — making downstream defect detection position-independent.
2. **The seal is a thin ring** (the heat-sealed flange = outer flange edge − inner well
   edge). Region-overlap metrics (Dice) hide thin-ring boundary errors, so evaluation also
   uses **Boundary-IoU, HD95, ASSD**.

## 2. Data flow

```
raw NIR frame (grayscale, line-scan)
      │
      ▼
 pack_bbox ─► crop to tray ─► normalize (1st/99.5th percentile contrast stretch)
      │
      ▼
╔═══════════════════════════════╗
║ Stage 1 · Seal U-Net          ║   MobileNetV3-small encoder (ImageNet), 3.59 M params
║ input 3 × 1280 × 1280         ║   → per-pixel seal-ring probability map
╚═══════════════════════════════╝
      │  mask_to_ring  → outer + inner ring contours
      ▼
 unroll_maps ─► flatten the ring along the outer contour's normal (correspondence-free)
      │
      ▼   128 × 1536 strip
╔═══════════════════════════════╗
║ Stage 2 · Defect U-Net        ║   ResNet18 encoder (ImageNet), 14.33 M params
║ input 3 × 128 × 1536          ║   (or TinyUNet, grayscale 1 × 128 × 1536, 0.93 M)
╚═══════════════════════════════╝
      │  score = max( GaussianBlur( sigmoid, σ=2 ) )
      ▼
  verdict:  DEFECT if score ≥ threshold (0.5)  else  GOOD
      │  detections are back-projected onto the pack via the same unroll maps
```

All shared imaging/geometry primitives live in [`seal_inspection/core.py`](seal_inspection/core.py);
the end-to-end inference driver is [`seal_inspection/pipeline.py`](seal_inspection/pipeline.py).

## 3. Preprocessing

- **`normalize`** — percentile (1 / 99.5) contrast stretch to 0–255. Robust min/max avoids a
  few hot/dead pixels squashing contrast, keeping the network input consistent across scans.
- **`pack_bbox`** — classical (no learning) tray detector: builds a per-column *conveyor
  reference* from the top/bottom strips (which are conveyor, not pack), subtracts it,
  thresholds, cleans morphologically (open 11×11 → close 41×41), takes the largest blob and
  adds a 40 px margin. The crop makes the seal fill the frame so resolution is not wasted.

## 4. Stage 1 — Seal segmentation

- **Model**: U-Net with a **MobileNetV3-small** encoder (ImageNet-pretrained), 1-channel
  content replicated to 3 channels, ImageNet-normalized. **3.59 M** params, trained at
  **1280 px** (native resolution chosen by the resolution ablation — boundary quality
  improves monotonically with input size; see `REPRODUCE.md` Table 4.2).
- **Output → ring** (`mask_to_ring`): morphological open 9 / close 35, then
  - **outer** = the largest external contour (kept *raw* so it follows the wavy flange edge),
  - **inner** = the hole inside the ring, or — if the net failed to open the centre — the
    outer eroded by a typical band width (`band_px = 90`) so a closed ring is still returned.
  - `_clean_contour` resamples each contour to **360 arc-length points** and circularly
    smooths it, turning a ragged pixel boundary into a clean, hand-drawn-like ring.
- **Why lightweight**: the seal stage is the latency bottleneck; a MobileNetV3-small encoder
  matches a ResNet34 baseline's Dice with 6.8× fewer parameters and ~2.3× lower CPU latency
  (Table 4.8), enabling GPU-free edge deployment.

## 5. Unrolling — flattening the ring (`unroll_maps`)

The production method is **perpendicular-to-outer** and *correspondence-free*:

1. Resample the smoothed outer contour to `strip_w = 1536` points; compute the local
   **normal** at each point (oriented inward via a probe against the filled outer mask).
2. Band depth `L` at each column = distance from the outer contour to the inner edge
   (`distanceTransform` of the inner mask), circularly smoothed.
3. Sample `strip_h = 128` rows along each normal at `a = linspace(-0.15, 1.15)`
   (row 0 = outer edge, row 127 = inner edge, ±15 % margin past both).

This produces `(map_x, map_y)` grids that `cv2.remap` applies to the image. The **same maps
back-project** predicted defects onto the original pack (no train/inference mismatch, and the
defect deforms identically to the pixels it sits on). A legacy variant (`unroll_maps_legacy`,
linear outer↔inner interpolation) is kept as the second branch of an exploratory ensemble.

## 6. Stage 2 — Defect segmentation

- **Model**: U-Net with a **ResNet18** encoder (ImageNet), **14.33 M** params, run on the
  128×1536 strip (3-channel, ImageNet-normalized). A compact alternative, **`TinyUNet`**
  (grayscale-native 1-channel, channels [16, 32, 64, 128], **0.93 M** params), nearly matches
  it — a capacity ablation (Table 4.6). See [`seal_inspection/tiny_unet.py`](seal_inspection/tiny_unet.py).
- **Pack score** = `max( GaussianBlur( sigmoid(logits), σ=2 ) )` — the peak smoothed defect
  probability over the strip. **Verdict** = DEFECT if score ≥ threshold (operating point 0.5;
  the sensitivity/false-alarm trade-off is tunable — Table 4.4).

## 7. Training

| Stage | Script | Key configuration |
|---|---|---|
| Seal | [`training/train_seal.py`](training/train_seal.py) | Fine-tune from an ImageNet MobileNetV3-small base; 1280², AdamW 1e-4, cosine, 40 epochs, batch 2, BCE+Dice, best-checkpoint (no early stopping), copy-paste augmentation P=0.8 |
| Defect | [`training/train_defect.py`](training/train_defect.py) | ResNet18/ImageNet on 128×1536 strips; AdamW 2e-4, ReduceLROnPlateau, **early stopping** (patience 25), BCE(pos_weight=20)+Dice, batch 8, 1200 steps/epoch, piece-level 15 % validation, copy-paste P=0.7, `--sealjit`/`--roll` strip augmentations |
| Compact | [`training/train_tiny.py`](training/train_tiny.py) | TinyUNet from scratch, early stopping |

**Copy-paste augmentation** pastes real defect cut-outs onto clean strips (random scale,
rotation, intensity, feathered edge) with the mask updated accordingly — multiplying defect
variety without capturing more defective packs.

## 8. Data and leakage-free evaluation

- Only **reviewed** pieces (human-verified ring masks, 119 packs) train the seal model.
- The global **hold-out** is composed entirely of **non-reviewed** pieces → the seal stage has
  never seen them, so the two-stage system is **leakage-free by construction**. The hold-out is
  never used for training, model/early-stopping selection, threshold setting, or the copy-paste
  library. See `REPRODUCE.md` for the exact result-to-script mapping.
- **Assisted labeling**: polygon → mask → model inference → mask → polygon → re-imported into
  CVAT as a pre-annotation a human only touches up ([`seal_inspection/cvat.py`](seal_inspection/cvat.py),
  [`data_prep/predict_to_cvat*.py`](data_prep)).

## 9. Deployment

The deployed pipeline runs FP32 ONNX on CPU (seal @1280 ≈ 342 ms, full pack ≈ 630 ms → ~100 packs/min at 4 threads; `results/latency.json`). The fast option is **FP32 @384 px** — same E2E AUROC 0.977 / 21-of-23 recall as @1280 at ~26 ms (13× faster), trading only contour precision (Dice 0.936 vs 0.963). **Static INT8 is not deployable as-is**: it keeps ~97% pixel agreement but fragments the thin ring, so ring localisation fails on most packs (`results/int8_quality.json`) — a documented negative result. A Rust `ort` benchmark stub in [`rust_infer/`](rust_infer) (source only) times the seal net on edge x86.
An inference-time **quality score** (geometry + probability-map confidence, no ground truth
needed — [`deploy/quality_score.py`](deploy/quality_score.py)) flags low-confidence
predictions for review. CPU latencies are in `REPRODUCE.md` (Table 4.8).

## 10. Design rationale, in one line each

- **Follow edges, don't rectify** — a free-running camera's distortion is non-rigid.
- **Two stages** — localize the seal first so defect detection sees a canonical strip.
- **Lightweight encoders** — the target is GPU-free CPU/edge inference.
- **Correspondence-free unroll** — robust to small contour errors; stable for thin defects.
- **Boundary metrics** — Dice hides thin-ring edge errors; Boundary-IoU/HD95/ASSD do not.
- **Leakage-free hold-out** — the honest measure of a two-stage system.
