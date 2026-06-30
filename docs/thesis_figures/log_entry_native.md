## I. Native-resolution retrain + 3-way resolution sweep (2026-06-27, latest)

New reviewed export `annotations 8_32.xml` → **19 reviewed prod2 packs** (44 good, 8 defect, 3 exclude). Uploaded to box as `data/annotations/prod2_reviewed.xml`.

**"Native" = 1280×1280 input** (the pack crop is ~1220×1330 median across products, max ~1550; 1280 ≈ crop-native, divisible by 32, minimal resampling). The model is fully-convolutional (MobileNetV3-small U-Net), so it takes any /32 size. Added **AMP (mixed precision)** so 1280² fits in 16 GB at batch 2; smoke-tested 1 epoch before the full run. `train_reviewed.py` now takes `--img/--batch/--epochs/--samples`. Native run: `--img 1280 --batch 2 --epochs 40 --samples 240`, fine-tuned from `best_lite.pt`, ~25 s/epoch → `best_lite_reviewed_1280.pt` + `seal_lite_reviewed_1280.onnx`. **val 0.959** (prod1 0.969, prod2 0.948, prod3 0.954, prod4 0.963, prod5 0.962).

**3-way boundary sweep (same 19 reviewed packs, `eval_boundary.py` reads ck["img"]):**

prod2 (19 packs):
| metric | 384 | 512 | 1280 (native) |
|---|---|---|---|
| Dice | 0.918 | 0.934 | 0.952 |
| Boundary-IoU | 0.497 | 0.564 | 0.663 |
| HD95 (px) | 5.05 | 4.23 | 3.23 |
| ASSD (px) | 2.04 | 1.66 | 1.23 |

overall (59 packs): Boundary-IoU 0.495 → 0.573 → **0.722**; ASSD 2.04 → 1.61 → **0.98** (sub-pixel mean edge error).

**Conclusion:** edge quality improves **monotonically** with input resolution; native gives the biggest jump (prod2 B-IoU +0.10 over 512, ASSD −26%). Figure: `fig_resolution_sweep.png`. **Trade-off:** 1280 inference ≈6× the 512 cost on CPU/Rust — so native is the **accuracy upper bound / best label-generator**; for the shipped model stay at 384–512 or use two-stage (coarse-locate → crop seal band → high-res refine).

**Label regeneration:** native model preann → prod2 empties **8** (was 31 at 512, 119 originally). Re-thinned to **~77 pts/polygon** (curvature-adaptive, `thin_polys.py`, target ~80).

**Merge rule updated (`merge_annotations.py`):** packs tagged **reviewed/good/defect are now all protected** (existing polygons kept as GT); `exclude` kept empty; only **untagged** packs get new native-model polygons. Latest `prod2_merged.xml`: 52 protected, 568 new model polygons, 8 fallback, all 74 tags preserved. CVAT gotcha still applies: Upload REPLACES the task → always export first, then merge.
