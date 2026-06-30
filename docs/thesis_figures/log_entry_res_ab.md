## H. Higher-resolution retrain + resolution A/B (2026-06-27, later)

New reviewed export `annotations 7_44.xml` → **17 reviewed prod2 packs** (was 11), 44 good, 8 defect, 3 exclude. Uploaded to box as `data/annotations/prod2_reviewed.xml`.

`src/train_reviewed.py` parametrized with `--img`/`--batch`; output names `best_lite_reviewed_<IMG>.pt`. Trained BOTH resolutions on the SAME 17-reviewed set (clean A/B, only resolution differs), fine-tuned from `best_lite.pt`:
- **384** (batch 16): val 0.938 (prod2 0.949) → `best_lite_reviewed_384.pt`
- **512** (batch 12): val 0.942 (prod2 0.950) → `best_lite_reviewed_512.pt` + `seal_lite_reviewed_512.onnx`

**Boundary A/B (`eval_boundary.py`, reads ck["img"] for input size):**

| metric | 384 (prod2/overall) | 512 (prod2/overall) |
|---|---|---|
| Dice | 0.919 / 0.922 | 0.934 / 0.938 |
| Boundary-IoU | 0.500 / 0.496 | 0.566 / 0.574 |
| HD95 (px) | 4.99 / 4.84 | 4.13 / 4.00 |
| ASSD (px) | 2.03 / 2.03 | 1.65 / 1.61 |

**Conclusion:** raising input resolution 384→512 measurably sharpens edges — Boundary-IoU +0.07, ASSD −19%, HD95 −17%, Dice barely moves (as expected). Trade-off: 512 inference ≈1.8× slower on the CPU/Rust target. Use 512 for label-generation quality; revisit res for the shipped model based on the latency budget.

**Label regeneration + merge:**
- `predict_to_cvat_lite.py` now reads ck["img"] (predicts at native res) and uses HALF the points (simp step 9→18, straight_every 16→32) → avg **47 pts/polygon** (was ~85). 512 model: prod2 empties **31** (was 42 at 384-reviewed, 119 originally).
- `merge_annotations.py` → `prod2_merged.xml`: 17 reviewed packs' polygons kept as GT, 3 exclude kept, 581 non-reviewed given new 512 polygons, 30 empties; all 72 tags (reviewed/good/defect/exclude) preserved. Safe to re-import to CVAT.

**Workflow rule learned:** CVAT "Upload annotations" REPLACES the task; always export first, then merge (preserve tags + reviewed GT) before re-importing.
