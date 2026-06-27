# Seal inspection — session log 2026-06-27

Everything tried this session, with results. Figures referenced live in `docs/thesis_figures/`.

## A. Methodology decisions (thesis)
- **Single-annotator ground truth** is acceptable for a Master's thesis, with: a written annotation protocol, anchoring to producer/QC labels where they exist, an intra-annotator consistency re-label of a subset, ideally a second annotator on a sample (Cohen's κ), and disclosure as a limitation. Visual seal labels are a *proxy* for functional seal integrity.
- **Stickers over the seal** → tag image-level `exclude` (do not delete); pipeline filters `exclude` out of train/val/test. Documented exclusion criterion.
- **Defect definition** → location in the seal weld, NOT size. Product material *in the weld band* = defect regardless of size (capillary leak path); material outside the weld line = cosmetic. Truly borderline → `uncertain` tag + exclude from clean sets. Bias toward calling defect when in doubt (false negative worse in food safety).
- **Provenance** → one class `sellado`; track human-verified packs with an image-level `reviewed` tag. NEVER train on unreviewed model output (feedback-loop drift). CVAT also records `source="auto"|"manual"` per shape (secondary signal).

## B. Evaluation: boundary metrics matter (Dice hides edge errors)
- `src/eval_boundary.py` — adds Boundary-IoU, HD95, ASSD to Dice/IoU; per-product; marks trainer's held-out val. `--split val|train|all --bd <px> --model <pt>`.
- **Finding:** on held-out val, **Dice ~0.92 but Boundary-IoU ~0.47, HD95 ~4.9px, ASSD ~2.2px** → region overlap looks excellent while the thin-ring *edge* match is mediocre. Report boundary metrics alongside Dice in the thesis.

## C. Inference-time quality score (no GT) — review/active-learning queue
- `src/seal_quality.py` — geometry (closed ring? band-width CV?) + prob-map confidence (uncertain-fraction, edge sharpness) + optional `--tta`. CSV sorted worst-first.
- prod2: flags 50/631 (8%); broken `no_closed_ring` packs rank dead-last (0.05 vs 0.90+).
- Caveat: validated for catching gross failures; fine-grained ranking unproven (labeled packs all good; no GT on bad ones).

## D. LLM-judge polygon-refine loop (proven on 1 pack)
- Concept: specialized models GENERATE candidate rings (pixel work); the LLM JUDGES/diagnoses/selects (never outputs coordinates). LLM strong at critiquing an overlay, weak at producing precise coordinates.
- `src/refine_candidates.py --image <name> [--model <pt>]` — candidates: A seg@0.50, B seg@0.35, C morph-close@0.40, D TTA@0.50, E geometric outer+offset, F hybrid (convex-hull outer + inner = real∩offset), G D with deep-only convexity-defect bridging.
- On `seal_159` (seg model totally failed = no ring): user judged **D (TTA) closest to truth** (follows real edge) but it pinches at the bottom product-overflow and has a top-left outer notch. E/F too rigid (user rejects geometric for wavy packs). G bridged the notch but a single 30px defect spanned a whole side.
- **Fig:** `fig_candidate_comparison_DEF.png` (D|E|F).

## E. Learn the corner from reviewed packs → RETRAIN (the real fix)
- Studied reviewed GT corners (packs 157,161,155): correct outer corner is a tight rounded corner hugging the true tray edge, outside the barcode, uniform band. **Fig:** `fig_reviewed_GT_corners.png`.
- Conclusion: the corner was a **labels problem**, not a heuristic one. Per-pack geometry hacks are brittle; teach the model from reviewed GT instead.
- `src/train_reviewed.py` (copy of train_multiprod) fine-tunes prod2 on the **11 human-`reviewed` packs only** (filters: require `reviewed`, skip `exclude`, `seal_mask` uses label=='sellado' so defect polygons are ignored) + prod1/3/4/5 → `models/best_lite_reviewed.pt` + `seal_lite_reviewed.onnx`. Kept SEPARATE from `best_lite_multiprod.pt` for A/B.
- **Result val 0.937 (prod2 0.943).** Re-running candidates on seal_159 with the reviewed model: plain seg@0.50 went FAIL(no ring) → **closes the ring on its own**; outer corner-notch depth 30px → 24px (gone, no heuristic). **Fig:** `fig_corner_fix_OLDvsREV.png` (old vs reviewed model, candidate D — corner notch fixed).

## F. Quantitative A/B (old `best_lite_multiprod` vs `best_lite_reviewed`)
- **Label regeneration:** prod2 pre-annotation empties **119 → 42** (589/631 now produce a polygon). Big reduction in manual drawing. Output: `outputs/preannotations_seal/prod2_reviewed.xml`.
- **Boundary metrics on the 11 reviewed prod2 packs** (`eval_boundary --split all`):
  | metric | old | reviewed |
  |---|---|---|
  | Dice | 0.921 | 0.921 |
  | Boundary-IoU | 0.548 | 0.497 |
  | HD95 (px) | 6.11 | 5.02 |
  | ASSD (px) | 2.02 | 2.03 |
- **Honest read:** retrain reduced worst-case edge error (HD95 ↓) and fixed gross failures (ring closure, corner divert), but aggregate Boundary-IoU is flat-to-slightly-down. NOT a clean test: n=11, 9 in the reviewed model's training. Clean generalization number needs a larger held-out reviewed set (the next review batch).

## G. Next steps
1. Review the regenerated `prod2_reviewed.xml` (42 empties + spot-check), tag `reviewed`; build a proper held-out test set (human-drawn, never pre-annotated).
2. Retrain on the bigger reviewed set; re-measure boundary metrics for a clean generalization number.
3. Reconsider `exclude` tags: seal_159 is recoverable (not a sticker); bottom product-in-band may be a DEFECT.
4. Later: unrolled-strip defect dataset + classifier; drift monitoring + retraining triggers (MLOps framing).
