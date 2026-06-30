
## N. END-TO-END deployment pipeline (2026-06-28)

`src/defect_pipeline.py`: on an UNLABELED pack -> [seal model best_lite_reviewed_1280.pt] predict seal ring -> ring_contours (convex outer + hole/erode inner) -> unroll -> [defect model defect_strip.pt] -> defect mask -> map detections back to raw via unroll maps -> composite (pack crop + predicted seal mask cyan + red circles on defects; predicted-seal strip below). Verdict = DEFECT if any detection else OK.
Run on the 6 held-out defect test packs + 8 random packs: **6/6 defects flagged DEFECT, 8/8 random packs OK (no false positives)** — using the PREDICTED seal (no GT). Composites `outputs/defect_pipeline/` (pulled to ~/Downloads/defect_pipeline). Fig `docs/thesis_figures/fig_pipeline_endtoend.png`. This is the real two-stage product: raw -> seal seg -> unroll -> defect seg -> pack verdict. (Small sample; same scarcity caveats as Model B.)
