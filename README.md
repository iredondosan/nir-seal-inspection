# NIR Seal Inspection

**Deep-learning inspection of the heat-sealed flange of food trays, from NIR line-scan images — optimized for CPU / edge deployment without a GPU.**

A NIR line-scan camera images trays crossing the gap between two conveyors. The goal is to inspect the heat-sealed flange (the *seal*) of each tray for contamination and sealing defects. Because the camera is **free-running** (no encoder trigger), packs appear with non-rigid wavy distortion, so the pipeline **follows the real seal edges** instead of globally rectifying the image.

This repository accompanies a Master's thesis (TFM). It contains the full source used to **train, evaluate and deploy** the system, and — crucially — a table that maps **every number and figure in the thesis to the exact script that produced it** (see [`REPRODUCE.md`](REPRODUCE.md)). The full system design and a data-flow diagram are in [`ARCHITECTURE.md`](ARCHITECTURE.md). Each experiment's design and caveats are in [`EXPERIMENTS.md`](EXPERIMENTS.md).

---

## Two-stage pipeline

1. **Pre-process** — remove the black side bands, detect the tray (per-column conveyor reference + background subtraction), percentile contrast stretch.
2. **Stage 1 · Seal segmentation** — a lightweight **U-Net (MobileNetV3-small, 3.59 M)** segments the seal *ring* (outer flange edge − inner well edge) at 1280 px.
3. **Unroll** — the ring is flattened into a **128×1536 strip** by marching perpendicular to the outer contour (correspondence-free).
4. **Stage 2 · Defect segmentation** — a **U-Net (ResNet18, 14.33 M)** segments defects on the strip; the pack score is `max(GaussianBlur(sigmoid, σ=2))`.
5. **Verdict** — DEFECT if score ≥ threshold, else GOOD.

## Headline results

| Metric | Value | Backed by |
|---|---|---|
| Seal Dice (validation) | **0.967** | `evaluation/eval_seal.py` |
| Seal Dice, zero-shot across products (LOPO) | **0.955 ± 0.013** | `experiments/lopo_seal.py` |
| Defect AUROC (isolated GT strip) | **0.978** | `evaluation/eval_e2e.py` |
| End-to-end AUROC (deployed) | **0.968** | `evaluation/eval_e2e.py` |
| End-to-end AUROC (5-fold CV) | **0.977 ± 0.004** | `experiments/kfold_cv.py` |
| Operating point @0.5 | recall **21/23**, FP **8/156 (5.1 %)** | `evaluation/eval_thresholds.py` |
| PatchCore baseline (unsupervised, greedy coreset) | AUROC **0.776** | `experiments/baseline_patchcore.py` |
| Deployed latency (CPU, ONNX) | seal **342 ms** @1280 / **26 ms** @384; full pipeline **~630 ms → ~100 packs/min**; INT8 not deployable | `demo/bench_latency.py` |

Full table-by-table traceability in [`REPRODUCE.md`](REPRODUCE.md).

## Repository layout

```
seal_inspection/     Deployment library — pipeline primitives
  core.py            normalize, pack_bbox, mask_to_ring, unroll_maps
  pipeline.py        end-to-end two-stage inference
  tiny_unet.py       compact grayscale defect U-Net
  cvat.py            CVAT polygon <-> mask helpers
  reference/         didactic, simplified trainers (fixed epochs, no early stopping)
training/            train_seal.py · train_defect.py · train_tiny.py
experiments/         lopo_seal.py · kfold_cv.py · ablation_*.py · baseline_patchcore.py
evaluation/          eval_seal.py · eval_boundary.py · eval_thresholds.py · eval_e2e.py
data_prep/           make_{masks,strips,holdout}.py · predict_to_cvat*.py
deploy/              quantize_int8.py · bench_cpu.py · export_demo_onnx.py · pipeline_e2e.py
figures/             thesis-figure scripts (need the private data)
demo/                Interactive Streamlit demo (ONNX, CPU; assets via Releases)
docs/                SOURCE_OF_TRUTH.md (result-to-code traceability ledger)
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # installs the seal_inspection package + pinned deps
# or: pip install -r requirements.txt
```

## Reproducing the results

Each thesis table/figure has a named script and a `make` target:

```bash
make table-4.1        # per-product seal Dice
make table-4.4        # defect threshold sweep (operating point)
make table-4.7        # augmentation ablation
make deploy-onnx      # export + INT8 quantize + CPU benchmark
make help             # list every target
```

These run the real scripts and **require the dataset and trained weights locally** (not in git — see below). The exact command and expected number for every result are listed in [`REPRODUCE.md`](REPRODUCE.md).

## Try it — interactive demo

An interactive **Streamlit** app runs the whole pipeline on CPU (**ONNX Runtime, no PyTorch**): pick a sample pack or upload your own NIR image, move the decision-threshold slider live, and watch the seal → unrolled strip → defect verdict with a per-stage latency breakdown. Switch seal/defect models on the fly to see the resolution and capacity ablations in action.

```bash
pip install -r demo/requirements.txt
for f in seal.onnx seal_512.onnx seal_384.onnx defect.onnx defect_tiny.onnx; do curl -sL -o demo/models/$f https://github.com/iredondosan/nir-seal-inspection/releases/download/v1.0.0/$f; done
cd demo
streamlit run app.py
```

The app opens at http://localhost:8501 — use the *upload* mode for your own NIR pack.

The client's NIR images are proprietary and are **not** shared, so the demo ships **no sample images** — use the *upload* mode with your own grayscale NIR pack. Only the model weights (5 ONNX, published under [Releases](../../releases)) are provided. The demo is self-contained — it does not require the training stack.

## Deployment

The deployed pipeline runs **FP32 ONNX on CPU** (no GPU). On an i7-12700K (4 threads) the two stages take **~342 ms** (seal @1280 px) and **~65 ms** (defect); with cropping and unrolling, a **full pack takes ~630 ms → ~100 packs/min**, within typical line cadences (`results/latency.json`, `demo/bench_latency.py`).

For tighter compute budgets, running the **seal at 384 px** keeps the same end-to-end **AUROC (0.977, recall 21/23)** while cutting the seal stage **13×** (342 → 26 ms) and the full pack **~2.8×** (630 → 226 ms, ~264 packs/min), at the cost of some contour precision (Dice 0.936 vs 0.963; `results/int8_quality.json`).

**Static INT8 is not deployable as-is**: despite ~97 % pixel agreement with FP32 it fragments the thin seal ring, so ring localisation fails on most packs (61/179 @384, 0/179 @1280) — a documented negative result (`results/int8_quality.json`).

Because region overlap (Dice) hides thin-ring edge errors, evaluation also reports **Boundary-IoU, HD95 and ASSD**, and an inference-time **quality score** (`deploy/quality_score.py`, no ground truth needed) flags low-confidence predictions for review.

## Data availability

The NIR images and annotations are proprietary and are **not** released. The code is fully public and every result is traceable to its script; the demo bundle provides a runnable subset. Contact the author for research access to additional data.

## Contact

Questions, bug reports, or help reproducing results — please [open an issue](https://github.com/iredondosan/nir-seal-inspection/issues).

## Citation

```bibtex
@mastersthesis{redondo2026sealinspection,
  author = {Ignacio Redondo},
  title  = {Inspección automática de sellado alimentario con visión NIR y aprendizaje profundo},
  school = {Universidad Europea de Valencia},
  note   = {Máster de Formación Permanente en Inteligencia Artificial},
  year   = {2026}
}
```

## License

Code released under the [MIT License](LICENSE). Proprietary data is excluded.
