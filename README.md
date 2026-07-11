# NIR Seal Inspection

**Deep-learning inspection of the heat-sealed flange of food trays, from NIR line-scan images — optimized for CPU / edge (Rust) deployment without a GPU.**

A NIR line-scan camera images trays crossing the gap between two conveyors. The goal is to inspect the heat-sealed flange (the *seal*) of each tray for contamination and sealing defects. Because the camera is **free-running** (no encoder trigger), packs appear with non-rigid wavy distortion, so the pipeline **follows the real seal edges** instead of globally rectifying the image.

![Pipeline](docs/thesis_figures/fig_pipeline_endtoend.png)

This repository accompanies a Master's thesis (TFM). It contains the full source used to **train, evaluate and deploy** the system, and — crucially — a table that maps **every number and figure in the thesis to the exact script that produced it** (see [`REPRODUCE.md`](REPRODUCE.md)).

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
| Seal Dice, zero-shot across products (LOPO) | **0.955 ± 0.010** | `experiments/lopo_seal.py` |
| Defect AUROC (isolated GT strip) | **0.978** | `evaluation/eval_e2e.py` |
| End-to-end AUROC (deployed) | **0.968** | `evaluation/eval_e2e.py` |
| End-to-end AUROC (5-fold CV) | **0.975 ± 0.008** | `experiments/kfold_cv.py` |
| Operating point @0.5 | recall **21/23**, FP **8/156 (5.1 %)** | `evaluation/eval_thresholds.py` |
| PatchCore baseline (unsupervised) | AUROC 0.800 | `experiments/baseline_patchcore.py` |
| Deployed seal latency (CPU) | **68 ms** torch / **42 ms** ONNX / **19 ms** INT8 (@384, i7-12700K, 4 threads) | `deploy/bench_cpu.py` |

Full table-by-table traceability in [`REPRODUCE.md`](REPRODUCE.md).

## Repository layout

```
seal_inspection/     Deployment library: pipeline primitives (core, pipeline, unroll, cvat)
  reference/         Didactic, simplified trainers (fixed epochs, no early stopping)
training/            Real trainers that produce the deployed weights
experiments/         Ablations + cross-validation (one script per thesis result)
evaluation/          Evaluation / verification tools (Dice, boundary, thresholds, e2e)
data_prep/           Dataset construction + CVAT-assisted annotation
figures/             Scripts that regenerate the thesis figures
deploy/              ONNX export, INT8 quantization, CPU benchmarks, e2e demo
rust_infer/          Rust ONNX inference app (edge deployment)
demo/                Interactive Streamlit demo (ONNX, CPU; assets via Releases)
docs/                SOURCE_OF_TRUTH.md (internal ledger), project log, thesis figures
archive/exploratory/ One-off diagnostics and non-reported experiments (kept for history)
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
# download the demo bundle (ONNX models + sample packs) from the latest Release
# into demo/models/ and demo/samples/, then:
cd demo && streamlit run app.py    # opens http://localhost:8501
```

The images (client IP) and model weights are **not** in git; the demo assets (5 ONNX models + sample packs) are published under [Releases](../../releases). The demo is self-contained — it does not require the training stack.

## Deployment

The seal stage is exported to **INT8 ONNX (~4.2 MB)** and runs via the Rust `ort` app in `rust_infer/` on low-power x86 (no GPU), ~19 ms/pack single-thread. Region overlap (Dice) hides thin-ring edge errors, so evaluation also reports **Boundary-IoU, HD95, ASSD**, and an inference-time **quality score** (`deploy/quality_score.py`, no ground truth needed) flags low-confidence predictions for review.

## Data availability

The NIR images and annotations are proprietary and are **not** released. The code is fully public and every result is traceable to its script; the demo bundle provides a runnable subset. Contact the author for research access to additional data.

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
