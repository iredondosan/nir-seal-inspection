# Reproducción de los resultados de la memoria

Esta tabla ata **cada resultado del TFM** al script exacto que lo produce, el modelo y los datos
que necesita, y el número esperado. Es la versión pública de `docs/SOURCE_OF_TRUTH.md`.

## Requisitos

```bash
pip install -e .          # paquete + dependencias fijadas
```

- **Datos** (`data/`, ~2,2 GB) y **pesos** (`models/`, ~2,0 GB) **no están en git** (tamaño + propiedad
  industrial). Deben estar presentes localmente en la raíz del repo. Las rutas se resuelven con
  `seal_inspection/paths.py` (ancla `ROOT`), así que los scripts corren desde cualquier directorio.
- **Hold-out global vigente**: `data/holdout_labels.csv` = 179 packs (**156 correctos / 23 defectuosos**).
- **CPU vs GPU**: los *entrenamientos* usan GPU; todas las *evaluaciones* y *latencias* corren en CPU.
- **Determinismo**: los experimentos fijan `SEED=42`.

## Resultados → script → número esperado

| Resultado (memoria) | Comando | Modelo / datos | Valor esperado |
|---|---|---|---|
| **Tabla 4.1** · Dice del sellado por producto | `python evaluation/eval_seal.py` | `best_lite_reviewed_1280.pt` + `prod*_reviewed.xml` | prod1 0.971 · prod2 0.950 · prod3 0.969 · prod4 0.975 · prod5 0.972 · prod6 0.969 · **global 0.967** |
| **Tabla 4.2** · Resolución (Dice/B-IoU/HD95/ASSD) | 384/512: `python evaluation/eval_boundary.py --ckpt best_lite_reviewed_{384,512}.pt`  ·  1280: `python experiments/ablation_resolution_1280.py` | modelos `best_lite_reviewed_{384,512,1280}.pt` | 1280 (prod2): Dice 0.954 · B-IoU 0.674 · HD95 3.00 · ASSD 1.15 (monótono con 384/512) |
| **Tabla 4.3** · LOPO del sellado (zero-shot) | `python experiments/lopo_seal.py` | reentrena 5× desde ImageNet (≈85 min, GPU) | zero-shot 0.960/0.934/0.962/0.971/0.949 → **media 0.955 ± 0.013**; en muestra media 0.966; caída ~0.010 (+ métricas de borde) |
| **Tabla 4.4** · Barrido de umbral (punto de operación) | `python evaluation/eval_thresholds.py`  (verif.: `evaluation/eval_tables.py`) | seal + `defect_strip.pt` sobre hold-out | AUROC **0.968**; @0.50 TP21/FP8/TN148/FN2; @0.70 21/6/150/2; @0.85 17/5/151/6; @0.90 15/4/152/8; @0.95 14/2/154/9 |
| **Tabla 4.5** · Transferencia ImageNet vs desde cero | tira: `python experiments/ablation_transfer.py`  ·  e2e: `python experiments/ablation_transfer_e2e.py` | `scratch_seal_1280.pt`, `defect_scratch_es.pt` vs desplegados | sellado Dice 0.967/0.949 · defecto tira 0.978/0.972 · **e2e 0.968/0.891** · anillo cerrado 179/153 · ES 69/141 ép. |
| **Tabla 4.6** · TinyUNet (capacidad) | `python evaluation/eval_tiny_e2e.py`  (verif.: `evaluation/eval_tables.py`) | `tiny_defect.pt` (0.93 M) vs `defect_strip.pt` (14.33 M) | tira GT: resnet18 **0.978** / tiny **0.972**; e2e 0.968 / 0.966 |
| **Tabla 4.7** · Ablación copy-paste (PI-4) | `python training/train_defect.py --sealjit --nopaste --out models/defect_nopaste.pt` · `python evaluation/eval_copypaste.py` | copy-paste ON vs OFF, resto fijo | ON 0.984/0.975 (23/23) · OFF 0.961/0.948 (19/23) → **ΔE2E +0.027** fuera del ruido; PI-4 respondida |
| **Tabla 4.8** · Ablación de aumentación | `python experiments/ablation_augment.py` | `defect_{strip.noaug,roll,jit,rolljit}.pt` sobre hold-out (23 def) | base 0.975/0.953 · +roll 0.980/0.983 · **+sealjit 0.984/0.975 (23/23, 3.8 % FP)** · +ambos 0.976/0.976 |
| **Tabla 4.9** · Despliegue (latencias CPU) | `python demo/bench_latency.py`  ·  INT8: `python deploy/quantize_int8.py` | ONNX (i7-12700K, 4 hilos) | sellado ONNX **342 ms** @1280 / **26 ms** @384 (torch 68 ms ref.); defecto 65/40 ms; pipeline E2E ~630 ms → ~100/min; INT8 **no desplegable** |
| **CV de 5 pliegues** (AUROC e2e) | `python experiments/kfold_cv.py` | reentrena 5× el defecto (GPU) | **AUROC 0.977 ± 0.004** (re-run 07-12; entre corridas ~0.965–0.977 por no-determinismo GPU) |
| **PatchCore** (línea base, coreset greedy) | `python experiments/baseline_patchcore.py` | tiras correctas | AUROC **0.776** (greedy; banco completo 0.784; random 0.694) |
| **Tabla 4.10** · Sistemas E2E (homogéneo) | `python evaluation/eval_systems_e2e.py` | 6 configs, conjunto común 176 | desplegado 0.968·21/23; todas 0.964–0.970 (±0.01) |
| **INT8 (calidad)** | `python evaluation/eval_int8_quality.py` | INT8 vs FP32 @384/1280 | **no desplegable**: localiza 61/179 @384, 0/179 @1280 |
| **Latencia CPU (ONNX)** | `python demo/bench_latency.py` | i7-12700K, 4 hilos | seal 342/26 ms @1280/384; defecto 65/40 ms; E2E ~630 ms→~100/min |
| **§5.4** · Análisis de errores (FN/FP) | `python evaluation/error_analysis.py` | seal + `defect_strip.pt` | 2 FN: `seal_1313` (prod3), `seal_2381` (prod2), score ≈ 0.000 |
| **Fig. 4.2 / 4.6** (barrido resolución / umbrales) | `python figures/thresholds_plot.py`, `python figures/build_figs2.py` | — | figuras coinciden con sus tablas |

## Notas

- **Los 2 falsos negativos son el resultado FINAL** — aparecen al adoptar el split de validación +
  early stopping (protocolo sin fuga). El hold-out es íntegramente de piezas **no revisadas** → sin
  fuga entre etapas por construcción.
- La **"CV de 5 pliegues" del defecto no es CV clásica**: el test no rota; son cinco reentrenamientos
  con particiones internas train/val distintas, todos sobre el **mismo** hold-out fijo. La CV genuina
  es la **LOPO** (Tablas 4.3 / prod6).
- El modelo `sealjit` de la ablación (Tabla 4.8, 0.975 e2e) y el `defect_strip.pt` desplegado (0.968
  e2e) son reentrenamientos distintos, dentro de la variabilidad ±0.008.
- Detalle exhaustivo y números superados: `docs/SOURCE_OF_TRUTH.md`.
