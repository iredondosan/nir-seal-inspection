# TFM — Seal Inspection · SOURCE OF TRUTH

> Documento de referencia que ata el **código Python** y los **modelos entrenados** a los
> **resultados reportados en la memoria**. Ante cualquier duda, **manda el modelo desplegado
> + el script que lo produjo** (no los borradores ni los reports intermedios).
> Última reconciliación: estado del repo a **2026-07-06** (últimos entrenamientos con split de validación).

---

## 0. Regla de oro

- **Fuente de verdad = los `.pt` desplegados + el script `src/*.py` que los generó + el `holdout_labels.csv` vigente.**
- Los `docs/*.md` (LOG, SYSTEM_REPORT) son un **diario cronológico**; algunos números quedaron **superados** por entrenamientos posteriores (ver §6).
- El paquete `seal_inspection/train_*.py` son versiones **simplificadas/didácticas** (épocas fijas, sin early stopping); **los entrenamientos reales están en `src/`**.

---

## 1. Pipeline (dos etapas) — `seal_inspection/core.py`

```
imagen NIR cruda → [pack_bbox: recorte a la bandeja] → [normalize: estiramiento percentil 1/99.5]
   → Etapa 1: U-Net sellado → mapa prob. → umbral → mask_to_ring (contorno exterior+interior)
   → unroll_maps: desenrollado a tira 128×1536
   → Etapa 2: U-Net defecto sobre la tira → score = max(GaussianBlur(sigmoid, σ=2))
   → veredicto: DEFECTO si score ≥ umbral, si no CORRECTO
```

Funciones clave (`core.py`):
- `normalize` — estiramiento de contraste por percentiles (1 / 99.5) a 0–255.
- `pack_bbox` — detección de bandeja por sustracción de fondo (referencia de cinta en franjas superior/inferior), morfología (open 11×11 + close 41×41), mayor blob, margen 40 px.
- `polygons_to_band_mask` — rasteriza el sellado: relleno del polígono **exterior** menos el **interior** (anillo).
- `mask_to_ring` — de máscara predicha a contornos: morfología (open 9 + close 35), mayor contorno externo = exterior (contorno **crudo**, sigue el borde ondulado), agujero interior o erosión (banda 90 px); `_clean_contour` remuestrea a 360 pts + suavizado circular.
- `unroll_maps` (**producción, perpendicular-al-exterior**) — para cada una de las `WS=1536` posiciones sobre el contorno exterior suavizado, marcha hacia dentro por la **normal local** una profundidad = distancia al borde interior (`distanceTransform`); `a = linspace(-0.15, 1.15, 128)` (fila 0 = exterior, 127 = interior, ±15 % margen). **Correspondence-free**.
- `unroll_maps_legacy` — interpolación lineal exterior↔interior emparejados (2.ª rama del *ensemble* exploratorio).
- Las mismas `(map_x, map_y)` sirven para **retroproyectar** los defectos a la imagen original.

---

## 2. Scripts de ENTRENAMIENTO reales (`src/`) — fuente de verdad

| Script | Produce | Config real |
|---|---|---|
| `src/train_lite.py` | `best_lite.pt` (base sellado, 384) | MobileNetV3-small U-Net, ImageNet, 384², BCE+Dice, AdamW, 60 ép., batch 16, mejor checkpoint |
| `src/train_reviewed.py` | **`best_lite_reviewed_1280.pt`** (sellado desplegado) | Fine-tune de `best_lite.pt`; **1280²**, AdamW **1e-4**, wd 1e-4, **coseno**, **40 ép.**, batch 2, BCE+Dice, AMP, copy-paste **P=0.8** (contaminación/grafismo sin cambiar máscara), VAL_PER=2/producto, **mejor checkpoint** (sin early stopping). `--scratch` para la ablación. **NOTA (2026-07-11):** la config desplegada (1280 / batch 2 / 40 ép.) se pasa por **flags CLI**; los *defaults* del script son `--img 512 --batch 12 --epochs 60`. Verificable: el checkpoint guarda `img=1280`. |
| `src/train_defect.py` | **`defect_strip.pt`** (defecto desplegado) | ResNet18 U-Net, ImageNet, tira **128×1536**, AdamW **2e-4**, wd 1e-4, **ReduceLROnPlateau** (factor 0.5, pac. 8), **EARLY STOPPING** (paciencia **25** sobre pérdida de validación, **máx. 200 ép.**), BCE(**pos_weight=20**)+Dice, batch 8, 1200 pasos/ép., **val 15 % a nivel de pieza** (fuera de train y de la biblioteca copy-paste), sobremuestreo 50 %, copy-paste **P=0.7**, **`--sealjit`** (jitter vertical del sellado), `--roll`, `--kfold`. Test evaluado **una vez**. |
| `src/train_tiny.py` | `tiny_defect.pt` (TinyUNet) | TinyUNet 1-canal (0.93 M), 4 niveles [16,32,64,128], **desde cero**, early stopping (paciencia 30). |
| `src/kfold_cv.py` | `defect_kf0..kf4.pt` | Validación cruzada de **5 pliegues** del defecto → **AUROC 0.975 ± 0.008**. |
| `src/lopo_cv.py` | (eval) | Sellado: **leave-one-product-out** (Dice zero-shot 0.955±0.010). prod6/prod6: **leave-one-pack-out** (9/9 capturas, 3/3 packs). |
| `src/anomaly_patchcore.py` | (línea base) | PatchCore sobre tiras correctas → **AUROC 0.800**. |
| `src/train_resnet34.py` | `defect` base pesado | Línea base de mayor capacidad para §4.10 (24.4 M). |
| `src/quantize_int8.py` | ONNX INT8 | Cuantización INT8 del sellado (384², ~4.2 MB, ~20 ms/4hilos). |

Eval: `src/eval_e2e.py`, `src/eval_thresholds.py` (umbral/OP), `src/eval_boundary.py` (Boundary-IoU/HD95/ASSD), `src/eval_tta.py` (TTA). Datos: `src/make_masks.py`, `src/make_strips.py`, `src/make_holdout.py`, `src/predict_to_cvat*.py` (pre-anotación asistida).

---

## 3. Modelos desplegados (los que reporta la memoria)

| Modelo | Fecha | Métrica guardada | Rol |
|---|---|---|---|
| **`best_lite_reviewed_1280.pt`** | 06-30 22:42 | `val_dice = 0.9672` (**Dice 0.967**) | **Sellado en producción** (6 productos, 1280²) |
| **`defect_strip.pt`** | 07-05 22:20 | `stop_epoch = 69`, `score_thr = 0.5` | **Defecto en producción** (= `defect_imagenet_es`, +sealjit, early stop @69) |
| `defect_strip.prev.pt` | 06-30 22:59 | — | 2.ª rama del *ensemble* (unroll legacy), exploratorio |
| `tiny_defect.pt` | 07-05 22:12 | `stop_epoch = 136` | TinyUNet (ablación de capacidad) |

**Ablación transferencia (§4.7):** `scratch_seal_1280.pt` (Dice **0.949**), `defect_scratch_es.pt` (`stop_epoch = 141`). → El "**69 vs 141 épocas**" son las **paradas tempranas del modelo de DEFECTO** (ImageNet vs desde cero), no del sellado.

**E2E de la ablación (verificado 2026-07-10, hold-out actual 156/23):** ImageNet **AUROC 0.968** (recall 21/23, FP 8/156, sellado falla en **0** piezas) vs desde cero **AUROC 0.895** (recall 20/23, FP 11/156, **sellado falla en 26 piezas**). El sistema completo **NO es igual**: el sellado desde cero (Dice 0.949) fragmenta el anillo y falla el cierre en 26 piezas -> puntúan 0 -> hunde el AUROC e2e. La cabeza de defecto sí es init-agnostic en tira de referencia (0.978 vs 0.972), pero el beneficio del sellado **se propaga** al e2e.

**5-fold CV (§4.6):** `defect_kf0..kf4.pt` (07-06), stop_epoch 129/93/107/112/67 → **AUROC 0.975 ± 0.008**.

---

## 4. RESULTADOS AUTORITATIVOS (estado final, split de validación) 

Hold-out global vigente: **179 packs = 156 correctos / 23 defectuosos** (`data/holdout_labels.csv`, 07-05, ya con las 3 correcciones de etiqueta aplicadas). prod6 NO aporta al hold-out.

| Métrica | Valor | Modelo / script |
|---|---|---|
| Sellado — Dice validación | **0.967** | `best_lite_reviewed_1280.pt` |
| Sellado — Dice zero-shot (LOPO) | 0.955 ± 0.010 | **`_lopo.py`** (no lopo_cv.py, que es el de prod6) |
| Defecto aislado — AUROC (tira GT) | **0.978** | `defect_strip.pt` |
| Extremo a extremo — AUROC (desplegado) | **0.968** | seal + `defect_strip.pt` |
| Extremo a extremo — AUROC (5-fold CV) | **0.975 ± 0.008** | `defect_kf0..4` |
| Punto de operación @0.50 | **recall 21/23**, FP **8/156 = 5.1 %** | 2 FN: `seal_1313` (prod3), `seal_2381` (prod2), ambos score ≈ 0.000 |
| PatchCore (línea base anomalía) | AUROC 0.800 | `anomaly_patchcore.py` |
| TinyUNet | AUROC 0.972 · 0.93 M (15.5× menos) | `tiny_defect.pt` |
| prod6/prod6 — LOPO por pack | 9/9 capturas, 3/3 packs (≥0.989) | `lopo_cv.py` |

### Verificación tabla a tabla (2026-07-10, modelos + hold-out actuales)

- **tab:dice-producto (4.1)** ✅ per-producto desplegado (`eval_seal.py`): prod1 0.971 · prod2 0.950 · prod3 0.969 · prod4 0.975 · prod5 0.972 · prod6 0.969 · **global 0.967**.
- **tab:resolucion (4.2)** ✅ 384/512 (`log_entry_res_ab.md`) y **1280 RE-MEDIDO** (`eval_boundary_1280.py`, 2026-07-10): prod2 Dice 0.954 / B-IoU 0.674 / HD95 3.00 / ASSD 1.15 (tesis 0.952/0.663/3.23/1.23 ✓); global ASSD 0.87 (sub-píxel).
- **tab:lopo (4.3)** ✅ **respaldado por código `_lopo.py`** (determinista, SEED=42): entrena el sellado en 4 productos desde ImageNet (sin fuga) y evalúa zero-shot en el excluido, ×5. Columna **en muestra** hardcodeada en `_lopo.py` (`INS`): prod1 0.965 · prod2 0.945 · prod3 0.966 · prod4 0.967 · prod5 0.957 (media 0.960) = tab:lopo ✓. Zero-shot per-producto 0.957/0.939/0.962/0.968/0.949 (media **0.955 ± 0.010**), caída 0.005. OJO: el `INS` de `_lopo.py` es la corrida antigua (media 0.960); el modelo desplegado actual da per-producto distinto (media 0.967, tab:dice-producto). El zero-shot (el resultado clave) es sólido; la comparación 'en muestra' usa esa referencia antigua.
- **tab:umbral (4.4)** ✅ **EXACTO** (`eval_tables.py`): AUROC 0.968; @0.50 TP21/FP8/TN148/FN2; @0.70 21/6/150/2; @0.85 17/5/151/6; @0.90 15/4/152/8; @0.95 14/2/154/9.
- **tab:pretrain (4.5)** ✅ corregido: 0.967/0.949 · 0.978/0.972 · **e2e 0.968/0.895** · anillo cerrado 179/153 · FP 5.1%/7.1% · ES 69/141.
- **tab:tiny (4.6)** ✅ tira GT resnet18 0.978 / tiny 0.972; e2e 0.968 / 0.969; latencias i7 4hilos ~151/83 ms (memoria 142/90.6, dentro de ruido).
- **tab:aug (4.7)** ✅ corregido (23 def): baseline 0.975/0.969 · +roll 0.980/0.967 · **+sealjit 0.984/0.982** · +both 0.976/0.966.
- **tab:despliegue (4.8)** ✅ **RE-MEDIDO en CPU** (i7-12700K, 4 hilos, sin GPU) 2026-07-10. El 16.6/467/28× del LOG (junio) NO reproduce. Latencias actuales torch: Lite MobileNetV3-small (3.59 M) @384 **68 ms** / @512 130 / @1280 924; ResNet34-UNet (24.44 M) @384 **157 ms** / @512 285 → **~2.3× a igual resolución (384)**, 6.8× menos parámetros. ONNX Lite: @384 42 ms, @1280 **341 ms**, INT8@384 **19 ms**. Defecto resnet18 151 ms, TinyUNet 83 ms (tira 128×1536). Tabla y §4.10 actualizadas a estas cifras medidas.
- **fig:resolucion (4.2)** y **fig:thresholds (4.6)** ✅ coinciden con sus tablas.

**Los 2 falsos negativos son el resultado FINAL** — aparecieron al adoptar el **split de validación + early stopping** (protocolo sin fuga). El §5.4 (análisis de errores) es correcto.

---

## 5. Aumentaciones (ablación defecto — `src/train_defect.py --roll/--sealjit`)

| Variante | AUROC tira GT | AUROC E2E | recall | FP | (hold-out 23 defectos, verificado 2026-07-10) |
|---|:--:|:--:|:--:|:--:|:--|
| baseline (`defect_strip.noaug`) | 0.975 | 0.969 | 22/23 | 5.8 % | |
| +roll (`defect_roll`) | 0.980 | 0.967 | 22/23 | 8.3 % | competitivo aislado, sube FP e2e |
| **+sealjit** (`defect_jit`) | **0.984** | **0.982** | **23/23** | **3.2 %** | mejor en TODO; variante desplegada |
| +both (`defect_rolljit`) | 0.976 | 0.966 | 23/23 | 9.6 % | |

Lección: **validar la aumentación de extremo a extremo, no en la etapa aislada** (`roll` sube las falsas alarmas e2e). En el hold-out corregido, **sealjit es el mejor en todas las métricas** (0.984 GT / 0.982 e2e / 23-23 / 3.2 % FP). OJO: el `defect_jit` de la ablación da 0.982 e2e; el modelo desplegado `defect_strip.pt` (reentreno posterior) da 0.968 — dentro de ±0.008. *(Cifras del hold-out **pre-corrección** de etiquetas, 26 defectos. **Ya incorporadas a la memoria en §4.9** (Tabla `tab:aug`), etiquetadas explícitamente como "conjunto de desarrollo". Pendiente opcional: reevaluar los 4 modelos (`defect_strip.noaug`/`defect_roll`/`defect_jit`/`defect_rolljit`) sobre el hold-out corregido (23 defectos) para consistencia total — como se hizo con la ablación de transferencia.)*

---

## 6. Números SUPERADOS (no usar — quedaron obsoletos)

| Fuente | Decía | Realidad final |
|---|---|---|
| SYSTEM_REPORT §5 (cuerpo) | E2E AUROC 0.935, defecto 0.964, hold-out 154/26, OP 0.43 | Pre-corrección de etiquetas |
| SYSTEM_REPORT Addendum §D | 100 % recall (23/23), AUROC 0.976 | Estado intermedio (~19:25); **superado** por `defect_strip.pt` de las 22:20 (split de validación) → **21/23, 0.968** |
| Tesis Tabla 4.5 (fila e2e AUROC) | 0.968 / 0.968 (idéntico) | **ERROR**: el e2e desde cero es **0.895** (el sellado desde cero falla el cierre en 26 piezas), no 0.968. Corregido a **0.968 / 0.895** |
| Logs `train_seal_p4/p5` | Dice 0.961/0.963 | Corridas de 40 ép. anteriores; el desplegado (mejor corrida) = **0.967** |
| §4.7 "parada temprana 69/141" | atribuido al **sellado** | son las épocas ES del **defecto** (ImageNet 69 / scratch 141) |

---

## 7. Estado de incorporación a la memoria

**Ya incorporado** (memoria v3):
- ✅ Ablación de aumentación (roll/sealjit) → §4.9 (Tabla `tab:aug`) + lección "validar de extremo a extremo".
- ✅ INT8 en despliegue → §4.10: seal INT8@384 ≈ **19 ms/4 hilos** (i7-12700K) vs fp32@1280 ≈ 341 ms (~18×), 4,2 MB (3,4× menos). Medido consistente con SYSTEM_REPORT §6.

**Pendiente / posible incorporación futura:**
- Despliegue en **Rust** (`rust_infer/`) — la memoria lo cita como línea futura; existe código real.
- `defect_rebal_*` (rebalanceo, oversample defectos pequeños) — experimento no reportado.
- `eval_tta.py` (test-time augmentation) — no reportado.
