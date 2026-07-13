# Experimentos

Diseño de cada experimento del proyecto: qué pregunta responde, protocolo, métrica y matices.
Complementa a [`REPRODUCE.md`](REPRODUCE.md) (tabla resultado→comando→número) y a
[`ARCHITECTURE.md`](ARCHITECTURE.md) (diseño del sistema). Los scripts viven en `experiments/`;
requieren los datos (`data/`) y los pesos (`models/`) locales.

> Nota transversal: todas las evaluaciones se hacen sobre el **hold-out global sin fuga**
> (179 piezas = 156 correctas / 23 defectuosas), formado íntegramente por piezas **no revisadas**
> — la etapa de sellado (entrenada solo con piezas revisadas) nunca las vio, así que el sistema de
> dos etapas queda libre de fuga *por construcción*.

---

## LOPO del sellado — generalización a un producto no visto (Tabla 4.3)

**Script:** `experiments/lopo_seal.py`

- **Objetivo:** ¿la segmentación del sellado generaliza a un **producto nuevo**, no visto en
  entrenamiento? (PI-3.)
- **Protocolo:** validación cruzada *dejando un producto fuera* (LOPO). Para cada uno de prod1–prod5:
  se entrena una U-Net de sellado con los **otros cuatro** productos, **desde el codificador
  ImageNet** (sin fuga entre productos), y se evalúa en el producto excluido con **cero de sus
  etiquetas** (*zero-shot*). Determinista (`SEED=42`).
- **Métrica:** Dice por pieza (`dice_one`: `sigmoid > 0.5`, Dice suavizado) promediado sobre las
  piezas del producto excluido → columna **zero-shot**. La columna **en muestra** se
  **calcula** entrenando un modelo con los 5 productos (mismo setup ImageNet) y midiendo el Dice de
  validación de cada producto —comparación consistente—; se añaden métricas de **borde** (Boundary-IoU/HD95/ASSD).
- **Resultado:** zero-shot **0,955 ± 0,013** frente a **0,966** en muestra (calculada, mismo setup) →
  caída media **~0,010** Dice (prod4 iguala su rendimiento en muestra, caída 0,000). En **borde** el hueco
  es algo mayor (B-IoU zero-shot 0,699 vs 0,738; ASSD 1,18 vs 0,90 px), coherente con que el Dice atenúa
  los errores de contorno (Tabla 4.2). El sellado no memoriza cada formato: aprende la señal estructural
  común (producto oscuro → reborde brillante → cinta).
- **Coste:** entrena **6 modelos** (5 zero-shot + 1 in-sample) a 1280 px (~2,5 h GPU). Los modelos se
  **guardan** (`models/lopo_*.pt`) → cualquier métrica futura es una evaluación rápida, sin re-entrenar. JSON: `results/lopo_seal.json`.

## LOPO de prod6 — detección de defecto en un producto *few-shot*

**Script:** `experiments/lopo_prod6.py`

- **Objetivo:** validar la detección de defecto en prod6, cuyas pocas piezas defectuosas son
  escaneos repetidos de tres piezas físicas y **no aportan al hold-out**.
- **Protocolo:** *dejar-un-pack-fuera* sobre los 3 packs defectuosos físicos de prod6: para cada
  pack se reconstruyen las tiras, se **reentrena** el defecto (+sealjit) sin ese pack y se puntúa
  extremo a extremo (sellado predicho → desenrollado → defecto) el pack excluido.
- **Métrica:** detección a nivel de captura/pack en el punto de operación.
- **Resultado:** **9/9 capturas, 3/3 packs** detectados (score ≥ 0,989).

## Estabilidad del AUROC — cinco reentrenamientos (Sección 4.6)

**Script:** `experiments/kfold_cv.py`

- **Objetivo:** comprobar que el AUROC de extremo a extremo **no depende de una partición
  afortunada** (crítico con tan pocas piezas defectuosas).
- **Protocolo:** se reentrena la **cabeza de defecto** con **cinco particiones internas
  train/validación distintas** (parada temprana en cada una) y se evalúa cada modelo sobre el
  **mismo hold-out externo fijo**.
- **⚠️ No es validación cruzada clásica:** el conjunto de prueba **no rota**. Rotarlo destruiría la
  propiedad de *sin fuga por construcción* (que depende de la distinción revisada/no revisada) y
  obligaría a reentrenar el sellado en cada pliegue, consumiendo las escasas máscaras revisadas.
  Por eso se fija el test y solo varía la partición interna del defecto. (La memoria detalla esto
  en la Sección 4.6.)
- **Resultado:** AUROC **0,977 ± 0,004** (estable entre pliegues; JSON `results/kfold_cv.json`, re-run 2026-07-12). Entre reentrenamientos COMPLETOS la media fluctúa levemente (rango 0,965–0,977; el ±std mide la dispersión entre pliegues de UNA corrida, no entre corridas) → leer como **≈0,97**. La **sensibilidad** en el punto de operación,
  en cambio, oscila entre **17 y 22 de 23** según el pliegue: varias piezas defectuosas diminutas
  caen cerca del umbral. Por eso el AUROC (independiente del umbral) es la métrica principal.

## Transferencia de aprendizaje — ImageNet vs desde cero (Tabla 4.5)

**Scripts:** `experiments/ablation_transfer.py` (por etapa, tira aislada) ·
`experiments/ablation_transfer_e2e.py` (extremo a extremo)

- **Objetivo:** cuantificar el valor de partir de pesos ImageNet, por etapa y en el sistema completo.
- **Protocolo:** se reentrenan ambas etapas **sin** inicialización preentrenada (mismo protocolo de
  validación y parada temprana) y se evalúan sobre el mismo hold-out.
- **Resultado:**
  - Sellado (dato escaso): Dice **0,967 vs 0,949**; el modelo desde cero **fragmenta el anillo** y
    falla su cierre en **26 de 179** piezas (frente a 0 con ImageNet).
  - Defecto (tira de referencia): AUROC **0,978 vs 0,972** — casi insensible a la inicialización *en
    exactitud*; sí converge más rápido (parada temprana ép. **69** vs **141**).
  - **Extremo a extremo: 0,968 vs 0,891.** El beneficio del preentrenamiento es **selectivo y
    acoplado**: ayuda a la etapa con datos escasos (sellado) y ese beneficio **se propaga** al
    sistema completo, porque la calidad del sellado condiciona la del defecto.

## Ablación de copy-paste — ¿responde PI-4? (Tabla 4.7)

**Script:** `evaluation/eval_copypaste.py` · modelo sin copy-paste: `python training/train_defect.py --sealjit --nopaste --out models/defect_nopaste.pt`

- **Objetivo:** aislar la aportación de la aumentación **copy-paste de defectos reales** (núcleo de PI-4): activarla/desactivarla con el resto del protocolo fijo (ambos `--sealjit`, mismo seed/split, sellado @1280; flag `--nopaste`).
- **Resultado:** copy-paste **ON** (`defect_jit`) 0.984 GT / **0.975 e2e** / 23-23 / 6 FP; **OFF** (`defect_nopaste`) 0.961 GT / **0.948 e2e** / 19-23 / 4 FP. **ΔE2E +0.027, ΔGT +0.023** — fuera del ruido de reentrenamiento (±0.01), a diferencia de roll/sealjit.
- **Lectura:** el copy-paste es **decisivo** con datos escasos (solo ~84 tiras defectuosas reales); sin él el detector subdetecta. **Responde afirmativamente a PI-4 por ablación.** `results/ablation_copypaste.json`.

## Ablación de aumentación de la tira — roll / sealjit (Tabla 4.8)

**Script:** `experiments/ablation_augment.py`

- **Objetivo:** medir el efecto de dos aumentaciones específicas de la tira: **roll** (desplazamiento
  circular horizontal) y **sealjit** (jitter vertical que emula el sellado *predicho* imperfecto).
- **Protocolo:** se evalúan las cuatro variantes (base / +roll / +sealjit / +ambas) sobre la tira de
  referencia (GT) **y** de extremo a extremo, sobre el hold-out de 23 defectos.
- **Resultado (origen del desenrollado anclado, 2026-07-12):** la aumentación es **determinante**
  (base 0,953 e2e, 20/23 → cualquier variante 0,975–0,983, equivalentes ±0,01). **sealjit** es la
  variante desplegada: mejor AUROC en la tira aislada (0,984) y el **mejor punto de operación**
  (23/23 detectadas, **3,8 %** de falsas alarmas), motivada mecánicamente (emula el sellado *predicho*).
  *roll* logra el AUROC e2e más alto (0,983) pero con menor recall (21/23). El orden aislado ≠ e2e:
  la aumentación debe validarse de extremo a extremo (antes del anclaje, la costura arbitraria
  penalizaba falsamente a *roll*).
- **Matiz:** el modelo `sealjit` de esta ablación da 0,975 e2e; el modelo desplegado (reentreno
  posterior) da **0,968**, dentro de la variabilidad ±0,008.

## Ablación de resolución — 384 / 512 / 1280 px (Tabla 4.2)

**Scripts:** `experiments/ablation_resolution_1280.py` · `evaluation/eval_boundary.py` (384/512)

- **Objetivo:** aislar el efecto de la resolución de entrada del sellado en la **calidad de borde**.
- **Protocolo:** se entrena el mismo modelo a tres resoluciones, sobre el mismo conjunto de piezas
  revisadas de **prod2**, variando solo la resolución.
- **Métrica:** además del Dice, métricas de **borde** — Boundary-IoU, HD95, ASSD
  (`evaluation/eval_boundary.py`), porque en un anillo delgado el Dice oculta errores de contorno.
- **Resultado:** la calidad de borde mejora **monótonamente** con la resolución (prod2 a 1280:
  Dice 0,954 · B-IoU 0,674 · HD95 3,00 · ASSD 1,15), mientras el Dice apenas se mueve. Coste: 1280 px
  es ~6× más costoso en CPU que 512; la resolución nativa se reserva como cota de calidad y para
  generar etiquetas.
- **Matiz:** es un experimento **independiente centrado en prod2**, por lo que su Dice puede diferir
  levemente del modelo multi-producto de la Tabla 4.1 (así lo indica la memoria).

## Línea base no supervisada — PatchCore

**Script:** `experiments/baseline_patchcore.py`

- **Objetivo:** contrastar el detector supervisado con una alternativa **no supervisada** (red de
  seguridad estilo PatchCore) sobre la tira desenrollada.
- **Protocolo:** un codificador ImageNet congelado convierte cada tira en una rejilla de descriptores;
  se modela la normalidad con tiras correctas y se puntúa por distancia.
- **Coreset:** k-center **greedy** (‘farthest-point sampling’), fiel a PatchCore (no aleatorio).
- **Resultado:** AUROC **0,776** (greedy; banco completo 0,784; el subconjunto aleatorio previo daba 0,694) —
  no supera al supervisado (0,978); línea base honesta, no punto de partida. El 0,800 previo salía de una
  configuración no reproducible (sin greedy / otro backbone).

## Cuantización INT8, comparación de sistemas y latencia

**Scripts:** `evaluation/eval_int8_quality.py` · `evaluation/eval_systems_e2e.py` · `demo/bench_latency.py` · `deploy/quantize_int8.py`

- **INT8 (estático):** reduce el sellado a 4,2 MB pero **no es desplegable** — fragmenta el anillo fino y la
  localización falla (61/179 @384, 0/179 @1280; `results/int8_quality.json`). La opción rápida real es **FP32 @384**.
- **Comparación de sistemas (Tabla 4.10):** 6 configuraciones (sellado 1280/512/384 × ResNet18/TinyUNet) sobre el
  conjunto común → AUROC E2E 0,964–0,970, equivalentes (±0,01); el desplegado reproduce tab:umbral (0,968·21/23).
  `results/systems_e2e.json`.
- **Latencia (ONNX, i7-12700K):** sellado 342 ms@1280 / 26 ms@384; defecto 65/40 ms; pipeline E2E ~630 ms →
  ~100 piezas/min. `results/latency.json`.

---

Los números esperados de cada experimento están en [`REPRODUCE.md`](REPRODUCE.md); el detalle de
trazabilidad y las corridas superadas, en `docs/SOURCE_OF_TRUTH.md`.
