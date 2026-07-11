# Demo — Inspección automática del sellado (NIR)

Demo interactiva del sistema de inspección de sellado de bandejas alimentarias a
partir de imágenes NIR retroiluminadas. Muestra, paso a paso, cómo el sistema
localiza el sellado, lo desenrolla en una tira y detecta los defectos, con el
veredicto final CORRECTO/DEFECTO. **Corre en CPU** (ONNX Runtime, sin GPU ni PyTorch).

## Instalación y ejecución

```bash
python -m venv .venv && source .venv/bin/activate      # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
streamlit run app.py
```

Se abre en el navegador (http://localhost:8501).

> **Assets del demo (no versionados).** Los modelos ONNX (`models/`) y las piezas de
> ejemplo (`samples/`) no están en git por tamaño y confidencialidad. Descarga el *demo
> bundle* desde el **Release** del repositorio y descomprímelo aquí (`demo/models/`, `demo/samples/`).

## Uso
- **Barra lateral:** elige una pieza de ejemplo (`samples/`) o sube tu propia imagen NIR.
- **Umbral de decisión:** slider que fija el punto de operación; al bajarlo, el sistema
  es más sensible (más detecciones y más falsas alarmas). Se actualiza en vivo.
- **Paneles:** (1) sellado predicho, (2) tira desenrollada con el defecto, (3) resultado
  sobre la pieza con los defectos marcados. Arriba: veredicto, puntuación y tiempo de inferencia.

## Estructura
```
app.py             UI Streamlit
pipeline.py        pipeline de 2 etapas (ONNX Runtime)
pipeline_core.py   geometría (recorte, mask->anillo, desenrollado) en numpy/OpenCV
models/            seal.onnx, defect.onnx
samples/           imágenes NIR de ejemplo
```

## Modelos (seleccionables en la barra lateral)
**Etapa 1 · sellado** (U-Net MobileNetV3-small, transferencia desde ImageNet)
- `seal.onnx` — **1280 px, desplegado**, máxima calidad. Dice de validación 0,967. ~700 ms/pieza en CPU.
- `seal_512.onnx` — **512 px, rápido** (~250 ms, Dice 0,942).
- `seal_384.onnx` — **384 px, más rápido** (~140 ms, Dice 0,938).

> ⚡ **Tramo velocidad/exactitud:** bajar la resolución del sellado acelera mucho, pero el detector de
> defecto se entrenó con el sellado a 1280 px; con un sellado más basto la tira desenrollada cambia y
> **pueden perderse defectos sutiles** (falsos negativos). El sistema **desplegado usa 1280 px**.

**Etapa 2 · detección de defecto** (tira 128×1536)
- `defect.onnx` — U-Net **ResNet18** (14,3 M, **desplegado**), AUROC extremo a extremo 0,968.
- `defect_tiny.onnx` — **TinyUNet** (0,93 M, ~15× menos parámetros; iguala el rendimiento —
  ablación de capacidad, y algo más rápida por tira).

El desglose de tiempos por etapa (que muestra la app) evidencia que **el sellado es el cuello de botella**.
Cambiar de modelo en vivo permite contemplar el ahorro de tiempo y las ablaciones del TFM.

> Las imágenes de sellado son propiedad del cliente y confidenciales; solo se incluyen
> unas pocas piezas de ejemplo con fines de demostración.
