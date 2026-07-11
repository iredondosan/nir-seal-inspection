"""Streamlit demo: automated seal inspection on backlit NIR tray images.

Run:  streamlit run app.py
CPU only (ONNX Runtime); no PyTorch required.
"""
import os
import glob
import numpy as np
import cv2
import streamlit as st
import pipeline as P

st.set_page_config(page_title="Inspección de sellado NIR", page_icon="🔍", layout="wide")

st.title("🔍 Inspección automática del sellado (NIR)")
st.caption("Pipeline en dos etapas: segmentación del sellado → desenrollado → detección de defectos. "
           "Inferencia en CPU con ONNX Runtime.")

# ---- model catalogue (only those present in ./models are shown) ----
SEAL_OPTS = {
    "MobileNetV3 · 1280 px (máx. calidad, desplegado)": "seal.onnx",
    "MobileNetV3 · 512 px (rápido)": "seal_512.onnx",
    "MobileNetV3 · 384 px (más rápido)": "seal_384.onnx",
}
DEFECT_OPTS = {
    "U-Net ResNet18 · 14,3 M (desplegado)": ("defect.onnx", "resnet"),
    "TinyUNet · 0,93 M (compacto)": ("defect_tiny.onnx", "tiny"),
}
SEAL_OPTS = {k: v for k, v in SEAL_OPTS.items() if os.path.exists(os.path.join("models", v))}
DEFECT_OPTS = {k: v for k, v in DEFECT_OPTS.items() if os.path.exists(os.path.join("models", v[0]))}


@st.cache_resource(show_spinner=False)
def _sess(path):
    return P.load_session(path)


# ---- sidebar ----
st.sidebar.header("Modelos")
seal_label = st.sidebar.selectbox("Etapa 1 · sellado", list(SEAL_OPTS)) if SEAL_OPTS else None
defect_label = st.sidebar.selectbox("Etapa 2 · detección de defecto", list(DEFECT_OPTS)) if DEFECT_OPTS else None
if not seal_label or not defect_label:
    st.error("Faltan modelos ONNX en ./models/ (se esperan al menos seal.onnx y defect.onnx).")
    st.stop()
seal_file = SEAL_OPTS[seal_label]
defect_file, defect_kind = DEFECT_OPTS[defect_label]
if seal_file != "seal.onnx":
    st.sidebar.warning("⚡ Sellado a resolución reducida: **más rápido**, pero puede **perder defectos sutiles**. "
                       "El detector se entrenó con el sellado a 1280 px; el sistema desplegado usa 1280 px.")

st.sidebar.header("Entrada")
samples = sorted(glob.glob(os.path.join("samples", "*")))
mode = st.sidebar.radio("Origen de la imagen", ["Ejemplos", "Subir imagen"], horizontal=True)
gray, name = None, None
if mode == "Ejemplos" and samples:
    sel = st.sidebar.selectbox("Pieza de ejemplo", [os.path.basename(s) for s in samples])
    path = next(s for s in samples if os.path.basename(s) == sel)
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE); name = sel
else:
    up = st.sidebar.file_uploader("Imagen NIR (escala de grises)", type=["png", "jpg", "jpeg", "tif", "tiff"])
    if up is not None:
        gray = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_GRAYSCALE); name = up.name

thr = st.sidebar.slider("Umbral de decisión", 0.0, 1.0, 0.50, 0.01)
st.sidebar.caption("↓ umbral → más sensible (más detecciones, más falsas alarmas). "
                   "En seguridad alimentaria interesa alta sensibilidad.")

if gray is None:
    st.info("Elige una pieza de ejemplo en la barra lateral o sube una imagen NIR.")
    st.stop()


@st.cache_data(show_spinner="Procesando pieza…")
def _run(img_bytes, seal_file, defect_file, defect_kind):
    g = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    return P.run(g, _sess(os.path.join("models", seal_file)),
                 _sess(os.path.join("models", defect_file)), defect_kind)


_ok, buf = cv2.imencode(".png", gray)
res = _run(buf.tobytes(), seal_file, defect_file, defect_kind)

if not res or not res.get("seal"):
    st.warning("El modelo de sellado seleccionado no localizó el anillo en esta pieza "
               "(el modelo *desde cero* falla en algunas piezas — es parte de la ablación de transferencia).")
    st.stop()

pan = P.overlays(res, thr)
score = res["score"]
defect = score >= thr

# ---- verdict + metrics ----
c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
with c1:
    st.markdown("### :red[🟥 DEFECTO]" if defect else "### :green[🟩 CORRECTO]")
c2.metric("Puntuación de defecto", f"{score:.3f}")
c3.metric("Detecciones", pan["n_det"])
c4.metric("Tiempo total (CPU)", f"{res['ms']:.0f} ms")

st.caption(f"**Sellado:** {seal_label}  ·  **Defecto:** {defect_label}")
st.caption(f"⏱️ Desglose por etapa: sellado **{res['ms_seal']:.0f} ms** · desenrollado {res['ms_unroll']:.0f} ms · "
           f"detección de defecto **{res['ms_defect']:.0f} ms**. "
           f"El sellado domina el tiempo; cambia el modelo de defecto (ResNet18 ↔ TinyUNet) para ver el ahorro en la Etapa 2.")
st.divider()

# ---- pipeline panels ----
cols = st.columns(3)
cols[0].image(pan["seal"], caption="1 · Sellado predicho (verde)", use_container_width=True)
cols[1].image(pan["strip"], caption="2 · Tira desenrollada + defecto (morado)", use_container_width=True)
cols[2].image(pan["final"], caption="3 · Resultado sobre la pieza (defectos en rojo)", use_container_width=True)

st.caption(f"Pieza: **{name}**  ·  score = máx. de la probabilidad de defecto suavizada (σ=2)  ·  "
           f"veredicto = DEFECTO si score ≥ umbral ({thr:.2f}).")
