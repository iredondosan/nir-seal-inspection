"""Latency benchmark of the deployed two-stage pipeline on CPU (ONNX Runtime).

Measures each model at its DEPLOYED resolution (seal @1280, defect @128x1536) plus
the full end-to-end pipeline, in FP32 and dynamic-INT8, on the i7-12700K (4 threads).
Warm-up + median of N. Writes ../results/latency.json.  Run from demo/:  python bench_latency.py
"""
import time, json, glob, os
import numpy as np, cv2, onnxruntime as ort
import pipeline, pipeline_core as C
from onnxruntime.quantization import quantize_dynamic, QuantType

THREADS = 4; N = 40; WARM = 8
np.random.seed(0)

def bench(path, inp, n=N, warm=WARM):
    s = pipeline.load_session(path, THREADS); name = s.get_inputs()[0].name
    for _ in range(warm): s.run(None, {name: inp})
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); s.run(None, {name: inp}); ts.append((time.perf_counter()-t0)*1000)
    ts.sort()
    return {"median_ms": round(ts[len(ts)//2], 1), "p90_ms": round(ts[int(len(ts)*0.9)], 1), "min_ms": round(ts[0], 1)}

IN = {
    "seal_1280": ("models/seal.onnx",      np.random.rand(1,3,1280,1280).astype(np.float32)),
    "seal_512":  ("models/seal_512.onnx",  np.random.rand(1,3,512,512).astype(np.float32)),
    "seal_384":  ("models/seal_384.onnx",  np.random.rand(1,3,384,384).astype(np.float32)),
    "defect_resnet18": ("models/defect.onnx",      np.random.rand(1,3,128,1536).astype(np.float32)),
    "defect_tiny":     ("models/defect_tiny.onnx", np.random.rand(1,1,128,1536).astype(np.float32)),
}
res = {"cpu": "i7-12700K", "threads": THREADS, "n": N, "engine": "onnxruntime " + ort.__version__,
       "note": "median of N single-inference runs after warm-up; random input (latency is value-independent)."}

# ---- FP32 per-model ----
res["fp32"] = {}
for k, (p, inp) in IN.items():
    res["fp32"][k] = bench(p, inp); print("FP32", k, res["fp32"][k])

# ---- dynamic INT8 per-model ----
os.makedirs("models/int8", exist_ok=True); res["int8_dynamic"] = {}
for k in ["seal_1280", "defect_resnet18", "defect_tiny"]:
    p, inp = IN[k]; out = f"models/int8/{k}_int8.onnx"
    try:
        quantize_dynamic(p, out, weight_type=QuantType.QInt8)
        res["int8_dynamic"][k] = bench(out, inp); print("INT8", k, res["int8_dynamic"][k])
    except Exception as e:
        res["int8_dynamic"][k] = {"error": str(e)[:120]}; print("INT8", k, "ERROR", e)

# ---- E2E full pipeline on real images ----
imgs = sorted(glob.glob("../data/images/prod*/*_raw.png"))
imgs = imgs[::max(1, len(imgs)//18)][:18]
seal_sess, def_sess = pipeline.load_models("models/seal.onnx", "models/defect.onnx", THREADS)
_, tiny_sess = pipeline.load_models("models/seal.onnx", "models/defect_tiny.onnx", THREADS)
g0 = cv2.imread(imgs[0], 0)
for _ in range(4): pipeline.run(g0, seal_sess, def_sess, "resnet")   # warm

def agg(rows, key="ms"):
    v = sorted(r[key] for r in rows); m = v[len(v)//2]
    return {"median_ms": round(m, 1), "n": len(v), "pieces_per_min_4c": round(60000/m)}

R, T = [], []
for p in imgs:
    g = cv2.imread(p, 0)
    r = pipeline.run(g, seal_sess, def_sess, "resnet")
    if r.get("seal"): R.append(r)
    rt = pipeline.run(g, seal_sess, tiny_sess, "tiny")
    if rt.get("seal"): T.append(rt)
res["e2e_resnet18"] = agg(R)
res["e2e_resnet18"]["stages_median_ms"] = {k: round(sorted(x[k] for x in R)[len(R)//2], 1)
                                           for k in ("ms_seal", "ms_unroll", "ms_defect")}
res["e2e_tiny"] = agg(T)
print("E2E resnet18:", res["e2e_resnet18"]); print("E2E tiny:", res["e2e_tiny"])

json.dump(res, open("../results/latency.json", "w"), indent=2)
print("\nwrote ../results/latency.json")
