import numpy as np, onnxruntime as ort, time, os
from seal_inspection.paths import ROOT as R
def bench(path, th):
    so=ort.SessionOptions(); so.intra_op_num_threads=th; so.inter_op_num_threads=1
    s=ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    inp=s.get_inputs()[0]; name=inp.name
    shp=[d if isinstance(d,int) else 1 for d in inp.shape]
    x=np.random.rand(*shp).astype(np.float32)
    for _ in range(3): s.run(None,{name:x})
    t=time.perf_counter()
    for _ in range(15): s.run(None,{name:x})
    return (time.perf_counter()-t)/15*1000, tuple(shp), os.path.getsize(path)/1e6
for tag,p in [("fp32 @384 (reviewed)","models/seal_lite_reviewed_384.onnx"),
              ("INT8 @384","models/seal_lite_int8.onnx")]:
    fp=f"{R}/{p}"
    if not os.path.exists(fp): print(tag,"(no existe)"); continue
    for th in (1,4):
        ms,shp,mb=bench(fp,th)
        print("%-22s th=%d  %6.1f ms  shape=%s  %.1f MB"%(tag,th,ms,shp,mb))
