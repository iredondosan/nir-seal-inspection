#!/usr/bin/env python3
"""CPU inference timing for the two models (torch, fp32). Reports 1-thread and all-thread latency."""
import time, os, numpy as np, torch
import segmentation_models_pytorch as smp
from seal_inspection.paths import ROOT
torch.set_grad_enabled(False)
def load(p):
    ck=torch.load(p,map_location="cpu",weights_only=False)
    m=smp.Unet(ck["encoder"],encoder_weights=None,in_channels=3,classes=1); m.load_state_dict(ck["state_dict"]); m.eval()
    return m,ck
seal,sk=load(f"{ROOT}/models/best_lite_reviewed_1280.pt"); SIMG=sk["img"]
defm,dk=load(f"{ROOT}/models/defect_strip.pt"); HS,WS=dk["HS"],dk["WS"]
xs=torch.randn(1,3,SIMG,SIMG); xd=torch.randn(1,3,HS,WS)
def bench(m,x,n=12):
    for _ in range(3): m(x)            # warmup
    t=[]
    for _ in range(n):
        s=time.perf_counter(); m(x); t.append((time.perf_counter()-s)*1000)
    return np.mean(t),np.std(t)
import multiprocessing as mp
print(f"CPU cores: {mp.cpu_count()}")
for nt in [1, mp.cpu_count()]:
    torch.set_num_threads(nt)
    sm,ss=bench(seal,xs); dm,ds=bench(defm,xd)
    print(f"--- threads={nt} ---")
    print(f"  Seal  (MobileNetV3-UNet @ {SIMG}x{SIMG}): {sm:6.1f} ms  (±{ss:.1f})")
    print(f"  Defect(ResNet18-UNet @ {HS}x{WS}):       {dm:6.1f} ms  (±{ds:.1f})")
    print(f"  Seal+Defect total:                       {sm+dm:6.1f} ms  (~{1000/(sm+dm):.1f} packs/s, model only)")
