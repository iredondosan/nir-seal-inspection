#!/usr/bin/env python3
"""Static INT8 quantization of a lite seal ONNX model + FP32-vs-INT8 CPU bench.
Calibration uses pack-cropped images spanning the given products.
  python quantize_int8.py --model models/seal_lite_multiprod.onnx --out models/seal_lite_multiprod_int8.onnx
"""
import os, glob, time, argparse, numpy as np, cv2
import onnx, onnxruntime as ort
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat

ROOT="/home/ubuntu/TFM/seal-inspection"
ap=argparse.ArgumentParser()
ap.add_argument("--model", default=f"{ROOT}/models/seal_lite_multiprod.onnx")
ap.add_argument("--out",   default=f"{ROOT}/models/seal_lite_multiprod_int8.onnx")
ap.add_argument("--prods", default="prod1,prod2,prod3,prod4,prod5")
ap.add_argument("--per",   type=int, default=8, help="calib images per product")
args=ap.parse_args()
FP32=args.model; INT8=args.out; PREP=FP32.replace(".onnx","_prep.onnx")
IMG=384; MARGIN=40
MEAN=np.array((.485,.456,.406),np.float32); STD=np.array((.229,.224,.225),np.float32)

def norm(g):
    lo,hi=np.percentile(g,[1,99.5]); hi=max(hi,lo+1)
    return np.clip((g.astype(np.float32)-lo)/(hi-lo)*255,0,255).astype(np.uint8)
def conveyor_cols(N):
    cm=np.median(N,0).astype(np.float32); on=np.where(cm>cm.max()*0.5)[0]; cL,cR=on.min(),on.max(); g=np.gradient(cm)
    return int(np.argmax(g[max(0,cL-60):cL+60])+max(0,cL-60)), int(np.argmin(g[cR-60:cR+60])+(cR-60))
def pack_bbox(gray):
    N=norm(gray); h,w=N.shape
    try: cL,cR=conveyor_cols(N)
    except Exception: cL,cR=0,w
    top=np.median(N[20:240,:],0); bot=np.median(N[h-240:h-20,:],0); ref=np.maximum(top,bot)
    diff=np.clip(np.tile(ref,(h,1))-N.astype(np.float32),0,255); diff[:,:cL]=0; diff[:,cR:]=0
    m=cv2.morphologyEx((diff>20).astype(np.uint8)*255,cv2.MORPH_OPEN,np.ones((11,11),np.uint8))
    m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((41,41),np.uint8))
    c=[c for c in cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0] if cv2.contourArea(c)>h*w*0.02]
    if not c: return 0,0,w,h
    x,y,bw,bh=cv2.boundingRect(max(c,key=cv2.contourArea))
    return max(0,x-MARGIN),max(0,y-MARGIN),min(w,x+bw+MARGIN),min(h,y+bh+MARGIN)
def preprocess(path):
    g=cv2.imread(path,cv2.IMREAD_GRAYSCALE); x0,y0,x1,y1=pack_bbox(g)
    gc=norm(g[y0:y1,x0:x1]); im=np.stack([gc,gc,gc],-1)
    im=cv2.resize(im,(IMG,IMG)).astype(np.float32)/255.0
    return ((im-MEAN)/STD).transpose(2,0,1)[None].astype(np.float32)

files=[]
for p in args.prods.split(","):
    files += sorted(glob.glob(f"{ROOT}/data/images/{p}/*_raw.png"))[:args.per]
print(f"calibration images: {len(files)} across {args.prods}")
class DR(CalibrationDataReader):
    def __init__(s): s.it=iter([{"input":preprocess(f)} for f in files])
    def get_next(s): return next(s.it,None)

onnx.save(onnx.shape_inference.infer_shapes(onnx.load(FP32)), PREP)
quantize_static(PREP, INT8, DR(), quant_format=QuantFormat.QDQ,
                per_channel=True, weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8)
print(f"FP32 {os.path.getsize(FP32)/1e6:.1f}MB  ->  INT8 {os.path.getsize(INT8)/1e6:.1f}MB")

def sess(m,thr):
    so=ort.SessionOptions(); so.intra_op_num_threads=thr; so.inter_op_num_threads=1
    return ort.InferenceSession(m, sess_options=so, providers=["CPUExecutionProvider"])
x=preprocess(files[0])
def bench(m,thr,it=40):
    s=sess(m,thr)
    for _ in range(6): s.run(None,{"input":x})
    t=time.time()
    for _ in range(it): s.run(None,{"input":x})
    return (time.time()-t)/it*1000
# accuracy: INT8 vs FP32 mask agreement across products
import random
agree=[]
for f in files[::5]:
    a=1/(1+np.exp(-sess(FP32,os.cpu_count()).run(None,{"input":preprocess(f)})[0]))>0.5
    b=1/(1+np.exp(-sess(INT8,os.cpu_count()).run(None,{"input":preprocess(f)})[0]))>0.5
    agree.append((a==b).mean())
print(f"INT8 vs FP32 mask agreement: {np.mean(agree)*100:.2f}%")
print("threads |  FP32 ms | INT8 ms | speedup")
for thr in [1,2,4,os.cpu_count()]:
    f32=bench(FP32,thr); i8=bench(INT8,thr)
    print(f"  {thr:3d}   | {f32:7.1f}  | {i8:7.1f} | {f32/i8:.2f}x")
print("DONE")
