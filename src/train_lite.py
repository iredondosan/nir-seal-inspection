#!/usr/bin/env python3
"""
Lightweight seal segmentation for embedded (Pi5 / Rust) deployment.
  - pack-crop preprocessing (classical band-removal + pack detection)
  - small input (384x384) + lightweight encoder (MobileNetV3 via smp)
  - trains on prod2, exports ONNX, benchmarks CPU
Compare against the ResNet34 @512x640 baseline (train_resnet34.py).
"""
import os, glob, random, time
import numpy as np, cv2, torch, torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
import xml.etree.ElementTree as ET
import segmentation_models_pytorch as smp

SEED=42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
ROOT="/home/ubuntu/TFM/seal-inspection"
XML=f"{ROOT}/data/annotations/annotations.xml"
IMGDIR=f"{ROOT}/data/images/prod2"
OUT=f"{ROOT}/outputs/training_lite"; os.makedirs(OUT,exist_ok=True)
os.makedirs(f"{ROOT}/models",exist_ok=True)
CKPT=f"{ROOT}/models/best_lite.pt"; ONNX=f"{ROOT}/models/seal_lite.onnx"
IMG=384; BATCH=16; EPOCHS=60; SAMPLES=320; VAL_N=2; THRESH=0.5; MARGIN=40
dev="cuda" if torch.cuda.is_available() else "cpu"
IM_MEAN=(.485,.456,.406); IM_STD=(.229,.224,.225)

def norm(g):
    lo,hi=np.percentile(g,[1,99.5]); hi=max(hi,lo+1)
    return np.clip((g.astype(np.float32)-lo)/(hi-lo)*255,0,255).astype(np.uint8)

def conveyor_cols(N):
    cm=np.median(N,axis=0).astype(np.float32); on=np.where(cm>cm.max()*0.5)[0]
    cL,cR=on.min(),on.max(); g=np.gradient(cm)
    cL=int(np.argmax(g[max(0,cL-60):cL+60])+max(0,cL-60)); cR=int(np.argmin(g[cR-60:cR+60])+(cR-60))
    return cL,cR

def pack_bbox(gray):
    N=norm(gray); h,w=N.shape
    try: cL,cR=conveyor_cols(N)
    except Exception: cL,cR=0,w
    top=np.median(N[20:240,:],0); bot=np.median(N[h-240:h-20,:],0); ref=np.maximum(top,bot)
    diff=np.clip(np.tile(ref,(h,1))-N.astype(np.float32),0,255); diff[:,:cL]=0; diff[:,cR:]=0
    m=(diff>20).astype(np.uint8)*255
    m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((11,11),np.uint8))
    m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((41,41),np.uint8))
    cnts,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts=[c for c in cnts if cv2.contourArea(c)>h*w*0.02]
    if not cnts: return 0,0,w,h
    x,y,bw,bh=cv2.boundingRect(max(cnts,key=cv2.contourArea))
    return max(0,x-MARGIN),max(0,y-MARGIN),min(w,x+bw+MARGIN),min(h,y+bh+MARGIN)

def parse_pts(s): return np.array([[float(a) for a in p.split(',')] for p in s.strip().split(';')],np.float32)
def seal_mask(node):
    W=int(node.get('width')); H=int(node.get('height')); pl=[parse_pts(pg.get('points')) for pg in node.findall('polygon')]
    if len(pl)<2: return None
    pl=sorted(pl,key=lambda p:cv2.contourArea(p.astype(np.float32)),reverse=True)
    m=np.zeros((H,W),np.uint8); cv2.fillPoly(m,[pl[0].astype(np.int32)],1); cv2.fillPoly(m,[pl[1].astype(np.int32)],0)
    return m

# ---- build cropped (image3ch, mask) samples ----
root=ET.parse(XML).getroot(); samples=[]
for node in root.findall('image'):
    p=os.path.join(IMGDIR,node.get('name')); m=seal_mask(node)
    if m is None or not os.path.exists(p): continue
    g=cv2.imread(p,cv2.IMREAD_GRAYSCALE)
    x0,y0,x1,y1=pack_bbox(g)
    gc=norm(g[y0:y1,x0:x1]); mc=m[y0:y1,x0:x1]
    samples.append((np.stack([gc,gc,gc],-1), mc))
print(f"samples: {len(samples)}  (pack-cropped)",flush=True)
order=list(range(len(samples))); random.Random(SEED).shuffle(order); vs=set(order[:VAL_N])
train=[samples[i] for i in range(len(samples)) if i not in vs]
val=[samples[i] for i in range(len(samples)) if i in vs]

def ttf():
    geo=[A.HorizontalFlip(p=.5),A.VerticalFlip(p=.5)]
    try: geo.append(A.Affine(scale=(.85,1.15),translate_percent=(0,.06),rotate=(-180,180),border_mode=cv2.BORDER_CONSTANT,fill=0,fill_mask=0,p=.9))
    except TypeError: geo.append(A.Affine(scale=(.85,1.15),translate_percent=(0,.06),rotate=(-180,180),p=.9))
    ph=[A.RandomBrightnessContrast(.4,.4,p=.8),A.RandomGamma((60,140),p=.6)]
    try: ph.append(A.GaussNoise(p=.2))
    except Exception: pass
    return A.Compose([A.Resize(IMG,IMG),*geo,*ph,A.Normalize(IM_MEAN,IM_STD),ToTensorV2()])
def etf(): return A.Compose([A.Resize(IMG,IMG),A.Normalize(IM_MEAN,IM_STD),ToTensorV2()])
class TR(torch.utils.data.Dataset):
    def __init__(s,smp_,L): s.d=smp_; s.tf=ttf(); s.L=L
    def __len__(s): return s.L
    def __getitem__(s,i): im,mk=s.d[random.randrange(len(s.d))]; o=s.tf(image=im,mask=mk); return o["image"],o["mask"].float().unsqueeze(0)
class EV(torch.utils.data.Dataset):
    def __init__(s,smp_): s.d=smp_; s.tf=etf()
    def __len__(s): return len(s.d)
    def __getitem__(s,i): im,mk=s.d[i]; o=s.tf(image=im,mask=mk); return o["image"],o["mask"].float().unsqueeze(0)
train_dl=torch.utils.data.DataLoader(TR(train,SAMPLES),batch_size=BATCH,shuffle=True)
val_dl=torch.utils.data.DataLoader(EV(val),batch_size=1)

# ---- model: lightweight encoder via smp ----
ENC=None
for cand in ["timm-mobilenetv3_small_100","tu-mobilenetv3_small_100","mobilenet_v2"]:
    try:
        model=smp.Unet(cand,encoder_weights="imagenet",in_channels=3,classes=1); ENC=cand; break
    except Exception as e: print("encoder",cand,"unavailable:",str(e)[:90],flush=True)
model=model.to(dev)
params=sum(p.numel() for p in model.parameters())/1e6
print(f"encoder={ENC}  params={params:.2f}M  input={IMG}x{IMG}  device={dev}",flush=True)

def dloss(l,t,e=1.): p=torch.sigmoid(l); return (1-((2*(p*t).sum((2,3))+e)/(p.sum((2,3))+t.sum((2,3))+e))).mean()
bce=nn.BCEWithLogitsLoss()
@torch.no_grad()
def dice(l,t,e=1.): p=(torch.sigmoid(l)>THRESH).float(); return ((2*(p*t).sum((2,3))+e)/(p.sum((2,3))+t.sum((2,3))+e)).mean().item()
opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
best=0.; best_state=None
for ep in range(1,EPOCHS+1):
    model.train(); tot=n=0
    for x,y in train_dl:
        x,y=x.to(dev),y.to(dev); opt.zero_grad(); lo=model(x); L=bce(lo,y)+dloss(lo,y); L.backward(); opt.step(); tot+=L.item()*x.size(0); n+=x.size(0)
    sch.step(); model.eval(); vd=vn=0
    with torch.no_grad():
        for x,y in val_dl: x,y=x.to(dev),y.to(dev); vd+=dice(model(x),y)*x.size(0); vn+=x.size(0)
    v=vd/max(vn,1)
    if v>=best: best=v; best_state={k:val_.detach().cpu().clone() for k,val_ in model.state_dict().items()}
    if ep%5==0 or ep==1: print(f"epoch {ep:3d}/{EPOCHS}  loss {tot/n:.4f}  VAL-dice {v:.3f}  best {best:.3f}",flush=True)
if best_state: model.load_state_dict(best_state)
torch.save({"state_dict":model.state_dict(),"encoder":ENC,"img":IMG,"thresh":THRESH,"mean":IM_MEAN,"std":IM_STD,"val_dice":best,"margin":MARGIN},CKPT)
print(f"\nBEST VAL DICE (lite): {best:.3f}   saved {CKPT}  ({os.path.getsize(CKPT)/1e6:.1f} MB)",flush=True)

# ---- ONNX export (static 1x3xIMGxIMG) ----
model.cpu().eval()
torch.onnx.export(model, torch.randn(1,3,IMG,IMG), ONNX, opset_version=17,
                  input_names=["input"], output_names=["logits"])
print(f"exported ONNX: {ONNX}  ({os.path.getsize(ONNX)/1e6:.1f} MB)",flush=True)

# ---- CPU benchmark (embedded-relevant) ----
import os as _os; torch.set_num_threads(_os.cpu_count())
def cpu_bench(sz,iters=15):
    x=torch.randn(1,3,sz,sz)
    with torch.no_grad():
        for _ in range(5): model(x)
    t=time.time()
    with torch.no_grad():
        for _ in range(iters): model(x)
    return (time.time()-t)/iters*1000
print("=== CPU torch (this desktop CPU; Pi5 ~8-12x slower) ===",flush=True)
for sz in [IMG,256]: print(f"  torch {sz}x{sz}: {cpu_bench(sz):.1f} ms",flush=True)
try:
    import onnxruntime as ort
    so=ort.SessionOptions(); so.intra_op_num_threads=_os.cpu_count()
    sess=ort.InferenceSession(ONNX,sess_options=so,providers=["CPUExecutionProvider"])
    xx=np.random.randn(1,3,IMG,IMG).astype(np.float32)
    for _ in range(5): sess.run(None,{"input":xx})
    t=time.time()
    for _ in range(20): sess.run(None,{"input":xx})
    print(f"  onnxruntime {IMG}x{IMG}: {(time.time()-t)/20*1000:.1f} ms",flush=True)
except Exception as e: print("onnxruntime bench failed:",e,flush=True)
print("DONE",flush=True)
