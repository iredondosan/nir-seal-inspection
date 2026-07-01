#!/usr/bin/env python3
"""Multi-product fine-tune: continue from best_lite.pt on prod2 + prod1 (pack-cropped, 384, MobileNetV3).
Reports per-product validation Dice. Saves best_lite_multiprod.pt + ONNX."""
import os, glob, random, time
import numpy as np, cv2, torch, torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
import xml.etree.ElementTree as ET
import segmentation_models_pytorch as smp

SEED=42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
ROOT="/home/ubuntu/TFM/seal-inspection"
DATASETS=[("data/annotations/prod2_reviewed.xml","data/images/prod2","prod2"),
          ("data/annotations/prod1_reviewed.xml","data/images/prod1","prod1"),
          ("data/annotations/prod3_reviewed.xml","data/images/prod3","prod3"),
          ("data/annotations/prod4_reviewed.xml","data/images/prod4","prod4"),
          ("data/annotations/prod5_reviewed.xml","data/images/prod5","prod5"),
          ("data/annotations/prod6_reviewed.xml","data/images/prod6", "prod6"),
          ("data/annotations/prod6_bad_reviewed.xml","data/images/prod6_bad","prod6")]
OUT=f"{ROOT}/outputs/training_reviewed"; os.makedirs(OUT,exist_ok=True)
BASE=f"{ROOT}/models/best_lite.pt"
CKPT=f"{ROOT}/models/best_lite_reviewed.pt"; ONNX=f"{ROOT}/models/seal_lite_reviewed.onnx"
import argparse
_ap=argparse.ArgumentParser(); _ap.add_argument("--img",type=int,default=512); _ap.add_argument("--batch",type=int,default=12); _ap.add_argument("--epochs",type=int,default=60); _ap.add_argument("--samples",type=int,default=400)
_ap.add_argument("--scratch",action="store_true",help="random init (no ImageNet/base) -> train from scratch for the ablation")
_ap.add_argument("--ckpt",default="",help="output checkpoint path (default models/best_lite_reviewed_<IMG>.pt)")
_a,_=_ap.parse_known_args()
IMG=_a.img; BATCH=_a.batch; EPOCHS=_a.epochs; SAMPLES=_a.samples; SCRATCH=_a.scratch; VAL_PER=2; THRESH=0.5; MARGIN=40
CKPT=_a.ckpt if _a.ckpt else f"{ROOT}/models/best_lite_reviewed_{IMG}.pt"; ONNX=CKPT.replace(".pt",".onnx")
P_CONTAM=0.8   # fraction of train samples that get contamination pasted on the seal band
CONTAM_XML=f"{ROOT}/data/annotations/contaminants.xml"
HOLD=set(l.strip() for l in open(f"{ROOT}/data/holdout.txt")) if os.path.exists(f"{ROOT}/data/holdout.txt") else set()
CONTAM_EXCLUDE={"seal_1302_1780665903828_raw.png"}|{f"{h}.png" for h in HOLD}|{f"{h}.jpg" for h in HOLD}   # +global hold-out
FORCE_TRAIN={"seal_1998_1780688689500_raw.png"}      # barcode-over-seal: keep in train, never val
P_PRINT=0.6   # fraction of train samples that get printed-graphic cut-outs pasted on the band (mask unchanged)
dev="cuda" if torch.cuda.is_available() else "cpu"
IM_MEAN=(.485,.456,.406); IM_STD=(.229,.224,.225)

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
def parse_pts(s): return np.array([[float(a) for a in p.split(',')] for p in s.strip().split(';')],np.float32)
def tags_of(node): return {t.get('label') for t in node.findall('tag')}
def seal_mask(node):
    W=int(node.get('width')); H=int(node.get('height'))
    pl=[parse_pts(pg.get('points')) for pg in node.findall('polygon') if pg.get('label')=='sellado']
    if len(pl)<2: return None
    pl=sorted(pl,key=lambda p:cv2.contourArea(p.astype(np.float32)),reverse=True)
    m=np.zeros((H,W),np.uint8); cv2.fillPoly(m,[pl[0].astype(np.int32)],1); cv2.fillPoly(m,[pl[1].astype(np.int32)],0)
    return m

# ---- build pack-cropped samples per product ----
train=[]; val=[]   # train: (img3,mask) ; val: (img3,mask,prod)
for xmlrel,imgrel,prod in DATASETS:
    samps=[]; reviewed_only="reviewed" in xmlrel   # prod2_reviewed.xml -> train only human-reviewed packs
    for node in ET.parse(f"{ROOT}/{xmlrel}").getroot().findall('image'):
        nm=node.get('name'); p=f"{ROOT}/{imgrel}/{nm}"; tg=tags_of(node)
        if "exclude" in tg: continue                                   # never train on excluded packs
        if reviewed_only and "reviewed" not in tg: continue            # only human-verified GT
        m=seal_mask(node)
        if m is None or not os.path.exists(p): continue
        g=cv2.imread(p,cv2.IMREAD_GRAYSCALE); x0,y0,x1,y1=pack_bbox(g)
        gc=norm(g[y0:y1,x0:x1]); samps.append((np.stack([gc,gc,gc],-1), m[y0:y1,x0:x1], nm))
    random.Random(SEED).shuffle(samps)
    forced=[s for s in samps if s[2] in FORCE_TRAIN]; rest=[s for s in samps if s[2] not in FORCE_TRAIN]
    vp = VAL_PER if len(rest) >= 6 else 0          # tiny products (e.g. prod6=3) -> all to train, no val
    val   += [(im,mk,prod) for im,mk,_ in rest[:vp]]
    train += [(im,mk) for im,mk,_ in rest[vp:]+forced]
    print(f"{prod}: {len(samps)} samples -> {vp} val / {len(samps)-vp} train (forced-train: {len(forced)})",flush=True)
print(f"TOTAL train {len(train)}  val {len(val)}",flush=True)

def add_contamination(img3, band):
    """Paste randomized dark/bright blobs onto the seal band (where band==1); mask is NOT changed.
    Teaches the model the seal region is geometric and invariant to contamination on it."""
    h,w=band.shape; ys,xs=np.where(band>0)
    if len(xs)==0: return img3
    out=img3.astype(np.float32); bandd=cv2.dilate(band,np.ones((7,7),np.uint8))
    for _ in range(random.randint(1,4)):
        k=random.randrange(len(xs)); cx,cy=int(xs[k]),int(ys[k]); blob=np.zeros((h,w),np.float32)
        for _ in range(random.randint(2,6)):
            ox=cx+random.randint(-30,30); oy=cy+random.randint(-30,30)
            cv2.ellipse(blob,(ox,oy),(random.randint(10,45),random.randint(10,45)),random.randint(0,180),0,360,1.0,-1)
        blob=cv2.GaussianBlur(blob,(0,0),random.uniform(4,10)); mx=blob.max()
        if mx<1e-6: continue
        blob=(blob/mx)*(bandd>0)
        val=random.uniform(5,40) if random.random()<0.8 else random.uniform(200,245)   # mostly dark (NIR contaminant)
        a=(blob*random.uniform(0.7,0.95))[...,None]
        tex=val+np.random.randn(h,w,1)*9
        out=out*(1-a)+tex*a
    return np.clip(out,0,255).astype(np.uint8)

def load_contaminants(xml=None):
    """Cut out EVERY labelled defect/liquid instance (patch + feathered alpha) from ALL product
    annotation files in DATASETS -> a large, realistic library of 'stuff on the band' to paste onto the
    seal (mask unchanged). Replaces the old standalone contaminants.xml so the augmentation draws on every
    labelled defect in the dataset, not a hand-picked file."""
    insts=[]
    for xmlrel,imgrel,prod in DATASETS:
        xmlp=f"{ROOT}/{xmlrel}"
        if not os.path.exists(xmlp): continue
        for node in ET.parse(xmlp).getroot().findall("image"):
            name=node.get("name")
            if name in CONTAM_EXCLUDE: continue
            polys=[pg for pg in node.findall("polygon") if pg.get("label") in ("defect","liquid")]
            if not polys: continue
            p=f"{ROOT}/{imgrel}/{name}"
            if not os.path.exists(p): continue
            g=cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if g is None: continue
            for pg in polys:
                pts=parse_pts(pg.get("points")).astype(np.int32)
                x,y,bw,bh=cv2.boundingRect(pts)
                if bw<4 or bh<4: continue
                patch=g[y:y+bh, x:x+bw].copy()
                al=np.zeros((bh,bw),np.uint8); cv2.fillPoly(al,[pts-[x,y]],255)
                al=cv2.GaussianBlur(al,(0,0),2).astype(np.float32)/255.0   # feather edges
                insts.append((patch,al))
    return insts

def paste_contaminants(img3, band, insts):
    """Paste real contaminant cut-outs onto the seal band at random scale/rotation/position. Mask unchanged."""
    if not insts: return img3
    h,w=band.shape; ys,xs=np.where(band>0)
    if len(xs)==0: return img3
    out=img3.copy()
    for _ in range(random.randint(2,5)):                           # more per image
        patch,al=random.choice(insts)
        s=random.uniform(0.12,0.55) if random.random()<0.7 else random.uniform(0.55,1.8)  # bias SMALL
        pw,ph=max(3,int(patch.shape[1]*s)),max(3,int(patch.shape[0]*s))
        p=cv2.resize(patch,(pw,ph)).astype(np.float32); a=cv2.resize(al,(pw,ph))
        M=cv2.getRotationMatrix2D((pw/2,ph/2),random.uniform(0,360),1.0)
        p=cv2.warpAffine(p,M,(pw,ph),borderValue=0); a=cv2.warpAffine(a,M,(pw,ph),borderValue=0)
        if random.random()<0.5: p=cv2.flip(p,1); a=cv2.flip(a,1)
        p=np.clip(p*random.uniform(0.7,1.2),0,255)                 # intensity jitter
        k=random.randrange(len(xs)); cx,cy=int(xs[k]),int(ys[k])   # random spot on the band
        x0,y0=cx-pw//2,cy-ph//2
        ix0,iy0=max(0,x0),max(0,y0); ix1,iy1=min(w,x0+pw),min(h,y0+ph)
        if ix1<=ix0 or iy1<=iy0: continue
        px0,py0=ix0-x0,iy0-y0; px1,py1=px0+(ix1-ix0),py0+(iy1-iy0)
        aa=(a[py0:py1,px0:px1]*random.uniform(0.8,1.0))[...,None]
        pp=p[py0:py1,px0:px1][...,None]
        reg=out[iy0:iy1,ix0:ix1].astype(np.float32)
        out[iy0:iy1,ix0:ix1]=np.clip(reg*(1-aa)+pp*aa,0,255).astype(np.uint8)
    return out

CONTAM=load_contaminants()
print(f"defect cut-outs for band augmentation (from all labelled packs): {len(CONTAM)} (excluded {CONTAM_EXCLUDE})",flush=True)

def load_prints():
    """Auto-extract printed-graphic cut-outs (barcode/nutrition/text) from prod2 flanges. No labels needed:
    printed content is reliably darker than the flange WITHIN the seal band."""
    insts=[]
    xmlp=f"{ROOT}/data/annotations/annotations.xml"; imgdir=f"{ROOT}/data/images/prod2"
    for node in ET.parse(xmlp).getroot().findall('image'):
        nm=node.get('name'); p=f"{imgdir}/{nm}"; m=seal_mask(node)
        if m is None or not os.path.exists(p): continue
        g=cv2.imread(p,cv2.IMREAD_GRAYSCALE); x0,y0,x1,y1=pack_bbox(g); crop=norm(g[y0:y1,x0:x1]); band=m[y0:y1,x0:x1]
        fl=band>0
        if fl.sum()<200: continue
        flmed=np.median(crop[fl])
        pm=(((flmed-crop.astype(np.float32))>28)&fl).astype(np.uint8)*255
        pm=cv2.dilate(pm,np.ones((13,13),np.uint8))
        n,lab,stats,_=cv2.connectedComponentsWithStats(pm)
        for i in range(1,n):
            x,y,bw,bh,area=stats[i]
            if 600<area<90000 and bw>20 and bh>12:
                pad=4; yy=max(0,y-pad); xx=max(0,x-pad); patch=crop[yy:y+bh+pad,xx:x+bw+pad].copy()
                sub=(lab[yy:y+bh+pad,xx:x+bw+pad]==i)
                al=(((flmed-patch.astype(np.float32))/max(1,flmed*0.6)).clip(0,1))*sub
                insts.append((patch, cv2.GaussianBlur(al.astype(np.float32),(0,0),1.5)))
    return insts

def paste_prints(img3, band, insts):
    """Paste printed-graphic cut-outs onto the band at varied scale/rotation; mask unchanged."""
    if not insts: return img3
    h,w=band.shape; ys,xs=np.where(band>0)
    if len(xs)==0: return img3
    out=img3.copy()
    for _ in range(random.randint(1,2)):
        patch,al=random.choice(insts); s=random.uniform(0.5,1.4)
        pw,ph=max(4,int(patch.shape[1]*s)),max(4,int(patch.shape[0]*s))
        p=cv2.resize(patch,(pw,ph)).astype(np.float32); a=cv2.resize(al,(pw,ph))
        M=cv2.getRotationMatrix2D((pw/2,ph/2),random.choice([0,90,180,270])+random.uniform(-15,15),1.0)
        p=cv2.warpAffine(p,M,(pw,ph)); a=cv2.warpAffine(a,M,(pw,ph))
        if random.random()<0.5: p=cv2.flip(p,1); a=cv2.flip(a,1)
        k=random.randrange(len(xs)); cx,cy=int(xs[k]),int(ys[k]); x0,y0=cx-pw//2,cy-ph//2
        ix0,iy0=max(0,x0),max(0,y0); ix1,iy1=min(w,x0+pw),min(h,y0+ph)
        if ix1<=ix0 or iy1<=iy0: continue
        px0,py0=ix0-x0,iy0-y0; px1,py1=px0+(ix1-ix0),py0+(iy1-iy0)
        aa=a[py0:py1,px0:px1][...,None]; pp_=p[py0:py1,px0:px1][...,None]
        reg=out[iy0:iy1,ix0:ix1].astype(np.float32); out[iy0:iy1,ix0:ix1]=np.clip(reg*(1-aa)+pp_*aa,0,255).astype(np.uint8)
    return out

PRINTS=load_prints()
print(f"printed-graphic instances loaded: {len(PRINTS)}",flush=True)

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
    def __init__(s,d,L): s.d=d; s.tf=ttf(); s.L=L
    def __len__(s): return s.L
    def __getitem__(s,i):
        im,mk=s.d[random.randrange(len(s.d))]
        if random.random()<P_PRINT:                                 # printed graphics on band (mask unchanged)
            im = paste_prints(im,mk,PRINTS)
        if random.random()<P_CONTAM:                                # contaminants on band (mask unchanged)
            im = paste_contaminants(im,mk,CONTAM) if CONTAM else add_contamination(im,mk)
        o=s.tf(image=im,mask=mk); return o["image"],o["mask"].float().unsqueeze(0)
train_dl=torch.utils.data.DataLoader(TR(train,SAMPLES),batch_size=BATCH,shuffle=True)
etf_=etf()

# ---- model: fine-tune from best_lite.pt, OR random init for the from-scratch ablation ----
base=torch.load(BASE,map_location="cpu"); ENC=base["encoder"]
model=smp.Unet(ENC,encoder_weights=None,in_channels=3,classes=1)
if SCRATCH:
    print(f"FROM SCRATCH: random-init {ENC} (no ImageNet, no base), {EPOCHS} epochs",flush=True)
else:
    model.load_state_dict(base["state_dict"]); print(f"fine-tuning {ENC} from {BASE} (prod2 val_dice {base['val_dice']:.3f})",flush=True)
model=model.to(dev)

def dloss(l,t,e=1.): p=torch.sigmoid(l); return (1-((2*(p*t).sum((2,3))+e)/(p.sum((2,3))+t.sum((2,3))+e))).mean()
bce=nn.BCEWithLogitsLoss()
@torch.no_grad()
def dice_one(im3,mk):
    o=etf_(image=im3,mask=mk); x=o["image"].unsqueeze(0).to(dev); y=o["mask"].float().unsqueeze(0).unsqueeze(0).to(dev)
    p=(torch.sigmoid(model(x))>THRESH).float()
    return ((2*(p*y).sum()+1)/(p.sum()+y.sum()+1)).item()
def val_dice():
    model.eval(); per={};
    for im,mk,prod in val: per.setdefault(prod,[]).append(dice_one(im,mk))
    return {k:np.mean(v) for k,v in per.items()}, np.mean([d for v in per.values() for d in v])
opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
use_amp=(dev=="cuda"); scaler=torch.amp.GradScaler("cuda",enabled=use_amp)
print(f"IMG={IMG} BATCH={BATCH} EPOCHS={EPOCHS} SAMPLES={SAMPLES} amp={use_amp}",flush=True)
best=0.; best_state=None
for ep in range(1,EPOCHS+1):
    model.train(); tot=n=0
    for x,y in train_dl:
        x,y=x.to(dev),y.to(dev); opt.zero_grad()
        with torch.amp.autocast("cuda",enabled=use_amp):
            lo=model(x); L=bce(lo,y)+dloss(lo,y)
        scaler.scale(L).backward(); scaler.step(opt); scaler.update(); tot+=L.item()*x.size(0); n+=x.size(0)
    sch.step(); per,ov=val_dice()
    if ov>=best: best=ov; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    if ep%5==0 or ep==1:
        ps=" ".join(f"{k}={v:.3f}" for k,v in sorted(per.items()))
        print(f"epoch {ep:3d}/{EPOCHS}  loss {tot/n:.4f}  VAL {ov:.3f} [{ps}]  best {best:.3f}",flush=True)
if best_state: model.load_state_dict(best_state)
per,ov=val_dice(); ps=" ".join(f"{k}={v:.3f}" for k,v in sorted(per.items()))
torch.save({"state_dict":model.state_dict(),"encoder":ENC,"img":IMG,"thresh":THRESH,"mean":IM_MEAN,"std":IM_STD,"val_dice":best,"products":[d[2] for d in DATASETS]},CKPT)
print(f"\nBEST multi-product VAL {best:.3f}  final per-product [{ps}]  saved {CKPT}",flush=True)
model.cpu().eval()
torch.onnx.export(model,torch.randn(1,3,IMG,IMG),ONNX,opset_version=17,input_names=["input"],output_names=["logits"],dynamo=False)
print(f"exported {ONNX} ({os.path.getsize(ONNX)/1e6:.1f}MB)",flush=True)
print("DONE",flush=True)
