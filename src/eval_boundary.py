#!/usr/bin/env python3
"""Offline boundary-aware evaluation of the seal segmenter against GT polygons.
Region metrics (Dice, IoU) hide thin-ring edge errors; this adds Boundary-IoU, HD95, ASSD.
Computes per-image + per-product means; marks which packs are the trainer's held-out VAL.
Usage: .venv/bin/python src/eval_boundary.py [--split val|train|all] [--bd 5] [--csv out.csv]"""
import os, glob, argparse, random
import numpy as np, cv2, torch
import xml.etree.ElementTree as ET
import segmentation_models_pytorch as smp
from scipy.ndimage import distance_transform_edt, binary_erosion

ROOT="/home/ubuntu/TFM/seal-inspection"; MODEL=f"{ROOT}/models/best_lite_multiprod.pt"
IMG=384; MARGIN=40; MEAN=np.array((.485,.456,.406),np.float32); STD=np.array((.229,.224,.225),np.float32)
SEED=42; VAL_PER=2; FORCE_TRAIN={"seal_1998_1780688689500_raw.png"}
DATASETS=[("data/annotations/prod2_reviewed.xml","data/images/prod2","prod2"),
          ("data/annotations/prod1.xml",      "data/images/prod1","prod1"),
          ("data/annotations/prod3.xml",      "data/images/prod3","prod3"),
          ("data/annotations/prod4.xml",      "data/images/prod4","prod4"),
          ("data/annotations/prod5.xml",      "data/images/prod5","prod5")]

def norm(g):
    lo,hi=np.percentile(g,[1,99.5]); hi=max(hi,lo+1); return np.clip((g.astype(np.float32)-lo)/(hi-lo)*255,0,255).astype(np.uint8)
def cc(N):
    cm=np.median(N,0).astype(np.float32); on=np.where(cm>cm.max()*0.5)[0]; cL,cR=on.min(),on.max(); g=np.gradient(cm)
    return int(np.argmax(g[max(0,cL-60):cL+60])+max(0,cL-60)),int(np.argmin(g[cR-60:cR+60])+(cR-60))
def pack_bbox(g):
    N=norm(g); h,w=N.shape
    try: cL,cR=cc(N)
    except Exception: cL,cR=0,w
    top=np.median(N[20:240,:],0); bot=np.median(N[h-240:h-20,:],0); ref=np.maximum(top,bot)
    d=np.clip(np.tile(ref,(h,1))-N.astype(np.float32),0,255); d[:,:cL]=0; d[:,cR:]=0
    mm=cv2.morphologyEx((d>20).astype(np.uint8)*255,cv2.MORPH_OPEN,np.ones((11,11),np.uint8)); mm=cv2.morphologyEx(mm,cv2.MORPH_CLOSE,np.ones((41,41),np.uint8))
    c=[c for c in cv2.findContours(mm,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0] if cv2.contourArea(c)>h*w*0.02]
    if not c: return 0,0,w,h
    x,y,bw,bh=cv2.boundingRect(max(c,key=cv2.contourArea)); return max(0,x-MARGIN),max(0,y-MARGIN),min(w,x+bw+MARGIN),min(h,y+bh+MARGIN)
def pp(s): return np.array([[float(a) for a in p.split(',')] for p in s.strip().split(';')],np.float32)
def tags_of(node): return {t.get('label') for t in node.findall('tag')}
def seal_mask(node):
    W=int(node.get('width')); H=int(node.get('height')); pl=[pp(pg.get('points')) for pg in node.findall('polygon') if pg.get('label')=='sellado']
    if len(pl)<2: return None
    pl=sorted(pl,key=lambda p:cv2.contourArea(p.astype(np.float32)),reverse=True)
    m=np.zeros((H,W),np.uint8); cv2.fillPoly(m,[pl[0].astype(np.int32)],1); cv2.fillPoly(m,[pl[1].astype(np.int32)],0); return m

def bd(m):  # boundary pixels of a binary mask
    e=binary_erosion(m.astype(bool),iterations=1,border_value=0); return m.astype(bool)&~e
def metrics(gt,pr,bdpx):
    gt=(gt>0).astype(np.uint8); pr=(pr>0).astype(np.uint8)
    inter=(gt&pr).sum(); union=(gt|pr).sum(); gs=gt.sum(); ps=pr.sum()
    dice=2*inter/(gs+ps) if (gs+ps)>0 else 0.0
    iou=inter/union if union>0 else 0.0
    bg=bd(gt); bp=bd(pr)
    if bg.sum()==0 or bp.sum()==0:
        return dict(dice=dice,iou=iou,biou=0.0,hd=np.inf,hd95=np.inf,assd=np.inf)
    dtg=distance_transform_edt(~bg); dtp=distance_transform_edt(~bp)
    d=np.concatenate([dtg[bp],dtp[bg]])
    Gb=gt.astype(bool)&(dtg<=bdpx); Pb=pr.astype(bool)&(dtp<=bdpx)
    bi=(Gb&Pb).sum(); bu=(Gb|Pb).sum(); biou=bi/bu if bu>0 else 0.0
    return dict(dice=dice,iou=iou,biou=biou,hd=float(d.max()),hd95=float(np.percentile(d,95)),assd=float(d.mean()))

def split_map(xmlrel,imgrel):
    """Replicate trainer split: XML-order valid samples, shuffle(SEED), forced->train, first VAL_PER of rest->val."""
    samps=[]; reviewed_only="reviewed" in xmlrel
    for node in ET.parse(f"{ROOT}/{xmlrel}").getroot().findall('image'):
        nm=node.get('name'); p=f"{ROOT}/{imgrel}/{nm}"; tg=tags_of(node)
        if "exclude" in tg or (reviewed_only and "reviewed" not in tg): continue
        if seal_mask(node) is None or not os.path.exists(p): continue
        samps.append(nm)
    random.Random(SEED).shuffle(samps)
    forced=[n for n in samps if n in FORCE_TRAIN]; rest=[n for n in samps if n not in FORCE_TRAIN]
    val=set(rest[:VAL_PER]); train=set(rest[VAL_PER:])|set(forced); return train,val

ap=argparse.ArgumentParser(); ap.add_argument("--split",default="val",choices=["val","train","all"])
ap.add_argument("--bd",type=float,default=5.0); ap.add_argument("--model",default=MODEL); ap.add_argument("--csv",default=f"{ROOT}/outputs/eval_boundary.csv"); a=ap.parse_args()
ck=torch.load(a.model,map_location="cpu",weights_only=False); THR=ck.get("thresh",0.5); IMG=ck.get("img",IMG); print(f"model: {a.model}  input res: {IMG}")
m=smp.Unet(ck["encoder"],encoder_weights=None,in_channels=3,classes=1); m.load_state_dict(ck["state_dict"]); m.eval()
dev="cuda" if torch.cuda.is_available() else "cpu"; m=m.to(dev)

rows=[]; per={}
for xmlrel,imgrel,prod in DATASETS:
    train,val=split_map(xmlrel,imgrel); reviewed_only="reviewed" in xmlrel
    for node in ET.parse(f"{ROOT}/{xmlrel}").getroot().findall('image'):
        nm=node.get('name'); p=f"{ROOT}/{imgrel}/{nm}"; tg=tags_of(node)
        if "exclude" in tg or (reviewed_only and "reviewed" not in tg): continue
        gt=seal_mask(node)
        if gt is None or not os.path.exists(p): continue
        inval = nm in val; sp = "val" if inval else "train"
        if a.split=="val" and not inval: continue
        if a.split=="train" and inval: continue
        g=cv2.imread(p,cv2.IMREAD_GRAYSCALE); x0,y0,x1,y1=pack_bbox(g); crop=g[y0:y1,x0:x1]; gtc=gt[y0:y1,x0:x1]; h,w=crop.shape
        im=cv2.resize(np.stack([norm(crop)]*3,-1),(IMG,IMG)).astype(np.float32)/255.0; x=((im-MEAN)/STD).transpose(2,0,1)[None]
        with torch.no_grad(): prob=torch.sigmoid(m(torch.from_numpy(x).to(dev)))[0,0].cpu().numpy()
        pred=(cv2.resize(prob,(w,h))>THR).astype(np.uint8)
        mt=metrics(gtc,pred,a.bd); mt.update(prod=prod,name=nm,split=sp)
        rows.append(mt); per.setdefault(prod,[]).append(mt)

def agg(lst,k):
    v=[r[k] for r in lst if np.isfinite(r[k])]; return np.mean(v) if v else float('nan')
hdr=["prod","split","dice","iou","biou","hd95","assd","name"]
os.makedirs(os.path.dirname(a.csv),exist_ok=True)
with open(a.csv,"w") as f:
    f.write(",".join(hdr)+"\n")
    for r in rows: f.write(f"{r['prod']},{r['split']},{r['dice']:.4f},{r['iou']:.4f},{r['biou']:.4f},{r['hd95']:.2f},{r['assd']:.3f},{r['name']}\n")
print(f"=== boundary eval  split={a.split}  bd={a.bd}px  ({len(rows)} packs) ===")
print(f"{'product':8s} {'n':>3s} {'Dice':>6s} {'IoU':>6s} {'B-IoU':>6s} {'HD95':>7s} {'ASSD':>6s}")
for prod in [d[2] for d in DATASETS]:
    if prod not in per: continue
    L=per[prod]; print(f"{prod:8s} {len(L):3d} {agg(L,'dice'):6.3f} {agg(L,'iou'):6.3f} {agg(L,'biou'):6.3f} {agg(L,'hd95'):7.2f} {agg(L,'assd'):6.2f}")
allr=rows
print(f"{'OVERALL':8s} {len(allr):3d} {agg(allr,'dice'):6.3f} {agg(allr,'iou'):6.3f} {agg(allr,'biou'):6.3f} {agg(allr,'hd95'):7.2f} {agg(allr,'assd'):6.2f}")
nbad=sum(1 for r in allr if not np.isfinite(r['hd95']))
if nbad: print(f"({nbad} packs had empty prediction -> HD95/ASSD undefined, excluded from means)")
print(f"wrote {a.csv}")
