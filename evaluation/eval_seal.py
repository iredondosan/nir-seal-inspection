#!/usr/bin/env python3
"""Re-evaluate the DEPLOYED seal model per-product on the reproduced validation split
(faithful to training/train_seal.py: SEED 42, VAL_PER 2, reviewed-only packs)."""
import os, random, numpy as np, cv2, torch
import xml.etree.ElementTree as ET
import albumentations as A
from albumentations.pytorch import ToTensorV2
from seal_inspection import core
from seal_inspection.paths import ROOT; dev="cuda" if torch.cuda.is_available() else "cpu"
SEED=42; VAL_PER=2; THRESH=0.5; MARGIN=40; IMG=1280
IM_MEAN=(.485,.456,.406); IM_STD=(.229,.224,.225)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DATASETS=[("data/annotations/prod2_reviewed.xml","data/images/prod2","prod2"),
          ("data/annotations/prod1_reviewed.xml","data/images/prod1","prod1"),
          ("data/annotations/prod3_reviewed.xml","data/images/prod3","prod3"),
          ("data/annotations/prod4_reviewed.xml","data/images/prod4","prod4"),
          ("data/annotations/prod5_reviewed.xml","data/images/prod5","prod5"),
          ("data/annotations/prod6_reviewed.xml","data/images/prod6","prod6")]
FORCE_TRAIN={"seal_1998_1780688689500_raw.png"}
def parse_pts(s): return np.array([[float(a) for a in p.split(',')] for p in s.strip().split(';')],np.float32)
def tags_of(node): return {t.get('label') for t in node.findall('tag')}
def seal_mask(node):
    W=int(node.get('width')); H=int(node.get('height'))
    pl=[parse_pts(pg.get('points')) for pg in node.findall('polygon') if pg.get('label')=='sellado']
    if len(pl)<2: return None
    pl=sorted(pl,key=lambda p:cv2.contourArea(p.astype(np.float32)),reverse=True)
    m=np.zeros((H,W),np.uint8); cv2.fillPoly(m,[pl[0].astype(np.int32)],1); cv2.fillPoly(m,[pl[1].astype(np.int32)],0)
    return m
val=[]
for xmlrel,imgrel,prod in DATASETS:
    samps=[]; reviewed_only="reviewed" in xmlrel
    for node in ET.parse(f"{ROOT}/{xmlrel}").getroot().findall('image'):
        nm=node.get('name'); p=f"{ROOT}/{imgrel}/{nm}"; tg=tags_of(node)
        if "exclude" in tg: continue
        if reviewed_only and "reviewed" not in tg: continue
        m=seal_mask(node)
        if m is None or not os.path.exists(p): continue
        g=cv2.imread(p,cv2.IMREAD_GRAYSCALE); x0,y0,x1,y1=core.pack_bbox(g)
        gc=core.normalize(g[y0:y1,x0:x1]); samps.append((np.stack([gc,gc,gc],-1), m[y0:y1,x0:x1], nm))
    random.Random(SEED).shuffle(samps)
    rest=[s for s in samps if s[2] not in FORCE_TRAIN]
    vp=VAL_PER if len(rest)>=6 else 0
    val+=[(im,mk,prod) for im,mk,_ in rest[:vp]]
    print(f"{prod:6s}: {len(samps)} packs -> {vp} val")
model,ck=core.load_unet(f"{ROOT}/models/best_lite_reviewed_1280.pt",dev)
etf=A.Compose([A.Resize(IMG,IMG),A.Normalize(IM_MEAN,IM_STD),ToTensorV2()])
@torch.no_grad()
def dice_one(im3,mk):
    o=etf(image=im3,mask=mk); x=o["image"].unsqueeze(0).to(dev); y=o["mask"].float().unsqueeze(0).unsqueeze(0).to(dev)
    p=(torch.sigmoid(model(x))>THRESH).float()
    return float(((2*(p*y).sum()+1)/(p.sum()+y.sum()+1)).item())
per={}
for im,mk,prod in val: per.setdefault(prod,[]).append(dice_one(im,mk))
print("\nPER-PRODUCT val Dice (deployed model):")
for k in ["prod1","prod2","prod3","prod4","prod5","prod6"]:
    if k in per: print("  %s = %.4f  (n=%d)"%(k,np.mean(per[k]),len(per[k])))
allv=[d for v in per.values() for d in v]
print("  GLOBAL (mean over %d val imgs) = %.4f"%(len(allv),np.mean(allv)))
print("  stored val_dice in checkpoint = %.4f"%ck.get("val_dice",float("nan")))

try:
    from seal_inspection.results import save_results
    save_results("eval_seal", {
        "per_product": {k: float(sum(per[k])/len(per[k])) for k in per},
        "global": float(sum(allv)/len(allv)),
        "checkpoint_val_dice": float(ck.get("val_dice", float("nan"))),
    })
except Exception as _e:
    print('[results] skip:', _e)
