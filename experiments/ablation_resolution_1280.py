#!/usr/bin/env python3
"""Métricas de borde a 1280 (modelo desplegado) para prod2 y global, verificando la fila 1280 de tab:resolucion."""
import os, random, numpy as np, cv2, torch
import xml.etree.ElementTree as ET
from scipy.ndimage import binary_erosion, distance_transform_edt
from seal_inspection import core
from seal_inspection.paths import ROOT as R; dev="cuda" if torch.cuda.is_available() else "cpu"
SEED=42; VAL_PER=2; THRESH=0.5; MARGIN=40; BDPX=5.0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DATASETS=[("data/annotations/prod2_reviewed.xml","data/images/prod2","prod2"),
          ("data/annotations/prod1_reviewed.xml","data/images/prod1","prod1"),
          ("data/annotations/prod3_reviewed.xml","data/images/prod3","prod3"),
          ("data/annotations/prod4_reviewed.xml","data/images/prod4","prod4"),
          ("data/annotations/prod5_reviewed.xml","data/images/prod5","prod5"),
          ("data/annotations/prod6_reviewed.xml","data/images/prod6","prod6")]
FORCE_TRAIN={"seal_1998_1780688689500_raw.png"}
def pp(s): return np.array([[float(a) for a in p.split(',')] for p in s.strip().split(';')],np.float32)
def tags_of(n): return {x.get('label') for x in n.findall('tag')}
def seal_mask(node):
    W=int(node.get('width')); H=int(node.get('height'))
    pl=[pp(pg.get('points')) for pg in node.findall('polygon') if pg.get('label')=='sellado']
    if len(pl)<2: return None
    pl=sorted(pl,key=lambda p:cv2.contourArea(p.astype(np.float32)),reverse=True)
    m=np.zeros((H,W),np.uint8); cv2.fillPoly(m,[pl[0].astype(np.int32)],1); cv2.fillPoly(m,[pl[1].astype(np.int32)],0); return m
def bd(m):
    e=binary_erosion(m.astype(bool),iterations=1,border_value=0); return m.astype(bool)&~e
def metrics(gt,pr):
    gt=(gt>0).astype(np.uint8); pr=(pr>0).astype(np.uint8)
    inter=(gt&pr).sum(); dice=2*inter/(gt.sum()+pr.sum()) if (gt.sum()+pr.sum())>0 else 0
    bg=bd(gt); bp=bd(pr)
    if bg.sum()==0 or bp.sum()==0: return dice,0,np.inf,np.inf
    dtg=distance_transform_edt(~bg); dtp=distance_transform_edt(~bp)
    d=np.concatenate([dtg[bp],dtp[bg]])
    Gb=gt.astype(bool)&(dtg<=BDPX); Pb=pr.astype(bool)&(dtp<=BDPX)
    biou=(Gb&Pb).sum()/max(1,(Gb|Pb).sum())
    return dice,biou,float(np.percentile(d,95)),float(d.mean())
seal,sk=core.load_unet(f"{R}/models/best_lite_reviewed_1280.pt",dev)
val=[]
for xmlrel,imgrel,prod in DATASETS:
    samps=[]
    for node in ET.parse(f"{R}/{xmlrel}").getroot().findall('image'):
        nm=node.get('name'); p=f"{R}/{imgrel}/{nm}"; tg=tags_of(node)
        if "exclude" in tg or "reviewed" not in tg: continue
        m=seal_mask(node)
        if m is None or not os.path.exists(p): continue
        samps.append((nm,p,m))
    random.Random(SEED).shuffle(samps)
    rest=[s for s in samps if s[0] not in FORCE_TRAIN]
    for nm,p,m in rest[:VAL_PER if len(rest)>=6 else 0]: val.append((prod,p,m))
res={}
for prod,p,m in val:
    g=cv2.imread(p,0); x0,y0,x1,y1=core.pack_bbox(g)
    prob=core.predict_probability(seal,g[y0:y1,x0:x1],sk["img"],dev)
    pred=(cv2.resize(prob,(x1-x0,y1-y0))>THRESH).astype(np.uint8)
    gt=m[y0:y1,x0:x1]
    res.setdefault(prod,[]).append(metrics(gt,pred))
allm=[x for v in res.values() for x in v]
def agg(rows): return np.mean([r[0] for r in rows]),np.mean([r[1] for r in rows]),np.mean([r[2] for r in rows]),np.mean([r[3] for r in rows])
print("BOUNDARY @1280 (modelo desplegado):")
if "prod2" in res:
    d,b,h,a=agg(res["prod2"]); print("  prod2:   Dice %.3f  B-IoU %.3f  HD95 %.2f  ASSD %.2f  (n=%d)"%(d,b,h,a,len(res["prod2"])))
d,b,h,a=agg(allm); print("  overall: Dice %.3f  B-IoU %.3f  HD95 %.2f  ASSD %.2f  (n=%d)"%(d,b,h,a,len(allm)))
