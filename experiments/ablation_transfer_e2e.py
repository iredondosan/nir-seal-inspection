#!/usr/bin/env python3
"""AUROC de extremo a extremo (sellado predicho -> desenrollo -> defecto) para
ImageNet vs desde cero, sobre el hold-out global. Responde: ¿es realmente igual?"""
import glob, os, numpy as np, cv2, torch
from seal_inspection import core
from seal_inspection.paths import ROOT as R; dev="cuda" if torch.cuda.is_available() else "cpu"
MEAN,STD=core.IMAGENET_MEAN,core.IMAGENET_STD
sig=lambda z:1/(1+np.exp(-z))
toin3=lambda g:((np.stack([g]*3,-1)/255.0-MEAN)/STD).transpose(2,0,1)[None].astype(np.float32)
lab={ln.split(",")[0]:int(ln.split(",")[1]) for ln in open(f"{R}/data/holdout_labels.csv").read().splitlines()[1:] if ln.strip()}

def dscore(defm,strip):
    x=torch.from_numpy(toin3(strip)).to(dev)
    with torch.no_grad(): p=torch.sigmoid(defm(x))[0,0].cpu().numpy()
    return float(cv2.GaussianBlur(p,(0,0),2).max())

def e2e_scores(seal_p,def_p):
    seal,sk=core.load_unet(f"{R}/{seal_p}",dev); defm,dk=core.load_unet(f"{R}/{def_p}",dev)
    HS,WS=dk["HS"],dk["WS"]; S=[];L=[];fails=0
    for nm,l in lab.items():
        h=glob.glob(f"{R}/data/images/*/{nm}.png")+glob.glob(f"{R}/data/images/*/{nm}.jpg")
        if not h: continue
        g=cv2.imread(h[0],0); H,W=g.shape; x0,y0,x1,y1=core.pack_bbox(g)
        prob=core.predict_probability(seal,g[y0:y1,x0:x1],sk["img"],dev)
        full=np.zeros((H,W),np.float32); full[y0:y1,x0:x1]=cv2.resize(prob,(x1-x0,y1-y0))
        O,I=core.mask_to_ring((full>sk.get("thresh",.5)).astype(np.uint8)*255)
        if O is None:
            S.append(0.0); L.append(l); fails+=1; continue   # seal falla -> score 0
        mx,my=core.unroll_maps(O,I,HS,WS); strip=cv2.remap(core.normalize(g),mx,my,cv2.INTER_LINEAR,borderValue=0)
        S.append(dscore(defm,strip)); L.append(l)
    S,L=np.array(S),np.array(L); pos,neg=S[L==1],S[L==0]
    au=float(np.mean([(a>b)+0.5*(a==b) for a in pos for b in neg]))
    tp=int(((S>=0.5)&(L==1)).sum()); fp=int(((S>=0.5)&(L==0)).sum())
    return au,tp,int(L.sum()),fp,int((L==0).sum()),fails

_res={}
for tag,sp,dp in [("ImageNet   ","models/best_lite_reviewed_1280.pt","models/defect_strip.pt"),
                  ("desde cero ","models/scratch_seal_1280.pt","models/defect_scratch_es.pt")]:
    au,tp,nd,fp,ng,fails=e2e_scores(sp,dp)
    print("%s  E2E AUROC=%.4f  recall %d/%d  FP %d/%d  (sellado falla en %d piezas)"%(tag,au,tp,nd,fp,ng,fails))
    _res[tag.strip()]={"e2e_auroc":float(au),"recall":int(tp),"n_def":int(nd),"fp":int(fp),"n_good":int(ng),"seal_fail":int(fails)}
try:
    from seal_inspection.results import save_results
    save_results("ablation_transfer_e2e", _res)
except Exception as _e:
    print("[results] skip:", _e)
