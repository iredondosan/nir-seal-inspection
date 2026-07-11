#!/usr/bin/env python3
"""Genera una figura que ilustra la aumentación copy-paste de defectos sobre la tira:
tira correcta -> se pegan recortes de defectos reales -> tira aumentada + máscara actualizada."""
import glob, os, random, numpy as np, cv2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from seal_inspection.paths import ROOT as R; STR=f"{R}/data/strips"
random.seed(7); np.random.seed(7)

def load(split):
    items=[]
    for ip in sorted(glob.glob(f"{STR}/{split}/img/*.png")):
        mp=ip.replace("/img/","/mask/")
        if not os.path.exists(mp): continue
        items.append((cv2.imread(ip,0),(cv2.imread(mp,0)>127).astype(np.uint8)))
    return items

def build_lib(defs):
    lib=[]
    for img,mask in defs:
        n,lab,stats,_=cv2.connectedComponentsWithStats(mask)
        for i in range(1,n):
            x,y,w,h,area=stats[i]
            if area<6: continue
            lib.append((img[y:y+h,x:x+w].copy(),(lab[y:y+h,x:x+w]==i).astype(np.float32)))
    return lib

def paste(img,mask,lib,k):
    out=img.copy().astype(np.float32); mo=mask.copy(); H,W=img.shape
    for _ in range(k):
        patch,alpha=random.choice(lib); s=random.uniform(0.7,1.5)
        pw,ph=max(3,int(patch.shape[1]*s)),max(3,int(patch.shape[0]*s))
        p=cv2.resize(patch,(pw,ph)).astype(np.float32); a=cv2.resize(alpha,(pw,ph))
        if random.random()<0.5: p,a=cv2.flip(p,1),cv2.flip(a,1)
        a=cv2.GaussianBlur(a,(0,0),1.0)
        x0,y0=random.randint(0,max(0,W-pw)),random.randint(0,max(0,H-ph))
        x1,y1=min(W,x0+pw),min(H,y0+ph); aa=a[:y1-y0,:x1-x0]; pp=p[:y1-y0,:x1-x0]
        out[y0:y1,x0:x1]=out[y0:y1,x0:x1]*(1-aa)+pp*aa
        mo[y0:y1,x0:x1]=np.maximum(mo[y0:y1,x0:x1],(aa>0.3).astype(np.uint8))
    return np.clip(out,0,255).astype(np.uint8),mo

train=load("train")
defs=[t for t in train if t[1].sum()>0]; good=[t for t in train if t[1].sum()==0]
lib=build_lib(defs)
print("defect strips:",len(defs)," good:",len(good)," cut-outs:",len(lib))

def overlay(img,mask):
    v=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    v[mask>0]=(0,0,220)
    return cv2.cvtColor(v,cv2.COLOR_BGR2RGB)

picks=random.sample(range(len(good)),2)
rows=[]
for gi in picks:
    g,gm=good[gi]
    aug,am=paste(g,gm,lib,random.randint(2,3))
    rows.append(("Tira correcta (original)",cv2.cvtColor(g,cv2.COLOR_GRAY2RGB)))
    rows.append(("Tras copy-paste de defectos reales (máscara en rojo)",overlay(aug,am)))

fig,axs=plt.subplots(len(rows),1,figsize=(13,1.4*len(rows)))
for ax,(title,im) in zip(axs,rows):
    ax.imshow(im,aspect="auto"); ax.set_title(title,fontsize=10,loc="left"); ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
out=f"{R}/docs/thesis_figures/fig_copypaste.png"
plt.savefig(out,dpi=130,bbox_inches="tight"); print("wrote",out)
