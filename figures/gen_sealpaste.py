#!/usr/bin/env python3
"""Ilustra la copy-paste de la ETAPA DE SELLADO: se pegan recortes de contaminación/grafismo
sobre la banda, pero la máscara del anillo NO cambia (enseña que el sellado es geométrico)."""
import os, random, numpy as np, cv2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from seal_inspection import train_seal as ts, cvat, core
from seal_inspection.paths import ROOT as R; random.seed(3); np.random.seed(3)
lib=ts.load_cutouts(R); print("cut-outs contaminación:",len(lib))
pack=None
for xml,imgrel,prod in ts.DATASETS:
    p=os.path.join(R,xml)
    if not os.path.exists(p): continue
    for node in cvat.iter_images(p):
        tg=cvat.tags(node)
        if "exclude" in tg: continue
        if "reviewed" in xml and "reviewed" not in tg: continue
        bp=ts.build_pack(R,imgrel,node)
        if bp is not None: pack=bp; break
    if pack is not None: break
rgb,mask,name=pack; print("pack:",name,rgb.shape)
pasted=ts.paste_cutouts(rgb.copy(),mask,lib)
# outline of the seal band (same for both)
cnts,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
def draw(img):
    v=img.copy()
    cv2.drawContours(v,cnts,-1,(0,220,0),4)
    return v
fig,ax=plt.subplots(1,2,figsize=(13,6.5))
ax[0].imshow(draw(rgb)); ax[0].set_title("Original (banda de sellado en verde)",fontsize=11)
ax[1].imshow(draw(pasted)); ax[1].set_title("Tras copy-paste de contaminación/grafismo\n(la máscara del sellado NO cambia)",fontsize=11)
for a in ax: a.set_xticks([]); a.set_yticks([])
plt.tight_layout()
out=f"{R}/docs/thesis_figures/fig_sealpaste.png"
plt.savefig(out,dpi=120,bbox_inches="tight"); print("wrote",out)
