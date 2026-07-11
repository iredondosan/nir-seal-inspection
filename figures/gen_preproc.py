#!/usr/bin/env python3
"""Ilustra el preprocesado clásico: imagen NIR cruda -> normalización por percentiles
-> detección de bandeja (pack_bbox, sustracción de fondo) -> recorte a la pieza."""
import glob, numpy as np, cv2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from seal_inspection import core
from seal_inspection.paths import ROOT as R
# pick a raw image with a clear tray (prod2 has many)
cands=sorted(glob.glob(f"{R}/data/images/prod2/*.png"))+sorted(glob.glob(f"{R}/data/images/prod4/*.png"))
g=None
for p in cands:
    im=cv2.imread(p,0)
    if im is None: continue
    x0,y0,x1,y1=core.pack_bbox(im)
    if (x1-x0)>400 and (y1-y0)>400: g=im; box=(x0,y0,x1,y1); name=p.split("/")[-1]; break
print("imagen:",name,g.shape,"bbox:",box)
raw_disp=core.normalize(g)                      # percentile stretch for display
x0,y0,x1,y1=box
withbox=cv2.cvtColor(raw_disp,cv2.COLOR_GRAY2BGR)
cv2.rectangle(withbox,(x0,y0),(x1,y1),(0,220,0),8)
crop=core.normalize(g[y0:y1,x0:x1])
fig,ax=plt.subplots(1,3,figsize=(14,5.2))
ax[0].imshow(raw_disp,cmap="gray"); ax[0].set_title("(1) NIR normalizada\n(estiramiento por percentiles)",fontsize=10)
ax[1].imshow(cv2.cvtColor(withbox,cv2.COLOR_BGR2RGB)); ax[1].set_title("(2) Detección de bandeja\n(sustracción de fondo)",fontsize=10)
ax[2].imshow(crop,cmap="gray"); ax[2].set_title("(3) Recorte a la pieza\n(entrada de la etapa de sellado)",fontsize=10)
for a in ax: a.set_xticks([]); a.set_yticks([])
plt.tight_layout()
out=f"{R}/docs/thesis_figures/fig_preproc.png"
plt.savefig(out,dpi=125,bbox_inches="tight"); print("wrote",out)
