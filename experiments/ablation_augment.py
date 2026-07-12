#!/usr/bin/env python3
"""Reevalúa la ablación de aumentación (baseline/roll/sealjit/both) sobre el hold-out
ACTUAL (23 defectos): E2E (sellado predicho) y tira GT (data/strips/test)."""
import glob, os, numpy as np, cv2, torch
from seal_inspection import core
from seal_inspection.paths import ROOT as R; dev="cuda" if torch.cuda.is_available() else "cpu"
MEAN,STD=core.IMAGENET_MEAN,core.IMAGENET_STD
toin3=lambda g:((np.stack([g]*3,-1)/255.0-MEAN)/STD).transpose(2,0,1)[None].astype(np.float32)
MODELS={"baseline":"models/defect_strip.noaug.pt","+roll":"models/defect_roll.pt",
        "+sealjit":"models/defect_jit.pt","+both":"models/defect_rolljit.pt"}
defs={k:core.load_unet(f"{R}/{p}",dev)[0] for k,p in MODELS.items()}
HS,WS=128,1536
def dscore(m,strip):
    with torch.no_grad(): p=torch.sigmoid(m(torch.from_numpy(toin3(strip)).to(dev)))[0,0].cpu().numpy()
    return float(cv2.GaussianBlur(p,(0,0),2).max())
def auroc(S,L):
    pos,neg=S[L==1],S[L==0]; return float(np.mean([(a>b)+0.5*(a==b) for a in pos for b in neg]))

# ---- E2E (sellado predicho) sobre holdout_labels.csv actual ----
seal,sk=core.load_unet(f"{R}/models/best_lite_reviewed_1280.pt",dev)
lab={ln.split(",")[0]:int(ln.split(",")[1]) for ln in open(f"{R}/data/holdout_labels.csv").read().splitlines()[1:] if ln.strip()}
Sc={k:[] for k in MODELS}; L=[]
for nm,l in lab.items():
    h=glob.glob(f"{R}/data/images/*/{nm}.png")+glob.glob(f"{R}/data/images/*/{nm}.jpg")
    if not h: continue
    g=cv2.imread(h[0],0); H,W=g.shape; x0,y0,x1,y1=core.pack_bbox(g)
    prob=core.predict_probability(seal,g[y0:y1,x0:x1],sk["img"],dev)
    full=np.zeros((H,W),np.float32); full[y0:y1,x0:x1]=cv2.resize(prob,(x1-x0,y1-y0))
    O,I=core.mask_to_ring((full>sk.get("thresh",.5)).astype(np.uint8)*255)
    L.append(l)
    if O is None:
        for k in MODELS: Sc[k].append(0.0); continue
    mx,my=core.unroll_maps(O,I,HS,WS); strip=cv2.remap(core.normalize(g),mx,my,cv2.INTER_LINEAR,borderValue=0)
    for k,m in defs.items(): Sc[k].append(dscore(m,strip))
L=np.array(L); nd=int(L.sum()); ng=int((L==0).sum())
print("E2E sobre hold-out actual (%d defecto / %d correcto):"%(nd,ng))
for k in MODELS:
    S=np.array(Sc[k]); au=auroc(S,L)
    tp=int(((S>=0.5)&(L==1)).sum()); fp=int(((S>=0.5)&(L==0)).sum())
    print("  %-9s E2E AUROC %.3f  recall %d/%d  FP %d/%d (%.1f%%)"%(k,au,tp,nd,fp,ng,100*fp/ng))

# ---- tira GT (data/strips/test) ----
tg=sorted(glob.glob(f"{R}/data/strips/test/img/*.png"))
Sg={k:[] for k in MODELS}; Lg=[]
for ip in tg:
    mp=ip.replace("/img/","/mask/")
    if not os.path.exists(mp): continue
    s=cv2.imread(ip,0); mk=cv2.imread(mp,0)
    Lg.append(1 if (mk>127).sum()>0 else 0)
    for k,m in defs.items(): Sg[k].append(dscore(m,s))
Lg=np.array(Lg)
print("\nTira GT (data/strips/test, %d strips, %d defecto):"%(len(Lg),int(Lg.sum())))
for k in MODELS:
    print("  %-9s AUROC tira GT %.3f"%(k,auroc(np.array(Sg[k]),Lg)))

try:
    from seal_inspection.results import save_results
    save_results("ablation_augment", {
        "n_def": int(L.sum()), "n_good": int((L == 0).sum()),
        "variants": {k: {
            "e2e_auroc": float(auroc(np.array(Sc[k]), L)),
            "gt_auroc": float(auroc(np.array(Sg[k]), Lg)),
            "recall": int(((np.array(Sc[k]) >= 0.5) & (L == 1)).sum()),
            "fp": int(((np.array(Sc[k]) >= 0.5) & (L == 0)).sum()),
        } for k in MODELS},
    })
except Exception as _e:
    print('[results] skip:', _e)
