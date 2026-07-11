#!/usr/bin/env python3
"""Verifica tab:umbral (barrido de umbral, modelo desplegado) y tab:tiny (TinyUNet e2e + tira GT)
sobre el hold-out actual. Calcula el sellado una vez por pieza y comparte la tira."""
import glob, os, numpy as np, cv2, torch
from seal_inspection import core
from seal_inspection.tiny_unet import TinyUNet
from seal_inspection.paths import ROOT as R; dev="cuda" if torch.cuda.is_available() else "cpu"
MEAN,STD=core.IMAGENET_MEAN,core.IMAGENET_STD
toin3=lambda g:((np.stack([g]*3,-1)/255.0-MEAN)/STD).transpose(2,0,1)[None].astype(np.float32)
def auroc(S,L):
    pos,neg=S[L==1],S[L==0]; return float(np.mean([(a>b)+0.5*(a==b) for a in pos for b in neg]))
seal,sk=core.load_unet(f"{R}/models/best_lite_reviewed_1280.pt",dev)
defm,dk=core.load_unet(f"{R}/models/defect_strip.pt",dev); HS,WS=dk["HS"],dk["WS"]
tiny=TinyUNet(base=16,in_ch=1).to(dev)
tiny.load_state_dict(torch.load(f"{R}/models/tiny_defect.pt",map_location=dev,weights_only=False)["state_dict"]); tiny.eval()
def sc_res(strip):
    with torch.no_grad(): p=torch.sigmoid(defm(torch.from_numpy(toin3(strip)).to(dev)))[0,0].cpu().numpy()
    return float(cv2.GaussianBlur(p,(0,0),2).max())
def sc_tiny(strip):
    x=torch.from_numpy(((strip/255.0-0.5)/0.5).astype(np.float32))[None,None].to(dev)
    with torch.no_grad(): p=torch.sigmoid(tiny(x))[0,0].cpu().numpy()
    return float(cv2.GaussianBlur(p,(0,0),2).max())
lab={ln.split(",")[0]:int(ln.split(",")[1]) for ln in open(f"{R}/data/holdout_labels.csv").read().splitlines()[1:] if ln.strip()}
Sr=[];Sti=[];L=[]
for nm,l in lab.items():
    h=glob.glob(f"{R}/data/images/*/{nm}.png")+glob.glob(f"{R}/data/images/*/{nm}.jpg")
    if not h: continue
    g=cv2.imread(h[0],0); H,W=g.shape; x0,y0,x1,y1=core.pack_bbox(g)
    prob=core.predict_probability(seal,g[y0:y1,x0:x1],sk["img"],dev)
    full=np.zeros((H,W),np.float32); full[y0:y1,x0:x1]=cv2.resize(prob,(x1-x0,y1-y0))
    O,I=core.mask_to_ring((full>sk.get("thresh",.5)).astype(np.uint8)*255); L.append(l)
    if O is None: Sr.append(0.0); Sti.append(0.0); continue
    mx,my=core.unroll_maps(O,I,HS,WS); strip=cv2.remap(core.normalize(g),mx,my,cv2.INTER_LINEAR,borderValue=0)
    Sr.append(sc_res(strip)); Sti.append(sc_tiny(strip))
Sr,Sti,L=np.array(Sr),np.array(Sti),np.array(L); nd=int(L.sum()); ng=int((L==0).sum())
print("== tab:umbral (deployed resnet18, E2E, %d def / %d good) AUROC=%.4f =="%(nd,ng,auroc(Sr,L)))
print("umbral  TP FP TN FN  Prec  Sens  F1   Exac")
for th in [0.50,0.70,0.85,0.90,0.95]:
    tp=int(((Sr>=th)&(L==1)).sum()); fp=int(((Sr>=th)&(L==0)).sum()); fn=nd-tp; tn=ng-fp
    pr=tp/(tp+fp) if tp+fp else 0; se=tp/nd; f1=2*pr*se/(pr+se) if pr+se else 0; ac=(tp+tn)/(nd+ng)
    print("%.2f    %2d %2d %3d %2d  %.2f  %.2f  %.2f %.3f"%(th,tp,fp,tn,fn,pr,se,f1,ac))
print("\n== tab:tiny (TinyUNet, E2E) AUROC=%.4f  recall@0.5 %d/%d  FP %d/%d =="%(
    auroc(Sti,L),int(((Sti>=0.5)&(L==1)).sum()),nd,int(((Sti>=0.5)&(L==0)).sum()),ng))
# tira GT para ambos
tg=sorted(glob.glob(f"{R}/data/strips/test/img/*.png")); Rg=[];Tg=[];Lg=[]
for ip in tg:
    mp=ip.replace("/img/","/mask/")
    if not os.path.exists(mp): continue
    s=cv2.imread(ip,0); mk=cv2.imread(mp,0); Lg.append(1 if (mk>127).sum()>0 else 0)
    Rg.append(sc_res(s)); Tg.append(sc_tiny(s))
Lg=np.array(Lg)
print("tira GT: resnet18 AUROC=%.4f  TinyUNet AUROC=%.4f  (%d strips, %d def)"%(auroc(np.array(Rg),Lg),auroc(np.array(Tg),Lg),len(Lg),int(Lg.sum())))
