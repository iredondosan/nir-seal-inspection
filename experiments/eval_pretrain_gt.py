#!/usr/bin/env python3
"""GT-strip AUROC of the defect head, ImageNet-pretrained vs from-scratch, on the reference
test strips. Completes the 'tira de referencia' row of tab:pretrain (only E2E was saved).
Adds gt_strip_auroc to results/ablation_transfer_e2e.json."""
import glob, json, numpy as np, cv2, torch
from seal_inspection import core
from seal_inspection.paths import ROOT as R
dev="cuda" if torch.cuda.is_available() else "cpu"
MEAN,STD=core.IMAGENET_MEAN,core.IMAGENET_STD

def load_strips(split):
    items=[]
    for ip in sorted(glob.glob(f"{R}/data/strips/{split}/img/*.png")):
        m=cv2.imread(ip.replace("/img/","/mask/"),0); img=cv2.imread(ip,0)
        items.append((img, 1 if (m is not None and m.sum()>0) else 0))
    return items
test=load_strips("test")

@torch.no_grad()
def dscore(model,strip):
    x=((np.stack([strip]*3,-1)/255.-MEAN)/STD).transpose(2,0,1)[None].astype(np.float32)
    p=torch.sigmoid(model(torch.from_numpy(x).to(dev)))[0,0].cpu().numpy()
    return float(cv2.GaussianBlur(p,(0,0),2).max())

def auroc(model):
    sc=np.array([dscore(model,s) for s,_ in test]); lb=np.array([l for _,l in test])
    pos,neg=sc[lb==1],sc[lb==0]
    return float(np.mean([(x>y)+0.5*(x==y) for x in pos for y in neg])),int((lb==1).sum()),int((lb==0).sum())

res={}
for tag,p in [("imagenet","models/defect_strip.pt"),("scratch","models/defect_scratch_es.pt")]:
    m,_=core.load_unet(f"{R}/{p}",dev)
    au,nd,ng=auroc(m); res[tag]=round(au,4)
    print("%s: GT-strip AUROC %.4f  (%d def / %d good)"%(tag,au,nd,ng),flush=True)

jp=f"{R}/results/ablation_transfer_e2e.json"
d=json.load(open(jp,encoding="utf-8"))
d["gt_strip_auroc"]={"imagenet":res["imagenet"],"scratch":res["scratch"]}
json.dump(d,open(jp,"w",encoding="utf-8"),indent=2,ensure_ascii=False)
print("added gt_strip_auroc to results/ablation_transfer_e2e.json",flush=True)
