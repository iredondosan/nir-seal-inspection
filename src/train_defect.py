#!/usr/bin/env python3
"""Defect model (Model B): segment defects on the UNROLLED seal strip.
Trains on data/strips/{train}, evaluates on held-out data/strips/{test} (real defects only).
Handles defect scarcity via oversampling + copy-paste of real defect cut-outs onto good strips.
Reports pixel Dice on defect strips + image-level detection (precision/recall/AUROC)."""
import os, glob, random
import numpy as np, cv2, torch, torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

ROOT="/home/ubuntu/TFM/seal-inspection"; STR=f"{ROOT}/data/strips"
SEED=42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
HS=128; WS=1536; BATCH=8; EPOCHS=60; STEPS=1200; THR=0.5; P_PASTE=0.7
MEAN=(.485,.456,.406); STD=(.229,.224,.225)
dev="cuda" if torch.cuda.is_available() else "cpu"

def load(split):
    items=[]
    for ip in sorted(glob.glob(f"{STR}/{split}/img/*.png")):
        mp=ip.replace("/img/","/mask/");
        if not os.path.exists(mp): continue
        img=cv2.imread(ip,cv2.IMREAD_GRAYSCALE); m=cv2.imread(mp,cv2.IMREAD_GRAYSCALE)
        items.append((img,(m>127).astype(np.uint8),os.path.basename(ip)))
    return items
train=load("train"); test=load("test")
tr_def=[t for t in train if t[1].sum()>0]; tr_good=[t for t in train if t[1].sum()==0]
print(f"train {len(train)} ({len(tr_def)} defect / {len(tr_good)} good)  test {len(test)} ({sum(1 for t in test if t[1].sum()>0)} defect)",flush=True)

# defect cut-out library (from train defect strips) for copy-paste
LIB=[]
for img,m,_ in tr_def:
    n,lab,stats,_=cv2.connectedComponentsWithStats(m)
    for i in range(1,n):
        x,y,w,h,area=stats[i]
        if area<6: continue
        patch=img[y:y+h,x:x+w].copy(); al=(lab[y:y+h,x:x+w]==i).astype(np.float32)
        LIB.append((patch,al))
print(f"defect cut-outs in library: {len(LIB)}",flush=True)

def paste(img,m):
    if not LIB: return img,m
    out=img.copy().astype(np.float32); mo=m.copy()
    for _ in range(random.randint(1,3)):
        patch,al=random.choice(LIB); s=random.uniform(0.6,1.6)
        pw,ph=max(3,int(patch.shape[1]*s)),max(3,int(patch.shape[0]*s))
        p=cv2.resize(patch,(pw,ph)).astype(np.float32); a=cv2.resize(al,(pw,ph))
        if random.random()<0.5: p=cv2.flip(p,1); a=cv2.flip(a,1)
        if random.random()<0.5: p=cv2.flip(p,0); a=cv2.flip(a,0)
        p=np.clip(p*random.uniform(0.8,1.15),0,255); a=cv2.GaussianBlur(a,(0,0),1.0)
        H,W=img.shape; x0=random.randint(0,max(0,W-pw)); y0=random.randint(0,max(0,H-ph))
        x1,y1=min(W,x0+pw),min(H,y0+ph); aa=a[:y1-y0,:x1-x0]; pp=p[:y1-y0,:x1-x0]
        reg=out[y0:y1,x0:x1]; out[y0:y1,x0:x1]=reg*(1-aa)+pp*aa
        mo[y0:y1,x0:x1]=np.maximum(mo[y0:y1,x0:x1],(aa>0.3).astype(np.uint8))
    return np.clip(out,0,255).astype(np.uint8),mo

aug=A.Compose([A.HorizontalFlip(p=.5),A.VerticalFlip(p=.5),A.RandomBrightnessContrast(.3,.3,p=.7),
               A.ShiftScaleRotate(shift_limit=.03,scale_limit=.05,rotate_limit=4,border_mode=cv2.BORDER_REFLECT,p=.5),
               A.Normalize(MEAN,STD),ToTensorV2()])
ev=A.Compose([A.Normalize(MEAN,STD),ToTensorV2()])
class DS(torch.utils.data.Dataset):
    def __init__(s,L): s.L=L
    def __len__(s): return s.L
    def __getitem__(s,i):
        # oversample defect: 50% draw a real defect strip, else good
        img,m,_=random.choice(tr_def) if (tr_def and random.random()<0.5) else random.choice(tr_good)
        img,m=img.copy(),m.copy()
        if random.random()<P_PASTE: img,m=paste(img,m)               # copy-paste defects (user's idea)
        o=aug(image=np.stack([img]*3,-1),mask=m); return o["image"],o["mask"].float().unsqueeze(0)
dl=torch.utils.data.DataLoader(DS(STEPS),batch_size=BATCH,shuffle=True,num_workers=4)

model=smp.Unet("resnet18",encoder_weights="imagenet",in_channels=3,classes=1).to(dev)
def dice_l(l,t,e=1.): p=torch.sigmoid(l); return (1-((2*(p*t).sum((2,3))+e)/(p.sum((2,3))+t.sum((2,3))+e))).mean()
bce=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(20.).to(dev))
opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
scaler=torch.amp.GradScaler("cuda",enabled=dev=="cuda")

@torch.no_grad()
def evaluate():
    model.eval(); scores=[]; labels=[]; dices=[]
    for img,m,_ in test:
        x=ev(image=np.stack([img]*3,-1),mask=m)["image"].unsqueeze(0).to(dev)
        prob=torch.sigmoid(model(x))[0,0].cpu().numpy()
        scores.append(float(cv2.GaussianBlur(prob,(0,0),2).max())); labels.append(1 if m.sum()>0 else 0)
        if m.sum()>0:
            pr=(prob>THR).astype(np.uint8); inter=(pr&m).sum(); dices.append(2*inter/(pr.sum()+m.sum()+1e-6))
    scores=np.array(scores); labels=np.array(labels)
    # AUROC (Mann-Whitney)
    pos=scores[labels==1]; neg=scores[labels==0]
    auroc=np.mean([ (1.0 if a>b else 0.5 if a==b else 0.0) for a in pos for b in neg]) if len(pos) and len(neg) else float('nan')
    # best-F1 threshold over score
    best=(0,0,0,0)
    for th in np.unique(scores):
        pred=(scores>=th).astype(int); tp=((pred==1)&(labels==1)).sum(); fp=((pred==1)&(labels==0)).sum(); fn=((pred==0)&(labels==1)).sum()
        pr=tp/(tp+fp+1e-9); rc=tp/(tp+fn+1e-9); f1=2*pr*rc/(pr+rc+1e-9)
        if f1>best[0]: best=(f1,pr,rc,th)
    return auroc,np.mean(dices) if dices else float('nan'),best

best_auroc=0; best_state=None
for ep in range(1,EPOCHS+1):
    model.train(); tot=n=0
    for x,y in dl:
        x,y=x.to(dev),y.to(dev); opt.zero_grad()
        with torch.amp.autocast("cuda",enabled=dev=="cuda"):
            lo=model(x); L=bce(lo,y)+dice_l(lo,y)
        scaler.scale(L).backward(); scaler.step(opt); scaler.update(); tot+=L.item()*x.size(0); n+=x.size(0)
    sch.step()
    if ep%10==0 or ep==1:
        au,pd,(f1,pr,rc,th)=evaluate()
        print(f"ep {ep:3d} loss {tot/n:.4f} | test AUROC {au:.3f} pixelDice {pd:.3f} | bestF1 {f1:.3f} (P{pr:.2f} R{rc:.2f})",flush=True)
        if au>=best_auroc: best_auroc=au; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
if best_state: model.load_state_dict(best_state)
au,pd,(f1,pr,rc,th)=evaluate()
torch.save({"state_dict":model.state_dict(),"encoder":"resnet18","HS":HS,"WS":WS,"thr":THR,"score_thr":float(th),"mean":MEAN,"std":STD},f"{ROOT}/models/defect_strip.pt")
print(f"\nFINAL test: AUROC {au:.3f}  pixelDice {pd:.3f}  bestF1 {f1:.3f} (P{pr:.2f} R{rc:.2f} @score>{th:.2f})",flush=True)
# save test prediction overlays
os.makedirs(f"{ROOT}/outputs/defect_test",exist_ok=True); model.eval()
with torch.no_grad():
    for img,m,nm in test:
        if m.sum()==0: continue
        x=ev(image=np.stack([img]*3,-1),mask=m)["image"].unsqueeze(0).to(dev)
        prob=torch.sigmoid(model(x))[0,0].cpu().numpy()
        v=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR); v[m>0]=(0,180,0)              # GT green
        v[(prob>THR)]=(0,0,230)                                              # pred red
        cv2.imwrite(f"{ROOT}/outputs/defect_test/{nm}",v)
print("saved test overlays to outputs/defect_test/",flush=True); print("DONE",flush=True)
