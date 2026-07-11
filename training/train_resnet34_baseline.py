import matplotlib; matplotlib.use("Agg")
import os, glob, random, math, time, sys
import numpy as np, cv2, torch
import torch.nn as nn, torch.nn.functional as F, torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

SEED=42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
from seal_inspection.paths import ROOT
XML=f"{ROOT}/data/annotations/annotations.xml"
IMGDIR=f"{ROOT}/data/images/prod2"
PRED_OUT=f"{ROOT}/outputs/training"; os.makedirs(PRED_OUT,exist_ok=True)
os.makedirs(f"{ROOT}/models",exist_ok=True)
IMG_H,IMG_W=640,512; BATCH=4; EPOCHS=40; SAMPLES_PER_EPOCH=320; VAL_N=2; THRESH=0.5
device="mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print("device:",device,flush=True)

def pts(s): return np.array([[float(a) for a in p.split(',')] for p in s.strip().split(';')],np.float32)
def seal(node):
    H=int(node.get('height')); W=int(node.get('width')); pl=[pts(pg.get('points')) for pg in node.findall('polygon')]
    if len(pl)<2: return None
    pl=sorted(pl,key=lambda q:cv2.contourArea(q.astype(np.float32)),reverse=True)
    m=np.zeros((H,W),np.uint8); cv2.fillPoly(m,[pl[0].astype(np.int32)],1); cv2.fillPoly(m,[pl[1].astype(np.int32)],0); return m
root=ET.parse(XML).getroot(); ann=[]
for n in root.findall('image'):
    p=os.path.join(IMGDIR,n.get('name')); m=seal(n)
    if m is not None and os.path.exists(p): ann.append((p,m))
order=list(range(len(ann))); random.Random(SEED).shuffle(order); vs=set(order[:VAL_N])
train=[ann[i] for i in range(len(ann)) if i not in vs]; val=[ann[i] for i in range(len(ann)) if i in vs]
print(f"annotated {len(ann)} -> train {len(train)} / val {len(val)}",flush=True)
print("VAL:",[os.path.basename(p) for p,_ in val],flush=True)

IM_MEAN=(.485,.456,.406); IM_STD=(.229,.224,.225)
def g3(p):
    im=cv2.imread(p,0); lo,hi=np.percentile(im,[1,99.5]); hi=max(hi,lo+1)
    n=np.clip((im.astype(np.float32)-lo)/(hi-lo)*255,0,255).astype(np.uint8); return np.stack([n]*3,-1)
def ttf():
    geo=[A.HorizontalFlip(p=.5),A.VerticalFlip(p=.5)]
    try: geo.append(A.Affine(scale=(.85,1.15),translate_percent=(0,.06),rotate=(-180,180),border_mode=cv2.BORDER_CONSTANT,fill=0,fill_mask=0,p=.9))
    except TypeError: geo.append(A.Affine(scale=(.85,1.15),translate_percent=(0,.06),rotate=(-180,180),p=.9))
    ph=[A.RandomBrightnessContrast(.4,.4,p=.8),A.RandomGamma((60,140),p=.6)]
    try: ph.append(A.GaussNoise(p=.2))
    except Exception: pass
    return A.Compose([A.Resize(IMG_H,IMG_W),*geo,*ph,A.Normalize(IM_MEAN,IM_STD),ToTensorV2()])
def etf(): return A.Compose([A.Resize(IMG_H,IMG_W),A.Normalize(IM_MEAN,IM_STD),ToTensorV2()])
class TR(torch.utils.data.Dataset):
    def __init__(s,smp,L): s.d=[(g3(p),m) for p,m in smp]; s.tf=ttf(); s.L=L
    def __len__(s): return s.L
    def __getitem__(s,i): img,mk=s.d[random.randrange(len(s.d))]; o=s.tf(image=img,mask=mk); return o["image"],o["mask"].float().unsqueeze(0)
class EV(torch.utils.data.Dataset):
    def __init__(s,smp): s.d=[(g3(p),m) for p,m in smp]; s.tf=etf()
    def __len__(s): return len(s.d)
    def __getitem__(s,i): img,mk=s.d[i]; o=s.tf(image=img,mask=mk); return o["image"],o["mask"].float().unsqueeze(0)
train_dl=torch.utils.data.DataLoader(TR(train,SAMPLES_PER_EPOCH),batch_size=BATCH,shuffle=True)
val_dl=torch.utils.data.DataLoader(EV(val),batch_size=1)

class D(nn.Module):
    def __init__(s,i,k,o): super().__init__(); s.c=nn.Sequential(nn.Conv2d(i+k,o,3,padding=1,bias=False),nn.BatchNorm2d(o),nn.ReLU(True),nn.Conv2d(o,o,3,padding=1,bias=False),nn.BatchNorm2d(o),nn.ReLU(True))
    def forward(s,x,sk): x=F.interpolate(x,size=sk.shape[-2:],mode="bilinear",align_corners=False); return s.c(torch.cat([x,sk],1))
class U(nn.Module):
    def __init__(s,pre=True):
        super().__init__()
        try: b=torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.DEFAULT if pre else None)
        except Exception as e: print("weight DL failed, random init:",e,flush=True); b=torchvision.models.resnet34(weights=None)
        s.stem=nn.Sequential(b.conv1,b.bn1,b.relu); s.pool=b.maxpool; s.l1,s.l2,s.l3,s.l4=b.layer1,b.layer2,b.layer3,b.layer4
        s.d4=D(512,256,256); s.d3=D(256,128,128); s.d2=D(128,64,64); s.d1=D(64,64,32)
        s.f=nn.Sequential(nn.Conv2d(32,32,3,padding=1,bias=False),nn.BatchNorm2d(32),nn.ReLU(True),nn.Conv2d(32,1,1))
    def forward(s,x):
        e0=s.stem(x);e1=s.l1(s.pool(e0));e2=s.l2(e1);e3=s.l3(e2);e4=s.l4(e3)
        d=s.d4(e4,e3);d=s.d3(d,e2);d=s.d2(d,e1);d=s.d1(d,e0);d=F.interpolate(d,scale_factor=2,mode="bilinear",align_corners=False);return s.f(d)
model=U(True).to(device)
def dloss(l,t,e=1.): p=torch.sigmoid(l); return (1-((2*(p*t).sum((2,3))+e)/(p.sum((2,3))+t.sum((2,3))+e))).mean()
bce=nn.BCEWithLogitsLoss()
@torch.no_grad()
def dice(l,t,thr=THRESH,e=1.): p=(torch.sigmoid(l)>thr).float(); return ((2*(p*t).sum((2,3))+e)/(p.sum((2,3))+t.sum((2,3))+e)).mean().item()
opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)

best=0.; best_state=None
for ep in range(1,EPOCHS+1):
    t0=time.time(); model.train(); tot=dsc=n=0
    for x,y in train_dl:
        x,y=x.to(device),y.to(device); opt.zero_grad(); lo=model(x); L=bce(lo,y)+dloss(lo,y); L.backward(); opt.step()
        tot+=L.item()*x.size(0); dsc+=dice(lo,y)*x.size(0); n+=x.size(0)
    sch.step(); model.eval(); vd=vn=0
    with torch.no_grad():
        for x,y in val_dl: x,y=x.to(device),y.to(device); vd+=dice(model(x),y)*x.size(0); vn+=x.size(0)
    v=vd/max(vn,1)
    if v>=best: best=v; best_state={k:val.detach().cpu().clone() for k,val in model.state_dict().items()}
    print(f"epoch {ep:3d}/{EPOCHS}  loss {tot/n:.4f}  train-dice {dsc/n:.3f}  VAL-dice {v:.3f}  best {best:.3f}  ({time.time()-t0:.0f}s)",flush=True)
if best_state: model.load_state_dict(best_state)
CKPT=f"{ROOT}/models/best.pt"
torch.save({"state_dict":model.state_dict(),"img_h":IMG_H,"img_w":IMG_W,"thresh":THRESH,
            "mean":IM_MEAN,"std":IM_STD,"val_dice":best}, CKPT)
print(f"\nBEST VAL DICE: {best:.3f}",flush=True)
print("saved checkpoint:",CKPT,flush=True)

# save val panel
E=etf(); model.eval()
fig,ax=plt.subplots(len(val),3,figsize=(12,4*len(val)))
if len(val)==1: ax=ax[None,:]
for i,(p,m) in enumerate(val):
    o=E(image=g3(p),mask=m)
    with torch.no_grad(): pr=(torch.sigmoid(model(o["image"].unsqueeze(0).to(device)))[0,0]>THRESH).cpu().numpy()
    im=o["image"].numpy().transpose(1,2,0); im=(im*IM_STD+IM_MEAN).clip(0,1); gt=o["mask"].numpy()
    ax[i,0].imshow(im);ax[i,0].set_title(os.path.basename(p)[:20]);ax[i,0].axis("off")
    g=im.copy();g[gt>0]=[0,1,0];ax[i,1].imshow(.6*im+.4*g);ax[i,1].set_title("ground truth");ax[i,1].axis("off")
    r=im.copy();r[pr>0]=[1,0,0];ax[i,2].imshow(.6*im+.4*r);ax[i,2].set_title("prediction");ax[i,2].axis("off")
plt.tight_layout(); plt.savefig(f"{PRED_OUT}/val_panel.png",dpi=90); plt.close()

# eyeball test
used={p for p,_ in ann}; pool=[p for p in glob.glob(f"{IMGDIR}/*_raw.png") if p not in used]
random.Random(SEED).shuffle(pool); test=pool[:12]
cols=3; rows=math.ceil(len(test)/cols); fig,ax=plt.subplots(rows,cols,figsize=(4*cols,5*rows)); ax=np.array(ax).reshape(-1)
for k,p in enumerate(test):
    o=E(image=g3(p))
    with torch.no_grad(): pr=(torch.sigmoid(model(o["image"].unsqueeze(0).to(device)))[0,0]>THRESH).cpu().numpy()
    im=o["image"].numpy().transpose(1,2,0); im=(im*IM_STD+IM_MEAN).clip(0,1); ov=im.copy(); ov[pr]=[1,0,0]; vis=.6*im+.4*ov
    ax[k].imshow(vis); ax[k].set_title(os.path.basename(p)[:22],fontsize=8); ax[k].axis("off")
    cv2.imwrite(f"{PRED_OUT}/{os.path.basename(p).replace('_raw.png','_pred.png')}",(vis[...,::-1]*255).astype(np.uint8))
for k in range(len(test),len(ax)): ax[k].axis("off")
plt.tight_layout(); plt.savefig(f"{PRED_OUT}/eyeball.png",dpi=90); plt.close()
print("saved val_panel.png and eyeball.png to",PRED_OUT,flush=True)
print("DONE",flush=True)
