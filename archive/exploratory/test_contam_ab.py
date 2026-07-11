import cv2, numpy as np, torch, random, xml.etree.ElementTree as ET, os
import segmentation_models_pytorch as smp
random.seed(11); np.random.seed(11)
from seal_inspection.paths import ROOT
OLD=f"{ROOT}/models/best_lite_multiprod_noaug_backup.pt"
NEW=f"{ROOT}/models/best_lite_multiprod.pt"
XML=f"{ROOT}/data/annotations/annotations.xml"; IMGDIR=f"{ROOT}/data/images/prod2"
IMG=384; MARGIN=40; MEAN=np.array((.485,.456,.406),np.float32); STD=np.array((.229,.224,.225),np.float32)
def norm(g):
    lo,hi=np.percentile(g,[1,99.5]); hi=max(hi,lo+1)
    return np.clip((g.astype(np.float32)-lo)/(hi-lo)*255,0,255).astype(np.uint8)
def cc(N):
    cm=np.median(N,0).astype(np.float32); on=np.where(cm>cm.max()*0.5)[0]; cL,cR=on.min(),on.max(); g=np.gradient(cm)
    return int(np.argmax(g[max(0,cL-60):cL+60])+max(0,cL-60)),int(np.argmin(g[cR-60:cR+60])+(cR-60))
def pack_bbox(g):
    N=norm(g); h,w=N.shape
    try: cL,cR=cc(N)
    except Exception: cL,cR=0,w
    top=np.median(N[20:240,:],0); bot=np.median(N[h-240:h-20,:],0); ref=np.maximum(top,bot)
    d=np.clip(np.tile(ref,(h,1))-N.astype(np.float32),0,255); d[:,:cL]=0; d[:,cR:]=0
    m=cv2.morphologyEx((d>20).astype(np.uint8)*255,cv2.MORPH_OPEN,np.ones((11,11),np.uint8))
    m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((41,41),np.uint8))
    c=[c for c in cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0] if cv2.contourArea(c)>h*w*0.02]
    if not c: return 0,0,w,h
    x,y,bw,bh=cv2.boundingRect(max(c,key=cv2.contourArea))
    return max(0,x-MARGIN),max(0,y-MARGIN),min(w,x+bw+MARGIN),min(h,y+bh+MARGIN)
def pp(s): return np.array([[float(a) for a in p.split(',')] for p in s.split(';')],np.float32)

node=ET.parse(XML).getroot().findall("image")[2]; name=node.get("name")
g=cv2.imread(f"{IMGDIR}/{name}",0)
pl=sorted([pp(pg.get("points")) for pg in node.findall("polygon")],key=lambda p:cv2.contourArea(p.astype(np.float32)),reverse=True)
H,W=g.shape; band=np.zeros((H,W),np.uint8); cv2.fillPoly(band,[pl[0].astype(np.int32)],1); cv2.fillPoly(band,[pl[1].astype(np.int32)],0)
x0,y0,x1,y1=pack_bbox(g); g=g[y0:y1,x0:x1]; band=band[y0:y1,x0:x1]; h,w=g.shape

# add a couple of clear dark contaminants ON the band (spanning radially toward product)
def contaminate(img,band):
    out=img.astype(np.float32); ys,xs=np.where(band>0); bandd=cv2.dilate(band,np.ones((11,11),np.uint8))
    for _ in range(2):
        k=random.randrange(len(xs)); cx,cy=int(xs[k]),int(ys[k]); blob=np.zeros((h,w),np.float32)
        for _ in range(5):
            cv2.ellipse(blob,(cx+random.randint(-25,25),cy+random.randint(-25,25)),(random.randint(25,55),random.randint(25,55)),random.randint(0,180),0,360,1.0,-1)
        blob=cv2.GaussianBlur(blob,(0,0),7); blob=(blob/blob.max())*(bandd>0)
        a=blob*0.92; tex=15+np.random.randn(h,w)*8
        out=out*(1-a)+tex*a
    return np.clip(out,0,255).astype(np.uint8)
cont=contaminate(g,band)

def predict(ckpt,img):
    c=torch.load(ckpt,map_location="cpu",weights_only=False)
    m=smp.Unet(c["encoder"],encoder_weights=None,in_channels=3,classes=1); m.load_state_dict(c["state_dict"]); m.eval()
    im=cv2.resize(np.stack([norm(img)]*3,-1),(IMG,IMG)).astype(np.float32)/255.0
    x=((im-MEAN)/STD).transpose(2,0,1)[None]
    with torch.no_grad(): p=torch.sigmoid(m(torch.from_numpy(x)))[0,0].numpy()
    return (cv2.resize(p,(w,h))>c.get("thresh",0.5)).astype(np.uint8)*255
def ov(img,mask,col):
    v=cv2.cvtColor(norm(img),cv2.COLOR_GRAY2BGR); v[mask>0]=np.clip(0.45*v[mask>0]+np.array(col),0,255).astype(np.uint8)
    cnts,_=cv2.findContours(mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(v,cnts,-1,col,2); return v

old=predict(OLD,cont); new=predict(NEW,cont)
gtv=cv2.cvtColor(norm(cont),cv2.COLOR_GRAY2BGR)
cnts,_=cv2.findContours(band,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(gtv,cnts,-1,(0,255,0),2)
cells=[("contaminated input (GT band green)",gtv),("OLD model (no aug)",ov(cont,old,(0,0,180))),("NEW model (contam aug)",ov(cont,new,(0,0,180)))]
out=[]
for t,im in cells:
    im=cv2.resize(im,(440,int(im.shape[0]*440/im.shape[1])))
    lab=np.full((30,440,3),40,np.uint8); cv2.putText(lab,t,(6,21),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
    out.append(np.vstack([lab,im]))
Hc=max(c.shape[0] for c in out); out=[np.vstack([c,np.full((Hc-c.shape[0],c.shape[1],3),255,np.uint8)]) for c in out]
cv2.imwrite(f"{ROOT}/outputs/contam_ab.png",np.hstack(out)); print("wrote outputs/contam_ab.png on",name)
