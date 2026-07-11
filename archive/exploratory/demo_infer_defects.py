import cv2, numpy as np, torch, glob, random, xml.etree.ElementTree as ET
import segmentation_models_pytorch as smp
random.seed(2); np.random.seed(2)
from seal_inspection.paths import ROOT; MODEL=f"{ROOT}/models/best_lite_multiprod.pt"
CONTAM_XML=f"{ROOT}/data/annotations/contaminants.xml"; IMG=384; MARGIN=40
MEAN=np.array((.485,.456,.406),np.float32); STD=np.array((.229,.224,.225),np.float32)
def norm(g):
    lo,hi=np.percentile(g,[1,99.5]); hi=max(hi,lo+1); return np.clip((g.astype(np.float32)-lo)/(hi-lo)*255,0,255).astype(np.uint8)
def cc(N):
    cm=np.median(N,0).astype(np.float32); on=np.where(cm>cm.max()*0.5)[0]; cL,cR=on.min(),on.max(); g=np.gradient(cm)
    return int(np.argmax(g[max(0,cL-60):cL+60])+max(0,cL-60)),int(np.argmin(g[cR-60:cR+60])+(cR-60))
def pack_bbox(g):
    N=norm(g); h,w=N.shape
    try: cL,cR=cc(N)
    except Exception: cL,cR=0,w
    top=np.median(N[20:240,:],0); bot=np.median(N[h-240:h-20,:],0); ref=np.maximum(top,bot)
    d=np.clip(np.tile(ref,(h,1))-N.astype(np.float32),0,255); d[:,:cL]=0; d[:,cR:]=0
    mm=cv2.morphologyEx((d>20).astype(np.uint8)*255,cv2.MORPH_OPEN,np.ones((11,11),np.uint8)); mm=cv2.morphologyEx(mm,cv2.MORPH_CLOSE,np.ones((41,41),np.uint8))
    c=[c for c in cv2.findContours(mm,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0] if cv2.contourArea(c)>h*w*0.02]
    if not c: return 0,0,w,h
    x,y,bw,bh=cv2.boundingRect(max(c,key=cv2.contourArea)); return max(0,x-MARGIN),max(0,y-MARGIN),min(w,x+bw+MARGIN),min(h,y+bh+MARGIN)
def pp(s): return np.array([[float(a) for a in p.split(',')] for p in s.split(';')],np.float32)
def load_contaminants(xml):
    insts=[]
    for node in ET.parse(xml).getroot().findall("image"):
        hits=glob.glob(f"{ROOT}/data/images/*/{node.get('name')}")
        if not hits: continue
        gg=cv2.imread(hits[0],0)
        for pg in node.findall("polygon"):
            pts=pp(pg.get("points")).astype(np.int32); x,y,bw,bh=cv2.boundingRect(pts)
            if bw<4 or bh<4: continue
            patch=gg[y:y+bh,x:x+bw].copy(); al=np.zeros((bh,bw),np.uint8); cv2.fillPoly(al,[pts-[x,y]],255)
            insts.append((patch,cv2.GaussianBlur(al,(0,0),2).astype(np.float32)/255.0))
    return insts
def paste(img3,band,insts,n):
    h,w=band.shape; ys,xs=np.where(band>0); out=img3.copy(); dmap=np.zeros((h,w),np.float32)
    for _ in range(n):
        patch,al=random.choice(insts); s=random.uniform(0.2,0.9) if random.random()<0.7 else random.uniform(0.9,1.6)
        pw,ph=max(3,int(patch.shape[1]*s)),max(3,int(patch.shape[0]*s)); p=cv2.resize(patch,(pw,ph)).astype(np.float32); a=cv2.resize(al,(pw,ph))
        M=cv2.getRotationMatrix2D((pw/2,ph/2),random.uniform(0,360),1.0); p=cv2.warpAffine(p,M,(pw,ph)); a=cv2.warpAffine(a,M,(pw,ph))
        k=random.randrange(len(xs)); cx,cy=int(xs[k]),int(ys[k]); x0,y0=cx-pw//2,cy-ph//2
        ix0,iy0=max(0,x0),max(0,y0); ix1,iy1=min(w,x0+pw),min(h,y0+ph)
        if ix1<=ix0 or iy1<=iy0: continue
        px0,py0=ix0-x0,iy0-y0; px1,py1=px0+(ix1-ix0),py0+(iy1-iy0)
        aa=a[py0:py1,px0:px1]; pp_=np.clip(p[py0:py1,px0:px1]*random.uniform(0.7,1.2),0,255)
        reg=out[iy0:iy1,ix0:ix1].astype(np.float32)
        out[iy0:iy1,ix0:ix1]=np.clip(reg*(1-aa[...,None])+pp_[...,None]*aa[...,None],0,255).astype(np.uint8)
        dmap[iy0:iy1,ix0:ix1]=np.maximum(dmap[iy0:iy1,ix0:ix1],aa)
    return out,dmap
ck=torch.load(MODEL,map_location="cpu",weights_only=False); m=smp.Unet(ck["encoder"],encoder_weights=None,in_channels=3,classes=1); m.load_state_dict(ck["state_dict"]); m.eval()
def predict(crop):
    h,w=crop.shape; im=cv2.resize(np.stack([norm(crop)]*3,-1),(IMG,IMG)).astype(np.float32)/255.0; x=((im-MEAN)/STD).transpose(2,0,1)[None]
    with torch.no_grad(): p=torch.sigmoid(m(torch.from_numpy(x)))[0,0].numpy()
    return (cv2.resize(p,(w,h))>ck.get("thresh",0.5)).astype(np.uint8)*255
insts=load_contaminants(CONTAM_XML); print("instances:",len(insts))
SAMP=[("prod1",f"{ROOT}/data/annotations/prod1.xml"),("prod2",f"{ROOT}/data/annotations/annotations.xml"),("prod3",f"{ROOT}/data/annotations/prod3.xml"),("prod4",f"{ROOT}/data/annotations/prod4.xml"),("prod5",f"{ROOT}/data/annotations/prod5.xml")]
rows=[]
for prod,xmlp in SAMP:
    node=ET.parse(xmlp).getroot().findall("image")[1]; nm=node.get("name")
    g=cv2.imread(f"{ROOT}/data/images/{prod}/{nm}",0)
    pl=sorted([pp(pg.get("points")) for pg in node.findall("polygon")],key=lambda q:cv2.contourArea(q.astype(np.float32)),reverse=True)
    H,W=g.shape; band=np.zeros((H,W),np.uint8); cv2.fillPoly(band,[pl[0].astype(np.int32)],1); cv2.fillPoly(band,[pl[1].astype(np.int32)],0)
    x0,y0,x1,y1=pack_bbox(g); crop=norm(g[y0:y1,x0:x1]); band=band[y0:y1,x0:x1]
    cont,dmap=paste(np.stack([crop]*3,-1),band,insts,random.randint(3,5))
    pred=predict(cv2.cvtColor(cont,cv2.COLOR_BGR2GRAY) if cont.ndim==3 else cont)
    inp=cont.copy()
    dc,_=cv2.findContours((dmap>0.3).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(inp,dc,-1,(0,255,255),2)
    pv=cont.copy(); pv[pred>0]=np.clip(0.4*pv[pred>0]+np.array([0,0,185]),0,255).astype(np.uint8)
    pc,_=cv2.findContours(pred,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(pv,pc,-1,(0,0,255),2); cv2.drawContours(pv,dc,-1,(0,255,255),2)
    cells=[]
    for t,v in [(f"{prod}: defects applied (yellow)",inp),(f"{prod}: model prediction (red)",pv)]:
        v=cv2.resize(v,(360,int(v.shape[0]*360/v.shape[1]))); lab=np.full((26,360,3),45,np.uint8); cv2.putText(lab,t,(5,18),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,255),1); cells.append(np.vstack([lab,v]))
    Hc=max(c.shape[0] for c in cells); cells=[np.vstack([c,np.full((Hc-c.shape[0],c.shape[1],3),255,np.uint8)]) for c in cells]
    rows.append(np.hstack(cells))
Wm=max(r.shape[1] for r in rows); rows=[np.hstack([r,np.full((r.shape[0],Wm-r.shape[1],3),255,np.uint8)]) for r in rows]
cv2.imwrite(f"{ROOT}/outputs/infer_defects.png",np.vstack(rows)); print("wrote outputs/infer_defects.png")
