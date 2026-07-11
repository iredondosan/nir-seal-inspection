import sys, cv2, numpy as np, torch, os
import segmentation_models_pytorch as smp
from seal_inspection.paths import ROOT
OLD=f"{ROOT}/models/best_lite_multiprod_prev_backup.pt"; NEW=f"{ROOT}/models/best_lite_multiprod.pt"
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
    x,y,bw,bh=cv2.boundingRect(max(c,key=cv2.contourArea)); return max(0,x-MARGIN),max(0,y-MARGIN),min(w,x+bw+MARGIN),min(h,y+bh+MARGIN)
def predict(ckpt,crop):
    c=torch.load(ckpt,map_location="cpu",weights_only=False)
    m=smp.Unet(c["encoder"],encoder_weights=None,in_channels=3,classes=1); m.load_state_dict(c["state_dict"]); m.eval()
    hh,ww=crop.shape; im=cv2.resize(np.stack([norm(crop)]*3,-1),(IMG,IMG)).astype(np.float32)/255.0
    x=((im-MEAN)/STD).transpose(2,0,1)[None]
    with torch.no_grad(): p=torch.sigmoid(m(torch.from_numpy(x)))[0,0].numpy()
    return (cv2.resize(p,(ww,hh))>c.get("thresh",0.5)).astype(np.uint8)*255
def ov(crop,mask):
    v=cv2.cvtColor(norm(crop),cv2.COLOR_GRAY2BGR); v[mask>0]=np.clip(0.4*v[mask>0]+np.array([0,0,185]),0,255).astype(np.uint8)
    cnts,_=cv2.findContours(mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(v,cnts,-1,(0,0,255),2); return v

path=sys.argv[1]; g=cv2.imread(path,0); x0,y0,x1,y1=pack_bbox(g); crop=g[y0:y1,x0:x1]; h,w=crop.shape
old=predict(OLD,crop); new=predict(NEW,crop)
inp=cv2.cvtColor(norm(crop),cv2.COLOR_GRAY2BGR); oldv=ov(crop,old); newv=ov(crop,new)
def row(cells,W=460):
    out=[]
    for t,im in cells:
        im=cv2.resize(im,(W,int(im.shape[0]*W/im.shape[1]))); lab=np.full((28,W,3),40,np.uint8)
        cv2.putText(lab,t,(6,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1); out.append(np.vstack([lab,im]))
    H=max(c.shape[0] for c in out); out=[np.vstack([c,np.full((H-c.shape[0],c.shape[1],3),255,np.uint8)]) for c in out]
    return np.hstack(out)
full=row([("input (contaminated)",inp),("OLD model (no aug)",oldv),("NEW model (contam aug)",newv)])
# zoom on region where masks differ
diff=cv2.bitwise_xor(old,new); ys,xs=np.where(diff>0)
if len(xs)>30:
    pad=60; zx0,zy0=max(0,xs.min()-pad),max(0,ys.min()-pad); zx1,zy1=min(w,xs.max()+pad),min(h,ys.max()+pad)
    zin=inp[zy0:zy1,zx0:zx1]; zo=oldv[zy0:zy1,zx0:zx1]; zn=newv[zy0:zy1,zx0:zx1]
    zoom=row([("ZOOM input",zin),("ZOOM OLD (diverts?)",zo),("ZOOM NEW",zn)],460)
    sep=np.full((6,full.shape[1],3),255,np.uint8)
    canvas=np.vstack([full,sep,zoom])
else:
    canvas=full; print("NOTE: old and new masks barely differ (diff px=%d)"%len(xs))
stem=os.path.basename(path).replace(".png","")
cv2.imwrite(f"{ROOT}/outputs/real_ab_{stem}.png",canvas); print("diff px=",len(xs),"wrote outputs/real_ab_"+stem+".png")
