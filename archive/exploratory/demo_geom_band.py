import cv2, numpy as np, torch, glob
import segmentation_models_pytorch as smp
ROOT="/home/ubuntu/TFM/seal-inspection"; MODEL=f"{ROOT}/models/best_lite_multiprod.pt"
IMG=384; MARGIN=40; MEAN=np.array((.485,.456,.406),np.float32); STD=np.array((.229,.224,.225),np.float32)
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
    m=cv2.morphologyEx((d>20).astype(np.uint8)*255,cv2.MORPH_OPEN,np.ones((11,11),np.uint8)); m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((41,41),np.uint8))
    c=[c for c in cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0] if cv2.contourArea(c)>h*w*0.02]
    if not c: return 0,0,w,h
    x,y,bw,bh=cv2.boundingRect(max(c,key=cv2.contourArea)); return max(0,x-MARGIN),max(0,y-MARGIN),min(w,x+bw+MARGIN),min(h,y+bh+MARGIN)
ck=torch.load(MODEL,map_location="cpu",weights_only=False); m=smp.Unet(ck["encoder"],encoder_weights=None,in_channels=3,classes=1); m.load_state_dict(ck["state_dict"]); m.eval()
g=cv2.imread(glob.glob(f"{ROOT}/data/images/*/seal_1302_1780665903828_raw.png")[0],0)
x0,y0,x1,y1=pack_bbox(g); crop=g[y0:y1,x0:x1]; h,w=crop.shape
im=cv2.resize(np.stack([norm(crop)]*3,-1),(IMG,IMG)).astype(np.float32)/255.0; x=((im-MEAN)/STD).transpose(2,0,1)[None]
with torch.no_grad(): p=torch.sigmoid(m(torch.from_numpy(x)))[0,0].numpy()
mask=(cv2.resize(p,(w,h))>ck.get("thresh",0.5)).astype(np.uint8)*255

# GEOMETRIC BAND: smooth outer (convex hull) + fixed inward offset = avg ring width
cnts,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
allp=np.vstack([c.reshape(-1,2) for c in cnts]); hull=cv2.convexHull(allp)
outer=np.zeros_like(mask); cv2.drawContours(outer,[hull],-1,255,-1)
W=max(20,int(mask.sum()/255/ (cv2.arcLength(hull,True)+1e-6)))   # avg ring width = area/perimeter
k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*W+1,2*W+1)); inner=cv2.erode(outer,k)
band=cv2.subtract(outer,inner)
print("avg band width W=",W)
def ov(mk,col=(0,0,185)):
    v=cv2.cvtColor(norm(crop),cv2.COLOR_GRAY2BGR); v[mk>0]=np.clip(0.4*v[mk>0]+np.array(col),0,255).astype(np.uint8)
    cnts,_=cv2.findContours(mk,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(v,cnts,-1,(0,0,255),2); return v
cells=[("input",cv2.cvtColor(norm(crop),cv2.COLOR_GRAY2BGR)),("RAW model mask (wavy divert)",ov(mask)),("GEOMETRIC band (outer+offset)",ov(band,(0,150,0)))]
out=[]
for t,v in cells:
    v=cv2.resize(v,(440,int(v.shape[0]*440/v.shape[1]))); lab=np.full((28,440,3),40,np.uint8); cv2.putText(lab,t,(6,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1); out.append(np.vstack([lab,v]))
H=max(c.shape[0] for c in out); out=[np.vstack([c,np.full((H-c.shape[0],c.shape[1],3),255,np.uint8)]) for c in out]
cv2.imwrite(f"{ROOT}/outputs/geom_band_seal1302.png",np.hstack(out)); print("wrote outputs/geom_band_seal1302.png")
