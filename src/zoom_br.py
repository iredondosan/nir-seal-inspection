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
    mm=cv2.morphologyEx((d>20).astype(np.uint8)*255,cv2.MORPH_OPEN,np.ones((11,11),np.uint8)); mm=cv2.morphologyEx(mm,cv2.MORPH_CLOSE,np.ones((41,41),np.uint8))
    c=[c for c in cv2.findContours(mm,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0] if cv2.contourArea(c)>h*w*0.02]
    if not c: return 0,0,w,h
    x,y,bw,bh=cv2.boundingRect(max(c,key=cv2.contourArea)); return max(0,x-MARGIN),max(0,y-MARGIN),min(w,x+bw+MARGIN),min(h,y+bh+MARGIN)
ck=torch.load(MODEL,map_location="cpu",weights_only=False); m=smp.Unet(ck["encoder"],encoder_weights=None,in_channels=3,classes=1); m.load_state_dict(ck["state_dict"]); m.eval()
g=cv2.imread(glob.glob(f"{ROOT}/data/images/*/seal_1998_1780688689500_raw.png")[0],0)
x0,y0,x1,y1=pack_bbox(g); crop=g[y0:y1,x0:x1]; h,w=crop.shape
im=cv2.resize(np.stack([norm(crop)]*3,-1),(IMG,IMG)).astype(np.float32)/255.0; x=((im-MEAN)/STD).transpose(2,0,1)[None]
with torch.no_grad(): p=torch.sigmoid(m(torch.from_numpy(x)))[0,0].numpy()
mask=(cv2.resize(p,(w,h))>ck.get("thresh",0.5)).astype(np.uint8)*255
v=cv2.cvtColor(norm(crop),cv2.COLOR_GRAY2BGR); v[mask>0]=np.clip(0.4*v[mask>0]+np.array([0,0,185]),0,255).astype(np.uint8)
cnts,_=cv2.findContours(mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(v,cnts,-1,(0,0,255),2)
# full + bottom-right zoom
br=v[int(h*0.55):,int(w*0.5):]
full=cv2.resize(v,(420,int(v.shape[0]*420/v.shape[1]))); brz=cv2.resize(br,(420,int(br.shape[0]*420/br.shape[1])))
def lab(im,t): l=np.full((26,im.shape[1],3),40,np.uint8); cv2.putText(l,t,(6,18),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1); return np.vstack([l,im])
full=lab(full,"new model: full seal mask"); brz=lab(brz,"ZOOM bottom-right (dark label)")
Hh=max(full.shape[0],brz.shape[0]); 
full=np.vstack([full,np.full((Hh-full.shape[0],full.shape[1],3),255,np.uint8)]); brz=np.vstack([brz,np.full((Hh-brz.shape[0],brz.shape[1],3),255,np.uint8)])
cv2.imwrite(f"{ROOT}/outputs/seal1998_br.png",np.hstack([full,brz])); print("wrote outputs/seal1998_br.png")
