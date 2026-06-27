import cv2, numpy as np, torch, glob, xml.etree.ElementTree as ET
import segmentation_models_pytorch as smp
ROOT="/home/ubuntu/TFM/seal-inspection"; MODEL=f"{ROOT}/models/best_lite_multiprod.pt"
XML=f"{ROOT}/data/annotations/annotations.xml"; NAME="seal_1998_1780688689500_raw.png"
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
def pp(s): return np.array([[float(a) for a in p.split(',')] for p in s.split(';')],np.float32)
g=cv2.imread(glob.glob(f"{ROOT}/data/images/*/{NAME}")[0],0)
# GT mask
node=[im for im in ET.parse(XML).getroot().findall("image") if im.get("name")==NAME][0]
pl=sorted([pp(pg.get("points")) for pg in node.findall("polygon")],key=lambda q:cv2.contourArea(q.astype(np.float32)),reverse=True)
H,W=g.shape; gt=np.zeros((H,W),np.uint8); cv2.fillPoly(gt,[pl[0].astype(np.int32)],1); cv2.fillPoly(gt,[pl[1].astype(np.int32)],0)
x0,y0,x1,y1=pack_bbox(g); crop=g[y0:y1,x0:x1]; gt=gt[y0:y1,x0:x1]; h,w=crop.shape
# pred
ck=torch.load(MODEL,map_location="cpu",weights_only=False); m=smp.Unet(ck["encoder"],encoder_weights=None,in_channels=3,classes=1); m.load_state_dict(ck["state_dict"]); m.eval()
im=cv2.resize(np.stack([norm(crop)]*3,-1),(IMG,IMG)).astype(np.float32)/255.0; x=((im-MEAN)/STD).transpose(2,0,1)[None]
with torch.no_grad(): p=torch.sigmoid(m(torch.from_numpy(x)))[0,0].numpy()
pred=(cv2.resize(p,(w,h))>ck.get("thresh",0.5)).astype(np.uint8)
v=cv2.cvtColor(norm(crop),cv2.COLOR_GRAY2BGR)
gc,_=cv2.findContours(gt,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(v,gc,-1,(0,220,0),2)         # GT green
pc,_=cv2.findContours(pred,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(v,pc,-1,(0,0,235),2)       # pred red
br=v[int(h*0.5):,int(w*0.5):]
def lab(im,t): l=np.full((26,im.shape[1],3),40,np.uint8); cv2.putText(l,t,(6,18),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1); return np.vstack([l,im])
full=lab(cv2.resize(v,(380,int(v.shape[0]*380/v.shape[1]))),"GT green vs PRED red")
brz=lab(cv2.resize(br,(380,int(br.shape[0]*380/br.shape[1]))),"ZOOM bottom-right")
Hh=max(full.shape[0],brz.shape[0]); full=np.vstack([full,np.full((Hh-full.shape[0],full.shape[1],3),255,np.uint8)]); brz=np.vstack([brz,np.full((Hh-brz.shape[0],brz.shape[1],3),255,np.uint8)])
cv2.imwrite(f"{ROOT}/outputs/gt_vs_pred_1998.png",np.hstack([full,brz])); print("wrote outputs/gt_vs_pred_1998.png")
