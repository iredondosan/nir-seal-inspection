import cv2, numpy as np, torch, glob, xml.etree.ElementTree as ET
import segmentation_models_pytorch as smp
ROOT="/home/ubuntu/TFM/seal-inspection"; MODEL=f"{ROOT}/models/best_lite_multiprod.pt"
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
    m=cv2.morphologyEx((d>20).astype(np.uint8)*255,cv2.MORPH_OPEN,np.ones((11,11),np.uint8)); m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((41,41),np.uint8))
    c=[c for c in cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0] if cv2.contourArea(c)>h*w*0.02]
    if not c: return 0,0,w,h
    x,y,bw,bh=cv2.boundingRect(max(c,key=cv2.contourArea)); return max(0,x-MARGIN),max(0,y-MARGIN),min(w,x+bw+MARGIN),min(h,y+bh+MARGIN)
def pp(s): return np.array([[float(a) for a in p.split(',')] for p in s.split(';')],np.float32)
# clean prod3 sellado image + band
node=ET.parse(f"{ROOT}/data/annotations/prod3.xml").getroot().findall("image")[3]; nm=node.get("name")
g=cv2.imread(f"{ROOT}/data/images/prod3/{nm}",0)
pl=sorted([pp(pg.get("points")) for pg in node.findall("polygon")],key=lambda q:cv2.contourArea(q.astype(np.float32)),reverse=True)
H,W=g.shape; band=np.zeros((H,W),np.uint8); cv2.fillPoly(band,[pl[0].astype(np.int32)],1); cv2.fillPoly(band,[pl[1].astype(np.int32)],0)
x0,y0,x1,y1=pack_bbox(g); g=g[y0:y1,x0:x1]; band=band[y0:y1,x0:x1]; h,w=g.shape
# seal_1302 defect cut-out
dn=ET.parse(CONTAM_XML).getroot()
patch=al=None
for node in dn.findall("image"):
    if node.get("name")=="seal_1302_1780665903828_raw.png":
        gg=cv2.imread(glob.glob(f"{ROOT}/data/images/*/seal_1302_1780665903828_raw.png")[0],0)
        pts=pp(node.find("polygon").get("points")).astype(np.int32); bx,by,bw,bh=cv2.boundingRect(pts)
        patch=gg[by:by+bh,bx:bx+bw].copy(); al=np.zeros((bh,bw),np.uint8); cv2.fillPoly(al,[pts-[bx,by]],255); al=cv2.GaussianBlur(al,(0,0),2).astype(np.float32)/255.0
print("defect size:",patch.shape)
def paste_at(img,cx,cy,scale=1.0):
    out=img.copy().astype(np.float32); pw,ph=int(patch.shape[1]*scale),int(patch.shape[0]*scale)
    p=cv2.resize(patch,(pw,ph)).astype(np.float32); a=cv2.resize(al,(pw,ph))
    x0,y0=cx-pw//2,cy-ph//2; ix0,iy0=max(0,x0),max(0,y0); ix1,iy1=min(w,x0+pw),min(h,y0+ph)
    px0,py0=ix0-x0,iy0-y0; px1,py1=px0+(ix1-ix0),py0+(iy1-iy0)
    aa=a[py0:py1,px0:px1]; pp_=p[py0:py1,px0:px1]
    out[iy0:iy1,ix0:ix1]=out[iy0:iy1,ix0:ix1]*(1-aa)+pp_*aa; return np.clip(out,0,255).astype(np.uint8)
# positions: snap bbox corners + edge mids to nearest band pixel
ys,xs=np.where(band>0); bx,by,bw,bh=cv2.boundingRect((band>0).astype(np.uint8))
cands={"TL corner":(bx,by),"TR corner":(bx+bw,by),"BL corner":(bx,by+bh),"BR corner":(bx+bw,by+bh),
       "top edge":(bx+bw//2,by),"left edge":(bx,by+bh//2)}
pts=np.stack([xs,ys],1)
def snap(t): d=((pts-np.array(t))**2).sum(1); k=d.argmin(); return int(pts[k,0]),int(pts[k,1])
ck=torch.load(MODEL,map_location="cpu",weights_only=False); m=smp.Unet(ck["encoder"],encoder_weights=None,in_channels=3,classes=1); m.load_state_dict(ck["state_dict"]); m.eval()
def predict(crop):
    im=cv2.resize(np.stack([norm(crop)]*3,-1),(IMG,IMG)).astype(np.float32)/255.0; x=((im-MEAN)/STD).transpose(2,0,1)[None]
    with torch.no_grad(): p=torch.sigmoid(m(torch.from_numpy(x)))[0,0].numpy()
    return (cv2.resize(p,(w,h))>ck.get("thresh",0.5)).astype(np.uint8)*255
cells=[]
for lbl,t in cands.items():
    cx,cy=snap(t); cont=paste_at(g,cx,cy,1.2); mask=predict(cont)
    v=cv2.cvtColor(norm(cont),cv2.COLOR_GRAY2BGR); v[mask>0]=np.clip(0.4*v[mask>0]+np.array([0,0,185]),0,255).astype(np.uint8)
    cnts,_=cv2.findContours(mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(v,cnts,-1,(0,0,255),2)
    R=130; zx0,zy0=max(0,cx-R),max(0,cy-R); zx1,zy1=min(w,cx+R),min(h,cy+R)
    z=v[zy0:zy1,zx0:zx1]; z=cv2.resize(z,(300,int(z.shape[0]*300/z.shape[1])))
    lab=np.full((26,z.shape[1],3),45,np.uint8); cv2.putText(lab,lbl,(5,18),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
    cells.append(np.vstack([lab,z]))
hh=max(c.shape[0] for c in cells); ww=max(c.shape[1] for c in cells)
cells=[np.vstack([cv2.copyMakeBorder(c,0,hh-c.shape[0],0,ww-c.shape[1],cv2.BORDER_CONSTANT,value=(255,255,255))]) for c in cells]
r1=np.hstack(cells[:3]); r2=np.hstack(cells[3:]); cv2.imwrite(f"{ROOT}/outputs/position_test.png",np.vstack([r1,r2])); print("wrote outputs/position_test.png")
