#!/usr/bin/env python3
"""Generate CVAT 'sellado' pre-annotations using the LITE multiprod model (pack-crop->predict->map to full res).
Outputs outer+inner polygons in raw-image coords. Usage: --input <imgdir> --output <xml>"""
import os, glob, argparse, numpy as np, cv2, torch
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
def ring_polys(mask, band_px=85):
    m=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((15,15),np.uint8))
    cnts,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if not cnts: return None
    outer=max(cnts,key=cv2.contourArea); fill=np.zeros_like(m); cv2.drawContours(fill,[outer],-1,255,-1)
    hole=cv2.subtract(fill,m)
    hc=[c for c in cv2.findContours(hole,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0] if cv2.contourArea(c)>0.15*cv2.contourArea(outer)]
    if hc: inner=max(hc,key=cv2.contourArea)
    else:
        k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*band_px+1,2*band_px+1)); ic,_=cv2.findContours(cv2.erode(fill,k),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        if not ic: return None
        inner=max(ic,key=cv2.contourArea)
    return outer,inner
def simp(c,ow,oh, step=9.0, corner_deg=7.0, straight_every=16, win=2):
    # curvature-adaptive via WINDOWED turning angle (robust to rasterization noise):
    # dense points through the rounded corners, sparse on straight edges.
    P=c.reshape(-1,2).astype(np.float32)
    loop=np.r_[P,P[:1]]; d=np.r_[0,np.cumsum(np.hypot(*np.diff(loop,axis=0).T))]; L=d[-1]
    if L<10: return P
    n=max(40,int(L/step)); t=np.linspace(0,L,n,endpoint=False)
    Pr=np.stack([np.interp(t,d,loop[:,0]),np.interp(t,d,loop[:,1])],1)
    v1=Pr-np.roll(Pr,win,0); v2=np.roll(Pr,-win,0)-Pr          # direction over +/- win points
    dth=np.abs((np.arctan2(v2[:,1],v2[:,0])-np.arctan2(v1[:,1],v1[:,0])+np.pi)%(2*np.pi)-np.pi)
    thr=np.deg2rad(corner_deg)
    keep=[i for i in range(n) if dth[i]>thr or i%straight_every==0]
    out=Pr[keep]
    out[:,0]=np.clip(out[:,0],0,ow-1); out[:,1]=np.clip(out[:,1],0,oh-1); return out
def pstr(p): return ";".join(f"{x:.2f},{y:.2f}" for x,y in p)

ap=argparse.ArgumentParser(); ap.add_argument("--input",required=True); ap.add_argument("--output",required=True); ap.add_argument("--model",default=MODEL); ap.add_argument("--step",type=float,default=9.0); a=ap.parse_args()
ck=torch.load(a.model,map_location="cpu",weights_only=False); THR=ck.get("thresh",0.5); IMG=ck.get("img",IMG); print(f"input res: {IMG}")
m=smp.Unet(ck["encoder"],encoder_weights=None,in_channels=3,classes=1); m.load_state_dict(ck["state_dict"]); m.eval()
dev="cuda" if torch.cuda.is_available() else "cpu"; m=m.to(dev)
files=(sorted(glob.glob(os.path.join(a.input,"*_raw.png"))) or sorted(glob.glob(os.path.join(a.input,"*.jpg"))) or sorted(glob.glob(os.path.join(a.input,"*.png")))); os.makedirs(os.path.dirname(a.output) or ".",exist_ok=True)
rows=[]; ok=skip=0
for i,p in enumerate(files):
    g=cv2.imread(p,cv2.IMREAD_GRAYSCALE)
    if g is None: skip+=1; continue
    oh,ow=g.shape; x0,y0,x1,y1=pack_bbox(g); crop=g[y0:y1,x0:x1]; ch,cwd=crop.shape
    im=cv2.resize(np.stack([norm(crop)]*3,-1),(IMG,IMG)).astype(np.float32)/255.0; x=((im-MEAN)/STD).transpose(2,0,1)[None]
    with torch.no_grad(): prob=torch.sigmoid(m(torch.from_numpy(x).to(dev)))[0,0].cpu().numpy()
    full=np.zeros((oh,ow),np.float32); full[y0:y1,x0:x1]=cv2.resize(prob,(cwd,ch)); mask=(full>THR).astype(np.uint8)*255
    r=ring_polys(mask); polys=""
    if r is not None and len(r[0])>=8 and len(r[1])>=8:
        for cnt in r: polys+=f'    <polygon label="sellado" source="auto" occluded="0" points="{pstr(simp(cnt,ow,oh,step=a.step))}" z_order="0"></polygon>\n'
        ok+=1
    else: skip+=1
    rows.append(f'  <image id="{i}" name="{os.path.basename(p)}" width="{ow}" height="{oh}">\n{polys}  </image>')
open(a.output,"w").write('<?xml version="1.0" encoding="utf-8"?>\n<annotations>\n  <version>1.1</version>\n'+"\n".join(rows)+"\n</annotations>\n")
print(f"{a.output}: {len(files)} images, {ok} with seal polygons, {skip} empty")
