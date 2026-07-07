#!/usr/bin/env python3
"""FULL end-to-end deployment pipeline on UNLABELED packs:
 raw image -> [seal model] predict seal ring -> unroll -> [defect model] -> defects
 Output composite: pack crop + predicted seal mask (cyan) + red circles on detected defects; unrolled strip below.
Usage: .venv/bin/python src/defect_pipeline.py --input data/images/prod2 [--names a.png,b.png] [--sample 8]"""
import os, glob, argparse, random
import numpy as np, cv2, torch
import segmentation_models_pytorch as smp
ROOT="/home/ubuntu/TFM/seal-inspection"
SEAL=f"{ROOT}/models/best_lite_reviewed_1280.pt"; DEFECT=f"{ROOT}/models/defect_strip.pt"
OUT=f"{ROOT}/outputs/defect_pipeline"; MARGIN=40
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
def ring_contours(mask, band_px=90):
    m=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((9,9),np.uint8)); m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((35,35),np.uint8))
    cnts,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if not cnts: return None,None
    outer=max(cnts,key=cv2.contourArea)  # raw contour follows wave (was convexHull)
    fill=np.zeros_like(m); cv2.drawContours(fill,[outer],-1,255,-1); hole=cv2.subtract(fill,m)
    c2=[c for c in cv2.findContours(hole,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0] if cv2.contourArea(c)>0.2*cv2.contourArea(outer)]
    if c2: inner=max(c2,key=cv2.contourArea)
    else:
        k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*band_px+1,2*band_px+1)); ic,_=cv2.findContours(cv2.erode(fill,k),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        if not ic: return None,None
        inner=max(ic,key=cv2.contourArea)
    return outer.reshape(-1,2).astype(np.float32), inner.reshape(-1,2).astype(np.float32)
def resample(poly,n):
    p=np.r_[poly,poly[:1]]; d=np.r_[0,np.cumsum(np.hypot(*np.diff(p,axis=0).T))]
    t=np.linspace(0,d[-1],n,endpoint=False); return np.stack([np.interp(t,d,p[:,0]),np.interp(t,d,p[:,1])],1)
def csmooth(a,k=15): ker=np.ones(k)/k; return np.convolve(np.r_[a[-k:],a,a[:k]],ker,"same")[k:-k]
def ccw(p): return cv2.contourArea(p.astype(np.float32),oriented=True)>0
def unroll_maps(outer,inner,Hs,Ws):
    O=resample(outer,Ws); I=resample(inner,Ws)
    if ccw(O)!=ccw(I): I=I[::-1]
    j=int(np.argmin(np.hypot(I[:,0]-O[0,0],I[:,1]-O[0,1]))); I=np.roll(I,-j,axis=0)
    for arr in (O,I): arr[:,0]=csmooth(arr[:,0]); arr[:,1]=csmooth(arr[:,1])
    a=np.linspace(0,1,Hs)[:,None]
    return (O[:,0][None,:]*(1-a)+I[:,0][None,:]*a).astype(np.float32),(O[:,1][None,:]*(1-a)+I[:,1][None,:]*a).astype(np.float32)
def banner(w,t,h=30):
    b=np.full((h,w,3),35,np.uint8); cv2.putText(b,t,(8,21),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1,cv2.LINE_AA); return b

ap=argparse.ArgumentParser(); ap.add_argument("--input",default=f"{ROOT}/data/images/prod2")
ap.add_argument("--names",default=""); ap.add_argument("--sample",type=int,default=8); a=ap.parse_args()
dev="cuda" if torch.cuda.is_available() else "cpu"
sk=torch.load(SEAL,map_location="cpu",weights_only=False); SIMG=sk["img"]; STHR=sk.get("thresh",0.5)
seal=smp.Unet(sk["encoder"],encoder_weights=None,in_channels=3,classes=1); seal.load_state_dict(sk["state_dict"]); seal=seal.to(dev).eval()
dk=torch.load(DEFECT,map_location="cpu",weights_only=False); HS,WS,DTHR=dk["HS"],dk["WS"],dk.get("thr",0.5)
defm=smp.Unet(dk["encoder"],encoder_weights=None,in_channels=3,classes=1); defm.load_state_dict(dk["state_dict"]); defm=defm.to(dev).eval()
os.makedirs(OUT,exist_ok=True)
names=[n for n in a.names.split(",") if n]
allf=sorted(glob.glob(os.path.join(a.input,"*_raw.png"))) or sorted(glob.glob(os.path.join(a.input,"*.jpg")))
files=[os.path.join(a.input,n) for n in names]+[f for f in random.Random(1).sample(allf,min(a.sample,len(allf))) if os.path.basename(f) not in names]
n=0
for p in files:
    g=cv2.imread(p,cv2.IMREAD_GRAYSCALE)
    if g is None: continue
    H,W=g.shape; x0,y0,x1,y1=pack_bbox(g); crop=g[y0:y1,x0:x1]; ch,cwd=crop.shape
    im=cv2.resize(np.stack([norm(crop)]*3,-1),(SIMG,SIMG)).astype(np.float32)/255.0; x=((im-MEAN)/STD).transpose(2,0,1)[None]
    with torch.no_grad(): sp=torch.sigmoid(seal(torch.from_numpy(x).to(dev)))[0,0].cpu().numpy()
    full=np.zeros((H,W),np.float32); full[y0:y1,x0:x1]=cv2.resize(sp,(cwd,ch)); smask=(full>STHR).astype(np.uint8)*255
    outer,inner=ring_contours(smask)
    base=os.path.splitext(os.path.basename(p))[0]
    if outer is None: print("no ring:",base); continue
    mapx,mapy=unroll_maps(outer,inner,HS,WS)
    strip=cv2.remap(norm(g),mapx,mapy,cv2.INTER_LINEAR,borderValue=0)
    xs=((np.stack([strip]*3,-1).astype(np.float32)/255.0-MEAN)/STD).transpose(2,0,1)[None]
    with torch.no_grad(): dp=torch.sigmoid(defm(torch.from_numpy(xs).to(dev)))[0,0].cpu().numpy()
    dmask=(dp>DTHR).astype(np.uint8)
    # pack panel (seal bbox)
    bx,by,bw,bh=cv2.boundingRect(outer.astype(np.int32)); pad=60
    px0,py0=max(0,bx-pad),max(0,by-pad); px1,py1=min(W,bx+bw+pad),min(H,by+bh+pad)
    panel=cv2.cvtColor(norm(g[py0:py1,px0:px1]),cv2.COLOR_GRAY2BGR)
    band=np.zeros((H,W),np.uint8); cv2.drawContours(band,[outer.astype(np.int32)],-1,255,-1); cv2.drawContours(band,[inner.astype(np.int32)],-1,0,-1)
    bc=band[py0:py1,px0:px1]; panel[bc>0]=np.clip(0.7*panel[bc>0]+np.array([60,60,0]),0,255).astype(np.uint8)
    nc,lab,stats,cent=cv2.connectedComponentsWithStats(dmask); ndet=0
    for i in range(1,nc):
        if stats[i,4]<8: continue
        cy,cx=int(cent[i][1]),int(cent[i][0]); r=int(np.clip(np.sqrt(stats[i,4])*1.6,18,80))
        cv2.circle(panel,(int(mapx[cy,cx])-px0,int(mapy[cy,cx])-py0),r,(0,0,235),3); ndet+=1
    sv=cv2.cvtColor(strip,cv2.COLOR_GRAY2BGR); sv[dmask>0]=(0,0,235)
    pw=900; panel=cv2.resize(panel,(pw,int(panel.shape[0]*pw/panel.shape[1]))); sv=cv2.resize(sv,(pw,int(sv.shape[0]*pw/sv.shape[1])))
    verdict="DEFECT" if ndet>0 else "OK"
    comp=np.vstack([banner(pw,f"END-TO-END  {base}  ->  {verdict} ({ndet} detection(s))"),panel,
                    banner(pw,"Predicted seal unrolled + defect (red)"),sv])
    cv2.imwrite(f"{OUT}/{base}.png",comp); n+=1; print(f"{base}: {verdict} ({ndet})")
print(f"wrote {n} composites to {OUT}")
