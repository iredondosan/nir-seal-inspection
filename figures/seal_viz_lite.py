#!/usr/bin/env python3
"""QC visualization for the LITE (smp) model: pack + predicted mask on top,
4-row unrolled seal below. Predicts on the pack-crop, maps mask back to full raw res.
  python seal_viz_lite.py --weights models/best_lite_multiprod.pt --input data/images/prod1 --output outputs/viz_multiprod/prod1 --limit 6
"""
import os, glob, argparse
import numpy as np, cv2, torch
import segmentation_models_pytorch as smp

IM_MEAN=np.array((.485,.456,.406),np.float32); IM_STD=np.array((.229,.224,.225),np.float32); MARGIN=40

def norm(g):
    lo,hi=np.percentile(g,[1,99.5]); hi=max(hi,lo+1)
    return np.clip((g.astype(np.float32)-lo)/(hi-lo)*255,0,255).astype(np.uint8)
def conveyor_cols(N):
    cm=np.median(N,0).astype(np.float32); on=np.where(cm>cm.max()*0.5)[0]; cL,cR=on.min(),on.max(); g=np.gradient(cm)
    return int(np.argmax(g[max(0,cL-60):cL+60])+max(0,cL-60)), int(np.argmin(g[cR-60:cR+60])+(cR-60))
def pack_bbox(g):
    N=norm(g); h,w=N.shape
    try: cL,cR=conveyor_cols(N)
    except Exception: cL,cR=0,w
    top=np.median(N[20:240,:],0); bot=np.median(N[h-240:h-20,:],0); ref=np.maximum(top,bot)
    diff=np.clip(np.tile(ref,(h,1))-N.astype(np.float32),0,255); diff[:,:cL]=0; diff[:,cR:]=0
    m=cv2.morphologyEx((diff>20).astype(np.uint8)*255,cv2.MORPH_OPEN,np.ones((11,11),np.uint8))
    m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((41,41),np.uint8))
    c=[c for c in cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0] if cv2.contourArea(c)>h*w*0.02]
    if not c: return 0,0,w,h
    x,y,bw,bh=cv2.boundingRect(max(c,key=cv2.contourArea))
    return max(0,x-MARGIN),max(0,y-MARGIN),min(w,x+bw+MARGIN),min(h,y+bh+MARGIN)

def ring_contours(mask, band_px=90):
    m=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((9,9),np.uint8))
    m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((35,35),np.uint8))
    cnts,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if not cnts: return None,None,False
    outer=cv2.convexHull(np.vstack([c.reshape(-1,2) for c in cnts]))
    fill=np.zeros_like(m); cv2.drawContours(fill,[outer],-1,255,-1)
    hole=cv2.subtract(fill,m)
    c2=[c for c in cv2.findContours(hole,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0] if cv2.contourArea(c)>0.2*cv2.contourArea(outer)]
    if c2: inner=max(c2,key=cv2.contourArea); clean=True
    else:
        k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*band_px+1,2*band_px+1))
        ic,_=cv2.findContours(cv2.erode(fill,k),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        if not ic: return None,None,False
        inner=max(ic,key=cv2.contourArea); clean=False
    return outer.reshape(-1,2).astype(np.float32), inner.reshape(-1,2).astype(np.float32), clean
def resample(poly,n):
    p=np.r_[poly,poly[:1]]; d=np.r_[0,np.cumsum(np.hypot(*np.diff(p,axis=0).T))]
    t=np.linspace(0,d[-1],n,endpoint=False)
    return np.stack([np.interp(t,d,p[:,0]),np.interp(t,d,p[:,1])],1)
def csmooth(a,k=15): ker=np.ones(k)/k; return np.convolve(np.r_[a[-k:],a,a[:k]],ker,"same")[k:-k]
def ccw(p): return cv2.contourArea(p.astype(np.float32),oriented=True)>0
def unroll(gray,outer,inner,Hs,Ws):
    O=resample(outer,Ws); I=resample(inner,Ws)
    if ccw(O)!=ccw(I): I=I[::-1]
    j=int(np.argmin(np.hypot(I[:,0]-O[0,0],I[:,1]-O[0,1]))); I=np.roll(I,-j,axis=0)
    for arr in (O,I): arr[:,0]=csmooth(arr[:,0]); arr[:,1]=csmooth(arr[:,1])
    a=np.linspace(0,1,Hs)[:,None]
    mapx=(O[:,0][None,:]*(1-a)+I[:,0][None,:]*a).astype(np.float32)
    mapy=(O[:,1][None,:]*(1-a)+I[:,1][None,:]*a).astype(np.float32)
    return cv2.remap(gray,mapx,mapy,cv2.INTER_LINEAR,borderValue=0)
def banner(w,t,h=34):
    b=np.full((h,w,3),35,np.uint8); cv2.putText(b,t,(8,24),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1,cv2.LINE_AA); return b
def compose(pack_bgr,strip,rows,clean):
    """Native-resolution composite: full-res cropped pack panel on top, strip split into `rows`.
    strip width is expected to be rows*pack_width so each row matches the pack panel exactly (no downsizing)."""
    panel_w=pack_bgr.shape[1]
    sv=cv2.cvtColor(norm(strip),cv2.COLOR_GRAY2BGR); seg=sv.shape[1]//rows; parts=[]
    for k in range(rows):
        row=sv[:,k*seg:(k+1)*seg]
        if row.shape[1]!=panel_w: row=cv2.resize(row,(panel_w,sv.shape[0]))
        parts.append(row)
        if k<rows-1: parts.append(np.full((4,panel_w,3),255,np.uint8))
    tag="" if clean else "  [APPROX inner edge]"
    return np.vstack([banner(panel_w,"Pack + predicted seal mask (native resolution)"),pack_bgr,
                      banner(panel_w,f"Unrolled seal, {rows} rows = perimeter L->R"+tag),np.vstack(parts)])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--weights",required=True); ap.add_argument("--input",required=True)
    ap.add_argument("--output",required=True); ap.add_argument("--limit",type=int,default=0)
    ap.add_argument("--rows",type=int,default=4); ap.add_argument("--strip-h",type=int,default=180)
    a=ap.parse_args()
    dev="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    ck=torch.load(a.weights,map_location=dev,weights_only=False); IMG=ck["img"]; THR=ck.get("thresh",0.5)
    model=smp.Unet(ck["encoder"],encoder_weights=None,in_channels=3,classes=1); model.load_state_dict(ck["state_dict"]); model=model.to(dev).eval()
    os.makedirs(a.output,exist_ok=True)
    files=(sorted(glob.glob(os.path.join(a.input,"*_raw.png"))) or sorted(glob.glob(os.path.join(a.input,"*.jpg"))) or sorted(glob.glob(os.path.join(a.input,"*.png"))))
    if a.limit: files=files[:a.limit]
    ok=0
    for p in files:
        g=cv2.imread(p,cv2.IMREAD_GRAYSCALE)
        if g is None: continue
        h,w=g.shape; x0,y0,x1,y1=pack_bbox(g)
        gc=norm(g[y0:y1,x0:x1]); im=np.stack([gc,gc,gc],-1)
        xin=cv2.resize(im,(IMG,IMG)).astype(np.float32)/255.0; xin=((xin-IM_MEAN)/IM_STD).transpose(2,0,1)[None]
        with torch.no_grad(): prob=torch.sigmoid(model(torch.from_numpy(xin).to(dev)))[0,0].cpu().numpy()
        full=np.zeros((h,w),np.float32); full[y0:y1,x0:x1]=cv2.resize(prob,(x1-x0,y1-y0))
        mask=(full>THR).astype(np.uint8)*255
        outer,inner,clean=ring_contours(mask)
        if outer is None: print("skip",os.path.basename(p)); continue
        # full-res cropped pack panel + mask overlay (no downsizing)
        pc=g[y0:y1,x0:x1]; mc=mask[y0:y1,x0:x1]
        pv=cv2.cvtColor(norm(pc),cv2.COLOR_GRAY2BGR); pv[mc>0]=np.clip(0.5*pv[mc>0]+np.array([0,0,160]),0,255).astype(np.uint8)
        cw=pv.shape[1]
        # strip at native res: sample from full raw image along ring; width = rows*crop_w so each row = crop_w (no resize)
        strip=unroll(g,outer,inner,a.strip_h,a.rows*cw)
        base=os.path.splitext(os.path.basename(p))[0]
        if base.endswith("_raw"): base=base[:-4]
        cv2.imwrite(os.path.join(a.output,base+"_viz.png"),
                    compose(pv,strip,a.rows,clean)); ok+=1
    print(f"{a.output}: {ok}/{len(files)} written")

if __name__=="__main__": main()
