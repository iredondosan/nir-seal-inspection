#!/usr/bin/env python3
"""Measure pack-crop dimensions per product to pick a 'native' input size."""
import glob, numpy as np, cv2
ROOT="/home/ubuntu/TFM/seal-inspection"; MARGIN=40
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
for prod in ["prod1","prod2","prod3","prod4","prod5"]:
    fs=sorted(glob.glob(f"{ROOT}/data/images/{prod}/*_raw.png"))[:30]
    ws=[];hs=[]
    for p in fs:
        g=cv2.imread(p,cv2.IMREAD_GRAYSCALE)
        if g is None: continue
        x0,y0,x1,y1=pack_bbox(g); ws.append(x1-x0); hs.append(y1-y0)
    if ws: print(f"{prod}: n={len(ws)}  W median={int(np.median(ws))} max={max(ws)}  H median={int(np.median(hs))} max={max(hs)}")
print(f"\nALL max dim observed -> choose IMG >= that, divisible by 32")
