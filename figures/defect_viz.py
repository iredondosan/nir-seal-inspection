#!/usr/bin/env python3
"""End-to-end defect QC viz for the held-out TEST packs:
 top  = cropped pack with seal mask (cyan) + RED CIRCLES where the defect model fired (mapped back from the strip)
 bottom = unrolled seal strip with predicted defect (red) [GT green].
Uses GT seal polygons (test packs are labeled) to unroll, then runs models/defect_strip.pt."""
import os, glob, argparse
import numpy as np, cv2, torch
import xml.etree.ElementTree as ET
import segmentation_models_pytorch as smp
from seal_inspection.paths import ROOT
XML=f"{ROOT}/data/annotations/prod2_reviewed.xml"; IMGDIR=f"{ROOT}/data/images/prod2"
TESTIMG=f"{ROOT}/data/strips/test/img"; TESTMSK=f"{ROOT}/data/strips/test/mask"
OUT=f"{ROOT}/outputs/defect_viz"; MEAN=np.array((.485,.456,.406),np.float32); STD=np.array((.229,.224,.225),np.float32)

def norm(g):
    lo,hi=np.percentile(g,[1,99.5]); hi=max(hi,lo+1); return np.clip((g.astype(np.float32)-lo)/(hi-lo)*255,0,255).astype(np.uint8)
def pp(s): return np.array([[float(a) for a in p.split(',')] for p in s.strip().split(';')],np.float32)
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

ap=argparse.ArgumentParser(); ap.add_argument("--limit-good",type=int,default=6); a=ap.parse_args()
nodes={im.get("name"):im for im in ET.parse(XML).getroot().findall("image")}
ck=torch.load(f"{ROOT}/models/defect_strip.pt",map_location="cpu",weights_only=False)
HS,WS,THR=ck["HS"],ck["WS"],ck.get("thr",0.5)
m=smp.Unet(ck["encoder"],encoder_weights=None,in_channels=3,classes=1); m.load_state_dict(ck["state_dict"]); m.eval()
dev="cuda" if torch.cuda.is_available() else "cpu"; m=m.to(dev)
os.makedirs(OUT,exist_ok=True)
files=sorted(glob.glob(f"{TESTIMG}/*.png")); ngood=0; n=0
for ip in files:
    base=os.path.splitext(os.path.basename(ip))[0]; name=base+".png"
    if name not in nodes: continue
    gt=cv2.imread(f"{TESTMSK}/{base}.png",cv2.IMREAD_GRAYSCALE); is_def=gt is not None and gt.sum()>0
    if not is_def:
        if ngood>=a.limit_good: continue
        ngood+=1
    im=nodes[name]; sell=sorted([pp(p.get("points")) for p in im.findall("polygon") if p.get("label")=="sellado"],key=lambda q:cv2.contourArea(q.astype(np.float32)),reverse=True)
    if len(sell)<2: continue
    outer,inner=sell[0],sell[1]
    g=cv2.imread(f"{IMGDIR}/{name}",cv2.IMREAD_GRAYSCALE)
    if g is None: continue
    mapx,mapy=unroll_maps(outer,inner,HS,WS)
    strip=cv2.imread(ip,cv2.IMREAD_GRAYSCALE)
    x=((np.stack([strip]*3,-1).astype(np.float32)/255.0-MEAN)/STD).transpose(2,0,1)[None]
    with torch.no_grad(): prob=torch.sigmoid(m(torch.from_numpy(x).to(dev)))[0,0].cpu().numpy()
    dmask=(prob>THR).astype(np.uint8)
    # ---- pack panel (crop to seal bbox) ----
    H,W=g.shape; bx,by,bw,bh=cv2.boundingRect(outer.astype(np.int32)); pad=60
    x0,y0=max(0,bx-pad),max(0,by-pad); x1,y1=min(W,bx+bw+pad),min(H,by+bh+pad)
    panel=cv2.cvtColor(norm(g[y0:y1,x0:x1]),cv2.COLOR_GRAY2BGR)
    band=np.zeros((H,W),np.uint8); cv2.drawContours(band,[outer.astype(np.int32)],-1,255,-1); cv2.drawContours(band,[inner.astype(np.int32)],-1,0,-1)
    bandc=band[y0:y1,x0:x1]; panel[bandc>0]=np.clip(0.7*panel[bandc>0]+np.array([60,60,0]),0,255).astype(np.uint8)  # seal tint (cyan)
    # circles where defect fired: map strip components back to raw coords
    nc,lab,stats,cent=cv2.connectedComponentsWithStats(dmask)
    for i in range(1,nc):
        area=stats[i,4]
        if area<8: continue
        cy,cx=cent[i][1],cent[i][0]; r=int(np.clip(np.sqrt(area)*1.6,18,80))
        rx=int(mapx[int(cy),int(cx)])-x0; ry=int(mapy[int(cy),int(cx)])-y0
        cv2.circle(panel,(rx,ry),r,(0,0,235),3)
    # ---- strip panel ----
    sv=cv2.cvtColor(strip,cv2.COLOR_GRAY2BGR)
    if gt is not None: sv[gt>127]=(0,170,0)            # GT green
    sv[dmask>0]=(0,0,235)                              # pred red
    # compose (match widths)
    pw=900; panel=cv2.resize(panel,(pw,int(panel.shape[0]*pw/panel.shape[1]))); sv=cv2.resize(sv,(pw,int(sv.shape[0]*pw/sv.shape[1])))
    tag=f"{base}  [{'DEFECT' if is_def else 'good'}]  detections={max(0,nc-1)}"
    comp=np.vstack([banner(pw,"Pack + seal mask (cyan) + defect detections (red circles)"),panel,
                    banner(pw,"Unrolled seal: GT green / predicted defect red"),sv])
    cv2.putText(comp,tag,(8,comp.shape[0]-10),cv2.FONT_HERSHEY_SIMPLEX,0.55,(40,220,220),1,cv2.LINE_AA)
    cv2.imwrite(f"{OUT}/{base}.png",comp); n+=1
print(f"wrote {n} composites to {OUT} (HS={HS} WS={WS} thr={THR})")
