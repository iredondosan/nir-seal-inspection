#!/usr/bin/env python3
"""Inference-time seal-prediction QUALITY score (NO ground truth needed).
Combines: (1) geometry sanity (closed ring? band-width uniformity? plausible area/aspect),
(2) probability-map confidence (uncertain-pixel fraction, mean prob in mask, edge sharpness),
(3) optional TTA disagreement (flip/rotate -> mask agreement).
Emits a 0-1 score + flag + reasons, CSV sorted worst-first = human-review / active-learning queue.
Usage: .venv/bin/python src/seal_quality.py --input <imgdir> [--tta] [--csv out.csv] [--thresh 0.55]"""
import os, glob, argparse
import numpy as np, cv2, torch
import segmentation_models_pytorch as smp
from scipy.ndimage import distance_transform_edt

ROOT="/home/ubuntu/TFM/seal-inspection"; MODEL=f"{ROOT}/models/best_lite_multiprod.pt"
IMG=384; MARGIN=40; MEAN=np.array((.485,.456,.406),np.float32); STD=np.array((.229,.224,.225),np.float32)
# quality-score normalisation knobs (tuned to give ~1 for clean, ~0 for broken)
CV_MAX=0.55      # band-width coeff-of-variation at which uniformity score hits 0 (pinch/divert)
U_MAX=0.14       # uncertain-pixel fraction at which confidence score hits 0
SHARP_REF=0.020  # prob-gradient/px at the boundary considered "sharp" (->1)
W={"ring":0.30,"cv":0.25,"uncert":0.20,"sharp":0.10,"tta":0.15}  # weights (renormalised if no TTA)
FLAG=0.55        # composite below this -> flag for review

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
    hole=cv2.subtract(fill,m); had_hole=False
    hc=[c for c in cv2.findContours(hole,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0] if cv2.contourArea(c)>0.15*cv2.contourArea(outer)]
    if hc: inner=max(hc,key=cv2.contourArea); had_hole=True
    else:
        k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*band_px+1,2*band_px+1)); ic,_=cv2.findContours(cv2.erode(fill,k),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        if not ic: return None
        inner=max(ic,key=cv2.contourArea)
    return outer,inner,had_hole

def predict_prob(model,dev,crop):
    im=cv2.resize(np.stack([norm(crop)]*3,-1),(IMG,IMG)).astype(np.float32)/255.0; x=((im-MEAN)/STD).transpose(2,0,1)[None]
    with torch.no_grad(): return torch.sigmoid(model(torch.from_numpy(x).to(dev)))[0,0].cpu().numpy()  # 384x384

def tta_agreement(model,dev,crop,thr):
    """Predict under flips, undo, measure IoU agreement vs base. Returns mean IoU in [0,1]."""
    base=(predict_prob(model,dev,crop)>thr).astype(np.uint8)
    ious=[]
    for fn,inv in [(lambda a:cv2.flip(a,1),lambda a:cv2.flip(a,1)),
                   (lambda a:cv2.flip(a,0),lambda a:cv2.flip(a,0)),
                   (lambda a:cv2.flip(a,-1),lambda a:cv2.flip(a,-1))]:
        p=(inv(predict_prob(model,dev,fn(crop)))>thr).astype(np.uint8)
        i=(base&p).sum(); u=(base|p).sum(); ious.append(i/u if u>0 else 0.0)
    return float(np.mean(ious))

def clamp01(v): return max(0.0,min(1.0,v))

def assess(prob, thr, do_tta, model, dev, crop):
    """prob: 384x384 sigmoid map. Returns (score, subscores dict, reasons list)."""
    H,Wd=prob.shape; mask=(prob>thr).astype(np.uint8)*255
    reasons=[]
    # --- geometry ---
    r=ring_polys(mask)
    if r is None or len(r[0])<8:
        return 0.05, dict(ring=0,cv=0,uncert=0,sharp=0,tta=0,bandw=0,areafrac=float(mask.mean()/255)), ["no_closed_ring"]
    outer,inner,had_hole=r
    fill=np.zeros((H,Wd),np.uint8); cv2.drawContours(fill,[outer],-1,1,-1)
    innerfill=np.zeros((H,Wd),np.uint8); cv2.drawContours(innerfill,[inner],-1,1,-1)
    band=(fill&~innerfill).astype(np.uint8)
    s_ring=1.0 if had_hole else 0.45            # erosion-fallback ring is weaker evidence
    if not had_hole: reasons.append("ring_not_closed(hole_inferred)")
    # band-width uniformity: distance from each outer-boundary px to the inner contour
    ib=np.zeros((H,Wd),np.uint8); cv2.drawContours(ib,[inner],-1,1,1)
    dt_in=distance_transform_edt(1-ib)
    ob=np.zeros((H,Wd),np.uint8); cv2.drawContours(ob,[outer],-1,1,1)
    widths=dt_in[ob>0]
    if len(widths)<10 or widths.mean()<1e-3:
        cvw=1.0; bandw=0.0
    else:
        bandw=float(widths.mean()); cvw=float(widths.std()/widths.mean())
    s_cv=clamp01(1-cvw/CV_MAX)
    if cvw>0.35: reasons.append(f"band_width_uneven(cv={cvw:.2f})")
    areafrac=float(fill.mean())            # outer area / crop  (sanity, not scored hard)
    # --- probability-map confidence ---
    unc=float(((prob>0.30)&(prob<0.70)).mean()); s_unc=clamp01(1-unc/U_MAX)
    if unc>0.10: reasons.append(f"high_uncertain_frac({unc:.2f})")
    meanp=float(prob[mask>0].mean()) if (mask>0).any() else 0.0
    gx=cv2.Sobel(prob,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(prob,cv2.CV_32F,0,1,ksize=3)
    gmag=np.sqrt(gx*gx+gy*gy); edgeband=cv2.dilate(ob,np.ones((5,5),np.uint8))
    sharp=float(gmag[edgeband>0].mean()) if (edgeband>0).any() else 0.0
    s_sharp=clamp01(sharp/SHARP_REF)
    if sharp<0.010: reasons.append(f"fuzzy_boundary(sharp={sharp:.3f})")
    # --- TTA ---
    if do_tta:
        s_tta=tta_agreement(model,dev,crop,thr)
        if s_tta<0.85: reasons.append(f"tta_disagreement(iou={s_tta:.2f})")
        w=W
    else:
        s_tta=0.0; w={k:v for k,v in W.items() if k!="tta"}
    tot=sum(w.values()); sub=dict(ring=s_ring,cv=s_cv,uncert=s_unc,sharp=s_sharp,tta=s_tta)
    score=sum(w[k]*sub[k] for k in w)/tot
    sub.update(bandw=bandw,areafrac=areafrac,meanp=meanp)
    return float(score), sub, reasons

ap=argparse.ArgumentParser(); ap.add_argument("--input",required=True); ap.add_argument("--tta",action="store_true")
ap.add_argument("--thresh",type=float,default=None); ap.add_argument("--csv",default=f"{ROOT}/outputs/seal_quality.csv"); a=ap.parse_args()
ck=torch.load(MODEL,map_location="cpu",weights_only=False); THR=a.thresh if a.thresh is not None else ck.get("thresh",0.5)
m=smp.Unet(ck["encoder"],encoder_weights=None,in_channels=3,classes=1); m.load_state_dict(ck["state_dict"]); m.eval()
dev="cuda" if torch.cuda.is_available() else "cpu"; m=m.to(dev)
files=sorted(glob.glob(os.path.join(a.input,"*_raw.png")))
rows=[]
for p in files:
    g=cv2.imread(p,cv2.IMREAD_GRAYSCALE)
    if g is None: continue
    x0,y0,x1,y1=pack_bbox(g); crop=g[y0:y1,x0:x1]
    prob=predict_prob(m,dev,crop)
    score,sub,reasons=assess(prob,THR,a.tta,m,dev,crop)
    rows.append((os.path.basename(p),score,sub,reasons))
rows.sort(key=lambda r:r[1])  # worst first
os.makedirs(os.path.dirname(a.csv),exist_ok=True)
with open(a.csv,"w") as f:
    f.write("name,score,flag,ring,cv,uncert,sharp,tta,bandw,reasons\n")
    for nm,sc,sub,rs in rows:
        f.write(f"{nm},{sc:.4f},{int(sc<FLAG)},{sub['ring']:.2f},{sub['cv']:.2f},{sub['uncert']:.2f},{sub['sharp']:.2f},{sub.get('tta',0):.2f},{sub.get('bandw',0):.1f},{'|'.join(rs)}\n")
nflag=sum(1 for r in rows if r[1]<FLAG)
print(f"=== seal quality  ({len(rows)} packs, TTA={'on' if a.tta else 'off'}, thr={THR})  flag<{FLAG} ===")
print(f"flagged for review: {nflag}/{len(rows)} ({100*nflag/max(1,len(rows)):.0f}%)")
print("\nworst 15 (review-queue head):")
print(f"{'score':>6s} {'name':45s} reasons")
for nm,sc,sub,rs in rows[:15]: print(f"{sc:6.3f} {nm:45s} {','.join(rs)}")
print(f"\nwrote {a.csv} (sorted worst-first)")
