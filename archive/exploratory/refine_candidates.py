#!/usr/bin/env python3
"""Generate several candidate seal-ring polygons for ONE pack (for the LLM-judge refine loop).
Candidates use the multiprod lite seg model + morphology + a geometric outer+offset fallback,
so when the inner edge is occluded (product overflow) at least one candidate still closes the ring.
Saves an overview montage + per-candidate overlays + the polygons (raw coords) as JSON.
Usage: .venv/bin/python src/refine_candidates.py --image <name> [--prod prod2] [--outdir outputs/refine]"""
import os, glob, argparse, json
import numpy as np, cv2, torch
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
def ring_from_mask(mask, band_px=85):
    """Return (outer,inner,had_hole) contours from a binary mask, or None."""
    m=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((15,15),np.uint8))
    cnts,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if not cnts: return None
    outer=max(cnts,key=cv2.contourArea); fill=np.zeros_like(m); cv2.drawContours(fill,[outer],-1,255,-1)
    hole=cv2.subtract(fill,m)
    hc=[c for c in cv2.findContours(hole,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0] if cv2.contourArea(c)>0.15*cv2.contourArea(outer)]
    if hc: return outer,max(hc,key=cv2.contourArea),True
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*band_px+1,2*band_px+1)); ic,_=cv2.findContours(cv2.erode(fill,k),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if not ic: return None
    return outer,max(ic,key=cv2.contourArea),False
def predict_prob(model,dev,crop):
    im=cv2.resize(np.stack([norm(crop)]*3,-1),(IMG,IMG)).astype(np.float32)/255.0; x=((im-MEAN)/STD).transpose(2,0,1)[None]
    with torch.no_grad(): return torch.sigmoid(model(torch.from_numpy(x).to(dev)))[0,0].cpu().numpy()
def tta_prob(model,dev,crop):
    ps=[predict_prob(model,dev,crop)]
    for fn,inv in [(lambda a:cv2.flip(a,1),lambda a:cv2.flip(a,1)),(lambda a:cv2.flip(a,0),lambda a:cv2.flip(a,0)),(lambda a:cv2.flip(a,-1),lambda a:cv2.flip(a,-1))]:
        ps.append(inv(predict_prob(model,dev,fn(crop))))
    return np.mean(ps,0)

ap=argparse.ArgumentParser(); ap.add_argument("--image",required=True); ap.add_argument("--prod",default="prod2"); ap.add_argument("--outdir",default=f"{ROOT}/outputs/refine"); ap.add_argument("--model",default=MODEL); a=ap.parse_args()
os.makedirs(a.outdir,exist_ok=True)
hits=glob.glob(f"{ROOT}/data/images/*/{a.image}"); assert hits, f"image not found: {a.image}"
g=cv2.imread(hits[0],cv2.IMREAD_GRAYSCALE); oh,ow=g.shape
x0,y0,x1,y1=pack_bbox(g); crop=g[y0:y1,x0:x1]; ch,cw=crop.shape
ck=torch.load(a.model,map_location="cpu",weights_only=False); THR=ck.get("thresh",0.5); print(f"model: {a.model}")
m=smp.Unet(ck["encoder"],encoder_weights=None,in_channels=3,classes=1); m.load_state_dict(ck["state_dict"]); m.eval()
dev="cuda" if torch.cuda.is_available() else "cpu"; m=m.to(dev)
prob=cv2.resize(predict_prob(m,dev,crop),(cw,ch)); probT=cv2.resize(tta_prob(m,dev,crop),(cw,ch))

def mask_thr(p,t): return (p>t).astype(np.uint8)*255
def mask_close(p,t,k): mm=mask_thr(p,t); return cv2.morphologyEx(mm,cv2.MORPH_CLOSE,np.ones((k,k),np.uint8))
def geometric_offset(p,t,off):
    """Outer contour of the seal region, eroded inward by 'off' px -> guarantees a closed ring even if inner edge occluded."""
    mm=cv2.morphologyEx(mask_thr(p,t),cv2.MORPH_CLOSE,np.ones((25,25),np.uint8))
    cnts,_=cv2.findContours(mm,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if not cnts: return None
    outer=max(cnts,key=cv2.contourArea); fill=np.zeros_like(mm); cv2.drawContours(fill,[outer],-1,255,-1)
    inner=cv2.erode(fill,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*off+1,2*off+1)))
    return cv2.subtract(fill,inner)

# estimate a typical band width from the (possibly broken) baseline for the geometric fallback
base=ring_from_mask(mask_thr(prob,THR))
bw_est=85
if base is not None:
    of=np.zeros((ch,cw),np.uint8); cv2.drawContours(of,[base[0]],-1,1,-1); inf=np.zeros((ch,cw),np.uint8); cv2.drawContours(inf,[base[1]],-1,1,-1)
    bnd=of&~inf
    if bnd.sum()>0:
        from scipy.ndimage import distance_transform_edt; bw_est=int(np.clip(2*distance_transform_edt(bnd).max(),50,160))

CANDS=[
 ("A_seg@0.50", ring_from_mask(mask_thr(prob,0.50))),
 ("B_seg@0.35", ring_from_mask(mask_thr(prob,0.35))),
 ("C_close@0.40", ring_from_mask(mask_close(prob,0.40,31))),
 ("D_tta@0.50", ring_from_mask(mask_thr(probT,0.50))),
 ("E_geom_offset", ring_from_mask(geometric_offset(prob,0.40,bw_est) if geometric_offset(prob,0.40,bw_est) is not None else np.zeros((ch,cw),np.uint8))),
]
def hybrid(ringD, target):
    """Real inner edge where the band is healthy; geometric offset only where it pinches (occluded inner edge).
    Outer = convex hull of the seg outer (a seal outer is convex -> removes inward diverts at corners)."""
    if ringD is None: return None
    outer=cv2.convexHull(ringD[0])
    of=np.zeros((ch,cw),np.uint8); cv2.drawContours(of,[outer],-1,255,-1)
    real_hole=np.zeros((ch,cw),np.uint8); cv2.drawContours(real_hole,[ringD[1]],-1,255,-1)
    off_hole=cv2.erode(of,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*target+1,2*target+1)))
    final_hole=cv2.bitwise_and(real_hole,off_hole)          # keep hole only where BOTH agree -> band >= target everywhere
    return ring_from_mask(cv2.subtract(of,final_hole))
CANDS.append(("F_hybrid", hybrid(ring_from_mask(mask_thr(probT,0.50)), bw_est)))

def smooth_outer(outer, max_depth=25.0):
    """Bridge ONLY deep convexity defects (diverts/notches) with the hull chord; keep gentle waviness (<max_depth px)."""
    pts=outer.reshape(-1,2); n=len(pts)
    if n<4: return outer
    hull=cv2.convexHull(outer, returnPoints=False)
    defects=cv2.convexityDefects(outer, hull)
    if defects is None: return outer
    drop=set(); depths=[]
    for s,e,f,dpt in defects[:,0,:]:
        depths.append(dpt/256.0)
        if dpt/256.0>max_depth:
            i=(s+1)%n
            while i!=e: drop.add(i); i=(i+1)%n
    if depths: print(f"  outer convexity-defect depths(px): max={max(depths):.0f} bridged>{max_depth:.0f} -> dropped {len(drop)} pts")
    keep=np.array([pts[i] for i in range(n) if i not in drop],np.int32)
    return keep.reshape(-1,1,2) if len(keep)>=3 else outer
_rD=ring_from_mask(mask_thr(probT,0.50))
CANDS.append(("G_D_cornerfix", (smooth_outer(_rD[0]), _rD[1], _rD[2]) if _rD else None))

view=cv2.cvtColor(norm(crop),cv2.COLOR_GRAY2BGR)
def draw(base_img,ring,title):
    v=base_img.copy()
    if ring is not None:
        cv2.drawContours(v,[ring[0]],-1,(0,0,235),2)      # outer red
        cv2.drawContours(v,[ring[1]],-1,(0,220,235),2)    # inner yellow
        tag=f"{title}  ring_closed={ring[2]}"
    else: tag=f"{title}  FAILED(no ring)"
    l=np.full((26,v.shape[1],3),35,np.uint8); cv2.putText(l,tag,(6,18),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
    return np.vstack([l,v])
def fit(im,W=520): return cv2.resize(im,(W,int(im.shape[0]*W/im.shape[1])))

# save per-candidate full + bottom-zoom (the failure region), and the polygons
polys={}
for title,ring in CANDS:
    ov=draw(view,ring,title); cv2.imwrite(f"{a.outdir}/{a.image[:-4]}_{title}.png",fit(ov,640))
    if ring is not None: polys[title]={"outer":ring[0].reshape(-1,2).tolist(),"inner":ring[1].reshape(-1,2).tolist(),"closed":ring[2],"offset_x":x0,"offset_y":y0}
# heatmap + raw reference
hm=cv2.applyColorMap((prob*255).astype(np.uint8),cv2.COLORMAP_JET)
ref=np.hstack([fit(draw(view,None,"RAW crop")),fit(np.vstack([np.full((26,hm.shape[1],3),35,np.uint8),hm]))])
cv2.imwrite(f"{a.outdir}/{a.image[:-4]}_REF.png",ref)
# montage of all candidates
tiles=[fit(draw(view,r,t),360) for t,r in CANDS]
H=max(t.shape[0] for t in tiles); tiles=[np.vstack([t,np.full((H-t.shape[0],t.shape[1],3),255,np.uint8)]) for t in tiles]
cv2.imwrite(f"{a.outdir}/{a.image[:-4]}_MONTAGE.png",np.hstack(tiles))
json.dump(polys,open(f"{a.outdir}/{a.image[:-4]}_polys.json","w"))
print(f"band-width est: {bw_est}px"); print("candidates:", [f"{t}:{'ok' if r else 'FAIL'}{'(closed)' if r and r[2] else ''}" for t,r in CANDS])
print(f"wrote overlays + {a.image[:-4]}_polys.json to {a.outdir}")
