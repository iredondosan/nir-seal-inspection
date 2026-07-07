#!/usr/bin/env python3
"""Build the DEFECT dataset from MULTIPLE sources: unroll each labeled pack's seal into a strip
+ unroll its defect mask with the SAME mapping. Pack-level stratified split (no leakage),
with FORCE_TRAIN/FORCE_TEST for the few-distinct prod6 defects.
Output: data/strips/{train,test}/{img,mask}/<name>.png  + a few overlay viz."""
import os, glob, argparse, random
from collections import defaultdict
import numpy as np, cv2
import xml.etree.ElementTree as ET
ROOT="/home/ubuntu/TFM/seal-inspection"
HS=128; WS=1536; SEED=42; TEST_FRAC=0.2
# (xml, imgdir, mode): mode 'all'=good+defect; 'good'=negatives only; 'defect_reviewed'=only reviewed packs w/ defect
SOURCES=[("data/annotations/prod1_reviewed.xml","data/images/prod1","all"),
         ("data/annotations/prod2_reviewed.xml","data/images/prod2","all"),
         ("data/annotations/prod3_reviewed.xml","data/images/prod3","all"),
         ("data/annotations/prod4_reviewed.xml","data/images/prod4","all"),
         ("data/annotations/prod5_reviewed.xml","data/images/prod5","all"),
         ("data/annotations/prod6_reviewed.xml","data/images/prod6","all")]
FORCE_TEST=set()      # merged prod6: prod6 never in test (holdout.txt controls test)
FORCE_TRAIN=set()

def norm(g):
    lo,hi=np.percentile(g,[1,99.5]); hi=max(hi,lo+1); return np.clip((g.astype(np.float32)-lo)/(hi-lo)*255,0,255).astype(np.uint8)
def pp(s): return np.array([[float(a) for a in p.split(',')] for p in s.strip().split(';')],np.float32)
def resample(poly,n):
    p=np.r_[poly,poly[:1]]; d=np.r_[0,np.cumsum(np.hypot(*np.diff(p,axis=0).T))]
    t=np.linspace(0,d[-1],n,endpoint=False); return np.stack([np.interp(t,d,p[:,0]),np.interp(t,d,p[:,1])],1)
def csmooth(a,k=15): ker=np.ones(k)/k; return np.convolve(np.r_[a[-k:],a,a[:k]],ker,"same")[k:-k]
def ccw(p): return cv2.contourArea(p.astype(np.float32),oriented=True)>0
def unroll_maps(outer,inner,Hs,Ws):
    # Perpendicular-to-outer sampling: walk inward along the smoothed OUTER contour's normal
    # to a local depth = band width (distance to inner edge). Correspondence-free -> stable for
    # thin defects. MUST match seal_inspection/core.py unroll_maps so train/inference strips agree.
    O=resample(outer,Ws); O[:,0]=csmooth(O[:,0]); O[:,1]=csmooth(O[:,1])
    T=np.roll(O,-1,0)-np.roll(O,1,0); Tn=np.maximum(np.hypot(T[:,0],T[:,1]),1e-6)
    N=np.stack([-T[:,1]/Tn,T[:,0]/Tn],1)
    cw=int(max(outer[:,0].max(),inner[:,0].max()))+10; ch=int(max(outer[:,1].max(),inner[:,1].max()))+10
    fo=np.zeros((ch,cw),np.uint8); cv2.drawContours(fo,[outer.astype(np.int32)],-1,255,-1)
    pr=(O+4*N).astype(int)
    if (fo[np.clip(pr[:,1],0,ch-1),np.clip(pr[:,0],0,cw-1)]>0).mean()<0.5: N=-N
    fi=np.zeros((ch,cw),np.uint8); cv2.drawContours(fi,[inner.astype(np.int32)],-1,255,-1)
    dt=cv2.distanceTransform(255-fi,cv2.DIST_L2,5)
    L=csmooth(dt[np.clip(O[:,1].astype(int),0,ch-1),np.clip(O[:,0].astype(int),0,cw-1)])
    a=np.linspace(-0.15,1.15,Hs)[:,None]   # 0=outer edge, 1=inner edge, +/-15% margin
    return (O[:,0][None,:]+a*(L[None,:]*N[:,0][None,:])).astype(np.float32),(O[:,1][None,:]+a*(L[None,:]*N[:,1][None,:])).astype(np.float32)

def unroll_maps_legacy(outer,inner,Hs,Ws):
    # Legacy correspondence unroll (outer<->inner linear interpolation). Used to build strips for the
    # ensemble's second branch; MUST match seal_inspection/core.py unroll_maps_legacy.
    O=resample(outer,Ws); I=resample(inner,Ws)
    if ccw(O)!=ccw(I): I=I[::-1]
    j=int(np.argmin(np.hypot(I[:,0]-O[0,0],I[:,1]-O[0,1]))); I=np.roll(I,-j,axis=0)
    for arr in (O,I): arr[:,0]=csmooth(arr[:,0]); arr[:,1]=csmooth(arr[:,1])
    a=np.linspace(-0.15,1.15,Hs)[:,None]
    return (O[:,0][None,:]*(1-a)+I[:,0][None,:]*a).astype(np.float32),(O[:,1][None,:]*(1-a)+I[:,1][None,:]*a).astype(np.float32)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",default=f"{ROOT}/data/strips")
    ap.add_argument("--legacy",action="store_true",help="use the legacy correspondence unroll (ensemble 2nd branch)")
    ap.add_argument("--holdout",default=f"{ROOT}/data/holdout.txt",help="basenames forced to the TEST split (the global hold-out)")
    a=ap.parse_args()
    unr=unroll_maps_legacy if a.legacy else unroll_maps
    HOLD=set()
    if os.path.exists(a.holdout):
        HOLD=set(l.strip() for l in open(a.holdout) if l.strip())
        print(f"hold-out: {len(HOLD)} packs forced to TEST (from {os.path.basename(a.holdout)}); unroll={'legacy' if a.legacy else 'perpendicular'}")
    for sp in ["train","test"]:
        for k in ["img","mask"]: os.makedirs(f"{a.out}/{sp}/{k}",exist_ok=True)
    os.makedirs(f"{a.out}/viz",exist_ok=True)
    nodes=[]
    for xmlrel,imgrel,mode in SOURCES:
        path=f"{ROOT}/{xmlrel}"
        if not os.path.exists(path): print("MISSING",xmlrel); continue
        cnt=0
        for im in ET.parse(path).getroot().findall("image"):
            tg={t.get("label") for t in im.findall("tag")}
            if "exclude" in tg: continue
            sell=[pp(p.get("points")) for p in im.findall("polygon") if p.get("label")=="sellado"]
            if len(sell)<2: continue
            defs=[pp(p.get("points")) for p in im.findall("polygon") if p.get("label") in ("defect","liquid")]  # liquid IS a defect
            if mode=="good":
                if not (tg & {"good","reviewed"}): continue
                defs=[]; kind="good"
            elif mode=="defect_reviewed":
                if "reviewed" not in tg or not defs: continue
                kind="defect"
            else:  # all
                if not (tg & {"good","defect","reviewed"}): continue
                kind="defect" if defs else "good"
            nodes.append((im.get("name"),int(im.get("width")),int(im.get("height")),sell,defs,kind,imgrel)); cnt+=1
        print(f"{xmlrel} [{mode}] -> {cnt} packs")
    test_set=set()
    if HOLD:                                                 # GLOBAL hold-out: test = hold-out packs, everything else train
        for n in nodes:
            if os.path.splitext(n[0])[0] in HOLD: test_set.add(id(n))
    else:                                                    # fallback: per-product stratified split (per imgdir+kind)
        rng=random.Random(SEED)
        buckets=defaultdict(list)
        for n in nodes: buckets[(n[6],n[5])].append(n)
        for key,lst in sorted(buckets.items()):
            rng.shuffle(lst)
            for n in lst:
                if n[0] in FORCE_TEST: test_set.add(id(n))
            rest=[n for n in lst if n[0] not in FORCE_TEST and n[0] not in FORCE_TRAIN]
            k=int(len(rest)*TEST_FRAC)
            if key[1]=="defect" and len(rest)>=2 and k==0: k=1
            for n in rest[:k]: test_set.add(id(n))
    counts={"train":[0,0],"test":[0,0]}; nviz=0
    for n in nodes:
        name,W,H,sell,defs,kind,imgrel=n
        sp="test" if id(n) in test_set else "train"
        g=cv2.imread(f"{ROOT}/{imgrel}/{name}",cv2.IMREAD_GRAYSCALE)
        if g is None: continue
        sell=sorted(sell,key=lambda q:cv2.contourArea(q.astype(np.float32)),reverse=True); outer,inner=sell[0],sell[1]
        dm=np.zeros((H,W),np.uint8)
        for d in defs: cv2.fillPoly(dm,[d.astype(np.int32)],255)
        mapx,mapy=unr(outer,inner,HS,WS)
        strip=cv2.remap(norm(g),mapx,mapy,cv2.INTER_LINEAR,borderValue=0)
        smask=(cv2.remap(dm,mapx,mapy,cv2.INTER_LINEAR,borderValue=0)>127).astype(np.uint8)*255
        base=os.path.splitext(name)[0]
        cv2.imwrite(f"{a.out}/{sp}/img/{base}.png",strip); cv2.imwrite(f"{a.out}/{sp}/mask/{base}.png",smask)
        counts[sp][0 if kind=="good" else 1]+=1
        if defs and nviz<14:
            v=cv2.cvtColor(strip,cv2.COLOR_GRAY2BGR); v[smask>0]=(0,0,230); cv2.imwrite(f"{a.out}/viz/{base}_{sp}.png",v); nviz+=1
    print(f"train: {counts['train'][0]} good / {counts['train'][1]} defect")
    print(f"test:  {counts['test'][0]} good / {counts['test'][1]} defect")
if __name__=="__main__": main()
