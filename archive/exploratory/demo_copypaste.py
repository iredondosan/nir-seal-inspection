import cv2, numpy as np, random, glob, os, xml.etree.ElementTree as ET
random.seed(5); np.random.seed(5)
from seal_inspection.paths import ROOT; CONTAM_XML=f"{ROOT}/data/annotations/contaminants.xml"
MARGIN=40
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
def load_contaminants(xml,exclude=set()):
    insts=[]
    for node in ET.parse(xml).getroot().findall("image"):
        nm=node.get("name")
        if nm in exclude: continue
        hits=glob.glob(f"{ROOT}/data/images/*/{nm}")
        if not hits: continue
        g=cv2.imread(hits[0],0)
        for pg in node.findall("polygon"):
            pts=pp(pg.get("points")).astype(np.int32); x,y,bw,bh=cv2.boundingRect(pts)
            if bw<4 or bh<4: continue
            patch=g[y:y+bh,x:x+bw].copy(); al=np.zeros((bh,bw),np.uint8); cv2.fillPoly(al,[pts-[x,y]],255)
            insts.append((patch,cv2.GaussianBlur(al,(0,0),2).astype(np.float32)/255.0))
    return insts
def paste(img3,band,insts,n):
    h,w=band.shape; ys,xs=np.where(band>0); out=img3.copy()
    for _ in range(n):
        patch,al=random.choice(insts); s=random.uniform(0.3,2.0)
        pw,ph=max(3,int(patch.shape[1]*s)),max(3,int(patch.shape[0]*s))
        p=cv2.resize(patch,(pw,ph)).astype(np.float32); a=cv2.resize(al,(pw,ph))
        M=cv2.getRotationMatrix2D((pw/2,ph/2),random.uniform(0,360),1.0); p=cv2.warpAffine(p,M,(pw,ph)); a=cv2.warpAffine(a,M,(pw,ph))
        if random.random()<0.5: p=cv2.flip(p,1); a=cv2.flip(a,1)
        p=np.clip(p*random.uniform(0.7,1.2),0,255); k=random.randrange(len(xs)); cx,cy=int(xs[k]),int(ys[k]); x0,y0=cx-pw//2,cy-ph//2
        ix0,iy0=max(0,x0),max(0,y0); ix1,iy1=min(w,x0+pw),min(h,y0+ph)
        if ix1<=ix0 or iy1<=iy0: continue
        px0,py0=ix0-x0,iy0-y0; px1,py1=px0+(ix1-ix0),py0+(iy1-iy0)
        aa=(a[py0:py1,px0:px1]*random.uniform(0.8,1.0))[...,None]; pp_=p[py0:py1,px0:px1][...,None]
        reg=out[iy0:iy1,ix0:ix1].astype(np.float32); out[iy0:iy1,ix0:ix1]=np.clip(reg*(1-aa)+pp_*aa,0,255).astype(np.uint8)
    return out
def band_of(xmlpath,prod):
    node=ET.parse(xmlpath).getroot().findall("image")[0]; nm=node.get("name")
    g=cv2.imread(f"{ROOT}/data/images/{prod}/{nm}",0)
    pl=sorted([pp(pg.get("points")) for pg in node.findall("polygon")],key=lambda q:cv2.contourArea(q.astype(np.float32)),reverse=True)
    H,W=g.shape; m=np.zeros((H,W),np.uint8); cv2.fillPoly(m,[pl[0].astype(np.int32)],1); cv2.fillPoly(m,[pl[1].astype(np.int32)],0)
    x0,y0,x1,y1=pack_bbox(g); return np.stack([norm(g[y0:y1,x0:x1])]*3,-1), m[y0:y1,x0:x1], nm

insts=load_contaminants(CONTAM_XML,exclude={"seal_1302_1780665903828_raw.png"})
print("instances:",len(insts))
prods=[("prod1",f"{ROOT}/data/annotations/prod1.xml"),("prod2",f"{ROOT}/data/annotations/annotations.xml"),
       ("prod4",f"{ROOT}/data/annotations/prod4.xml"),("prod5",f"{ROOT}/data/annotations/prod5.xml")]
def outline(im3,band):
    v=im3.copy(); cnts,_=cv2.findContours(band,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(v,cnts,-1,(0,255,0),2); return v
rows=[]
for prod,xmlp in prods:
    im3,band,nm=band_of(xmlp,prod)
    variants=[("%s clean"%prod,outline(im3,band))]
    for j in range(3):
        aug=paste(im3,band,insts,random.randint(2,4)); variants.append(("%s + contaminants"%prod,outline(aug,band)))
    cells=[]
    for t,im in variants:
        im=cv2.resize(im,(300,int(im.shape[0]*300/im.shape[1]))); lab=np.full((26,300,3),45,np.uint8); cv2.putText(lab,t,(5,18),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,255),1)
        cells.append(np.vstack([lab,im]))
    H=max(c.shape[0] for c in cells); cells=[np.vstack([c,np.full((H-c.shape[0],c.shape[1],3),255,np.uint8)]) for c in cells]
    rows.append(np.hstack(cells))
W=max(r.shape[1] for r in rows); rows=[np.hstack([r,np.full((r.shape[0],W-r.shape[1],3),255,np.uint8)]) for r in rows]
cv2.imwrite(f"{ROOT}/outputs/copypaste_demo.png",np.vstack(rows)); print("wrote outputs/copypaste_demo.png")
