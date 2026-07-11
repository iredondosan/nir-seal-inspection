import cv2, numpy as np, random, glob, xml.etree.ElementTree as ET
random.seed(4); np.random.seed(4)
ROOT="/home/ubuntu/TFM/seal-inspection"
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
MARGIN=40
def pp(s): return np.array([[float(a) for a in p.split(',')] for p in s.split(';')],np.float32)
def load(name,prod,xmlp):
    g=cv2.imread(glob.glob(f"{ROOT}/data/images/{prod}/{name}")[0],0)
    node=[im for im in ET.parse(xmlp).getroot().findall("image") if im.get("name")==name][0]
    pl=sorted([pp(pg.get("points")) for pg in node.findall("polygon")],key=lambda q:cv2.contourArea(q.astype(np.float32)),reverse=True)
    H,W=g.shape; band=np.zeros((H,W),np.uint8); cv2.fillPoly(band,[pl[0].astype(np.int32)],1); cv2.fillPoly(band,[pl[1].astype(np.int32)],0)
    x0,y0,x1,y1=pack_bbox(g); return norm(g[y0:y1,x0:x1]), band[y0:y1,x0:x1]

# --- extract printed-graphic cut-outs from prod2 flanges (dark structured content ON the band) ---
def extract_prints(name):
    crop,band=load(name,"prod2",f"{ROOT}/data/annotations/annotations.xml")
    fl=band>0; flmed=np.median(crop[fl])
    pm=(((flmed-crop.astype(np.float32))>28)&fl).astype(np.uint8)*255
    pm=cv2.dilate(pm,np.ones((13,13),np.uint8))   # merge text strokes into printed blocks
    n,lab,stats,_=cv2.connectedComponentsWithStats(pm)
    insts=[]
    for i in range(1,n):
        x,y,bw,bh,area=stats[i]
        if 600<area<90000 and bw>20 and bh>12:
            pad=4; y0=max(0,y-pad); x0=max(0,x-pad); patch=crop[y0:y+bh+pad,x0:x+bw+pad].copy()
            sub=(lab[y0:y+bh+pad,x0:x+bw+pad]==i)
            al=(((flmed-patch.astype(np.float32))/max(1,flmed*0.6)).clip(0,1))*sub
            al=cv2.GaussianBlur(al.astype(np.float32),(0,0),1.5)
            insts.append((patch,al))
    return insts
prints=[]
for nm in ["seal_112_1780622750159_raw.png","seal_113_1780622758293_raw.png","seal_124_1780622881577_raw.png","seal_117_1780622824868_raw.png","seal_125_1780622889693_raw.png"]:
    prints+=extract_prints(nm)
print("print instances extracted:",len(prints))

def paste(crop,band,insts,n):
    h,w=band.shape; ys,xs=np.where(band>0); out=np.stack([crop]*3,-1).copy()
    for _ in range(n):
        patch,al=random.choice(insts); s=random.uniform(0.6,1.4)
        pw,ph=max(4,int(patch.shape[1]*s)),max(4,int(patch.shape[0]*s)); p=cv2.resize(patch,(pw,ph)).astype(np.float32); a=cv2.resize(al,(pw,ph))
        M=cv2.getRotationMatrix2D((pw/2,ph/2),random.choice([0,90,180,270])+random.uniform(-12,12),1.0); p=cv2.warpAffine(p,M,(pw,ph)); a=cv2.warpAffine(a,M,(pw,ph))
        k=random.randrange(len(xs)); cx,cy=int(xs[k]),int(ys[k]); x0,y0=cx-pw//2,cy-ph//2
        ix0,iy0=max(0,x0),max(0,y0); ix1,iy1=min(w,x0+pw),min(h,y0+ph)
        if ix1<=ix0 or iy1<=iy0: continue
        px0,py0=ix0-x0,iy0-y0; px1,py1=px0+(ix1-ix0),py0+(iy1-iy0)
        aa=a[py0:py1,px0:px1][...,None]; pp_=p[py0:py1,px0:px1][...,None]
        reg=out[iy0:iy1,ix0:ix1].astype(np.float32); out[iy0:iy1,ix0:ix1]=np.clip(reg*(1-aa)+pp_*aa,0,255).astype(np.uint8)
    return out
def outline(im3,band):
    v=im3.copy(); cnts,_=cv2.findContours(band,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(v,cnts,-1,(0,255,0),2); return v
TARGS=[("prod2","seal_115_1780622788753_raw.png",f"{ROOT}/data/annotations/annotations.xml"),
       ("prod1",None,f"{ROOT}/data/annotations/prod1.xml"),("prod4",None,f"{ROOT}/data/annotations/prod4.xml")]
rows=[]
for prod,nm,xmlp in TARGS:
    if nm is None: nm=[im.get("name") for im in ET.parse(xmlp).getroot().findall("image") if im.findall("polygon")][2]
    crop,band=load(nm,prod,xmlp)
    cells=[(f"{prod} clean",outline(np.stack([crop]*3,-1),band))]
    for _ in range(2): cells.append((f"{prod} + pasted print (mask unchanged)",outline(paste(crop,band,prints,random.randint(2,4)),band)))
    cc2=[]
    for t,im in cells:
        im=cv2.resize(im,(420,int(im.shape[0]*420/im.shape[1]))); l=np.full((26,420,3),45,np.uint8); cv2.putText(l,t,(5,18),cv2.FONT_HERSHEY_SIMPLEX,0.46,(255,255,255),1); cc2.append(np.vstack([l,im]))
    Hc=max(c.shape[0] for c in cc2); cc2=[np.vstack([c,np.full((Hc-c.shape[0],c.shape[1],3),255,np.uint8)]) for c in cc2]
    rows.append(np.hstack(cc2))
Wm=max(r.shape[1] for r in rows); rows=[np.hstack([r,np.full((r.shape[0],Wm-r.shape[1],3),255,np.uint8)]) for r in rows]
cv2.imwrite(f"{ROOT}/outputs/print_paste_demo.png",np.vstack(rows)); print("wrote outputs/print_paste_demo.png")
