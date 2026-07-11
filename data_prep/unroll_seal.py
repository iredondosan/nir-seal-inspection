import xml.etree.ElementTree as ET
import numpy as np, cv2, os

XML="/Users/nacho/Downloads/annotations 2.xml"
BASE="data/images/prod6"   # set to your pack-image folder
OUT="/Users/nacho/Downloads/seal_strips"; os.makedirs(OUT,exist_ok=True)
Hs,Ws=128,1024   # fixed output height (across seal) x width (perimeter)

def norm(img):
    lo,hi=np.percentile(img,[1,99.5]); hi=max(hi,lo+1)
    return np.clip((img.astype(np.float32)-lo)/(hi-lo)*255,0,255).astype(np.uint8)
def parse(s): return np.array([[float(a) for a in p.split(',')] for p in s.strip().split(';')],np.float32)
def resample(poly,n):
    p=np.r_[poly,poly[:1]]
    d=np.r_[0,np.cumsum(np.hypot(*np.diff(p,axis=0).T))]
    t=np.linspace(0,d[-1],n,endpoint=False)
    return np.stack([np.interp(t,d,p[:,0]),np.interp(t,d,p[:,1])],1)
def ccw(p): return cv2.contourArea(p.astype(np.float32),oriented=True)>0

root=ET.parse(XML).getroot()
previews=[]
for im in root.findall('image'):
    name=im.get('name'); polys=[parse(pg.get('points')) for pg in im.findall('polygon')]
    if len(polys)<2: continue
    polys=sorted(polys,key=lambda p:cv2.contourArea(p.astype(np.float32)),reverse=True)
    outer,inner=polys[0],polys[1]
    # make both same winding so point i<->i correspond
    if ccw(outer)!=ccw(inner): inner=inner[::-1]
    O=resample(outer,Ws); I=resample(inner,Ws)
    img=cv2.imread(f"{BASE}/{name}",0)
    # radial sample: for each column, H points from outer->inner
    a=np.linspace(0,1,Hs)[:,None]                       # (Hs,1)
    mapx=(O[:,0][None,:]*(1-a)+I[:,0][None,:]*a).astype(np.float32)  # (Hs,Ws)
    mapy=(O[:,1][None,:]*(1-a)+I[:,1][None,:]*a).astype(np.float32)
    strip=cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR,borderValue=0)
    out=f"{OUT}/{name.replace('_raw.png','_seal_strip.png')}"
    cv2.imwrite(out,strip)                               # raw values, fixed HsxWs
    cv2.imwrite(out.replace('.png','_view.png'),norm(strip))
    print(f"{name}: strip {strip.shape[1]}x{strip.shape[0]} -> {os.path.basename(out)}")
    previews.append(norm(strip))

stack=np.vstack(sum([[p,np.full((6,Ws),255,np.uint8)] for p in previews],[])[:-1])
cv2.imwrite("strips_preview.png",stack)
print("\nwrote",len(previews),"strips to",OUT)
