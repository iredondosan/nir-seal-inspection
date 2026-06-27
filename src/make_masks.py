import xml.etree.ElementTree as ET
import numpy as np, cv2, os

xml="/Users/nacho/Downloads/annotations 2.xml"
outdir="/Users/nacho/Downloads/seal_masks"
os.makedirs(outdir,exist_ok=True)
tree=ET.parse(xml); root=tree.getroot()

def parse_pts(s):
    return np.array([[float(a) for a in p.split(',')] for p in s.strip().split(';')],np.float32)
def area(p):
    return cv2.contourArea(p.astype(np.float32))

made=[]
for im in root.findall('image'):
    name=im.get('name'); W=int(im.get('width')); H=int(im.get('height'))
    polys=[parse_pts(pg.get('points')) for pg in im.findall('polygon')]
    if len(polys)<2:
        continue
    # outer = largest area, inner = next
    polys=sorted(polys,key=area,reverse=True)
    outer,inner=polys[0],polys[1]
    mask=np.zeros((H,W),np.uint8)
    cv2.fillPoly(mask,[outer.astype(np.int32)],255)      # fill outer
    cv2.fillPoly(mask,[inner.astype(np.int32)],0)        # subtract inner -> seal ring
    out=os.path.join(outdir, name.replace("_raw.png","_seal_mask.png"))
    cv2.imwrite(out,mask)
    made.append((name,out,int(area(outer)),int(area(inner)),int((mask>0).sum())))

for name,out,ao,ai,px in made:
    print(f"{name}: outer={ao}px inner={ai}px  seal={px}px  -> {os.path.basename(out)}")
print(f"\n{len(made)} masks written to {outdir}")
