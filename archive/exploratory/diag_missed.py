#!/usr/bin/env python3
"""Find the hold-out defects the clean ensemble misses, and diagnose WHY:
is the defect even inside the predicted seal band? is it visible in the unrolled strips?"""
import glob, os, numpy as np, cv2, torch
from seal_inspection import core, cvat
R = "/home/ubuntu/TFM/seal-inspection"; DTHR = 0.43
dev = "cuda" if torch.cuda.is_available() else "cpu"
seal, sk = core.load_unet(f"{R}/models/best_lite_reviewed_1280.pt", dev)
defm, dk = core.load_unet(f"{R}/models/defect_strip.pt", dev)          # perpendicular
legm, lk = core.load_unet(f"{R}/models/defect_strip.prev.pt", dev)     # legacy
HS, WS = dk["HS"], dk["WS"]; MEAN, STD = core.IMAGENET_MEAN, core.IMAGENET_STD

SRC = [("data/annotations/prod1_reviewed.xml", "data/images/prod1"),
       ("data/annotations/prod2_reviewed.xml", "data/images/prod2"),
       ("data/annotations/prod3_reviewed.xml", "data/images/prod3"),
       ("data/annotations/prod4_reviewed.xml", "data/images/prod4"),
       ("data/annotations/prod5_reviewed.xml", "data/images/prod5")]
node_of = {}
for xml, imgdir in SRC:
    for im in cvat.iter_images(f"{R}/{xml}"):
        node_of[os.path.splitext(im.get("name"))[0]] = (im, imgdir)

defect_names = [ln.split(",")[0] for ln in open(f"{R}/data/holdout_labels.csv").read().splitlines()[1:]
                if ln.split(",")[1] == "1"]

def score(model, strip):
    x = ((np.stack([strip]*3,-1)/255.0 - MEAN)/STD).transpose(2,0,1)[None].astype(np.float32)
    with torch.no_grad():
        p = torch.sigmoid(model(torch.from_numpy(x).to(dev)))[0,0].cpu().numpy()
    return float(cv2.GaussianBlur(p,(0,0),2).max()), p

rows = []
for nm in defect_names:
    node, imgdir = node_of[nm]
    g = cv2.imread(f"{R}/{imgdir}/{node.get('name')}", cv2.IMREAD_GRAYSCALE)
    if g is None: continue
    H, W = g.shape
    x0,y0,x1,y1 = core.pack_bbox(g)
    prob = core.predict_probability(seal, g[y0:y1,x0:x1], sk["img"], dev)
    full = np.zeros((H,W),np.float32); full[y0:y1,x0:x1] = cv2.resize(prob,(x1-x0,y1-y0))
    op, ip = core.mask_to_ring((full>sk.get("thresh",.5)).astype(np.uint8)*255)
    if op is None: continue
    d2d = np.zeros((H,W),np.uint8)
    for d in cvat.polygons(node,"defect")+cvat.polygons(node,"liquid"): cv2.fillPoly(d2d,[d.astype(np.int32)],255)
    band = core.polygons_to_band_mask(op, ip, H, W)
    inb = (d2d>0)&(band>0); inb_frac = inb.sum()/max(1,(d2d>0).sum())
    mxp,myp = core.unroll_maps(op,ip,HS,WS); sp_perp,_ = score(defm, cv2.remap(core.normalize(g),mxp,myp,cv2.INTER_LINEAR,borderValue=0))
    mxl,myl = core.unroll_maps_legacy(op,ip,HS,WS); sp_leg,_ = score(legm, cv2.remap(core.normalize(g),mxl,myl,cv2.INTER_LINEAR,borderValue=0))
    ens = max(sp_perp, sp_leg)
    rows.append((nm, sp_perp, sp_leg, ens, inb_frac, g, op, ip, d2d, mxp, myp, mxl, myl, imgdir, node.get("name")))

rows.sort(key=lambda r: r[3])
print(f"{'pack':40} {'perp':>6} {'legacy':>7} {'ens':>6} {'defect-in-pred-band%':>20}")
for r in rows:
    print(f"{r[0][:40]:40} {r[1]:6.3f} {r[2]:7.3f} {r[3]:6.3f} {100*r[4]:19.0f}%")

# diagnostic composite for the 3 lowest-scoring (the misses)
os.makedirs(f"{R}/outputs/missed", exist_ok=True)
for r in rows[:3]:
    nm, sp_perp, sp_leg, ens, inb, g, op, ip, d2d, mxp, myp, mxl, myl, imgdir, fname = r
    H,W = g.shape
    bx,by,bw,bh = cv2.boundingRect(op.astype(np.int32)); pad=50
    px0,py0,px1,py1 = max(0,bx-pad),max(0,by-pad),min(W,bx+bw+pad),min(H,by+bh+pad)
    panel = cv2.cvtColor(core.normalize(g[py0:py1,px0:px1]),cv2.COLOR_GRAY2BGR)
    band = core.polygons_to_band_mask(op,ip,H,W)[py0:py1,px0:px1]
    panel[band>0] = np.clip(0.7*panel[band>0]+np.array([60,60,0]),0,255).astype(np.uint8)
    panel[d2d[py0:py1,px0:px1]>0] = (0,230,0)
    pw=900; panel = cv2.resize(panel,(pw,int(panel.shape[0]*pw/panel.shape[1])))
    def strip_def(mx,my):
        s = cv2.remap(core.normalize(g),mx,my,cv2.INTER_LINEAR,borderValue=0)
        dm = cv2.remap(d2d,mx,my,cv2.INTER_NEAREST,borderValue=0)
        v = cv2.cvtColor(s,cv2.COLOR_GRAY2BGR); v[dm>127]=(0,230,0)
        return cv2.resize(v,(pw,int(v.shape[0]*pw/v.shape[1])))
    def banner(t):
        b=np.full((26,pw,3),35,np.uint8); cv2.putText(b,t,(8,18),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1); return b
    comp = np.vstack([banner(f"{nm[:38]}  ens={ens:.2f} (perp {sp_perp:.2f}/leg {sp_leg:.2f})  defect-in-band {100*inb:.0f}%  [GT defect=green]"),
                      panel, banner("perpendicular strip + GT defect (green)"), strip_def(mxp,myp),
                      banner("legacy strip + GT defect (green)"), strip_def(mxl,myl)])
    cv2.imwrite(f"{R}/outputs/missed/{nm}.png", comp)
print("wrote 3 diagnostics to outputs/missed/")
