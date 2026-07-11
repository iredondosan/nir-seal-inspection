#!/usr/bin/env python3
"""Ablation: ImageNet-pretrained vs from-scratch, evaluated on the SAME global hold-out.
Compares per-stage (seal val Dice, defect GT-strip AUROC) and single-branch (perpendicular)
end-to-end (predicted seal -> unroll -> defect score) for both initialisations."""
import glob, os, numpy as np, cv2, torch
from seal_inspection import core
from seal_inspection.paths import ROOT as R; DTHR = 0.43
dev = "cuda" if torch.cuda.is_available() else "cpu"

im_seal, im_sk = core.load_unet(f"{R}/models/best_lite_reviewed_1280.pt", dev)
im_def, im_dk = core.load_unet(f"{R}/models/defect_strip.pt", dev)
sc_seal, sc_sk = core.load_unet(f"{R}/models/scratch_seal_1280.pt", dev)
sc_def, sc_dk = core.load_unet(f"{R}/models/scratch_defect_perp.pt", dev)
HS, WS = im_dk["HS"], im_dk["WS"]; MEAN, STD = core.IMAGENET_MEAN, core.IMAGENET_STD

def dscore(model, strip):
    x = ((np.stack([strip]*3,-1)/255.0 - MEAN)/STD).transpose(2,0,1)[None].astype(np.float32)
    with torch.no_grad():
        p = torch.sigmoid(model(torch.from_numpy(x).to(dev)))[0,0].cpu().numpy()
    return float(cv2.GaussianBlur(p,(0,0),2).max())

def e2e_score(seal, sk, defm, g):
    H,W = g.shape; x0,y0,x1,y1 = core.pack_bbox(g)
    prob = core.predict_probability(seal, g[y0:y1,x0:x1], sk["img"], dev)
    full = np.zeros((H,W),np.float32); full[y0:y1,x0:x1] = cv2.resize(prob,(x1-x0,y1-y0))
    O,I = core.mask_to_ring((full>sk.get("thresh",.5)).astype(np.uint8)*255)
    if O is None: return None
    mx,my = core.unroll_maps(O,I,HS,WS)
    return dscore(defm, cv2.remap(core.normalize(g),mx,my,cv2.INTER_LINEAR,borderValue=0))

# hold-out labels
labrows = [ln.split(",") for ln in open(f"{R}/data/holdout_labels.csv").read().splitlines()[1:]]
def find_raw(nm):
    for e in (".png",".jpg"):
        h = glob.glob(f"{R}/data/images/*/{nm}{e}")
        if h: return h[0]
    return None

im_s, sc_s, labels = [], [], []
for nm, lb in labrows:
    rp = find_raw(nm)
    if rp is None: continue
    g = cv2.imread(rp, cv2.IMREAD_GRAYSCALE)
    a = e2e_score(im_seal, im_sk, im_def, g); b = e2e_score(sc_seal, sc_sk, sc_def, g)
    if a is None or b is None: continue
    im_s.append(a); sc_s.append(b); labels.append(int(lb))
im_s, sc_s, labels = np.array(im_s), np.array(sc_s), np.array(labels)

def stats(s):
    pos, neg = s[labels==1], s[labels==0]
    au = float(np.mean([(x>y)+0.5*(x==y) for x in pos for y in neg]))
    tp = int(((s>=DTHR)&(labels==1)).sum()); fp = int(((s>=DTHR)&(labels==0)).sum())
    return au, tp, fp, float(pos.min())

print(f"hold-out: {int((labels==1).sum())} defect / {int((labels==0).sum())} good   (single-branch perpendicular end-to-end)\n")
print(f"{'init':12} {'seal valDice':>12} {'E2E AUROC':>10} {'recall@.43':>11} {'FP@.43':>7} {'min-def':>8}")
for tag, seal_ck, s in [("ImageNet", im_sk, im_s), ("from-scratch", sc_sk, sc_s)]:
    au, tp, fp, mn = stats(s)
    vd = float(seal_ck.get("val_dice", float("nan")))
    print(f"{tag:12} {vd:12.3f} {au:10.3f} {str(tp)+'/'+str(int((labels==1).sum())):>11} {str(fp)+'/'+str(int((labels==0).sum())):>7} {mn:8.3f}")
