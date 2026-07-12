#!/usr/bin/env python3
"""Quality metrics for the INT8 seal vs FP32: Dice + boundary (val set) and E2E AUROC (hold-out).
Each seal variant is run via ONNX at its own input resolution; the defect head stays FP32 (torch).
Isolates: quantization cost (int8_384 vs fp32_384) and absolute edge quality (int8_384 vs deployed fp32_1280)."""
import os, glob, random, numpy as np, cv2, torch
import xml.etree.ElementTree as ET
import onnxruntime as ort
from scipy.ndimage import distance_transform_edt, binary_erosion
from seal_inspection import core
from seal_inspection.paths import ROOT as R
from seal_inspection.results import save_results

dev = "cuda" if torch.cuda.is_available() else "cpu"
SEED=42; VAL_PER=2; THRESH=0.5; BDPX=5.0
MEAN=np.array((.485,.456,.406),np.float32); STD=np.array((.229,.224,.225),np.float32)
FORCE_TRAIN={"seal_1998_1780688689500_raw.png"}
DATASETS=[("data/annotations/prod2_reviewed.xml","data/images/prod2","prod2"),
          ("data/annotations/prod1_reviewed.xml","data/images/prod1","prod1"),
          ("data/annotations/prod3_reviewed.xml","data/images/prod3","prod3"),
          ("data/annotations/prod4_reviewed.xml","data/images/prod4","prod4"),
          ("data/annotations/prod5_reviewed.xml","data/images/prod5","prod5"),
          ("data/annotations/prod6_reviewed.xml","data/images/prod6","prod6")]
VARIANTS={"fp32_1280":("demo/models/seal.onnx",1280), "fp32_384":("demo/models/seal_384.onnx",384),
          "int8_384":("/tmp/seal384_int8.onnx",384),  "int8_1280":("/tmp/seal1280_int8.onnx",1280)}

def _path(p): return p if p.startswith("/") else f"{R}/{p}"
def _sess(p):
    so=ort.SessionOptions(); so.intra_op_num_threads=4
    return ort.InferenceSession(_path(p),so,providers=["CPUExecutionProvider"])
SESS={k:(_sess(p),res) for k,(p,res) in VARIANTS.items() if os.path.exists(_path(p))}
print("variants:",list(SESS))

def seal_prob(k,norm_u8):
    s,res=SESS[k]
    im=cv2.resize(np.stack([norm_u8]*3,-1),(res,res)).astype(np.float32)/255.0
    x=((im-MEAN)/STD).transpose(2,0,1)[None].astype(np.float32)
    z=s.run(None,{s.get_inputs()[0].name:x})[0][0,0]
    return 1.0/(1.0+np.exp(-z))

def bd(m):
    e=binary_erosion(m.astype(bool),iterations=1,border_value=0); return m.astype(bool)&~e
def metrics(gt,pr,bdpx=BDPX):
    gt=(gt>0).astype(np.uint8); pr=(pr>0).astype(np.uint8)
    inter=(gt&pr).sum(); gs=gt.sum(); ps=pr.sum()
    dice=2*inter/(gs+ps) if (gs+ps)>0 else 0.0
    bg=bd(gt); bp=bd(pr)
    if bg.sum()==0 or bp.sum()==0: return dict(dice=dice,biou=0.0,hd95=np.inf,assd=np.inf)
    dtg=distance_transform_edt(~bg); dtp=distance_transform_edt(~bp); d=np.concatenate([dtg[bp],dtp[bg]])
    Gb=gt.astype(bool)&(dtg<=bdpx); Pb=pr.astype(bool)&(dtp<=bdpx); bu=(Gb|Pb).sum()
    return dict(dice=float(dice),biou=float((Gb&Pb).sum()/bu) if bu>0 else 0.0,
                hd95=float(np.percentile(d,95)),assd=float(d.mean()))

def pp(s): return np.array([[float(a) for a in q.split(',')] for q in s.strip().split(';')],np.float32)
def tags_of(n): return {t.get('label') for t in n.findall('tag')}
def seal_mask(n):
    W=int(n.get('width')); H=int(n.get('height'))
    pl=[pp(pg.get('points')) for pg in n.findall('polygon') if pg.get('label')=='sellado']
    if len(pl)<2: return None
    pl=sorted(pl,key=lambda p:cv2.contourArea(p.astype(np.float32)),reverse=True)
    m=np.zeros((H,W),np.uint8); cv2.fillPoly(m,[pl[0].astype(np.int32)],1); cv2.fillPoly(m,[pl[1].astype(np.int32)],0); return m

# ---- val set (eval_seal logic) ----
val=[]
for xmlrel,imgrel,prod in DATASETS:
    samps=[]
    for node in ET.parse(f"{R}/{xmlrel}").getroot().findall('image'):
        nm=node.get('name'); p=f"{R}/{imgrel}/{nm}"; tg=tags_of(node)
        if "exclude" in tg or "reviewed" not in tg: continue
        m=seal_mask(node)
        if m is None or not os.path.exists(p): continue
        g=cv2.imread(p,cv2.IMREAD_GRAYSCALE); x0,y0,x1,y1=core.pack_bbox(g)
        samps.append((core.normalize(g[y0:y1,x0:x1]), m[y0:y1,x0:x1], nm))
    random.Random(SEED).shuffle(samps)
    rest=[s for s in samps if s[2] not in FORCE_TRAIN]
    val+=[(gc,mk,prod) for gc,mk,_ in rest[:(VAL_PER if len(rest)>=6 else 0)]]
print(f"val set: {len(val)} imgs")

# ---- Part A: Dice + boundary ----
seal_q={}
for k in SESS:
    _,res=SESS[k]; rows=[]
    for gc,mk,prod in val:
        prob=seal_prob(k,gc); mkr=cv2.resize(mk,(res,res),interpolation=cv2.INTER_NEAREST)
        rows.append(metrics(mkr,(prob>THRESH).astype(np.uint8)))
    ag=lambda key:float(np.mean([r[key] for r in rows if np.isfinite(r[key])]))
    seal_q[k]={"res":res,"dice":round(ag("dice"),4),"biou":round(ag("biou"),3),"hd95":round(ag("hd95"),2),"assd":round(ag("assd"),2),"n":len(rows)}
    print(f"  SEAL {k}: dice={seal_q[k]['dice']} biou={seal_q[k]['biou']} hd95={seal_q[k]['hd95']} assd={seal_q[k]['assd']}")

# ---- Part B: E2E AUROC (torch defect head) ----
defm,dk=core.load_unet(f"{R}/models/defect_strip.pt",dev); HS,WS=dk["HS"],dk["WS"]
def dscore(strip):
    x=((np.stack([strip]*3,-1)/255.0-MEAN)/STD).transpose(2,0,1)[None].astype(np.float32)
    with torch.no_grad(): p=torch.sigmoid(defm(torch.from_numpy(x).to(dev)))[0,0].cpu().numpy()
    return float(cv2.GaussianBlur(p,(0,0),2).max())
lab={}
for ln in open(f"{R}/data/holdout_labels.csv").read().splitlines()[1:]:
    nm,l=ln.split(","); lab[nm]=int(l)

e2e_q={}
for k in SESS:
    scores,labels=[],[]
    for nm,l in lab.items():
        hits=glob.glob(f"{R}/data/images/*/{nm}.png")+glob.glob(f"{R}/data/images/*/{nm}.jpg")
        if not hits: continue
        g=cv2.imread(hits[0],cv2.IMREAD_GRAYSCALE); H,W=g.shape; x0,y0,x1,y1=core.pack_bbox(g)
        prob=seal_prob(k,core.normalize(g[y0:y1,x0:x1]))
        full=np.zeros((H,W),np.float32); full[y0:y1,x0:x1]=cv2.resize(prob,(x1-x0,y1-y0))
        O,I=core.mask_to_ring((full>0.5).astype(np.uint8)*255)
        if O is None: continue
        mx,my=core.unroll_maps(O,I,HS,WS)
        strip=cv2.remap(core.normalize(g),mx,my,cv2.INTER_LINEAR,borderValue=0)
        scores.append(dscore(strip)); labels.append(l)
    scores,labels=np.array(scores),np.array(labels); pos,neg=scores[labels==1],scores[labels==0]
    au=float(np.mean([(x>y)+0.5*(x==y) for x in pos for y in neg]))
    tp=int(((scores>=0.5)&(labels==1)).sum()); fp=int(((scores>=0.5)&(labels==0)).sum())
    e2e_q[k]={"auroc":round(au,3),"n_localised":len(scores),"recall":f"{tp}/{int((labels==1).sum())}","fp":f"{fp}/{int((labels==0).sum())}"}
    print(f"  E2E {k}: AUROC={au:.3f} recall={e2e_q[k]['recall']} fp={e2e_q[k]['fp']} n={len(scores)}")

save_results("int8_quality",{
  "note":"Seal via ONNX at its own resolution; defect head FP32 (torch). Dice/boundary on the reviewed val set "
         "(same as tab:dice-producto); E2E AUROC on the leakage-free hold-out. int8_384 vs fp32_384 = quantization "
         "cost; vs fp32_1280 = deployed baseline. Static QDQ INT8 (deploy/quantize_int8.py).",
  "seal_dice_boundary":seal_q,"e2e_auroc":e2e_q,"bdpx":BDPX})
print("saved results/int8_quality.json")
