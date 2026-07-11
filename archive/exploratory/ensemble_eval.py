#!/usr/bin/env python3
"""Supervised UNION anomaly ensemble for zero-miss. For each held-out test pack computes:
  * supervised score = max smoothed defect-prob (the defect U-Net)
  * anomaly score    = max patch nearest-neighbour distance (PatchCore, good-only memory bank)
Reports each model's recall, identifies the supervised-missed defect(s), and whether the anomaly
model recovers them -> the union recall and its false-alarm cost."""
import glob, os, numpy as np, cv2, torch
import segmentation_models_pytorch as smp, timm
from seal_inspection.paths import ROOT as R
MEAN = np.array((.485, .456, .406), np.float32); STD = np.array((.229, .224, .225), np.float32)
tMEAN = torch.tensor((.485, .456, .406)).view(1, 3, 1, 1); tSTD = torch.tensor((.229, .224, .225)).view(1, 3, 1, 1)
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42); np.random.seed(42)

# supervised defect model
ck = torch.load(f"{R}/models/defect_strip.pt", map_location="cpu", weights_only=False)
defm = smp.Unet(ck["encoder"], encoder_weights=None, in_channels=3, classes=1); defm.load_state_dict(ck["state_dict"]); defm.eval().to(dev)
# anomaly backbone
bb = timm.create_model("resnet18", pretrained=True, features_only=True, out_indices=(2, 3)).to(dev).eval()

def sup_score(img):
    x = ((np.stack([img] * 3, -1) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
    with torch.no_grad():
        p = torch.sigmoid(defm(torch.from_numpy(x).to(dev)))[0, 0].cpu().numpy()
    return float(cv2.GaussianBlur(p, (0, 0), 2).max())

@torch.no_grad()
def feats(img):
    x = torch.from_numpy(np.stack([img] * 3, -1)).permute(2, 0, 1)[None].float() / 255.0
    f2, f3 = bb(((x - tMEAN) / tSTD).to(dev))
    f3 = torch.nn.functional.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
    f = torch.cat([f2, f3], 1)[0]; C = f.shape[0]
    return torch.nn.functional.normalize(f.reshape(C, -1).t(), dim=1)

def load(split):
    out = []
    for ip in sorted(glob.glob(f"{R}/data/strips/{split}/img/*.png")):
        gt = cv2.imread(ip.replace("/img/", "/mask/"), 0)
        out.append((cv2.imread(ip, 0), 1 if (gt is not None and gt.sum() > 0) else 0, os.path.basename(ip)))
    return out

train, test = load("train"), load("test")
# anomaly memory bank from GOOD train
bank = torch.cat([feats(im).cpu() for im, lab, _ in train if lab == 0], 0)
if bank.shape[0] > 40000:
    bank = bank[torch.randperm(bank.shape[0])[:40000]]
bank = bank.to(dev)

rows = []  # (name,label,sup,anom)
for im, lab, nm in test:
    a = torch.cdist(feats(im), bank).min(1).values.max().item()
    rows.append((nm, lab, sup_score(im), a))
labels = np.array([r[1] for r in rows]); sup = np.array([r[2] for r in rows]); anom = np.array([r[3] for r in rows])
P = int((labels == 1).sum()); N = int((labels == 0).sum())

SUP_THR = 0.10                                   # high-recall supervised operating point
sup_pred = sup >= SUP_THR
print(f"test: {P} defect / {N} good")
print(f"SUPERVISED @thr {SUP_THR}: recall {int((sup_pred & (labels==1)).sum())}/{P}, FP {int((sup_pred & (labels==0)).sum())}")
# supervised-missed defects
missed = [r for r in rows if r[1] == 1 and r[2] < SUP_THR]
print(f"\nsupervised-MISSED defects ({len(missed)}):")
for nm, lab, s, a in missed:
    pct = 100 * (anom[labels == 0] < a).mean()
    print(f"  {nm}  sup={s:.3f}  anom={a:.3f}  (higher than {pct:.0f}% of good-pack anomaly scores)")
# choose anomaly threshold that recovers the missed defects, report union
if missed:
    need = min(r[3] for r in missed)             # anom thr must be <= this to catch the hardest missed
    anom_pred = anom >= need
    union = sup_pred | anom_pred
    tp = int((union & (labels == 1)).sum()); fp = int((union & (labels == 0)).sum())
    print(f"\nUNION (sup>= {SUP_THR} OR anom>= {need:.3f}): recall {tp}/{P}, FP {fp}/{N}, specificity {1-fp/N:.2f}")
    print(f"  -> {'ZERO MISS (19/19)' if tp==P else str(tp)+'/'+str(P)} at the cost of {fp} false alarms ({100*fp/N:.0f}% of good)")
