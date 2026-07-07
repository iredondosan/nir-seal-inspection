#!/usr/bin/env python3
"""Regenerate fig_defect_thresholds: ROC + metrics-vs-threshold + confusion matrix at the
operating point, from the deployed model's END-TO-END scores on the 179-pack hold-out."""
import glob, os, shutil, numpy as np, cv2, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from seal_inspection import core
R = "/home/ubuntu/TFM/seal-inspection"; dev = "cuda" if torch.cuda.is_available() else "cpu"
MEAN, STD = core.IMAGENET_MEAN, core.IMAGENET_STD; OP = 0.5
seal, sk = core.load_unet(f"{R}/models/best_lite_reviewed_1280.pt", dev)
defm, dk = core.load_unet(f"{R}/models/defect_strip.pt", dev); HS, WS = dk["HS"], dk["WS"]
lab = {ln.split(",")[0]: int(ln.split(",")[1]) for ln in open(f"{R}/data/holdout_labels.csv").read().splitlines()[1:]}
def dsc(strip):
    x = ((np.stack([strip]*3, -1)/255.0 - MEAN)/STD).transpose(2, 0, 1)[None].astype(np.float32)
    with torch.no_grad(): p = torch.sigmoid(defm(torch.from_numpy(x).to(dev)))[0, 0].cpu().numpy()
    return float(cv2.GaussianBlur(p, (0, 0), 2).max())
S, L = [], []
for nm, l in lab.items():
    h = glob.glob(f"{R}/data/images/*/{nm}.png") + glob.glob(f"{R}/data/images/*/{nm}.jpg")
    if not h: continue
    g = cv2.imread(h[0], 0); H, W = g.shape; x0, y0, x1, y1 = core.pack_bbox(g)
    prob = core.predict_probability(seal, g[y0:y1, x0:x1], sk["img"], dev)
    full = np.zeros((H, W), np.float32); full[y0:y1, x0:x1] = cv2.resize(prob, (x1-x0, y1-y0))
    O, I = core.mask_to_ring((full > sk.get("thresh", .5)).astype(np.uint8)*255)
    if O is None: continue
    mx, my = core.unroll_maps(O, I, HS, WS)
    S.append(dsc(cv2.remap(core.normalize(g), mx, my, cv2.INTER_LINEAR, borderValue=0))); L.append(l)
S, L = np.array(S), np.array(L); nd, ng = int(L.sum()), int((L == 0).sum())
pos, neg = S[L == 1], S[L == 0]
au = float(np.mean([(a > b)+0.5*(a == b) for a in pos for b in neg]))
ts = np.linspace(0, 1, 200)
tpr = [(pos >= t).mean() for t in ts]; fpr = [(neg >= t).mean() for t in ts]
prec = [((S >= t) & (L == 1)).sum()/max(1, (S >= t).sum()) for t in ts]
rec = [(pos >= t).mean() for t in ts]
f1 = [2*p*r/(p+r) if p+r else 0 for p, r in zip(prec, rec)]
tp = int(((S >= OP) & (L == 1)).sum()); fp = int(((S >= OP) & (L == 0)).sum()); fn = nd-tp; tn = ng-fp

fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
ax[0].plot(fpr, tpr, lw=2, color="#1f77b4"); ax[0].plot([0, 1], [0, 1], "--", color="gray", lw=1)
ax[0].set_title(f"Curva ROC (AUROC = {au:.3f})"); ax[0].set_xlabel("Tasa de falsos positivos"); ax[0].set_ylabel("Sensibilidad"); ax[0].grid(alpha=.3)
ax[1].plot(ts, prec, label="Precisión", color="#ff7f0e"); ax[1].plot(ts, rec, label="Sensibilidad", color="#2ca02c"); ax[1].plot(ts, f1, label="F1", color="#9467bd")
ax[1].axvline(OP, ls="--", color="gray", lw=1); ax[1].set_title("Métricas frente al umbral"); ax[1].set_xlabel("Umbral"); ax[1].legend(); ax[1].grid(alpha=.3)
cm = np.array([[tn, fp], [fn, tp]])
ax[2].imshow(cm, cmap="Blues"); ax[2].set_title(f"Matriz de confusión (umbral {OP})")
ax[2].set_xticks([0, 1]); ax[2].set_xticklabels(["Pred. correcto", "Pred. defecto"]); ax[2].set_yticks([0, 1]); ax[2].set_yticklabels(["Correcto", "Defecto"])
for i in range(2):
    for j in range(2):
        ax[2].text(j, i, cm[i, j], ha="center", va="center", fontsize=14, color="black")
plt.tight_layout()
out = f"{R}/docs/thesis_figures/fig_defect_thresholds.png"
if os.path.exists(out): shutil.copy2(out, out + ".prehullfix.bak")
plt.savefig(out, dpi=130); print(f"wrote fig_defect_thresholds  AUROC={au:.3f}  op@{OP}: TP{tp} FP{fp} TN{tn} FN{fn}")
