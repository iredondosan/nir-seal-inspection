#!/usr/bin/env python3
"""Image-level (pack pass/fail) threshold sweep for the defect model on the held-out test set.
Pack score = max of the smoothed defect-probability map. Prints confusion matrix + metrics per
threshold and AUROC; writes per-pack (score,label) to a CSV for plotting."""
import glob, os, numpy as np, cv2, torch
import segmentation_models_pytorch as smp
from seal_inspection.paths import ROOT as R
ck = torch.load(f"{R}/models/defect_strip.pt", map_location="cpu", weights_only=False)
m = smp.Unet(ck["encoder"], encoder_weights=None, in_channels=3, classes=1); m.load_state_dict(ck["state_dict"]); m.eval()
MEAN = np.array((.485, .456, .406), np.float32); STD = np.array((.229, .224, .225), np.float32)
dev = "cuda" if torch.cuda.is_available() else "cpu"; m = m.to(dev)

scores, labels = [], []
for ip in sorted(glob.glob(f"{R}/data/strips/test/img/*.png")):
    gt = cv2.imread(ip.replace("/img/", "/mask/"), cv2.IMREAD_GRAYSCALE)
    s = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
    x = ((np.stack([s] * 3, -1) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
    with torch.no_grad():
        p = torch.sigmoid(m(torch.from_numpy(x).to(dev)))[0, 0].cpu().numpy()
    scores.append(float(cv2.GaussianBlur(p, (0, 0), 2).max()))
    labels.append(1 if (gt is not None and gt.sum() > 0) else 0)
scores, labels = np.array(scores), np.array(labels)
P, N = int((labels == 1).sum()), int((labels == 0).sum())

# AUROC (Mann-Whitney) + AUPRC
pos, neg = scores[labels == 1], scores[labels == 0]
auroc = float(np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg]))
print(f"test set: {P} defect / {N} good   AUROC = {auroc:.3f}\n")
print(f"{'thr':>5} {'TP':>3} {'FP':>3} {'TN':>4} {'FN':>3} {'Prec':>6} {'Recall':>7} {'Spec':>6} {'F1':>6} {'Acc':>6}")
best = (0, None)
for thr in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]:
    pred = (scores >= thr).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum()); fp = int(((pred == 1) & (labels == 0)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum()); fn = int(((pred == 0) & (labels == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / len(labels)
    print(f"{thr:5.2f} {tp:3d} {fp:3d} {tn:4d} {fn:3d} {prec:6.2f} {rec:7.2f} {spec:6.2f} {f1:6.2f} {acc:6.3f}")
    if f1 > best[0]:
        best = (f1, thr)
print(f"\nbest-F1 threshold = {best[1]:.2f} (F1 {best[0]:.2f})")
# MAX threshold that still catches ALL defects (recall = 1.0)
thr0 = float(pos.min())
fp0 = int((neg >= thr0).sum())
print(f"\nZERO-MISS: max threshold with recall=1.0 is {thr0:.3f} (the lowest defect score)")
print(f"  @thr={thr0:.3f}: TP {P}/{P}, FN 0, FP {fp0}/{N}, specificity {(1-fp0/N):.2f}, precision {P/(P+fp0):.2f}")
print(f"  -> flags all {P} defects + {fp0} good packs ({100*fp0/N:.0f}% false-alarm rate)")
print(f"  lowest 3 defect scores: {[round(x,3) for x in np.sort(pos)[:3]]}")
with open(f"{R}/outputs/defect_scores.csv", "w") as f:
    f.write("score,label\n")
    for s, l in zip(scores, labels):
        f.write(f"{s:.5f},{l}\n")
print("wrote outputs/defect_scores.csv")
