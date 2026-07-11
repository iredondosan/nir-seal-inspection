#!/usr/bin/env python3
"""2-unroll ensemble. Per pack: score with NEW model on NEW (perp) strip AND OLD model on
OLD (correspondence) strip, max-pool. Old unroll reconstructed locally (matches the prev model).
Reports recall/FP/AUROC for OLD-only, NEW-only, and MAX ensemble, plus the weakest defects."""
import glob, os, numpy as np, cv2, torch
from seal_inspection import core
R = "/home/ubuntu/TFM/seal-inspection"
dev = "cuda" if torch.cuda.is_available() else "cpu"
seal, sk = core.load_unet(f"{R}/models/best_lite_reviewed_1280.pt", dev)
newm, dk = core.load_unet(f"{R}/models/defect_strip.pt", dev)
oldm, _  = core.load_unet(f"{R}/models/defect_strip.prev.pt", dev)
HS, WS = dk["HS"], dk["WS"]; MEAN, STD = core.IMAGENET_MEAN, core.IMAGENET_STD

def old_unroll(O, I):
    Or = core._resample_closed(O, WS); Ir = core._resample_closed(I, WS)
    if core._is_ccw(Or) != core._is_ccw(Ir): Ir = Ir[::-1]
    j = int(np.argmin(np.hypot(Ir[:, 0] - Or[0, 0], Ir[:, 1] - Or[0, 1]))); Ir = np.roll(Ir, -j, 0)
    for arr in (Or, Ir): arr[:, 0] = core._smooth_closed(arr[:, 0]); arr[:, 1] = core._smooth_closed(arr[:, 1])
    a = np.linspace(-0.15, 1.15, HS)[:, None]
    return ((Or[:, 0][None] * (1 - a) + Ir[:, 0][None] * a).astype(np.float32),
            (Or[:, 1][None] * (1 - a) + Ir[:, 1][None] * a).astype(np.float32))

def dscore(model, strip):
    x = ((np.stack([strip] * 3, -1) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
    with torch.no_grad():
        p = torch.sigmoid(model(torch.from_numpy(x).to(dev)))[0, 0].cpu().numpy()
    return float(cv2.GaussianBlur(p, (0, 0), 2).max())

old_s, new_s, labels, names = [], [], [], []
for ip in sorted(glob.glob(f"{R}/data/strips/test/img/*.png")):
    nm = os.path.splitext(os.path.basename(ip))[0]
    hits = glob.glob(f"{R}/data/images/*/{nm}.png") + glob.glob(f"{R}/data/images/*/{nm}.jpg")
    if not hits: continue
    g = cv2.imread(hits[0], cv2.IMREAD_GRAYSCALE); H, W = g.shape
    x0, y0, x1, y1 = core.pack_bbox(g)
    prob = core.predict_probability(seal, g[y0:y1, x0:x1], sk["img"], dev)
    full = np.zeros((H, W), np.float32); full[y0:y1, x0:x1] = cv2.resize(prob, (x1 - x0, y1 - y0))
    O, I = core.mask_to_ring((full > sk.get("thresh", .5)).astype(np.uint8) * 255)
    if O is None: continue
    mxn, myn = core.unroll_maps(O, I, HS, WS)             # new perp
    mxo, myo = old_unroll(O, I)                            # old correspondence
    gimg = core.normalize(g)
    new_s.append(dscore(newm, cv2.remap(gimg, mxn, myn, cv2.INTER_LINEAR, borderValue=0)))
    old_s.append(dscore(oldm, cv2.remap(gimg, mxo, myo, cv2.INTER_LINEAR, borderValue=0)))
    gt = cv2.imread(ip.replace("/img/", "/mask/"), 0); labels.append(1 if gt.sum() > 0 else 0); names.append(nm)
old_s, new_s, labels = np.array(old_s), np.array(new_s), np.array(labels)
ens = np.maximum(old_s, new_s)

def report(s, tag):
    pos, neg = s[labels == 1], s[labels == 0]
    au = float(np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg]))
    line = f"{tag:8s} AUROC {au:.3f} | "
    for thr in [0.43, 0.5, 0.7, 0.85]:
        tp = int(((s >= thr) & (labels == 1)).sum()); fp = int(((s >= thr) & (labels == 0)).sum())
        line += f"@{thr}: {tp}/{int((labels==1).sum())},FP{fp} | "
    print(line + f"min-defect {np.sort(pos)[0]:.3f}")

report(old_s, "OLD")
report(new_s, "NEW")
report(ens, "MAX")
for tag, s in [("OLD", old_s), ("NEW", new_s), ("MAX", ens)]:
    i = names.index("seal_2260_1780692167999_raw"); print(f"seal_2260 {tag}: {s[i]:.3f}")
idx = np.where(labels == 1)[0]
print("weakest defects (ensemble):")
for k in idx[np.argsort(ens[idx])][:5]:
    print(f"  {names[k][:38]:38s} old {old_s[k]:.3f}  new {new_s[k]:.3f}  ens {ens[k]:.3f}")
