#!/usr/bin/env python3
"""Evaluate unroll-TTA END-TO-END over the test set: predicted seal -> {baseline, jittered} unrolls,
max-pool the defect score. Compare baseline vs TTA (AUROC + recall/FP @0.43) and check seal_2260."""
import glob, os, numpy as np, cv2, torch
from seal_inspection import core, cvat
R = "/home/ubuntu/TFM/seal-inspection"
dev = "cuda" if torch.cuda.is_available() else "cpu"
seal, sk = core.load_unet(f"{R}/models/best_lite_reviewed_1280.pt", dev)
defm, dk = core.load_unet(f"{R}/models/defect_strip.pt", dev)
HS, WS = dk["HS"], dk["WS"]; MEAN = core.IMAGENET_MEAN; STD = core.IMAGENET_STD
np.random.seed(0)
VARIANTS = [(0, 1.15, 0.0), (8, 1.15, 1.5), (-8, 1.15, 1.5), (0, 1.20, 2.0), (6, 1.20, 2.0), (-6, 1.25, 2.0)]

def punroll(g, O, I, roll, a_hi, jit):
    Or = core._resample_closed(O, WS); Ir = core._resample_closed(I, WS)
    if core._is_ccw(Or) != core._is_ccw(Ir): Ir = Ir[::-1]
    j = int(np.argmin(np.hypot(Ir[:, 0] - Or[0, 0], Ir[:, 1] - Or[0, 1])))
    Ir = np.roll(np.roll(Ir, -j, 0), roll, 0)
    for arr in (Or, Ir):
        arr[:, 0] = core._smooth_closed(arr[:, 0]); arr[:, 1] = core._smooth_closed(arr[:, 1])
    if jit:
        Or = Or + np.random.randn(*Or.shape) * jit; Ir = Ir + np.random.randn(*Ir.shape) * jit
    a = np.linspace(-0.15, a_hi, HS)[:, None]
    mx = (Or[:, 0][None] * (1 - a) + Ir[:, 0][None] * a).astype(np.float32)
    my = (Or[:, 1][None] * (1 - a) + Ir[:, 1][None] * a).astype(np.float32)
    return cv2.remap(core.normalize(g), mx, my, cv2.INTER_LINEAR, borderValue=0)

def dscore(strip):
    x = ((np.stack([strip] * 3, -1) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
    with torch.no_grad():
        p = torch.sigmoid(defm(torch.from_numpy(x).to(dev)))[0, 0].cpu().numpy()
    return float(cv2.GaussianBlur(p, (0, 0), 2).max())

base, tta, labels, names = [], [], [], []
for ip in sorted(glob.glob(f"{R}/data/strips/test/img/*.png")):
    nm = os.path.splitext(os.path.basename(ip))[0]
    hits = glob.glob(f"{R}/data/images/*/{nm}.png") + glob.glob(f"{R}/data/images/*/{nm}.jpg")
    if not hits: continue
    g = cv2.imread(hits[0], cv2.IMREAD_GRAYSCALE); H, W = g.shape
    x0, y0, x1, y1 = core.pack_bbox(g); prob = core.predict_probability(seal, g[y0:y1, x0:x1], sk["img"], dev)
    full = np.zeros((H, W), np.float32); full[y0:y1, x0:x1] = cv2.resize(prob, (x1 - x0, y1 - y0))
    O, I = core.mask_to_ring((full > sk.get("thresh", .5)).astype(np.uint8) * 255)
    if O is None: continue
    ss = [dscore(punroll(g, O, I, r, a, j)) for (r, a, j) in VARIANTS]
    base.append(ss[0]); tta.append(max(ss))
    gt = cv2.imread(ip.replace("/img/", "/mask/"), 0); labels.append(1 if gt.sum() > 0 else 0); names.append(nm)
base, tta, labels = np.array(base), np.array(tta), np.array(labels)

def report(s, tag):
    pos, neg = s[labels == 1], s[labels == 0]
    au = float(np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg]))
    for thr in [0.43, 0.5]:
        tp = int(((s >= thr) & (labels == 1)).sum()); fp = int(((s >= thr) & (labels == 0)).sum())
        print(f"  {tag} @thr {thr}: recall {tp}/{int((labels==1).sum())}, FP {fp}/{int((labels==0).sum())}")
    print(f"  {tag} AUROC {au:.3f}")
print("END-TO-END (predicted seal):")
report(base, "baseline")
report(tta, "TTA     ")
i = names.index("seal_2260_1780692167999_raw")
print(f"\nseal_2260: baseline {base[i]:.3f} -> TTA {tta[i]:.3f}")
