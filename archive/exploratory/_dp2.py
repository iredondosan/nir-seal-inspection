import glob, os, numpy as np, cv2
from seal_inspection import core
R = "/home/ubuntu/TFM/seal-inspection"
H = set(l.split(",")[0] for l in open(f"{R}/data/holdout_labels.csv").read().splitlines()[1:] if l.strip())
RES = 192

def feats(p):
    g = cv2.imread(p, 0); x0, y0, x1, y1 = core.pack_bbox(g); c = core.normalize(g[y0:y1, x0:x1])
    t = cv2.resize(c, (RES, RES)).astype(np.float32)
    f0 = (t - t.mean()) / (t.std() + 1e-6)
    t2 = cv2.rotate(t, cv2.ROTATE_180); f1 = (t2 - t2.mean()) / (t2.std() + 1e-6)
    return f0.flatten(), f1.flatten()

print("Holdout-vs-train similarity per product (rotation-robust, crop-aligned, %dx%d)" % (RES, RES))
print("%-7s %4s %4s | %-28s | %-22s" % ("prod", "nH", "nT", "HOLDOUT best-match to TRAIN", "TRAIN-TRAIN (diff pkg)"))
print("%-7s %4s %4s | %6s %6s %6s %5s %5s | %6s %6s" % ("", "", "", "med", "p95", "max", ">.97", ">.99", "median", "p99"))
allmax = []
for prod in ["prod1", "prod2", "prod3", "prod4", "prod5"]:
    imgs = sorted(glob.glob(f"{R}/data/images/{prod}/*"))
    names = [os.path.splitext(os.path.basename(p))[0] for p in imgs]
    F0, F1 = [], []
    for p in imgs:
        a, b = feats(p); F0.append(a); F1.append(b)
    X = np.array(F0); X1 = np.array(F1); D = X.shape[1]
    Hi = [i for i, nm in enumerate(names) if nm in H]
    Ti = [i for i, nm in enumerate(names) if nm not in H]
    if not Hi or not Ti:
        print("%-7s %4d %4d | (skip)" % (prod, len(Hi), len(Ti))); continue
    XT = X[Ti]; XT1 = X1[Ti]
    best = []
    for h in Hi:
        s = np.maximum(XT @ X[h], XT1 @ X[h]) / D
        best.append(float(s.max()))
    best = np.array(best); allmax.append(best.max())
    # train-train baseline (different packages, same product): sample pairwise
    idx = Ti[:120]; XB = X[idx]; SB = XB @ XB.T / D; np.fill_diagonal(SB, -1)
    bb = SB[SB > -1]
    print("%-7s %4d %4d | %6.3f %6.3f %6.3f %5d %5d | %6.3f %6.3f" % (
        prod, len(Hi), len(Ti), np.median(best), np.percentile(best, 95), best.max(),
        int((best > .97).sum()), int((best > .99).sum()), np.median(bb), np.percentile(bb, 99)))
print("\nMax holdout->train similarity across all products: %.3f" % (max(allmax) if allmax else 0))
print("Interpretation: if HOLDOUT max ~= TRAIN-TRAIN level (< ~0.97) -> no rescanned-same-package leak.")
print("Values ~1.00 would indicate the SAME physical package split across train and test.")
open(f"{R}/outputs/dup2.done", "w").write("ok")
