#!/usr/bin/env python3
"""5-fold CV of the defect model: each fold trains with fold-i validation + early stopping,
then is evaluated end-to-end on the FIXED 179-pack test hold-out. Reports the distribution."""
import subprocess, glob, os, numpy as np, cv2, torch
from seal_inspection import core
R = "/home/ubuntu/TFM/seal-inspection"; PY = f"{R}/.venv/bin/python"; dev = "cuda" if torch.cuda.is_available() else "cpu"
MEAN, STD = core.IMAGENET_MEAN, core.IMAGENET_STD; K = 5; THR = 0.5
S1313 = "seal_1313_1780666315292_raw"
seal, sk = core.load_unet(f"{R}/models/best_lite_reviewed_1280.pt", dev)
lab = {ln.split(",")[0]: int(ln.split(",")[1]) for ln in open(f"{R}/data/holdout_labels.csv").read().splitlines()[1:]}

def eval_e2e(path):
    defm, dk = core.load_unet(path, dev); HS, WS = dk["HS"], dk["WS"]
    def dsc(strip):
        x = ((np.stack([strip]*3, -1)/255.0 - MEAN)/STD).transpose(2, 0, 1)[None].astype(np.float32)
        with torch.no_grad():
            p = torch.sigmoid(defm(torch.from_numpy(x).to(dev)))[0, 0].cpu().numpy()
        return float(cv2.GaussianBlur(p, (0, 0), 2).max())
    sc, lb, s1313 = [], [], None
    for nm, l in lab.items():
        h = glob.glob(f"{R}/data/images/*/{nm}.png") + glob.glob(f"{R}/data/images/*/{nm}.jpg")
        if not h: continue
        g = cv2.imread(h[0], 0); H, W = g.shape; x0, y0, x1, y1 = core.pack_bbox(g)
        prob = core.predict_probability(seal, g[y0:y1, x0:x1], sk["img"], dev)
        full = np.zeros((H, W), np.float32); full[y0:y1, x0:x1] = cv2.resize(prob, (x1-x0, y1-y0))
        O, I = core.mask_to_ring((full > sk.get("thresh", .5)).astype(np.uint8)*255)
        if O is None: continue
        mx, my = core.unroll_maps(O, I, HS, WS)
        sv = dsc(cv2.remap(core.normalize(g), mx, my, cv2.INTER_LINEAR, borderValue=0))
        sc.append(sv); lb.append(l)
        if nm == S1313: s1313 = sv
    sc, lb = np.array(sc), np.array(lb); pos, neg = sc[lb == 1], sc[lb == 0]
    au = float(np.mean([(a > b)+0.5*(a == b) for a in pos for b in neg]))
    rec = int((pos >= THR).sum()); fp = int((neg >= THR).sum())
    return au, rec, int((lb == 1).sum()), fp, int((lb == 0).sum()), s1313

rows = []
for i in range(K):
    out = f"models/defect_kf{i}.pt"
    subprocess.run([PY, "src/train_defect.py", "--sealjit", "--kfold", str(K), "--fold", str(i), "--out", out],
                   cwd=R, check=True, capture_output=True)
    au, rec, nd, fp, ng, s1313 = eval_e2e(f"{R}/{out}")
    rows.append((i, au, rec, nd, fp, ng, s1313))
    print(f"fold {i}: E2E AUROC {au:.3f}  recall {rec}/{nd}  FP {fp}/{ng}  seal_1313={s1313:.3f} ({'caught' if s1313>=THR else 'missed'})", flush=True)

aus = [r[1] for r in rows]; recs = [r[2] for r in rows]; fps = [r[4] for r in rows]
print("\n===== 5-FOLD CV SUMMARY (fixed 179-pack test hold-out) =====")
print(f"E2E AUROC : mean {np.mean(aus):.3f}  std {np.std(aus):.3f}  range [{min(aus):.3f}, {max(aus):.3f}]")
print(f"recall@0.5: range {min(recs)}-{max(recs)}/{rows[0][3]}   FP range {min(fps)}-{max(fps)}/{rows[0][5]}")
print(f"seal_1313 caught in {sum(1 for r in rows if r[6] >= THR)}/{K} folds")
open(f"{R}/outputs/kfold_done.flag", "w").write("done")
