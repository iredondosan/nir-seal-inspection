#!/usr/bin/env python3
"""Defect model (Model B): segment defects on the UNROLLED seal strip.
Trains on data/strips/train with a PIECE-LEVEL VALIDATION split (held out from training,
excluded from the copy-paste library) and EARLY STOPPING on validation loss; the
data/strips/test hold-out is evaluated ONCE at the end. Copy-paste + seal-jitter augmentation."""
import os, glob, random
import numpy as np, cv2, torch, torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import argparse

_ap = argparse.ArgumentParser()
_ap.add_argument("--strips", default="data/strips")
_ap.add_argument("--out", default="models/defect_strip.pt")
_ap.add_argument("--epochs", type=int, default=200, help="max epochs (early stopping usually stops sooner)")
_ap.add_argument("--patience", type=int, default=25, help="early-stopping patience on val loss")
_ap.add_argument("--val-frac", type=float, default=0.15)
_ap.add_argument("--scratch", action="store_true", help="random init (no ImageNet) for the ablation")
_ap.add_argument("--roll", action="store_true", help="horizontal circular roll (arbitrary seam)")
_ap.add_argument("--sealjit", action="store_true", help="vertical seal-misalignment jitter")
_ap.add_argument("--edgepaste", action="store_true", help="bias copy-paste toward the seal edges (top/bottom strip rows)")
_ap.add_argument("--kfold", type=int, default=0, help="if >0, use fold i of k as validation (k-fold CV)")
_ap.add_argument("--fold", type=int, default=0)
_ap.add_argument("--rebalance", action="store_true", help="oversample small defects + bias copy-paste small")
_a, _ = _ap.parse_known_args()
ROOT = "/home/ubuntu/TFM/seal-inspection"; STR = f"{ROOT}/{_a.strips}"; OUT = f"{ROOT}/{_a.out}"
SCRATCH = _a.scratch; ROLL = _a.roll; SEALJIT = _a.sealjit; EDGEPASTE = _a.edgepaste; KFOLD = _a.kfold; FOLD = _a.fold; REBALANCE = _a.rebalance
MAXEP = _a.epochs; PATIENCE = _a.patience; VALF = _a.val_frac
SEED = 42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
HS = 128; WS = 1536; BATCH = 8; STEPS = 1200; THR = 0.5; P_PASTE = 0.7
MEAN = (.485, .456, .406); STD = (.229, .224, .225)
dev = "cuda" if torch.cuda.is_available() else "cpu"

def load(split):
    items = []
    for ip in sorted(glob.glob(f"{STR}/{split}/img/*.png")):
        mp = ip.replace("/img/", "/mask/")
        if not os.path.exists(mp): continue
        items.append((cv2.imread(ip, 0), (cv2.imread(mp, 0) > 127).astype(np.uint8), os.path.basename(ip)))
    return items

alltrain = load("train"); test = load("test")
# piece-level stratified validation split, HELD OUT from training
rng = random.Random(SEED)
d_all = [t for t in alltrain if t[1].sum() > 0]; g_all = [t for t in alltrain if t[1].sum() == 0]
rng.shuffle(d_all); rng.shuffle(g_all)
if KFOLD > 0:
    def _fold(lst, k, i):
        a = round(len(lst)*i/k); b = round(len(lst)*(i+1)/k); return lst[a:b], lst[:a] + lst[b:]
    vd, td = _fold(d_all, KFOLD, FOLD); vg, tg = _fold(g_all, KFOLD, FOLD)
    val = vd + vg; train = td + tg; nvd = len(vd); nvg = len(vg)
    print(f"[k-fold {FOLD}/{KFOLD}]", flush=True)
else:
    nvd = max(1, round(len(d_all) * VALF)); nvg = max(1, round(len(g_all) * VALF))
    val = d_all[:nvd] + g_all[:nvg]; train = d_all[nvd:] + g_all[nvg:]
tr_def = [t for t in train if t[1].sum() > 0]; tr_good = [t for t in train if t[1].sum() == 0]
tr_def_w = [1.0/np.sqrt(max(1.0, float(t[1].sum()))) for t in tr_def]  # inverse-size weights (favor small defects)
print(f"train {len(train)} ({len(tr_def)} def/{len(tr_good)} good)  val {len(val)} ({nvd} def/{nvg} good)  "
      f"test {len(test)} ({sum(1 for t in test if t[1].sum()>0)} def)", flush=True)

# copy-paste library from TRAIN defects only (val never leaks in)
LIB = []
for img, m, _ in tr_def:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 6: continue
        LIB.append((img[y:y+h, x:x+w].copy(), (lab[y:y+h, x:x+w] == i).astype(np.float32)))
print(f"defect cut-outs in library: {len(LIB)}", flush=True)

def paste(img, m):
    if not LIB: return img, m
    out = img.copy().astype(np.float32); mo = m.copy()
    for _ in range(random.randint(1, 3)):
        patch, al = random.choice(LIB); s = random.uniform(0.15, 0.6) if REBALANCE else random.uniform(0.6, 1.6)
        pw, ph = max(3, int(patch.shape[1]*s)), max(3, int(patch.shape[0]*s))
        p = cv2.resize(patch, (pw, ph)).astype(np.float32); a = cv2.resize(al, (pw, ph))
        if random.random() < 0.5: p = cv2.flip(p, 1); a = cv2.flip(a, 1)
        if random.random() < 0.5: p = cv2.flip(p, 0); a = cv2.flip(a, 0)
        p = np.clip(p*random.uniform(0.8, 1.15), 0, 255); a = cv2.GaussianBlur(a, (0, 0), 1.0)
        H, W = img.shape; x0 = random.randint(0, max(0, W-pw))
        if EDGEPASTE and random.random() < 0.6:                                # bias paste toward the seal edges
            if random.random() < 0.5: y0 = random.randint(0, max(0, min(H-ph, 30)))
            else: y0 = random.randint(max(0, H-ph-30), max(0, H-ph))
        else:
            y0 = random.randint(0, max(0, H-ph))
        x1, y1 = min(W, x0+pw), min(H, y0+ph); aa = a[:y1-y0, :x1-x0]; pp = p[:y1-y0, :x1-x0]
        out[y0:y1, x0:x1] = out[y0:y1, x0:x1]*(1-aa) + pp*aa
        mo[y0:y1, x0:x1] = np.maximum(mo[y0:y1, x0:x1], (aa > 0.3).astype(np.uint8))
    return np.clip(out, 0, 255).astype(np.uint8), mo

def seal_jitter(img, m):
    H, W = img.shape; sy = random.uniform(0.90, 1.10); dy = random.uniform(-0.06, 0.06)*H
    M = np.float32([[1, 0, 0], [0, sy, dy+(1-sy)*H/2]])
    i2 = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    m2 = cv2.warpAffine(m, M, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return i2, m2

aug = A.Compose([A.HorizontalFlip(p=.5), A.VerticalFlip(p=.5), A.RandomBrightnessContrast(.3, .3, p=.7),
                 A.ShiftScaleRotate(shift_limit=.03, scale_limit=.05, rotate_limit=4, border_mode=cv2.BORDER_REFLECT, p=.5),
                 A.Normalize(MEAN, STD), ToTensorV2()])
ev = A.Compose([A.Normalize(MEAN, STD), ToTensorV2()])

class DS(torch.utils.data.Dataset):
    def __init__(s, L): s.L = L
    def __len__(s): return s.L
    def __getitem__(s, i):
        if tr_def and random.random() < 0.5:
            img, m, _ = random.choices(tr_def, weights=tr_def_w)[0] if REBALANCE else random.choice(tr_def)
        else:
            img, m, _ = random.choice(tr_good)
        img, m = img.copy(), m.copy()
        if random.random() < P_PASTE: img, m = paste(img, m)
        if ROLL: sh = random.randint(0, img.shape[1]-1); img = np.roll(img, sh, 1); m = np.roll(m, sh, 1)
        if SEALJIT and random.random() < 0.8: img, m = seal_jitter(img, m)
        o = aug(image=np.stack([img]*3, -1), mask=m); return o["image"], o["mask"].float().unsqueeze(0)
dl = torch.utils.data.DataLoader(DS(STEPS), batch_size=BATCH, shuffle=True, num_workers=4)

model = smp.Unet("resnet18", encoder_weights=(None if SCRATCH else "imagenet"), in_channels=3, classes=1).to(dev)
print(("FROM SCRATCH (random init)" if SCRATCH else "ImageNet-pretrained")
      + f" resnet18 | max {MAXEP} ep, patience {PATIENCE}, val {VALF:.0%} | roll={ROLL} sealjit={SEALJIT} edgepaste={EDGEPASTE} -> {OUT}", flush=True)
def dice_l(l, t, e=1.):
    p = torch.sigmoid(l); return (1 - ((2*(p*t).sum((2, 3))+e)/(p.sum((2, 3))+t.sum((2, 3))+e))).mean()
bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(20.).to(dev))
opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=8)
scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")

@torch.no_grad()
def val_loss():
    model.eval(); tot = 0.; n = 0
    for img, m, _ in val:
        o = ev(image=np.stack([img]*3, -1), mask=m)
        x = o["image"].unsqueeze(0).to(dev); y = o["mask"].float().unsqueeze(0).unsqueeze(0).to(dev)
        lo = model(x); tot += float(bce(lo, y) + dice_l(lo, y)); n += 1
    return tot/max(1, n)

@torch.no_grad()
def evaluate():   # TEST hold-out, only for the final report
    model.eval(); scores = []; labels = []; dices = []
    for img, m, _ in test:
        x = ev(image=np.stack([img]*3, -1), mask=m)["image"].unsqueeze(0).to(dev)
        prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
        scores.append(float(cv2.GaussianBlur(prob, (0, 0), 2).max())); labels.append(1 if m.sum() > 0 else 0)
        if m.sum() > 0:
            pr = (prob > THR).astype(np.uint8); dices.append(2*(pr & m).sum()/(pr.sum()+m.sum()+1e-6))
    scores = np.array(scores); labels = np.array(labels); pos = scores[labels == 1]; neg = scores[labels == 0]
    auroc = np.mean([(1.0 if a > b else 0.5 if a == b else 0.0) for a in pos for b in neg]) if len(pos) and len(neg) else float('nan')
    best = (0, 0, 0, 0)
    for th in np.unique(scores):
        pred = (scores >= th).astype(int); tp = ((pred == 1) & (labels == 1)).sum(); fp = ((pred == 1) & (labels == 0)).sum(); fn = ((pred == 0) & (labels == 1)).sum()
        pr = tp/(tp+fp+1e-9); rc = tp/(tp+fn+1e-9); f1 = 2*pr*rc/(pr+rc+1e-9)
        if f1 > best[0]: best = (f1, pr, rc, th)
    return auroc, (np.mean(dices) if dices else float('nan')), best

best_val = float('inf'); best_state = None; bad = 0; stop_ep = MAXEP
for ep in range(1, MAXEP+1):
    model.train(); tot = n = 0
    for x, y in dl:
        x, y = x.to(dev), y.to(dev); opt.zero_grad()
        with torch.amp.autocast("cuda", enabled=dev == "cuda"):
            lo = model(x); L = bce(lo, y) + dice_l(lo, y)
        scaler.scale(L).backward(); scaler.step(opt); scaler.update(); tot += L.item()*x.size(0); n += x.size(0)
    vl = val_loss(); sch.step(vl)
    improved = vl < best_val - 1e-4
    if improved: best_val = vl; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; bad = 0
    else: bad += 1
    if ep % 5 == 0 or ep == 1 or improved:
        print(f"ep {ep:3d}  train_loss {tot/n:.4f}  val_loss {vl:.4f}  best_val {best_val:.4f}  bad {bad}/{PATIENCE}", flush=True)
    if bad >= PATIENCE:
        stop_ep = ep; print(f"early stop at epoch {ep} (no val improvement for {PATIENCE})", flush=True); break
if best_state: model.load_state_dict(best_state)
au, pd, (f1, pr, rc, th) = evaluate()
torch.save({"state_dict": model.state_dict(), "encoder": "resnet18", "HS": HS, "WS": WS, "thr": THR,
            "score_thr": float(th), "mean": MEAN, "std": STD, "stop_epoch": stop_ep, "val_loss": best_val}, OUT)
print(f"\nFINAL (test, evaluated once): AUROC {au:.3f}  pixelDice {pd:.3f}  bestF1 {f1:.3f} "
      f"(P{pr:.2f} R{rc:.2f} @score>{th:.2f})  [stopped ep {stop_ep}]", flush=True)
print("DONE", flush=True)
