#!/usr/bin/env python3
"""Compact grayscale TinyUNet defect model, trained FROM SCRATCH with a piece-level
VALIDATION split + EARLY STOPPING on val loss; the test hold-out is evaluated ONCE.
Copy-paste + seal-jitter augmentation (matched to the resnet18 defect model).
Reports accuracy + params + CPU latency vs the resnet18-UNet."""
import os, glob, random, time, argparse
import numpy as np, cv2, torch, torch.nn as nn
import albumentations as A
from seal_inspection.tiny_unet import TinyUNet

_ap = argparse.ArgumentParser()
_ap.add_argument("--epochs", type=int, default=250)
_ap.add_argument("--patience", type=int, default=30)
_ap.add_argument("--val-frac", type=float, default=0.15)
_a, _ = _ap.parse_known_args()
from seal_inspection.paths import ROOT; STR = f"{ROOT}/data/strips"
SEED = 42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
HS, WS, BATCH, STEPS, THR, P_PASTE = 128, 1536, 8, 1200, 0.5, 0.7
MAXEP, PATIENCE, VALF = _a.epochs, _a.patience, _a.val_frac
dev = "cuda" if torch.cuda.is_available() else "cpu"

def load(split):
    out = []
    for ip in sorted(glob.glob(f"{STR}/{split}/img/*.png")):
        mp = ip.replace("/img/", "/mask/")
        if not os.path.exists(mp): continue
        out.append((cv2.imread(ip, 0), (cv2.imread(mp, 0) > 127).astype(np.uint8)))
    return out

alltrain, test = load("train"), load("test")
rng = random.Random(SEED)
d_all = [t for t in alltrain if t[1].sum() > 0]; g_all = [t for t in alltrain if t[1].sum() == 0]
rng.shuffle(d_all); rng.shuffle(g_all)
nvd = max(1, round(len(d_all)*VALF)); nvg = max(1, round(len(g_all)*VALF))
val = d_all[:nvd] + g_all[:nvg]; train = d_all[nvd:] + g_all[nvg:]
tr_def = [t for t in train if t[1].sum() > 0]; tr_good = [t for t in train if t[1].sum() == 0]
print(f"train {len(train)} ({len(tr_def)} def/{len(tr_good)} good)  val {len(val)} ({nvd} def/{nvg} good)  "
      f"test {len(test)} ({sum(1 for t in test if t[1].sum()>0)} def)", flush=True)

LIB = []
for img, m in tr_def:
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    for i in range(1, n):
        x, y, w, h, ar = st[i]
        if ar < 6: continue
        LIB.append((img[y:y+h, x:x+w].copy(), (lab[y:y+h, x:x+w] == i).astype(np.float32)))

def paste(img, m):
    if not LIB: return img, m
    out = img.copy().astype(np.float32); mo = m.copy()
    for _ in range(random.randint(1, 3)):
        p, a = random.choice(LIB); s = random.uniform(0.6, 1.6)
        pw, ph = max(3, int(p.shape[1]*s)), max(3, int(p.shape[0]*s))
        p = cv2.resize(p, (pw, ph)).astype(np.float32); a = cv2.resize(a, (pw, ph))
        if random.random() < .5: p = cv2.flip(p, 1); a = cv2.flip(a, 1)
        p = np.clip(p*random.uniform(.8, 1.15), 0, 255); a = cv2.GaussianBlur(a, (0, 0), 1.0)
        H, W = img.shape; x0 = random.randint(0, max(0, W-pw)); y0 = random.randint(0, max(0, H-ph))
        x1, y1 = min(W, x0+pw), min(H, y0+ph); aa = a[:y1-y0, :x1-x0]; pp = p[:y1-y0, :x1-x0]
        out[y0:y1, x0:x1] = out[y0:y1, x0:x1]*(1-aa) + pp*aa
        mo[y0:y1, x0:x1] = np.maximum(mo[y0:y1, x0:x1], (aa > .3).astype(np.uint8))
    return np.clip(out, 0, 255).astype(np.uint8), mo

def seal_jitter(img, m):
    H, W = img.shape; sy = random.uniform(0.90, 1.10); dy = random.uniform(-0.06, 0.06)*H
    M = np.float32([[1, 0, 0], [0, sy, dy+(1-sy)*H/2]])
    i2 = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    m2 = cv2.warpAffine(m, M, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return i2, m2

geo = A.Compose([A.HorizontalFlip(p=.5), A.VerticalFlip(p=.5), A.RandomBrightnessContrast(.3, .3, p=.7),
                 A.ShiftScaleRotate(shift_limit=.03, scale_limit=.05, rotate_limit=4, border_mode=cv2.BORDER_REFLECT, p=.5)])
def to_x(img): return torch.from_numpy(((img.astype(np.float32)/255.0 - 0.5)/0.5))[None]

class DS(torch.utils.data.Dataset):
    def __init__(s, L): s.L = L
    def __len__(s): return s.L
    def __getitem__(s, i):
        img, m = random.choice(tr_def) if (tr_def and random.random() < .5) else random.choice(tr_good)
        img, m = img.copy(), m.copy()
        if random.random() < P_PASTE: img, m = paste(img, m)
        if random.random() < 0.8: img, m = seal_jitter(img, m)
        o = geo(image=img, mask=m)
        return to_x(o["image"]), torch.from_numpy(o["mask"].astype(np.float32))[None]
dl = torch.utils.data.DataLoader(DS(STEPS), batch_size=BATCH, shuffle=True, num_workers=4)

model = TinyUNet(base=16, in_ch=1).to(dev)
nparam = sum(p.numel() for p in model.parameters())
print(f"TinyUNet params: {nparam/1e6:.3f}M | max {MAXEP} ep, patience {PATIENCE}, val {VALF:.0%}", flush=True)
def dice_l(l, t, e=1.):
    p = torch.sigmoid(l); return (1 - ((2*(p*t).sum((2, 3))+e)/(p.sum((2, 3))+t.sum((2, 3))+e))).mean()
bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(20.).to(dev))
opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)
scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")

@torch.no_grad()
def val_loss():
    model.eval(); tot = 0.; n = 0
    for img, m in val:
        x = to_x(img)[None].to(dev); y = torch.from_numpy(m.astype(np.float32))[None, None].to(dev)
        lo = model(x); tot += float(bce(lo, y) + dice_l(lo, y)); n += 1
    return tot/max(1, n)

@torch.no_grad()
def evaluate():
    model.eval(); sc = []; lb = []; dc = []
    for img, m in test:
        prob = torch.sigmoid(model(to_x(img)[None].to(dev)))[0, 0].cpu().numpy()
        sc.append(float(cv2.GaussianBlur(prob, (0, 0), 2).max())); lb.append(1 if m.sum() > 0 else 0)
        if m.sum() > 0:
            pr = (prob > THR).astype(np.uint8); dc.append(2*(pr & m).sum()/(pr.sum()+m.sum()+1e-6))
    sc, lb = np.array(sc), np.array(lb); pos, neg = sc[lb == 1], sc[lb == 0]
    au = float(np.mean([(a > b)+0.5*(a == b) for a in pos for b in neg]))
    best = (0, 0, 0, 0)
    for th in np.unique(sc):
        pr = (sc >= th).astype(int); tp = ((pr == 1) & (lb == 1)).sum(); fp = ((pr == 1) & (lb == 0)).sum(); fn = ((pr == 0) & (lb == 1)).sum()
        p, r = tp/(tp+fp+1e-9), tp/(tp+fn+1e-9); f1 = 2*p*r/(p+r+1e-9)
        if f1 > best[0]: best = (f1, p, r, th)
    return au, (np.mean(dc) if dc else float("nan")), best

best_val = float("inf"); best_state = None; bad = 0; stop_ep = MAXEP
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
    if ep % 10 == 0 or ep == 1 or improved:
        print(f"ep {ep:3d}  train_loss {tot/n:.4f}  val_loss {vl:.4f}  best_val {best_val:.4f}  bad {bad}/{PATIENCE}", flush=True)
    if bad >= PATIENCE:
        stop_ep = ep; print(f"early stop at epoch {ep}", flush=True); break
if best_state: model.load_state_dict(best_state)
au, pd, (f1, p, r, th) = evaluate()
torch.save({"state_dict": model.state_dict(), "arch": "TinyUNet", "base": 16, "in_ch": 1,
            "HS": HS, "WS": WS, "thr": THR, "score_thr": float(th), "stop_epoch": stop_ep}, f"{ROOT}/models/tiny_defect.pt")
print(f"FINAL TinyUNet (test, once): AUROC {au:.3f}  pixelDice {pd:.3f}  bestF1 {f1:.3f} (P{p:.2f} R{r:.2f} @score>{th:.2f})  [stopped ep {stop_ep}]", flush=True)

# CPU latency: TinyUNet (1ch) vs resnet18-UNet (3ch), single strip
import segmentation_models_pytorch as smp
def bench(m, ch):
    m = m.cpu().eval(); x = torch.randn(1, ch, HS, WS); torch.set_num_threads(4)
    with torch.no_grad():
        for _ in range(3): m(x)
        t = time.time()
        for _ in range(15): m(x)
    return (time.time()-t)/15*1000
r18 = smp.Unet("resnet18", encoder_weights=None, in_channels=3, classes=1)
p_tiny = sum(p.numel() for p in model.parameters()); p_r18 = sum(p.numel() for p in r18.parameters())
lat_tiny = bench(model, 1); lat_r18 = bench(r18, 3)
print(f"\nPARAMS  TinyUNet {p_tiny/1e6:.2f}M  |  resnet18-UNet {p_r18/1e6:.2f}M  ({p_r18/p_tiny:.1f}x smaller)", flush=True)
print(f"CPU latency/strip (4 threads)  TinyUNet {lat_tiny:.1f} ms  |  resnet18-UNet {lat_r18:.1f} ms  ({lat_r18/lat_tiny:.1f}x faster)", flush=True)
print("DONE", flush=True)
