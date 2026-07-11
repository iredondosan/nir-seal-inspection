"""
train_defect.py — Stage 2: segment defects on the unrolled seal strip.

Defects are rare and few-of-a-kind, so we lean on two tricks:
  * oversampling   — each batch draws defect strips ~50% of the time
  * copy-paste     — real defect cut-outs are pasted onto good strips with random
                     scale/flip/intensity, multiplying the positive signal

Trains on data/strips/train, evaluates on data/strips/test (held-out REAL
defects only — copy-paste is train-time only). Reports image-level detection
(AUROC, best-F1) and pixel overlap (Dice).

Usage:
    python -m seal_inspection.train_defect --root /path/to/project
"""
from __future__ import annotations
import os
import glob
import random
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from .core import IMAGENET_MEAN, IMAGENET_STD

SEED = 42
EPOCHS, BATCH, STEPS = 60, 8, 1200
THR = 0.5            # pixel threshold for the defect mask
P_PASTE = 0.7        # fraction of training samples that get pasted defects
POS_WEIGHT = 20.0    # BCE positive weight (defect pixels are a tiny minority)


def load_split(strips_dir: str, split: str):
    """Load (image, binary_mask, name) for every strip in a split."""
    items = []
    for ip in sorted(glob.glob(f"{strips_dir}/{split}/img/*.png")):
        mp = ip.replace("/img/", "/mask/")
        if not os.path.exists(mp):
            continue
        img = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
        mask = (cv2.imread(mp, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        items.append((img, mask, os.path.basename(ip)))
    return items


def build_defect_library(defect_items):
    """Cut out each connected defect region (patch + soft alpha) for copy-paste."""
    lib = []
    for img, mask, _ in defect_items:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < 6:
                continue
            patch = img[y:y + h, x:x + w].copy()
            alpha = (labels[y:y + h, x:x + w] == i).astype(np.float32)
            lib.append((patch, alpha))
    return lib


def paste_defects(img, mask, lib):
    """Paste 1-3 random library defects onto a strip; update the mask."""
    if not lib:
        return img, mask
    out = img.copy().astype(np.float32)
    mo = mask.copy()
    H, W = img.shape
    for _ in range(random.randint(1, 3)):
        patch, alpha = random.choice(lib)
        s = random.uniform(0.6, 1.6)
        pw, ph = max(3, int(patch.shape[1] * s)), max(3, int(patch.shape[0] * s))
        p = cv2.resize(patch, (pw, ph)).astype(np.float32)
        a = cv2.resize(alpha, (pw, ph))
        if random.random() < 0.5:
            p, a = cv2.flip(p, 1), cv2.flip(a, 1)
        if random.random() < 0.5:
            p, a = cv2.flip(p, 0), cv2.flip(a, 0)
        p = np.clip(p * random.uniform(0.8, 1.15), 0, 255)
        a = cv2.GaussianBlur(a, (0, 0), 1.0)              # feather edges (avoid teaching "seams")
        x0, y0 = random.randint(0, max(0, W - pw)), random.randint(0, max(0, H - ph))
        x1, y1 = min(W, x0 + pw), min(H, y0 + ph)
        aa, pp = a[:y1 - y0, :x1 - x0], p[:y1 - y0, :x1 - x0]
        out[y0:y1, x0:x1] = out[y0:y1, x0:x1] * (1 - aa) + pp * aa
        mo[y0:y1, x0:x1] = np.maximum(mo[y0:y1, x0:x1], (aa > 0.3).astype(np.uint8))
    return np.clip(out, 0, 255).astype(np.uint8), mo


class StripDataset(torch.utils.data.Dataset):
    """Samples strips with defect oversampling + copy-paste, then augments."""

    def __init__(self, defect_items, good_items, lib, length):
        self.defect, self.good, self.lib, self.length = defect_items, good_items, lib, length
        self.aug = A.Compose([
            A.HorizontalFlip(p=.5), A.VerticalFlip(p=.5),
            A.RandomBrightnessContrast(.3, .3, p=.7),
            A.ShiftScaleRotate(shift_limit=.03, scale_limit=.05, rotate_limit=4,
                               border_mode=cv2.BORDER_REFLECT, p=.5),
            A.Normalize(IMAGENET_MEAN, IMAGENET_STD), ToTensorV2(),
        ])

    def __len__(self):
        return self.length

    def __getitem__(self, _):
        # oversample: 50% of the time draw a real defect strip
        pool = self.defect if (self.defect and random.random() < 0.5) else self.good
        img, mask, _ = random.choice(pool)
        img, mask = img.copy(), mask.copy()
        if random.random() < P_PASTE:
            img, mask = paste_defects(img, mask, self.lib)
        o = self.aug(image=np.stack([img] * 3, -1), mask=mask)
        return o["image"], o["mask"].float().unsqueeze(0)


def dice_loss(logits, target, eps=1.):
    p = torch.sigmoid(logits)
    return (1 - ((2 * (p * target).sum((2, 3)) + eps) /
                 (p.sum((2, 3)) + target.sum((2, 3)) + eps))).mean()


@torch.no_grad()
def evaluate(model, test_items, device):
    """Image-level AUROC + best-F1 (defect/no-defect) and pixel-Dice on defects."""
    ev = A.Compose([A.Normalize(IMAGENET_MEAN, IMAGENET_STD), ToTensorV2()])
    scores, labels, dices = [], [], []
    for img, mask, _ in test_items:
        x = ev(image=np.stack([img] * 3, -1), mask=mask)["image"].unsqueeze(0).to(device)
        prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
        scores.append(float(cv2.GaussianBlur(prob, (0, 0), 2).max()))
        labels.append(1 if mask.sum() > 0 else 0)
        if mask.sum() > 0:
            pr = (prob > THR).astype(np.uint8)
            dices.append(2 * (pr & mask).sum() / (pr.sum() + mask.sum() + 1e-6))
    scores, labels = np.array(scores), np.array(labels)
    pos, neg = scores[labels == 1], scores[labels == 0]
    auroc = float(np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg])) if len(pos) and len(neg) else float("nan")
    best = (0, 0, 0, 0)
    for th in np.unique(scores):
        pred = (scores >= th).astype(int)
        tp = ((pred == 1) & (labels == 1)).sum()
        fp = ((pred == 1) & (labels == 0)).sum()
        fn = ((pred == 0) & (labels == 1)).sum()
        pr, rc = tp / (tp + fp + 1e-9), tp / (tp + fn + 1e-9)
        f1 = 2 * pr * rc / (pr + rc + 1e-9)
        if f1 > best[0]:
            best = (f1, pr, rc, th)
    return auroc, (np.mean(dices) if dices else float("nan")), best


def main():
    ap = argparse.ArgumentParser(description="Train the defect segmentation model.")
    ap.add_argument("--root", default="/home/ubuntu/TFM/seal-inspection")
    ap.add_argument("--encoder", default="resnet18")
    a = ap.parse_args()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    strips = os.path.join(a.root, "data/strips")

    train = load_split(strips, "train")
    test = load_split(strips, "test")
    tr_def = [t for t in train if t[1].sum() > 0]
    tr_good = [t for t in train if t[1].sum() == 0]
    lib = build_defect_library(tr_def)
    print(f"train {len(train)} ({len(tr_def)} defect / {len(tr_good)} good)  "
          f"test {len(test)} ({sum(m.sum() > 0 for _, m, _ in test)} defect)  lib {len(lib)}", flush=True)

    dl = torch.utils.data.DataLoader(StripDataset(tr_def, tr_good, lib, STEPS),
                                     batch_size=BATCH, shuffle=True, num_workers=4)
    model = smp.Unet(a.encoder, encoder_weights="imagenet", in_channels=3, classes=1).to(device)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(POS_WEIGHT).to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    best_auroc, best_state = 0.0, None
    for ep in range(1, EPOCHS + 1):
        model.train()
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                lo = model(x)
                loss = bce(lo, y) + dice_loss(lo, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sch.step()
        if ep % 10 == 0 or ep == 1:
            au, pd, (f1, pr, rc, th) = evaluate(model, test, device)
            print(f"ep {ep:3d} | AUROC {au:.3f}  pixelDice {pd:.3f}  bestF1 {f1:.3f} (P{pr:.2f} R{rc:.2f})", flush=True)
            if au >= best_auroc:
                best_auroc = au
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    au, pd, (f1, pr, rc, th) = evaluate(model, test, device)
    out = os.path.join(a.root, "models/defect_strip.pt")
    torch.save({"state_dict": model.state_dict(), "encoder": a.encoder,
                "HS": test[0][0].shape[0], "WS": test[0][0].shape[1],
                "thr": THR, "score_thr": float(th)}, out)
    print(f"\nFINAL test: AUROC {au:.3f}  pixelDice {pd:.3f}  bestF1 {f1:.3f} "
          f"(P{pr:.2f} R{rc:.2f} @score>{th:.2f})  saved {out}", flush=True)


if __name__ == "__main__":
    main()
