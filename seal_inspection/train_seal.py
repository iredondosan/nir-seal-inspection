"""
train_seal.py — Stage 1: segment the seal ring across all products.

A MobileNetV3-small U-Net is fine-tuned on pack-cropped images at (near-)native
resolution. Ground truth is the seal band (outer polygon minus inner polygon).

Training data per product comes from CVAT exports:
  * files named '*_reviewed.xml' are filtered to human-`reviewed` packs (GT);
  * other files (prod4/prod5) use all packs that have the two seal polygons.

Augmentation includes COPY-PASTE of real contaminant/printed-graphic cut-outs onto
the seal band WITHOUT changing the mask — this teaches the model that the seal is
a geometric region invariant to stuff sitting on it (contamination, barcodes).

Usage:
    python -m seal_inspection.train_seal --root /path --img 1280 --batch 2 --epochs 40
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
from . import core, cvat
from .core import IMAGENET_MEAN, IMAGENET_STD

SEED = 42
VAL_PER = 2          # packs held out per product for validation (0 if a product has <6)
SAMPLES = 240        # random samples drawn per epoch
P_PASTE = 0.8        # fraction of training samples that get a pasted cut-out

# (annotation xml, image folder, product label). '*_reviewed.xml' => reviewed-only.
DATASETS = [
    ("data/annotations/prod1_reviewed.xml", "data/images/prod1", "prod1"),
    ("data/annotations/prod2_reviewed.xml", "data/images/prod2", "prod2"),
    ("data/annotations/prod3_reviewed.xml", "data/images/prod3", "prod3"),
    ("data/annotations/prod4.xml", "data/images/prod4", "prod4"),
    ("data/annotations/prod5.xml", "data/images/prod5", "prod5"),
    ("data/annotations/prod6_reviewed.xml", "data/images/prod6", "prod6"),
    ("data/annotations/prod6_bad_reviewed.xml", "data/images/prod6_bad", "prod6"),
]
CONTAMINANTS_XML = "data/annotations/contaminants.xml"   # real 'defect' cut-outs for copy-paste


def build_pack(root, img_rel, node):
    """Crop a pack and rasterize its seal band mask. Returns (rgb, mask, name) or None."""
    name = node.get("name")
    seal = cvat.seal_outer_inner(node)
    path = os.path.join(root, img_rel, name)
    if seal is None or not os.path.exists(path):
        return None
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    full_mask = core.polygons_to_band_mask(seal[0], seal[1], gray.shape[0], gray.shape[1])
    x0, y0, x1, y1 = core.pack_bbox(gray)
    crop = core.normalize(gray[y0:y1, x0:x1])
    return np.stack([crop] * 3, -1), full_mask[y0:y1, x0:x1], name


def load_datasets(root):
    """Build train/val pack lists with a per-product hold-out."""
    rng = random.Random(SEED)
    train, val = [], []
    for xml_rel, img_rel, prod in DATASETS:
        path = os.path.join(root, xml_rel)
        if not os.path.exists(path):
            continue
        reviewed_only = "reviewed" in xml_rel
        packs = []
        for node in cvat.iter_images(path):
            tg = cvat.tags(node)
            if "exclude" in tg:
                continue
            if reviewed_only and "reviewed" not in tg:
                continue
            p = build_pack(root, img_rel, node)
            if p is not None:
                packs.append(p)
        rng.shuffle(packs)
        vp = VAL_PER if len(packs) >= 6 else 0      # tiny products go entirely to train
        val += [(im, m, prod) for im, m, _ in packs[:vp]]
        train += [(im, m) for im, m, _ in packs[vp:]]
        print(f"{prod}: {len(packs)} packs -> {vp} val / {len(packs) - vp} train", flush=True)
    return train, val


def load_cutouts(root):
    """Cut out real contaminant instances (patch + feathered alpha) for copy-paste."""
    lib = []
    path = os.path.join(root, CONTAMINANTS_XML)
    if not os.path.exists(path):
        return lib
    for node in cvat.iter_images(path):
        hits = glob.glob(os.path.join(root, "data/images", "*", node.get("name")))
        if not hits:
            continue
        gray = cv2.imread(hits[0], cv2.IMREAD_GRAYSCALE)
        for poly in cvat.polygons(node, "defect"):
            x, y, w, h = cv2.boundingRect(poly.astype(np.int32))
            if w < 4 or h < 4:
                continue
            alpha = np.zeros((h, w), np.uint8)
            cv2.fillPoly(alpha, [poly.astype(np.int32) - [x, y]], 255)
            lib.append((gray[y:y + h, x:x + w].copy(), cv2.GaussianBlur(alpha, (0, 0), 2).astype(np.float32) / 255.0))
    return lib


def paste_cutouts(rgb, band_mask, lib):
    """Paste real cut-outs onto the seal band; the MASK IS UNCHANGED (key idea)."""
    if not lib:
        return rgb
    ys, xs = np.where(band_mask > 0)
    if len(xs) == 0:
        return rgb
    out = rgb.copy()
    h, w = band_mask.shape
    for _ in range(random.randint(2, 5)):
        patch, alpha = random.choice(lib)
        s = random.uniform(0.12, 0.55) if random.random() < 0.7 else random.uniform(0.55, 1.8)  # bias small
        pw, ph = max(3, int(patch.shape[1] * s)), max(3, int(patch.shape[0] * s))
        p = cv2.resize(patch, (pw, ph)).astype(np.float32)
        a = cv2.resize(alpha, (pw, ph))
        M = cv2.getRotationMatrix2D((pw / 2, ph / 2), random.uniform(0, 360), 1.0)
        p, a = cv2.warpAffine(p, M, (pw, ph)), cv2.warpAffine(a, M, (pw, ph))
        k = random.randrange(len(xs))                 # land it on the band
        x0, y0 = int(xs[k]) - pw // 2, int(ys[k]) - ph // 2
        ix0, iy0, ix1, iy1 = max(0, x0), max(0, y0), min(w, x0 + pw), min(h, y0 + ph)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        aa = a[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0][..., None]
        pp = p[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0][..., None]
        reg = out[iy0:iy1, ix0:ix1].astype(np.float32)
        out[iy0:iy1, ix0:ix1] = np.clip(reg * (1 - aa) + pp * aa, 0, 255).astype(np.uint8)
    return out


class SealDataset(torch.utils.data.Dataset):
    def __init__(self, packs, lib, img, length):
        self.packs, self.lib, self.length = packs, lib, length
        self.tf = A.Compose([
            A.Resize(img, img), A.HorizontalFlip(p=.5), A.VerticalFlip(p=.5),
            A.Affine(scale=(.85, 1.15), translate_percent=(0, .06), rotate=(-180, 180), p=.9),
            A.RandomBrightnessContrast(.4, .4, p=.8), A.RandomGamma((60, 140), p=.6),
            A.Normalize(IMAGENET_MEAN, IMAGENET_STD), ToTensorV2(),
        ])

    def __len__(self):
        return self.length

    def __getitem__(self, _):
        im, m = self.packs[random.randrange(len(self.packs))]
        if random.random() < P_PASTE:
            im = paste_cutouts(im, m, self.lib)
        o = self.tf(image=im, mask=m)
        return o["image"], o["mask"].float().unsqueeze(0)


def dice_loss(logits, target, eps=1.):
    p = torch.sigmoid(logits)
    return (1 - ((2 * (p * target).sum((2, 3)) + eps) /
                 (p.sum((2, 3)) + target.sum((2, 3)) + eps))).mean()


def main():
    ap = argparse.ArgumentParser(description="Train the seal segmentation model.")
    ap.add_argument("--root", default="/home/ubuntu/TFM/seal-inspection")
    ap.add_argument("--base", default=None, help="checkpoint to fine-tune from (optional)")
    ap.add_argument("--encoder", default="timm-mobilenetv3_small_100")
    ap.add_argument("--img", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--thresh", type=float, default=0.5)
    a = ap.parse_args()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train, val = load_datasets(a.root)
    lib = load_cutouts(a.root)
    print(f"TOTAL train {len(train)}  val {len(val)}  | {len(lib)} contaminant cut-outs", flush=True)

    if a.base:
        model, ck = core.load_unet(os.path.join(a.root, a.base), device)
    else:
        model = smp.Unet(a.encoder, encoder_weights="imagenet", in_channels=3, classes=1).to(device)

    dl = torch.utils.data.DataLoader(SealDataset(train, lib, a.img, SAMPLES),
                                     batch_size=a.batch, shuffle=True)
    bce = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    eval_tf = A.Compose([A.Resize(a.img, a.img), A.Normalize(IMAGENET_MEAN, IMAGENET_STD), ToTensorV2()])

    @torch.no_grad()
    def val_dice():
        model.eval()
        per = {}
        for im, m, prod in val:
            o = eval_tf(image=im, mask=m)
            x = o["image"].unsqueeze(0).to(device)
            y = o["mask"].float()[None, None].to(device)
            p = (torch.sigmoid(model(x)) > a.thresh).float()
            per.setdefault(prod, []).append(((2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1)).item())
        return {k: float(np.mean(v)) for k, v in per.items()}, float(np.mean([d for v in per.values() for d in v]))

    best, best_state = 0.0, None
    for ep in range(1, a.epochs + 1):
        model.train()
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                lo = model(x)
                loss = bce(lo, y) + dice_loss(lo, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sch.step()
        per, ov = val_dice()
        if ov >= best:
            best, best_state = ov, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep % 5 == 0 or ep == 1:
            ps = " ".join(f"{k}={v:.3f}" for k, v in sorted(per.items()))
            print(f"epoch {ep:3d}/{a.epochs}  VAL {ov:.3f} [{ps}]  best {best:.3f}", flush=True)

    if best_state:
        model.load_state_dict(best_state)
    out = os.path.join(a.root, f"models/seal_{a.img}.pt")
    torch.save({"state_dict": model.state_dict(), "encoder": a.encoder, "img": a.img,
                "thresh": a.thresh, "val_dice": best}, out)
    print(f"\nBEST VAL {best:.3f}  saved {out}", flush=True)


if __name__ == "__main__":
    main()
