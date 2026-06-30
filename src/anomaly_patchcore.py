#!/usr/bin/env python3
"""anomaly_patchcore.py — UNSUPERVISED defect safety-net on the unrolled strip.

Idea (PatchCore-style): a frozen ImageNet backbone turns each strip into a grid of
patch feature vectors. We build a MEMORY BANK from GOOD strips only (no defect
labels). At test time, each patch's distance to its nearest memory vector is its
anomaly score; the strip score is the max over patches. Anything that doesn't look
like "normal seal" scores high — including defect types we never annotated.

Complements the supervised defect model: supervised = sharp on known defects,
anomaly = catches unknown ones. Trains on GOOD only; evaluated on the SAME
held-out test split (good vs real defects).

Usage: .venv/bin/python src/anomaly_patchcore.py --root /home/ubuntu/TFM/seal-inspection
"""
import os, glob, argparse, random
import numpy as np, cv2, torch
import timm
ROOT_DEFAULT = "/home/ubuntu/TFM/seal-inspection"
MEAN = torch.tensor((.485, .456, .406)).view(1, 3, 1, 1)
STD = torch.tensor((.229, .224, .225)).view(1, 3, 1, 1)
MEM_SIZE = 40000          # memory-bank vectors (random coreset of good patches)
SEED = 42


def load_strips(strips_dir, split):
    items = []
    for ip in sorted(glob.glob(f"{strips_dir}/{split}/img/*.png")):
        m = cv2.imread(ip.replace("/img/", "/mask/"), cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
        items.append((img, 1 if (m is not None and m.sum() > 0) else 0, os.path.basename(ip)))
    return items


@torch.no_grad()
def patch_features(model, img, device):
    """Return an (N_patches, C) array of L2-normalized patch embeddings for one strip.

    Concatenate two backbone stages (layer2 + layer3) — layer3 is upsampled to
    layer2's grid — so each patch mixes fine texture and coarser context.
    """
    x = torch.from_numpy(np.stack([img] * 3, -1)).permute(2, 0, 1)[None].float() / 255.0
    x = ((x - MEAN) / STD).to(device)
    f2, f3 = model(x)                                   # [1,C2,H2,W2], [1,C3,H3,W3]
    f3 = torch.nn.functional.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
    feat = torch.cat([f2, f3], 1)[0]                    # [C, H2, W2]
    C, H, W = feat.shape
    feat = feat.reshape(C, H * W).t()                   # [HW, C]
    feat = torch.nn.functional.normalize(feat, dim=1)
    return feat, (H, W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT_DEFAULT)
    a = ap.parse_args()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    strips = os.path.join(a.root, "data/strips")

    model = timm.create_model("resnet18", pretrained=True, features_only=True, out_indices=(2, 3)).to(device).eval()

    # --- build memory bank from GOOD train strips only ---
    train = load_strips(strips, "train")
    good = [t for t in train if t[1] == 0]
    bank = []
    for img, _, _ in good:
        feat, _ = patch_features(model, img, device)
        bank.append(feat.cpu())
    bank = torch.cat(bank, 0)
    if bank.shape[0] > MEM_SIZE:                        # random coreset for speed
        idx = torch.randperm(bank.shape[0])[:MEM_SIZE]
        bank = bank[idx]
    bank = bank.to(device)
    print(f"memory bank: {bank.shape[0]} patches x {bank.shape[1]} dims (from {len(good)} good strips)", flush=True)

    # --- score the held-out test split ---
    test = load_strips(strips, "test")
    scores, labels = [], []
    for img, lab, _ in test:
        feat, (H, W) = patch_features(model, img, device)
        # nearest-neighbour distance per patch (chunked cdist), strip score = max
        d = torch.cdist(feat, bank)                     # [HW, MEM]
        nn = d.min(1).values.reshape(H, W).cpu().numpy()
        nn = cv2.GaussianBlur(nn, (0, 0), 1.0)
        scores.append(float(nn.max())); labels.append(lab)
    scores, labels = np.array(scores), np.array(labels)

    pos, neg = scores[labels == 1], scores[labels == 0]
    auroc = float(np.mean([(x > y) + 0.5 * (x == y) for x in pos for y in neg]))
    # best-F1 operating point
    best = (0, 0, 0, 0)
    for th in np.unique(scores):
        pred = (scores >= th).astype(int)
        tp = ((pred == 1) & (labels == 1)).sum(); fp = ((pred == 1) & (labels == 0)).sum(); fn = ((pred == 0) & (labels == 1)).sum()
        pr, rc = tp / (tp + fp + 1e-9), tp / (tp + fn + 1e-9); f1 = 2 * pr * rc / (pr + rc + 1e-9)
        if f1 > best[0]:
            best = (f1, pr, rc, th)
    print(f"ANOMALY (good-only) test: AUROC {auroc:.3f}  | best-F1 {best[0]:.3f} (P{best[1]:.2f} R{best[2]:.2f})  "
          f"| {len(pos)} defect / {len(neg)} good", flush=True)


if __name__ == "__main__":
    main()
