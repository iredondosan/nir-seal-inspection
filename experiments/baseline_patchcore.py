#!/usr/bin/env python3
"""anomaly_patchcore.py — UNSUPERVISED defect safety-net on the unrolled strip.

Idea (PatchCore-style): a frozen ImageNet backbone turns each strip into a grid of
patch feature vectors. We build a MEMORY BANK from GOOD strips only (no defect
labels). At test time, each patch's distance to its nearest memory vector is its
anomaly score; the strip score is the max over patches.

Coreset: the memory bank is reduced by a GREEDY k-center coreset (farthest-point
sampling, faithful to PatchCore) rather than a random subsample. Reports the full
bank (no coreset), the greedy coreset, and a random coreset for reference.
Trains on GOOD only; evaluated on the SAME held-out test split (good vs defects).

Usage: .venv/bin/python experiments/baseline_patchcore.py --root .
"""
import os, glob, argparse, random
import numpy as np, cv2, torch
import timm
from seal_inspection.paths import ROOT as ROOT_DEFAULT
MEAN = torch.tensor((.485, .456, .406)).view(1, 3, 1, 1)
STD = torch.tensor((.229, .224, .225)).view(1, 3, 1, 1)
MEM_SIZE = 40000          # coreset size (greedy / random)
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
    x = torch.from_numpy(np.stack([img] * 3, -1)).permute(2, 0, 1)[None].float() / 255.0
    x = ((x - MEAN) / STD).to(device)
    f2, f3 = model(x)
    f3 = torch.nn.functional.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
    feat = torch.cat([f2, f3], 1)[0]
    C, H, W = feat.shape
    feat = feat.reshape(C, H * W).t()
    feat = torch.nn.functional.normalize(feat, dim=1)
    return feat, (H, W)


@torch.no_grad()
def greedy_coreset(feats, m, seed):
    """k-center greedy (farthest-point sampling). feats [N,D] L2-normalized. dist^2 = 2-2cos."""
    N = feats.shape[0]
    g = torch.Generator(device=feats.device).manual_seed(seed)
    first = int(torch.randint(0, N, (1,), generator=g, device=feats.device))
    sel = [first]
    min_d = (2 - 2 * (feats @ feats[first])).clamp_min_(0)
    min_d[first] = -1
    for _ in range(1, m):
        idx = int(torch.argmax(min_d))
        sel.append(idx)
        min_d = torch.minimum(min_d, (2 - 2 * (feats @ feats[idx])).clamp_min_(0))
        min_d[idx] = -1
    return torch.tensor(sel, device=feats.device)


@torch.no_grad()
def score_strips(bank, test, model, device, chunk=200000):
    scores, labels = [], []
    for img, lab, _ in test:
        feat, (H, W) = patch_features(model, img, device)
        feat = feat.half()
        maxcos = torch.full((feat.shape[0],), -2.0, device=device, dtype=torch.float16)
        for i in range(0, bank.shape[0], chunk):
            c = feat @ bank[i:i + chunk].t()
            maxcos = torch.maximum(maxcos, c.max(1).values)
        nn = (2 - 2 * maxcos.float()).clamp_min_(0).sqrt().reshape(H, W).cpu().numpy()
        nn = cv2.GaussianBlur(nn, (0, 0), 1.0)
        scores.append(float(nn.max())); labels.append(lab)
    return np.array(scores), np.array(labels)


def auroc_f1(scores, labels):
    pos, neg = scores[labels == 1], scores[labels == 0]
    au = float(np.mean([(x > y) + 0.5 * (x == y) for x in pos for y in neg]))
    best = (0., 0., 0., 0.)
    for th in np.unique(scores):
        pred = (scores >= th).astype(int)
        tp = ((pred == 1) & (labels == 1)).sum(); fp = ((pred == 1) & (labels == 0)).sum(); fn = ((pred == 0) & (labels == 1)).sum()
        pr, rc = tp / (tp + fp + 1e-9), tp / (tp + fn + 1e-9); f1 = 2 * pr * rc / (pr + rc + 1e-9)
        if f1 > best[0]: best = (f1, pr, rc, th)
    return au, best


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default=ROOT_DEFAULT); a = ap.parse_args()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    strips = os.path.join(a.root, "data/strips")
    model = timm.create_model("resnet18", pretrained=True, features_only=True, out_indices=(2, 3)).to(device).eval()

    good = [t for t in load_strips(strips, "train") if t[1] == 0]
    bank = torch.cat([patch_features(model, img, device)[0].half() for img, _, _ in good], 0)
    N = bank.shape[0]; print(f"full bank: {N} patches x {bank.shape[1]} dims (from {len(good)} good strips)", flush=True)
    test = load_strips(strips, "test")

    out = {}
    for name, idx in [("full", None),
                      ("greedy", greedy_coreset(bank, MEM_SIZE, SEED)),
                      ("random", torch.randperm(N, generator=torch.Generator(device=bank.device).manual_seed(SEED), device=bank.device)[:MEM_SIZE])]:
        b = bank if idx is None else bank[idx]
        s, l = score_strips(b, test, model, device); au, best = auroc_f1(s, l)
        out[name] = {"auroc": round(au, 4), "n_bank": int(b.shape[0]), "best_f1": round(float(best[0]), 3),
                     "precision": round(float(best[1]), 3), "recall": round(float(best[2]), 3)}
        print(f"{name.upper():7s} AUROC {au:.4f}  (N_bank={b.shape[0]})", flush=True)

    out["auroc"] = out["greedy"]["auroc"]      # headline = proper PatchCore (greedy coreset)
    out["n_def"] = int((l == 1).sum()); out["n_good"] = int((l == 0).sum())
    out["note"] = "Greedy k-center coreset (FPS) is the faithful PatchCore; 'full' = no coreset (upper bound); 'random' = old speed hack. resnet18 layer2+3, MEM_SIZE=40000, SEED=42."
    try:
        from seal_inspection.results import save_results
        save_results("baseline_patchcore", out)
        print("saved results/baseline_patchcore.json", flush=True)
    except Exception as e:
        print("[results] skip:", e)


if __name__ == "__main__":
    main()
