"""
pipeline.py — full end-to-end inference on an UNLABELED pack.

    raw image
      -> pack detection + crop          (core.pack_bbox)
      -> seal segmentation              (seal U-Net)
      -> ring extraction                (core.mask_to_ring)
      -> unroll the seal into a strip   (core.unroll_maps)
      -> defect segmentation            (defect U-Net)
      -> map detections back to the pack and decide DEFECT / OK

Produces a QC composite per pack: the cropped pack with the predicted seal mask
(cyan) and red circles where defects were found, and the unrolled strip below.

Usage:
    python -m seal_inspection.pipeline --seal models/seal.pt --defect models/defect.pt \
        --input data/images/prod2 --out outputs/pipeline [--limit 20]
"""
from __future__ import annotations
import os
import glob
import argparse
import numpy as np
import cv2
import torch
from . import core
from .core import IMAGENET_MEAN, IMAGENET_STD


def find_images(folder: str) -> list[str]:
    """Collect pack images (handles both '*_raw.png' and '*.jpg' products)."""
    for pat in ("*_raw.png", "*.jpg", "*.png"):
        files = sorted(glob.glob(os.path.join(folder, pat)))
        if files:
            return files
    return []


def detect_on_strip(defect_model, strip: np.ndarray, thr: float, device: str):
    """Run the defect model on a strip; return (binary defect mask, image-level score)."""
    x = ((np.stack([strip] * 3, -1).astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD)
    x = x.transpose(2, 0, 1)[None]
    with torch.no_grad():
        prob = torch.sigmoid(defect_model(torch.from_numpy(x).to(device)))[0, 0].cpu().numpy()
    # smooth before thresholding so the overlay verdict matches the image-level score
    # (max of the smoothed probability), i.e. the operating point from the threshold sweep
    sm = cv2.GaussianBlur(prob, (0, 0), 2)
    return (sm > thr).astype(np.uint8), float(sm.max())


def _banner(width: int, text: str, h: int = 30) -> np.ndarray:
    b = np.full((h, width, 3), 35, np.uint8)
    cv2.putText(b, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return b


def process_pack(gray, seal_model, seal_img, seal_thr, branches, device):
    """Run the whole pipeline on one grayscale frame.

    `branches` is a list of dicts, one per unroll/defect-model pair:
        {"name", "model", "unroll", "hs", "ws", "thr"}
    Each branch unrolls the predicted seal its own way and scores it with its
    matching model; a pack is DEFECT if ANY branch fires (image-level max-pool).
    Two branches (perpendicular + correspondence) are the production ensemble —
    each catches defects the other's unroll smears.

    Returns (composite_bgr, n_detections, score) or (None, 0, 0.0) if no ring.
    """
    H, W = gray.shape
    # 1) crop to the tray and predict the seal ring
    x0, y0, x1, y1 = core.pack_bbox(gray)
    crop = gray[y0:y1, x0:x1]
    prob = core.predict_probability(seal_model, crop, seal_img, device)
    full = np.zeros((H, W), np.float32)
    full[y0:y1, x0:x1] = cv2.resize(prob, (x1 - x0, y1 - y0))
    seal_mask = (full > seal_thr).astype(np.uint8) * 255
    outer, inner = core.mask_to_ring(seal_mask)
    if outer is None:
        return None, 0, 0.0

    # 2) pack panel: crop to the seal bbox and tint the band
    bx, by, bw, bh = cv2.boundingRect(outer.astype(np.int32))
    pad = 60
    px0, py0 = max(0, bx - pad), max(0, by - pad)
    px1, py1 = min(W, bx + bw + pad), min(H, by + bh + pad)
    panel = cv2.cvtColor(core.normalize(gray[py0:py1, px0:px1]), cv2.COLOR_GRAY2BGR)
    band = core.polygons_to_band_mask(outer, inner, H, W)[py0:py1, px0:px1]
    panel[band > 0] = np.clip(0.7 * panel[band > 0] + np.array([60, 60, 0]), 0, 255).astype(np.uint8)

    # 3) run every branch; collect detections (raw-image coords) and strip overlays
    dets, strip_rows, score = [], [], 0.0
    for br in branches:
        map_x, map_y = br["unroll"](outer, inner, br["hs"], br["ws"])
        strip = cv2.remap(core.normalize(gray), map_x, map_y, cv2.INTER_LINEAR, borderValue=0)
        defect_mask, s = detect_on_strip(br["model"], strip, br["thr"], device)
        score = max(score, s)
        n_comp, _, stats, centroids = cv2.connectedComponentsWithStats(defect_mask)
        for i in range(1, n_comp):
            if stats[i, 4] < 5:       # ignore tiny specks
                continue
            cy, cx = int(centroids[i][1]), int(centroids[i][0])
            radius = int(np.clip(np.sqrt(stats[i, 4]) * 1.6, 18, 80))
            rx, ry = int(map_x[cy, cx]) - px0, int(map_y[cy, cx]) - py0  # strip -> raw
            dets.append((rx, ry, radius))
        sb = cv2.cvtColor(strip, cv2.COLOR_GRAY2BGR)
        sb[defect_mask > 0] = (0, 0, 235)
        strip_rows.append((br["name"], sb))

    # 3b) merge near-duplicate detections (both branches circle the same real defect)
    merged = []
    for rx, ry, r in dets:
        if any((rx - mrx) ** 2 + (ry - mry) ** 2 < (max(r, mr) * 0.8) ** 2 for mrx, mry, mr in merged):
            continue
        merged.append((rx, ry, r))
    for rx, ry, r in merged:
        cv2.circle(panel, (rx, ry), r, (0, 0, 235), 3)
    n_det = len(merged)

    # 4) compose: pack panel (top) + one strip per branch below
    pw = 900
    verdict = "DEFECT" if n_det else "OK"
    panel = cv2.resize(panel, (pw, int(panel.shape[0] * pw / panel.shape[1])))
    parts = [_banner(pw, f"{verdict}  ({n_det} detection(s))  score {score:.2f}  |  pack + seal mask + defect circles"),
             panel]
    for name, sb in strip_rows:
        sb = cv2.resize(sb, (pw, int(sb.shape[0] * pw / sb.shape[1])))
        parts.append(_banner(pw, f"{name} unroll + defect (red)"))
        parts.append(sb)
    return np.vstack(parts), n_det, score


def main():
    ap = argparse.ArgumentParser(description="End-to-end seal+defect inference on unlabeled packs.")
    ap.add_argument("--seal", required=True, help="seal segmentation checkpoint (.pt)")
    ap.add_argument("--defect", required=True,
                    help="defect checkpoint trained on PERPENDICULAR-unroll strips (defect_strip.pt)")
    ap.add_argument("--defect-legacy", default=None,
                    help="defect checkpoint trained on CORRESPONDENCE-unroll strips (defect_strip.prev.pt). "
                         "If given, the two are ensembled (recommended).")
    ap.add_argument("--thr", type=float, default=None,
                    help="defect score threshold (default: each checkpoint's stored 'thr')")
    ap.add_argument("--input", required=True, help="folder of pack images")
    ap.add_argument("--out", required=True, help="output folder for composites")
    ap.add_argument("--limit", type=int, default=0, help="process at most N images (0 = all)")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seal_model, sk = core.load_unet(a.seal, device)
    seal_img, seal_thr = sk["img"], sk.get("thresh", 0.5)

    # branch 1: perpendicular-to-outer unroll (new model). branch 2 (optional): the
    # legacy correspondence unroll + its matching model -> the production ensemble.
    dm, dk = core.load_unet(a.defect, device)
    branches = [dict(name="perpendicular", model=dm, unroll=core.unroll_maps,
                     hs=dk["HS"], ws=dk["WS"], thr=a.thr if a.thr is not None else dk.get("thr", 0.5))]
    if a.defect_legacy:
        lm, lk = core.load_unet(a.defect_legacy, device)
        branches.append(dict(name="correspondence", model=lm, unroll=core.unroll_maps_legacy,
                             hs=lk["HS"], ws=lk["WS"], thr=a.thr if a.thr is not None else lk.get("thr", 0.5)))
    print(f"defect branches: {[b['name'] for b in branches]}  thr={[round(b['thr'],3) for b in branches]}")

    os.makedirs(a.out, exist_ok=True)
    files = find_images(a.input)
    if a.limit:
        files = files[:a.limit]

    for p in files:
        gray = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        composite, n_det, score = process_pack(gray, seal_model, seal_img, seal_thr, branches, device)
        base = os.path.splitext(os.path.basename(p))[0]
        if composite is None:
            print(f"{base}: no seal ring found")
            continue
        cv2.imwrite(os.path.join(a.out, base + ".png"), composite)
        print(f"{base}: {'DEFECT' if n_det else 'OK'} ({n_det}) score {score:.3f}")


if __name__ == "__main__":
    main()
