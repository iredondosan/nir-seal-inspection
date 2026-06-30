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
    """Collect pack images (handles both '*_raw.png' products and '*.jpg' prod6 sets)."""
    for pat in ("*_raw.png", "*.jpg", "*.png"):
        files = sorted(glob.glob(os.path.join(folder, pat)))
        if files:
            return files
    return []


def detect_on_strip(defect_model, strip: np.ndarray, thr: float, device: str):
    """Run the defect model on a strip; return the binary defect mask."""
    x = ((np.stack([strip] * 3, -1).astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD)
    x = x.transpose(2, 0, 1)[None]
    with torch.no_grad():
        prob = torch.sigmoid(defect_model(torch.from_numpy(x).to(device)))[0, 0].cpu().numpy()
    # smooth before thresholding so the overlay verdict matches the image-level score
    # (max of the smoothed probability), i.e. the operating point from the threshold sweep
    return (cv2.GaussianBlur(prob, (0, 0), 2) > thr).astype(np.uint8)


def _banner(width: int, text: str, h: int = 30) -> np.ndarray:
    b = np.full((h, width, 3), 35, np.uint8)
    cv2.putText(b, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return b


def process_pack(gray, seal_model, seal_img, seal_thr,
                 defect_model, defect_hs, defect_ws, defect_thr, device):
    """Run the whole pipeline on one grayscale frame.

    Returns (composite_bgr, n_detections) or (None, 0) if no seal ring was found.
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
        return None, 0

    # 2) unroll the seal and find defects on the strip
    map_x, map_y = core.unroll_maps(outer, inner, defect_hs, defect_ws)
    strip = cv2.remap(core.normalize(gray), map_x, map_y, cv2.INTER_LINEAR, borderValue=0)
    defect_mask = detect_on_strip(defect_model, strip, defect_thr, device)

    # 3) build the pack panel: crop to the seal bbox, tint the band, circle detections
    bx, by, bw, bh = cv2.boundingRect(outer.astype(np.int32))
    pad = 60
    px0, py0 = max(0, bx - pad), max(0, by - pad)
    px1, py1 = min(W, bx + bw + pad), min(H, by + bh + pad)
    panel = cv2.cvtColor(core.normalize(gray[py0:py1, px0:px1]), cv2.COLOR_GRAY2BGR)
    band = core.polygons_to_band_mask(outer, inner, H, W)[py0:py1, px0:px1]
    panel[band > 0] = np.clip(0.7 * panel[band > 0] + np.array([60, 60, 0]), 0, 255).astype(np.uint8)

    n_comp, _, stats, centroids = cv2.connectedComponentsWithStats(defect_mask)
    n_det = 0
    for i in range(1, n_comp):
        if stats[i, 4] < 5:           # ignore tiny specks
            continue
        cy, cx = int(centroids[i][1]), int(centroids[i][0])
        radius = int(np.clip(np.sqrt(stats[i, 4]) * 1.6, 18, 80))
        # map the strip-space detection centroid back to raw image coordinates
        rx, ry = int(map_x[cy, cx]) - px0, int(map_y[cy, cx]) - py0
        cv2.circle(panel, (rx, ry), radius, (0, 0, 235), 3)
        n_det += 1

    # 4) compose pack panel (top) + strip with defect overlay (bottom)
    strip_bgr = cv2.cvtColor(strip, cv2.COLOR_GRAY2BGR)
    strip_bgr[defect_mask > 0] = (0, 0, 235)
    pw = 900
    panel = cv2.resize(panel, (pw, int(panel.shape[0] * pw / panel.shape[1])))
    strip_bgr = cv2.resize(strip_bgr, (pw, int(strip_bgr.shape[0] * pw / strip_bgr.shape[1])))
    verdict = "DEFECT" if n_det else "OK"
    composite = np.vstack([
        _banner(pw, f"{verdict}  ({n_det} detection(s))   |  pack + seal mask + defect circles"),
        panel,
        _banner(pw, "predicted seal unrolled + defect (red)"),
        strip_bgr,
    ])
    return composite, n_det


def main():
    ap = argparse.ArgumentParser(description="End-to-end seal+defect inference on unlabeled packs.")
    ap.add_argument("--seal", required=True, help="seal segmentation checkpoint (.pt)")
    ap.add_argument("--defect", required=True, help="defect segmentation checkpoint (.pt)")
    ap.add_argument("--input", required=True, help="folder of pack images")
    ap.add_argument("--out", required=True, help="output folder for composites")
    ap.add_argument("--limit", type=int, default=0, help="process at most N images (0 = all)")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seal_model, sk = core.load_unet(a.seal, device)
    defect_model, dk = core.load_unet(a.defect, device)
    seal_img, seal_thr = sk["img"], sk.get("thresh", 0.5)
    defect_hs, defect_ws, defect_thr = dk["HS"], dk["WS"], dk.get("thr", 0.5)

    os.makedirs(a.out, exist_ok=True)
    files = find_images(a.input)
    if a.limit:
        files = files[:a.limit]

    for p in files:
        gray = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        composite, n_det = process_pack(gray, seal_model, seal_img, seal_thr,
                                        defect_model, defect_hs, defect_ws, defect_thr, device)
        base = os.path.splitext(os.path.basename(p))[0]
        if composite is None:
            print(f"{base}: no seal ring found")
            continue
        cv2.imwrite(os.path.join(a.out, base + ".png"), composite)
        print(f"{base}: {'DEFECT' if n_det else 'OK'} ({n_det})")


if __name__ == "__main__":
    main()
