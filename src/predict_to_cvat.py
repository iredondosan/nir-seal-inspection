#!/usr/bin/env python3
"""
Generate CVAT-importable polygon pre-annotations from model predictions.
Predicts the seal mask in FULL RAW-IMAGE coordinates, extracts the outer + inner
ring boundaries, simplifies them to polygons, and writes a 'CVAT for images 1.1' XML.
Workflow: upload raw images to a CVAT task -> import this XML -> correct -> export.

  python predict_to_cvat.py --weights models/best.pt --input data/images/prod1 \
      --output outputs/preannotations/prod1.xml
"""
import os, glob, argparse
import numpy as np, cv2, torch
from predict import U, load_gray3   # ResNet34 U-Net + percentile-norm preprocessing

def predict_mask(model, dev, img3, H, W, oh, ow, mean, std, thr):
    x = cv2.resize(img3, (W, H)).astype(np.float32) / 255.0
    x = ((x - mean) / std).transpose(2, 0, 1)[None]
    with torch.no_grad():
        prob = torch.sigmoid(model(torch.from_numpy(x).to(dev)))[0, 0].cpu().numpy()
    return (cv2.resize(prob, (ow, oh)) > thr).astype(np.uint8) * 255   # FULL raw-res mask

def ring_contours(mask, band_px=90):
    """outer + inner ring boundary in raw image coords (handles broken rings)."""
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((35, 35), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts: return None, None
    allpts = np.vstack([c.reshape(-1, 2) for c in cnts])
    outer = cv2.convexHull(allpts)
    fill = np.zeros_like(m); cv2.drawContours(fill, [outer], -1, 255, -1)
    hole = cv2.subtract(fill, m)
    cnts2, _ = cv2.findContours(hole, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts2 = [c for c in cnts2 if cv2.contourArea(c) > 0.2 * cv2.contourArea(outer)]
    if cnts2:
        inner = max(cnts2, key=cv2.contourArea)
    else:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*band_px+1, 2*band_px+1))
        ic, _ = cv2.findContours(cv2.erode(fill, k), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not ic: return None, None
        inner = max(ic, key=cv2.contourArea)
    return outer.reshape(-1, 2).astype(np.float32), inner.reshape(-1, 2).astype(np.float32)

def simplify(cnt, eps_frac=0.004):
    c = cnt.reshape(-1, 1, 2).astype(np.float32)
    peri = cv2.arcLength(c, True)
    ap = cv2.approxPolyDP(c, eps_frac * peri, True).reshape(-1, 2)
    return ap

def poly_str(pts): return ";".join(f"{x:.2f},{y:.2f}" for x, y in pts)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--label", default="sellado")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    ck = torch.load(args.weights, map_location=dev)
    H, W, THR = ck["img_h"], ck["img_w"], ck.get("thresh", 0.5)
    mean = np.array(ck.get("mean", (.485,.456,.406)), np.float32); std = np.array(ck.get("std", (.229,.224,.225)), np.float32)
    model = U().to(dev); model.load_state_dict(ck["state_dict"]); model.eval()

    files = sorted(glob.glob(os.path.join(args.input, "*_raw.png")))
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    rows = []; ok = skip = 0
    for i, p in enumerate(files):
        orig, img3 = load_gray3(p)
        if img3 is None: skip += 1; continue
        oh, ow = orig.shape
        mask = predict_mask(model, dev, img3, H, W, oh, ow, mean, std, THR)
        outer, inner = ring_contours(mask)
        polys = ""
        if outer is not None and inner is not None and len(outer) >= 8 and len(inner) >= 8:
            for cnt in (outer, inner):   # outer (larger) first, then inner
                pts = simplify(cnt)
                pts[:, 0] = np.clip(pts[:, 0], 0, ow-1); pts[:, 1] = np.clip(pts[:, 1], 0, oh-1)
                polys += f'    <polygon label="{args.label}" source="auto" occluded="0" points="{poly_str(pts)}" z_order="0"></polygon>\n'
            ok += 1
        else:
            skip += 1
        rows.append(f'  <image id="{i}" name="{os.path.basename(p)}" width="{ow}" height="{oh}">\n{polys}  </image>')
    xml = '<?xml version="1.0" encoding="utf-8"?>\n<annotations>\n  <version>1.1</version>\n' + "\n".join(rows) + "\n</annotations>\n"
    open(args.output, "w").write(xml)
    print(f"{args.output}: {len(files)} images, {ok} with polygons, {skip} empty")

if __name__ == "__main__":
    main()
