#!/usr/bin/env python3
"""
Per-pack seal visualization:
  top    = original image with predicted seal mask overlaid (red)
  bottom = the seal ring unrolled into a straight strip, split into 4 stacked rows

The unroll is derived from the predicted mask's two ring boundaries (outer + inner).

Usage:
  python seal_viz.py --weights best.pt --input prod2 --output seal_viz --limit 12
"""
import os, glob, argparse
import numpy as np, cv2, torch
from predict import U, load_gray3   # exact model + preprocessing

def norm_view(gray):
    lo, hi = np.percentile(gray, [1, 99.5]); hi = max(hi, lo + 1)
    return np.clip((gray.astype(np.float32) - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)

def predict_mask(model, dev, img3, H, W, oh, ow, mean, std, thr):
    x = cv2.resize(img3, (W, H)).astype(np.float32) / 255.0
    x = ((x - mean) / std).transpose(2, 0, 1)[None]
    with torch.no_grad():
        prob = torch.sigmoid(model(torch.from_numpy(x).to(dev)))[0, 0].cpu().numpy()
    return (cv2.resize(prob, (ow, oh)) > thr).astype(np.uint8) * 255

def ring_contours(mask, band_px=90):
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))     # drop specks
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((35, 35), np.uint8))  # bridge small gaps
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts: return None, None, False
    allpts = np.vstack([c.reshape(-1, 2) for c in cnts])     # all seal pixels (handles fragmented rings)
    outer = cv2.convexHull(allpts)
    fill = np.zeros_like(mask); cv2.drawContours(fill, [outer], -1, 255, -1)
    hole = cv2.subtract(fill, mask)
    cnts2, _ = cv2.findContours(hole, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts2 = [c for c in cnts2 if cv2.contourArea(c) > 0.2 * cv2.contourArea(outer)]
    if cnts2:                                   # clean ring: real inner hole
        inner = max(cnts2, key=cv2.contourArea); clean = True
    else:                                       # broken ring: offset outer inward by a fixed band
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band_px + 1, 2 * band_px + 1))
        ic, _ = cv2.findContours(cv2.erode(fill, k), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not ic: return None, None, False
        inner = max(ic, key=cv2.contourArea); clean = False
    return outer.reshape(-1, 2).astype(np.float32), inner.reshape(-1, 2).astype(np.float32), clean

def resample(poly, n):
    p = np.r_[poly, poly[:1]]
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    t = np.linspace(0, d[-1], n, endpoint=False)
    return np.stack([np.interp(t, d, p[:, 0]), np.interp(t, d, p[:, 1])], 1)

def csmooth(a, k=15):
    ker = np.ones(k) / k
    return np.convolve(np.r_[a[-k:], a, a[:k]], ker, "same")[k:-k]

def ccw(p): return cv2.contourArea(p.astype(np.float32), oriented=True) > 0

def unroll(gray, outer, inner, Hs, Ws):
    O = resample(outer, Ws); I = resample(inner, Ws)
    if ccw(O) != ccw(I): I = I[::-1]
    j = int(np.argmin(np.hypot(I[:, 0] - O[0, 0], I[:, 1] - O[0, 1])))  # align starts
    I = np.roll(I, -j, axis=0)
    for arr in (O, I):
        arr[:, 0] = csmooth(arr[:, 0]); arr[:, 1] = csmooth(arr[:, 1])
    a = np.linspace(0, 1, Hs)[:, None]
    mapx = (O[:, 0][None, :] * (1 - a) + I[:, 0][None, :] * a).astype(np.float32)
    mapy = (O[:, 1][None, :] * (1 - a) + I[:, 1][None, :] * a).astype(np.float32)
    return cv2.remap(gray, mapx, mapy, cv2.INTER_LINEAR, borderValue=0)

def banner(w, text, h=34):
    b = np.full((h, w, 3), 35, np.uint8)
    cv2.putText(b, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return b

def compose(orig, mask, strip, rows=4, panel_w=900, clean=True):
    # top: pack overlay
    packv = cv2.cvtColor(norm_view(orig), cv2.COLOR_GRAY2BGR)
    packv[mask > 0] = np.clip(0.5 * packv[mask > 0] + np.array([0, 0, 160]), 0, 255).astype(np.uint8)
    ph = int(packv.shape[0] * panel_w / packv.shape[1])
    packv = cv2.resize(packv, (panel_w, ph))
    # bottom: strip split into `rows` stacked segments
    sv = cv2.cvtColor(norm_view(strip), cv2.COLOR_GRAY2BGR)
    seg = sv.shape[1] // rows
    parts = []
    for k in range(rows):
        r = cv2.resize(sv[:, k * seg:(k + 1) * seg], (panel_w, sv.shape[0]))
        parts.append(r)
        if k < rows - 1: parts.append(np.full((3, panel_w, 3), 255, np.uint8))
    strip_panel = np.vstack(parts)
    tag = "" if clean else "  [APPROX: inner edge offset, ring was broken]"
    return np.vstack([banner(panel_w, "Pack + predicted seal mask"), packv,
                      banner(panel_w, f"Unrolled seal (outer=top edge), {rows} rows = perimeter L->R" + tag),
                      strip_panel])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--strip-h", type=int, default=130)
    ap.add_argument("--strip-w", type=int, default=3600)  # divisible by rows
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    ck = torch.load(args.weights, map_location=dev)
    H, W, THR = ck["img_h"], ck["img_w"], ck.get("thresh", 0.5)
    mean = np.array(ck["mean"], np.float32); std = np.array(ck["std"], np.float32)
    model = U().to(dev); model.load_state_dict(ck["state_dict"]); model.eval()
    os.makedirs(args.output, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.input, "*_raw.png")))
    if args.limit: files = files[:args.limit]
    print(f"device={dev}  {len(files)} packs -> {args.output}")
    ok = skip = 0
    for i, p in enumerate(files, 1):
        orig, img3 = load_gray3(p)
        if img3 is None: skip += 1; continue
        oh, ow = orig.shape
        mask = predict_mask(model, dev, img3, H, W, oh, ow, mean, std, THR)
        outer, inner, clean = ring_contours(mask)
        if outer is None or inner is None or len(outer) < 20 or len(inner) < 20:
            skip += 1; print("  skip (no seal found):", os.path.basename(p)); continue
        strip = unroll(orig, outer, inner, args.strip_h, args.strip_w)
        out = compose(orig, mask, strip, rows=args.rows, clean=clean)
        cv2.imwrite(os.path.join(args.output, os.path.basename(p).replace("_raw.png", "_viz.png")), out)
        ok += 1
        if i % 50 == 0 or i == len(files): print(f"  [{i}/{len(files)}]")
    print(f"done: {ok} written, {skip} skipped")

if __name__ == "__main__":
    main()
