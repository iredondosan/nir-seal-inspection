#!/usr/bin/env python3
"""Regenerate: fig_final_defect_multi (predicted seal, detected defects, excl. seal_1313),
and fig_defect_pred_test (GT seal strips, GT green vs predicted red) — current pipeline."""
import glob, os, shutil, numpy as np, cv2, torch
import xml.etree.ElementTree as ET
from seal_inspection import core
R = "/home/ubuntu/TFM/seal-inspection"; dev = "cuda" if torch.cuda.is_available() else "cpu"
FIG = f"{R}/docs/thesis_figures"; MEAN, STD = core.IMAGENET_MEAN, core.IMAGENET_STD
seal, sk = core.load_unet(f"{R}/models/best_lite_reviewed_1280.pt", dev)
defm, dk = core.load_unet(f"{R}/models/defect_strip.pt", dev); HS, WS = dk["HS"], dk["WS"]; DTHR = dk.get("thr", .5); OP = 0.5
lab = {ln.split(",")[0]: int(ln.split(",")[1]) for ln in open(f"{R}/data/holdout_labels.csv").read().splitlines()[1:]}
def pp(s): return np.array([[float(a) for a in p.split(",")] for p in s.strip().split(";")], np.float32)
def find(nm):
    for x in glob.glob(f"{R}/data/annotations/prod*_reviewed.xml"):
        for im in ET.parse(x).getroot().findall("image"):
            if os.path.splitext(im.get("name"))[0] == nm: return im
    return None
def prod_path(nm):
    h = glob.glob(f"{R}/data/images/*/{nm}.png") + glob.glob(f"{R}/data/images/*/{nm}.jpg")
    return (h[0].split("/images/")[1].split("/")[0], h[0]) if h else (None, None)
def dprob(strip):
    x = ((np.stack([strip]*3, -1)/255.0 - MEAN)/STD).transpose(2, 0, 1)[None].astype(np.float32)
    with torch.no_grad(): return torch.sigmoid(defm(torch.from_numpy(x).to(dev)))[0, 0].cpu().numpy()

# ---------- fig_final_defect_multi (predicted seal) ----------
def compose_pred(nm):
    prod, path = prod_path(nm); g = cv2.imread(path, 0); H, W = g.shape
    x0, y0, x1, y1 = core.pack_bbox(g)
    prob = core.predict_probability(seal, g[y0:y1, x0:x1], sk["img"], dev)
    full = np.zeros((H, W), np.float32); full[y0:y1, x0:x1] = cv2.resize(prob, (x1-x0, y1-y0))
    O, I = core.mask_to_ring((full > sk.get("thresh", .5)).astype(np.uint8)*255)
    if O is None: return None
    mx, my = core.unroll_maps(O, I, HS, WS); strip = cv2.remap(core.normalize(g), mx, my, cv2.INTER_LINEAR, borderValue=0)
    dp = dprob(strip); score = float(cv2.GaussianBlur(dp, (0, 0), 2).max()); dmask = (dp > DTHR).astype(np.uint8)
    pad = 30; cx0, cy0, cx1, cy1 = max(0, x0-pad), max(0, y0-pad), min(W, x1+pad), min(H, y1+pad)
    panel = cv2.cvtColor(core.normalize(g)[cy0:cy1, cx0:cx1], cv2.COLOR_GRAY2BGR)
    cv2.polylines(panel, [(O-[cx0, cy0]).astype(np.int32)], True, (255, 255, 0), 2)
    cv2.polylines(panel, [(I-[cx0, cy0]).astype(np.int32)], True, (0, 255, 255), 2)
    nl, lbc, st, cent = cv2.connectedComponentsWithStats(dmask)
    for i in range(1, nl):
        if st[i, 4] < 8: continue
        yy, xx = int(cent[i][1]), int(cent[i][0])
        cv2.circle(panel, (int(mx[yy, xx])-cx0, int(my[yy, xx])-cy0), int(np.clip(np.sqrt(st[i, 4])*1.6, 16, 70)), (0, 0, 235), 3)
    sv = cv2.cvtColor(strip, cv2.COLOR_GRAY2BGR); sv[dmask > 0] = (0, 0, 235)
    v = "DEFECT" if score >= OP else "GOOD"; col = (0, 0, 235) if v == "DEFECT" else (0, 200, 0)
    pw = 1200; rz = lambda im: cv2.resize(im, (pw, int(im.shape[0]*pw/im.shape[1])))
    ban = np.full((40, pw, 3), 30, np.uint8); cv2.putText(ban, f"{prod}/{nm[:22]}  score={score:.2f} -> {v}", (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
    return np.vstack([ban, rz(panel), rz(sv)])

defs = sorted(n for n, l in lab.items() if l == 1 and n != "seal_1313_1780666315292_raw")
seen = {}; pick = []
for n in defs:
    p = prod_path(n)[0]
    if seen.get(p, 0) < 2: pick.append(n); seen[p] = seen.get(p, 0)+1
    if len(pick) >= 6: break
tiles = [t for t in (compose_pred(n) for n in pick) if t is not None]
w = min(t.shape[1] for t in tiles); grid = np.vstack([t[:, :w] for t in tiles])
out = f"{FIG}/fig_final_defect_multi.png"
cv2.imwrite(out, grid); print(f"wrote fig_final_defect_multi ({len(tiles)} packs, no seal_1313)")

# ---------- fig_defect_pred_test (GT seal, GT green vs pred red) ----------
def gt_strip(nm):
    node = find(nm); prod, path = prod_path(nm); g = cv2.imread(path, 0); H, W = g.shape
    sell = sorted([pp(p.get("points")) for p in node.findall("polygon") if p.get("label") == "sellado"],
                  key=lambda q: cv2.contourArea(q.astype(np.float32)), reverse=True)
    defsp = [pp(p.get("points")) for p in node.findall("polygon") if p.get("label") in ("defect", "liquid")]
    dm = np.zeros((H, W), np.uint8)
    for d in defsp: cv2.fillPoly(dm, [d.astype(np.int32)], 255)
    mx, my = core.unroll_maps(sell[0], sell[1], HS, WS)
    strip = cv2.remap(core.normalize(g), mx, my, cv2.INTER_LINEAR, borderValue=0)
    gtm = (cv2.remap(dm, mx, my, cv2.INTER_LINEAR, borderValue=0) > 127).astype(np.uint8)
    dp = dprob(strip); pr = (dp > DTHR).astype(np.uint8)
    sv = cv2.cvtColor(strip, cv2.COLOR_GRAY2BGR); sv[gtm > 0] = (0, 200, 0); sv[pr > 0] = (0, 0, 235)
    pw = 1200; rz = lambda im: cv2.resize(im, (pw, int(im.shape[0]*pw/im.shape[1])))
    ban = np.full((34, pw, 3), 30, np.uint8); cv2.putText(ban, f"{prod}/{nm[:24]}  (GT green / pred red)", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 220, 220), 1, cv2.LINE_AA)
    return np.vstack([ban, rz(sv)])
tiles2 = [t for t in (gt_strip(n) for n in pick[:4]) if t is not None]
w2 = min(t.shape[1] for t in tiles2); grid2 = np.vstack([t[:, :w2] for t in tiles2])
out2 = f"{FIG}/fig_defect_pred_test.png"
if os.path.exists(out2): shutil.copy2(out2, out2 + ".prehullfix.bak")
cv2.imwrite(out2, grid2); print(f"wrote fig_defect_pred_test ({len(tiles2)} strips)")
