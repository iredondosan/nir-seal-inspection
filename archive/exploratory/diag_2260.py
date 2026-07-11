#!/usr/bin/env python3
"""Diagnose seal_2260: compare GT-seal strip vs PREDICTED-seal strip + defect heatmaps."""
import cv2, numpy as np, torch
from seal_inspection import core, cvat
from seal_inspection.paths import ROOT as R; NAME = "seal_2260_1780692167999_raw.png"
dev = "cuda" if torch.cuda.is_available() else "cpu"
seal, sk = core.load_unet(f"{R}/models/best_lite_reviewed_1280.pt", dev)
defm, dk = core.load_unet(f"{R}/models/defect_strip.pt", dev)
HS, WS = dk["HS"], dk["WS"]; MEAN = core.IMAGENET_MEAN; STD = core.IMAGENET_STD
g = cv2.imread(f"{R}/data/images/prod2/{NAME}", cv2.IMREAD_GRAYSCALE); H, W = g.shape

def defect_prob(strip):
    x = ((np.stack([strip] * 3, -1) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
    with torch.no_grad():
        p = torch.sigmoid(defm(torch.from_numpy(x).to(dev)))[0, 0].cpu().numpy()
    return p, float(cv2.GaussianBlur(p, (0, 0), 2).max())

# GT seal
node = [im for im in cvat.iter_images(f"{R}/data/annotations/prod2_reviewed.xml") if im.get("name") == NAME][0]
og, ig = cvat.seal_outer_inner(node)
mx, my = core.unroll_maps(og, ig, HS, WS); strip_gt = cv2.remap(core.normalize(g), mx, my, cv2.INTER_LINEAR)
pg, sg = defect_prob(strip_gt)

# PREDICTED seal
x0, y0, x1, y1 = core.pack_bbox(g); prob = core.predict_probability(seal, g[y0:y1, x0:x1], sk["img"], dev)
full = np.zeros((H, W), np.float32); full[y0:y1, x0:x1] = cv2.resize(prob, (x1 - x0, y1 - y0))
op, ip = core.mask_to_ring((full > sk.get("thresh", .5)).astype(np.uint8) * 255)
mx2, my2 = core.unroll_maps(op, ip, HS, WS); strip_pred = cv2.remap(core.normalize(g), mx2, my2, cv2.INTER_LINEAR)
pp, sp = defect_prob(strip_pred)
print(f"GT-seal strip defect score:   {sg:.3f}")
print(f"PRED-seal strip defect score: {sp:.3f}")
print(f"outer pts GT {len(og)} pred {len(op)} ; inner pts GT {len(ig)} pred {len(ip)}")

def panel(strip, prob, tag):
    v = cv2.cvtColor(strip, cv2.COLOR_GRAY2BGR)
    hm = cv2.applyColorMap((prob * 255).astype(np.uint8), cv2.COLORMAP_JET)
    o = cv2.addWeighted(v, 0.6, hm, 0.4, 0)
    b = np.full((24, o.shape[1], 3), 30, np.uint8); cv2.putText(b, tag, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return np.vstack([b, o])
out = np.vstack([panel(strip_gt, pg, f"GT-seal strip (score {sg:.2f})"),
                 np.full((6, WS, 3), 255, np.uint8),
                 panel(strip_pred, pp, f"PRED-seal strip (score {sp:.2f})")])
cv2.imwrite(f"{R}/outputs/diag_2260.png", out); print("wrote outputs/diag_2260.png")
