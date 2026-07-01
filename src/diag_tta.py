#!/usr/bin/env python3
"""Test unroll-TTA: score seal_2260 under several unroll parameterizations and max-pool.
A thin defect smeared at one parameterization should be sharp at another."""
import numpy as np, cv2, torch
from seal_inspection import core, cvat
R = "/home/ubuntu/TFM/seal-inspection"; NAME = "seal_2260_1780692167999_raw.png"
dev = "cuda" if torch.cuda.is_available() else "cpu"
seal, sk = core.load_unet(f"{R}/models/best_lite_reviewed_1280.pt", dev)
defm, dk = core.load_unet(f"{R}/models/defect_strip.pt", dev)
HS, WS = dk["HS"], dk["WS"]; MEAN = core.IMAGENET_MEAN; STD = core.IMAGENET_STD
g = cv2.imread(f"{R}/data/images/prod2/{NAME}", cv2.IMREAD_GRAYSCALE); H, W = g.shape

def punroll(O, I, roll=0, a_hi=1.15, jit=0.0):
    Or = core._resample_closed(O, WS); Ir = core._resample_closed(I, WS)
    if core._is_ccw(Or) != core._is_ccw(Ir):
        Ir = Ir[::-1]
    j = int(np.argmin(np.hypot(Ir[:, 0] - Or[0, 0], Ir[:, 1] - Or[0, 1])))
    Ir = np.roll(Ir, -j, 0); Ir = np.roll(Ir, roll, 0)
    for arr in (Or, Ir):
        arr[:, 0] = core._smooth_closed(arr[:, 0]); arr[:, 1] = core._smooth_closed(arr[:, 1])
    if jit:
        Or = Or + np.random.randn(*Or.shape) * jit; Ir = Ir + np.random.randn(*Ir.shape) * jit
    a = np.linspace(-0.15, a_hi, HS)[:, None]
    mx = (Or[:, 0][None] * (1 - a) + Ir[:, 0][None] * a).astype(np.float32)
    my = (Or[:, 1][None] * (1 - a) + Ir[:, 1][None] * a).astype(np.float32)
    return cv2.remap(core.normalize(g), mx, my, cv2.INTER_LINEAR, borderValue=0)

def score(strip):
    x = ((np.stack([strip] * 3, -1) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
    with torch.no_grad():
        p = torch.sigmoid(defm(torch.from_numpy(x).to(dev)))[0, 0].cpu().numpy()
    return float(cv2.GaussianBlur(p, (0, 0), 2).max())

# predicted seal ring
x0, y0, x1, y1 = core.pack_bbox(g); prob = core.predict_probability(seal, g[y0:y1, x0:x1], sk["img"], dev)
full = np.zeros((H, W), np.float32); full[y0:y1, x0:x1] = cv2.resize(prob, (x1 - x0, y1 - y0))
O, I = core.mask_to_ring((full > sk.get("thresh", .5)).astype(np.uint8) * 255)

np.random.seed(0)
variants = [(0, 1.15, 0), (-12, 1.15, 0), (12, 1.15, 0), (0, 1.30, 0), (-12, 1.30, 0), (12, 1.30, 0),
            (-6, 1.20, 1.5), (6, 1.20, 1.5), (0, 1.20, 2.0), (-20, 1.25, 0), (20, 1.25, 0), (0, 1.40, 0)]
scores = []
for roll, ahi, jit in variants:
    s = score(punroll(O, I, roll, ahi, jit)); scores.append(s)
    print(f"  roll={roll:+3d} a_hi={ahi:.2f} jit={jit}: score {s:.3f}")
print(f"\nbaseline (roll=0): {scores[0]:.3f}   |   TTA max-pool: {max(scores):.3f}  -> {'DETECTED' if max(scores)>=0.43 else 'missed'}")
