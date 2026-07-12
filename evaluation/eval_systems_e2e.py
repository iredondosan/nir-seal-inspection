#!/usr/bin/env python3
"""Consistent E2E eval of every deployable config with the THESIS pipeline (core.py, torch).

Uses the RESOLUTION-SPECIFIC seal checkpoints (best_lite_reviewed_{1280,512,384}.pt) so each
seal runs at its native resolution; defect head = ResNet18 or TinyUNet. All configs scored on
the COMMON set of hold-out packs localised by every seal, on core.py -> the deployed config
reproduces tab:umbral (~0.968, 21/23). Saves results/systems_e2e.json.
"""
import glob, numpy as np, cv2, torch
from seal_inspection import core
from seal_inspection.tiny_unet import TinyUNet
from seal_inspection.paths import ROOT as R
from seal_inspection.results import save_results
dev = "cuda" if torch.cuda.is_available() else "cpu"
MEAN, STD = core.IMAGENET_MEAN, core.IMAGENET_STD

SEAL_CKPT = {1280: "best_lite_reviewed_1280.pt", 512: "best_lite_reviewed_512.pt", 384: "best_lite_reviewed_384.pt"}
seals = {r: core.load_unet(f"{R}/models/{p}", dev)[0] for r, p in SEAL_CKPT.items()}
_, sk = core.load_unet(f"{R}/models/best_lite_reviewed_1280.pt", dev)  # for thresh
THR = sk.get("thresh", 0.5)
resnet, dk = core.load_unet(f"{R}/models/defect_strip.pt", dev); HS, WS = dk["HS"], dk["WS"]
tiny = TinyUNet(base=16, in_ch=1)
tiny.load_state_dict(torch.load(f"{R}/models/tiny_defect.pt", map_location=dev, weights_only=False)["state_dict"])
tiny.to(dev).eval()

@torch.no_grad()
def dscore_resnet(strip):
    x = ((np.stack([strip]*3, -1)/255. - MEAN)/STD).transpose(2, 0, 1)[None].astype(np.float32)
    p = torch.sigmoid(resnet(torch.from_numpy(x).to(dev)))[0, 0].cpu().numpy()
    return float(cv2.GaussianBlur(p, (0, 0), 2).max())

@torch.no_grad()
def dscore_tiny(strip):
    x = ((strip/255. - 0.5)/0.5).astype(np.float32)[None, None]
    p = torch.sigmoid(tiny(torch.from_numpy(x).to(dev)))[0, 0].cpu().numpy()
    return float(cv2.GaussianBlur(p, (0, 0), 2).max())

lab = {}
for ln in open(f"{R}/data/holdout_labels.csv").read().splitlines()[1:]:
    nm, l = ln.split(","); lab[nm] = int(l)

strips = {r: {} for r in SEAL_CKPT}; labels = {}
for nm, l in lab.items():
    hits = glob.glob(f"{R}/data/images/*/{nm}.png") + glob.glob(f"{R}/data/images/*/{nm}.jpg")
    if not hits: continue
    g = cv2.imread(hits[0], cv2.IMREAD_GRAYSCALE); H, W = g.shape; x0, y0, x1, y1 = core.pack_bbox(g)
    labels[nm] = l; crop = g[y0:y1, x0:x1]; gn = core.normalize(g)
    for r in SEAL_CKPT:
        prob = core.predict_probability(seals[r], crop, r, dev)
        full = np.zeros((H, W), np.float32); full[y0:y1, x0:x1] = cv2.resize(prob, (x1-x0, y1-y0))
        O, I = core.mask_to_ring((full > THR).astype(np.uint8)*255)
        if O is None:
            strips[r][nm] = None; continue
        mx, my = core.unroll_maps(O, I, HS, WS)
        strips[r][nm] = cv2.remap(gn, mx, my, cv2.INTER_LINEAR, borderValue=0)

common = [nm for nm in labels if all(strips[r].get(nm) is not None for r in SEAL_CKPT)]
loc = {r: int(sum(v is not None for v in strips[r].values())) for r in SEAL_CKPT}
ndef = sum(labels[nm] for nm in common); ngood = sum(labels[nm] == 0 for nm in common)
print(f"hold-out {len(labels)} -> localised {loc} -> COMMON {len(common)} ({ndef} def / {ngood} good)", flush=True)

def metrics(fn, res):
    sc = np.array([fn(strips[res][nm]) for nm in common]); lb = np.array([labels[nm] for nm in common])
    pos, neg = sc[lb == 1], sc[lb == 0]
    au = float(np.mean([(x > y) + 0.5*(x == y) for x in pos for y in neg]))
    sweep = {f"{th:.2f}": {"tp": int(((sc >= th) & (lb == 1)).sum()), "fp": int(((sc >= th) & (lb == 0)).sum())}
             for th in [0.30, 0.50, 0.70, 0.85, 0.90, 0.95]}
    return {"auroc": round(au, 4), "recall_0.50": sweep["0.50"]["tp"], "fp_0.50": sweep["0.50"]["fp"], "sweep": sweep}

out = {"harness": "core.py (torch) thesis pipeline; resolution-specific seal checkpoints; COMMON localised set (identical packs for all configs).",
       "test_set": {"holdout": len(labels), "localised_per_res": loc, "common": len(common), "n_def": ndef, "n_good": ngood},
       "configs": {}}
for res in SEAL_CKPT:
    for kind, fn in [("resnet18", dscore_resnet), ("tiny", dscore_tiny)]:
        k = f"seal{res}_{kind}"; out["configs"][k] = metrics(fn, res); m = out["configs"][k]
        print(f"  {k}: AUROC {m['auroc']}  recall@0.5 {m['recall_0.50']}/{ndef}  fp {m['fp_0.50']}/{ngood}", flush=True)

save_results("systems_e2e", out)
print("saved results/systems_e2e.json", flush=True)
