# -*- coding: utf-8 -*-
"""PI-4 ablation: copy-paste ON (defect_jit.pt) vs OFF (defect_nopaste.pt), everything else equal
(both --sealjit, same seed/split, same seal @1280). Scores isolated GT strip + E2E on the 179
hold-out with the identical harness (same predicted strip fed to both heads). Saves results JSON."""
import sys, os, glob, json, cv2, numpy as np, torch
R = "/home/ubuntu/TFM/seal-inspection"; sys.path.insert(0, R)
import seal_inspection.core as core
dev = "cuda" if torch.cuda.is_available() else "cpu"
MEAN, STD = core.IMAGENET_MEAN, core.IMAGENET_STD

seal, sk = core.load_unet(f"{R}/models/best_lite_reviewed_1280.pt", dev)
THR = sk.get("thresh", 0.5)
MODELS = {"copypaste_on": "models/defect_jit.pt", "copypaste_off": "models/defect_nopaste.pt"}
nets, HS, WS = {}, None, None
for k, p in MODELS.items():
    m, dk = core.load_unet(f"{R}/{p}", dev); nets[k] = m; HS, WS = dk["HS"], dk["WS"]

@torch.no_grad()
def dscore(m, strip):
    x = ((np.stack([strip]*3, -1)/255. - MEAN)/STD).transpose(2, 0, 1)[None].astype(np.float32)
    p = torch.sigmoid(m(torch.from_numpy(x).to(dev)))[0, 0].cpu().numpy()
    return float(cv2.GaussianBlur(p, (0, 0), 2).max())

def auroc(S, L):
    pos, neg = S[L == 1], S[L == 0]
    return float(np.mean([(a > b) + 0.5*(a == b) for a in pos for b in neg]))

# ---- E2E: predicted seal @1280, same strip to both heads ----
lab = {ln.split(",")[0]: int(ln.split(",")[1]) for ln in
       open(f"{R}/data/holdout_labels.csv").read().splitlines()[1:] if ln.strip()}
Se = {k: [] for k in MODELS}; Le = []; nloc = 0
for nm, l in lab.items():
    h = glob.glob(f"{R}/data/images/*/{nm}.png") + glob.glob(f"{R}/data/images/*/{nm}.jpg")
    if not h: continue
    g = cv2.imread(h[0], 0); H, W = g.shape; x0, y0, x1, y1 = core.pack_bbox(g)
    prob = core.predict_probability(seal, g[y0:y1, x0:x1], sk["img"], dev)
    full = np.zeros((H, W), np.float32); full[y0:y1, x0:x1] = cv2.resize(prob, (x1-x0, y1-y0))
    O, I = core.mask_to_ring((full > THR).astype(np.uint8)*255)
    Le.append(l)
    if O is None:
        for k in MODELS: Se[k].append(0.0)
        continue
    nloc += 1
    mx, my = core.unroll_maps(O, I, HS, WS)
    strip = cv2.remap(core.normalize(g), mx, my, cv2.INTER_LINEAR, borderValue=0)
    for k, m in nets.items(): Se[k].append(dscore(m, strip))
Le = np.array(Le)

# ---- GT strips (data/strips/test), same strip to both heads ----
Sg = {k: [] for k in MODELS}; Lg = []
for ip in sorted(glob.glob(f"{R}/data/strips/test/img/*.png")):
    mp = ip.replace("/img/", "/mask/")
    if not os.path.exists(mp): continue
    s = cv2.imread(ip, 0); mk = cv2.imread(mp, 0)
    Lg.append(1 if (mk > 127).sum() > 0 else 0)
    for k, m in nets.items(): Sg[k].append(dscore(m, s))
Lg = np.array(Lg)

out = {
    "question": "PI-4: aporta el copy-paste de defectos reales? copy-paste ON vs OFF, resto identico (ambos --sealjit, mismo seed/split, sellado best_lite_reviewed_1280.pt @1280).",
    "harness": "core.py torch pipeline; misma tira predicha a las dos cabezas; hold-out 179 (E2E) y data/strips/test (GT). Umbral 0.50.",
    "test": {"holdout_e2e": int(len(Le)), "n_def": int(Le.sum()), "n_good": int((Le == 0).sum()),
             "e2e_localised": nloc, "gt_strips": int(len(Lg)), "gt_def": int(Lg.sum())},
    "models": {}, "ckpts": MODELS,
}
for k in MODELS:
    Sa, Sga = np.array(Se[k]), np.array(Sg[k])
    out["models"][k] = {
        "gt_auroc": round(auroc(Sga, Lg), 4),
        "e2e_auroc": round(auroc(Sa, Le), 4),
        "recall": int(((Sa >= 0.5) & (Le == 1)).sum()), "n_def": int(Le.sum()),
        "fp": int(((Sa >= 0.5) & (Le == 0)).sum()), "n_good": int((Le == 0).sum()),
    }
on, off = out["models"]["copypaste_on"], out["models"]["copypaste_off"]
out["delta_e2e_auroc_on_minus_off"] = round(on["e2e_auroc"] - off["e2e_auroc"], 4)
out["delta_gt_auroc_on_minus_off"] = round(on["gt_auroc"] - off["gt_auroc"], 4)

os.makedirs(f"{R}/results", exist_ok=True)
json.dump(out, open(f"{R}/results/ablation_copypaste.json", "w"), indent=2, ensure_ascii=False)
print(json.dumps(out, indent=2, ensure_ascii=False))
