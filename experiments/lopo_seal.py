"""LOPO del sellado (Tabla 4.3): validación cruzada dejando un producto fuera.

Para cada producto (prod1..prod5) entrena la U-Net de sellado con los otros cuatro, desde el
codificador ImageNet (sin fuga entre productos), y evalúa zero-shot en el excluido (cero de
sus etiquetas). Determinista (SEED=42): Dice zero-shot por producto, media 0.955 +/- 0.010.
El dict INS (Dice "en muestra" de referencia) está fijado a mano. Ver docs/EXPERIMENTS.md.
"""
import os, glob, random, numpy as np, cv2, torch, torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
import xml.etree.ElementTree as ET
import segmentation_models_pytorch as smp

from seal_inspection.paths import ROOT as R; dev = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
IMG = 1280; BATCH = 2; EPOCHS = 60; SAMPLES = 400; VAL_PER = 2; THRESH = 0.5; MARGIN = 40; P_CONTAM = 0.8
IM_MEAN = (.485, .456, .406); IM_STD = (.229, .224, .225)
DATASETS = [("data/annotations/prod2_reviewed.xml", "data/images/prod2", "prod2"),
            ("data/annotations/prod1_reviewed.xml", "data/images/prod1", "prod1"),
            ("data/annotations/prod3_reviewed.xml", "data/images/prod3", "prod3"),
            ("data/annotations/prod4_reviewed.xml", "data/images/prod4", "prod4"),
            ("data/annotations/prod5_reviewed.xml", "data/images/prod5", "prod5"),
            ("data/annotations/prod6_reviewed.xml", "data/images/prod6", "prod6")]
FORCE_TRAIN = {"seal_1998_1780688689500_raw.png"}
ENC = torch.load(f"{R}/models/best_lite_reviewed_1280.pt", map_location="cpu", weights_only=False)["encoder"]
print("encoder:", ENC, "| clean ImageNet base (no seal-pretraining -> no cross-product leakage)", flush=True)

def norm(g):
    lo, hi = np.percentile(g, [1, 99.5]); hi = max(hi, lo + 1)
    return np.clip((g.astype(np.float32) - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
def conveyor_cols(N):
    cm = np.median(N, 0).astype(np.float32); on = np.where(cm > cm.max() * 0.5)[0]
    cL, cR = on.min(), on.max(); gd = np.gradient(cm)
    return int(np.argmax(gd[max(0, cL-60):cL+60]) + max(0, cL-60)), int(np.argmin(gd[cR-60:cR+60]) + (cR-60))
def pack_bbox(gray):
    N = norm(gray); h, w = N.shape
    try: cL, cR = conveyor_cols(N)
    except Exception: cL, cR = 0, w
    top = np.median(N[20:240, :], 0); bot = np.median(N[h-240:h-20, :], 0); ref = np.maximum(top, bot)
    diff = np.clip(np.tile(ref, (h, 1)) - N.astype(np.float32), 0, 255); diff[:, :cL] = 0; diff[:, cR:] = 0
    m = cv2.morphologyEx((diff > 20).astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((11, 11), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((41, 41), np.uint8))
    c = [c for c in cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] if cv2.contourArea(c) > h*w*0.02]
    if not c: return 0, 0, w, h
    x, y, bw, bh = cv2.boundingRect(max(c, key=cv2.contourArea))
    return max(0, x-MARGIN), max(0, y-MARGIN), min(w, x+bw+MARGIN), min(h, y+bh+MARGIN)
def parse_pts(s): return np.array([[float(a) for a in p.split(",")] for p in s.strip().split(";")], np.float32)
def tags_of(node): return {t.get("label") for t in node.findall("tag")}
def seal_mask(node):
    W = int(node.get("width")); H = int(node.get("height"))
    pl = [parse_pts(pg.get("points")) for pg in node.findall("polygon") if pg.get("label") == "sellado"]
    if len(pl) < 2: return None
    pl = sorted(pl, key=lambda p: cv2.contourArea(p.astype(np.float32)), reverse=True)
    m = np.zeros((H, W), np.uint8); cv2.fillPoly(m, [pl[0].astype(np.int32)], 1); cv2.fillPoly(m, [pl[1].astype(np.int32)], 0)
    return m

def load_per_product():
    per = {}
    for xmlrel, imgrel, prod in DATASETS:
        for node in ET.parse(f"{R}/{xmlrel}").getroot().findall("image"):
            nm = node.get("name"); p = f"{R}/{imgrel}/{nm}"; tg = tags_of(node)
            if "exclude" in tg: continue
            if "reviewed" not in tg: continue
            m = seal_mask(node)
            if m is None or not os.path.exists(p): continue
            g = cv2.imread(p, cv2.IMREAD_GRAYSCALE); x0, y0, x1, y1 = pack_bbox(g)
            gc = norm(g[y0:y1, x0:x1]); per.setdefault(prod, []).append((np.stack([gc, gc, gc], -1), m[y0:y1, x0:x1], nm))
    return per

def load_cutouts(products):
    insts = []
    for xmlrel, imgrel, prod in DATASETS:
        if prod not in products: continue
        for node in ET.parse(f"{R}/{xmlrel}").getroot().findall("image"):
            name = node.get("name"); polys = [pg for pg in node.findall("polygon") if pg.get("label") in ("defect", "liquid")]
            if not polys: continue
            p = f"{R}/{imgrel}/{name}";
            if not os.path.exists(p): continue
            g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if g is None: continue
            for pg in polys:
                pts = parse_pts(pg.get("points")).astype(np.int32); x, y, bw, bh = cv2.boundingRect(pts)
                if bw < 4 or bh < 4: continue
                patch = g[y:y+bh, x:x+bw].copy(); al = np.zeros((bh, bw), np.uint8); cv2.fillPoly(al, [pts-[x, y]], 255)
                insts.append((patch, cv2.GaussianBlur(al, (0, 0), 2).astype(np.float32)/255.0))
    return insts
def paste_contaminants(img3, band, insts):
    if not insts: return img3
    h, w = band.shape; ys, xs = np.where(band > 0)
    if len(xs) == 0: return img3
    out = img3.copy()
    for _ in range(random.randint(2, 5)):
        patch, al = random.choice(insts); s = random.uniform(0.12, 0.55) if random.random() < 0.7 else random.uniform(0.55, 1.8)
        pw, ph = max(3, int(patch.shape[1]*s)), max(3, int(patch.shape[0]*s))
        p = cv2.resize(patch, (pw, ph)).astype(np.float32); a = cv2.resize(al, (pw, ph))
        M = cv2.getRotationMatrix2D((pw/2, ph/2), random.uniform(0, 360), 1.0)
        p = cv2.warpAffine(p, M, (pw, ph), borderValue=0); a = cv2.warpAffine(a, M, (pw, ph), borderValue=0)
        if random.random() < 0.5: p = cv2.flip(p, 1); a = cv2.flip(a, 1)
        p = np.clip(p*random.uniform(0.7, 1.2), 0, 255)
        k = random.randrange(len(xs)); cx, cy = int(xs[k]), int(ys[k]); x0, y0 = cx-pw//2, cy-ph//2
        ix0, iy0 = max(0, x0), max(0, y0); ix1, iy1 = min(w, x0+pw), min(h, y0+ph)
        if ix1 <= ix0 or iy1 <= iy0: continue
        px0, py0 = ix0-x0, iy0-y0; px1, py1 = px0+(ix1-ix0), py0+(iy1-iy0)
        aa = (a[py0:py1, px0:px1]*random.uniform(0.8, 1.0))[..., None]; pp = p[py0:py1, px0:px1][..., None]
        reg = out[iy0:iy1, ix0:ix1].astype(np.float32); out[iy0:iy1, ix0:ix1] = np.clip(reg*(1-aa)+pp*aa, 0, 255).astype(np.uint8)
    return out

def ttf():
    geo = [A.HorizontalFlip(p=.5), A.VerticalFlip(p=.5)]
    try: geo.append(A.Affine(scale=(.85, 1.15), translate_percent=(0, .06), rotate=(-180, 180), border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=.9))
    except TypeError: geo.append(A.Affine(scale=(.85, 1.15), translate_percent=(0, .06), rotate=(-180, 180), p=.9))
    ph = [A.RandomBrightnessContrast(.4, .4, p=.8), A.RandomGamma((60, 140), p=.6)]
    try: ph.append(A.GaussNoise(p=.2))
    except Exception: pass
    return A.Compose([A.Resize(IMG, IMG), *geo, *ph, A.Normalize(IM_MEAN, IM_STD), ToTensorV2()])
def etf(): return A.Compose([A.Resize(IMG, IMG), A.Normalize(IM_MEAN, IM_STD), ToTensorV2()])
ETF = etf()

class TR(torch.utils.data.Dataset):
    def __init__(s, d, L, cut): s.d = d; s.tf = ttf(); s.L = L; s.cut = cut
    def __len__(s): return s.L
    def __getitem__(s, i):
        im, mk = s.d[random.randrange(len(s.d))]
        if s.cut and random.random() < P_CONTAM: im = paste_contaminants(im, mk, s.cut)
        o = s.tf(image=im, mask=mk); return o["image"], o["mask"].float().unsqueeze(0)

def dloss(l, t, e=1.):
    p = torch.sigmoid(l); return (1 - ((2*(p*t).sum((2, 3))+e)/(p.sum((2, 3))+t.sum((2, 3))+e))).mean()
bce = nn.BCEWithLogitsLoss()
@torch.no_grad()
def dice_one(model, im3, mk):
    o = ETF(image=im3, mask=mk); x = o["image"].unsqueeze(0).to(dev); y = o["mask"].float().unsqueeze(0).unsqueeze(0).to(dev)
    p = (torch.sigmoid(model(x)) > THRESH).float(); return ((2*(p*y).sum()+1)/(p.sum()+y.sum()+1)).item()

from scipy.ndimage import binary_erosion, distance_transform_edt
BDPX = 5.0
def _bd(m):
    return m.astype(bool) & ~binary_erosion(m.astype(bool), iterations=1, border_value=0)
@torch.no_grad()
def metrics_one(model, im3, mk):
    """Dice (suavizado, como dice_one) + Boundary-IoU/HD95/ASSD, a resolucion IMG."""
    o = ETF(image=im3, mask=mk); x = o["image"].unsqueeze(0).to(dev)
    gt = (o["mask"].cpu().numpy() > 0).astype(np.uint8)
    pr = (torch.sigmoid(model(x))[0, 0].cpu().numpy() > THRESH).astype(np.uint8)
    inter = int((gt & pr).sum()); dice = (2*inter + 1) / (int(pr.sum()) + int(gt.sum()) + 1)
    bg, bp = _bd(gt), _bd(pr)
    if bg.sum() == 0 or bp.sum() == 0:
        return {"dice": float(dice), "biou": 0.0, "hd95": float("inf"), "assd": float("inf")}
    dtg, dtp = distance_transform_edt(~bg), distance_transform_edt(~bp)
    dd = np.concatenate([dtg[bp], dtp[bg]])
    Gb, Pb = gt.astype(bool) & (dtg <= BDPX), pr.astype(bool) & (dtp <= BDPX)
    biou = (Gb & Pb).sum() / max(1, (Gb | Pb).sum())
    return {"dice": float(dice), "biou": float(biou), "hd95": float(np.percentile(dd, 95)), "assd": float(dd.mean())}
def _agg(rows):
    return {k: float(np.mean([r[k] for r in rows])) for k in ["dice", "biou", "hd95", "assd"]}
def _save_model(model, rel):
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "encoder": ENC, "img": IMG, "thresh": THRESH, "mean": IM_MEAN, "std": IM_STD}, f"{R}/{rel}")
    print("  [saved]", rel, flush=True)

def run_fold(heldout, per, cut):
    train, val = [], []
    for prod in per:
        if prod == heldout: continue
        samps = per[prod][:]; random.Random(SEED).shuffle(samps)
        forced = [s for s in samps if s[2] in FORCE_TRAIN]; rest = [s for s in samps if s[2] not in FORCE_TRAIN]
        vp = VAL_PER if len(rest) >= 6 else 0
        val += [(im, mk) for im, mk, _ in rest[:vp]]; train += [(im, mk) for im, mk, _ in rest[vp:]+forced]
    test = [(im, mk) for im, mk, _ in per[heldout]]
    dl = torch.utils.data.DataLoader(TR(train, SAMPLES, cut), batch_size=BATCH, shuffle=True)
    model = smp.Unet(ENC, encoder_weights="imagenet", in_channels=3, classes=1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    amp = (dev == "cuda"); scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best, best_state = 0., None
    for ep in range(1, EPOCHS+1):
        model.train()
        for x, y in dl:
            x, y = x.to(dev), y.to(dev); opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=amp): lo = model(x); L = bce(lo, y) + dloss(lo, y)
            scaler.scale(L).backward(); scaler.step(opt); scaler.update()
        sch.step(); model.eval()
        vd = np.mean([dice_one(model, im, mk) for im, mk in val]) if val else 0.
        if vd >= best: best = vd; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state: model.load_state_dict(best_state)
    model.eval()
    _save_model(model, f"models/lopo_zeroshot_{heldout}.pt")
    return _agg([metrics_one(model, im, mk) for im, mk in test]), len(test), float(best)

PRODS = ["prod1", "prod2", "prod3", "prod4", "prod5"]


def run_insample(per, cut):
    """In-sample reference, COMPUTED (not hardcoded): train ONE model on ALL 5 products
    (ImageNet base, same 1280/epochs/aug as the zero-shot folds), holding out VAL_PER pieces
    per product, and return {prod: validation Dice}. This is the apples-to-apples
    'con el producto incluido' baseline for the LOPO comparison (same setup as zero-shot)."""
    train, val_by = [], {}
    for prod in PRODS:
        samps = per[prod][:]; random.Random(SEED).shuffle(samps)
        forced = [s for s in samps if s[2] in FORCE_TRAIN]; rest = [s for s in samps if s[2] not in FORCE_TRAIN]
        vp = VAL_PER if len(rest) >= 6 else 0
        val_by[prod] = [(im, mk) for im, mk, _ in rest[:vp]]
        train += [(im, mk) for im, mk, _ in rest[vp:]+forced]
    allval = [x for v in val_by.values() for x in v]
    dl = torch.utils.data.DataLoader(TR(train, SAMPLES, cut), batch_size=BATCH, shuffle=True)
    model = smp.Unet(ENC, encoder_weights="imagenet", in_channels=3, classes=1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    amp = (dev == "cuda"); scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best, best_state = 0., None
    for ep in range(1, EPOCHS+1):
        model.train()
        for x, y in dl:
            x, y = x.to(dev), y.to(dev); opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=amp): lo = model(x); L = bce(lo, y) + dloss(lo, y)
            scaler.scale(L).backward(); scaler.step(opt); scaler.update()
        sch.step(); model.eval()
        vd = np.mean([dice_one(model, im, mk) for im, mk in allval]) if allval else 0.
        if vd >= best: best = vd; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state: model.load_state_dict(best_state)
    model.eval()
    _save_model(model, "models/lopo_insample.pt")
    return {prod: _agg([metrics_one(model, im, mk) for im, mk in vb]) for prod, vb in val_by.items()}


print("loading samples...", flush=True); per = load_per_product()
print("pieces per product:", {k: len(v) for k, v in per.items()}, flush=True)

print("=== IN-SAMPLE: training one model on ALL 5 products (ImageNet, same setup) ===", flush=True)
INS = run_insample(per, load_cutouts(PRODS))
print("in-sample per product:", {k: round(v["dice"], 3) for k, v in INS.items()}, flush=True)

res = {}
for heldout in PRODS:
    cut = load_cutouts([p for p in per if p != heldout])
    mm, n, bv = run_fold(heldout, per, cut); res[heldout] = mm
    print("LOPO %-6s zero-shot  Dice %.3f  B-IoU %.3f  HD95 %.2f  ASSD %.2f  (n=%d, best-val %.3f) | in-sample Dice %.3f  caida %.3f"
          % (heldout, mm["dice"], mm["biou"], mm["hd95"], mm["assd"], n, bv, INS[heldout]["dice"], INS[heldout]["dice"]-mm["dice"]), flush=True)

MK = ["dice", "biou", "hd95", "assd"]
zs_mean = {k: float(np.mean([res[p][k] for p in PRODS])) for k in MK}
is_mean = {k: float(np.mean([INS[p][k] for p in PRODS])) for k in MK}
zs_dice_std = float(np.std([res[p]["dice"] for p in PRODS]))
print("\nZERO-SHOT  Dice %.4f +/- %.4f  |  B-IoU %.3f  HD95 %.2f  ASSD %.2f"
      % (zs_mean["dice"], zs_dice_std, zs_mean["biou"], zs_mean["hd95"], zs_mean["assd"]), flush=True)
print("IN-SAMPLE  Dice %.4f  |  caida media (Dice) %.4f" % (is_mean["dice"], is_mean["dice"]-zs_mean["dice"]), flush=True)

import json
from seal_inspection.results import save_results
out = {"metric_keys": MK, "zero_shot": res, "in_sample": INS,
       "zero_shot_mean": zs_mean, "zero_shot_dice_std": zs_dice_std, "in_sample_mean": is_mean,
       "caida": {p: round(INS[p]["dice"] - res[p]["dice"], 4) for p in PRODS},
       "caida_media_dice": round(is_mean["dice"] - zs_mean["dice"], 4),
       "models": {"in_sample": "models/lopo_insample.pt",
                  "zero_shot": {p: f"models/lopo_zeroshot_{p}.pt" for p in PRODS}}}
json.dump(out, open(f"{R}/outputs/lopo_seal_results.json", "w"), indent=2)
save_results("lopo_seal", out)
open(f"{R}/outputs/lopo_seal.done", "w").write("done")
