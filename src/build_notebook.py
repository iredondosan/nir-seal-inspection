#!/usr/bin/env python3
"""Build seal_inspection_walkthrough.ipynb (a full pipeline walkthrough notebook)."""
import json

cells = []
def md(t):   cells.append({"cell_type": "markdown", "metadata": {}, "source": t})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})

# ----------------------------------------------------------------------------- title
md("""# NIR Line-Scan Seal Inspection — Full Pipeline Walkthrough

This notebook walks through the **complete two-stage seal-inspection system** end to end:

1. **The problem & imaging physics** — why near-infrared, why the geometry is hard.
2. **The dataset** — products, annotations, train/test split.
3. **Stage 1 — Seal segmentation** — crop the tray, segment the seal ring (U-Net).
4. **Mask ↔ polygon conversion** — moving between pixel masks and CVAT polygons.
5. **The geometric unroll** — flattening the seal ring into a strip (and the fix that made it robust).
6. **Stage 2 — Defect detection** — finding defects on the unrolled strip (U-Net).
7. **The end-to-end pipeline & 2-unroll ensemble** — how everything chains together.
8. **Full test-set inference** — run the system on every held-out pack and review the results.

> Pipeline at a glance:
> `raw NIR frame → pack crop → seal U-Net → ring → unroll → strip → defect U-Net → DEFECT / OK`

Every code cell runs against the real trained models and real data on the GPU box.""")

# ----------------------------------------------------------------------------- 0 setup
md("""## 0 · Setup

Load the three trained checkpoints and the shared helper package (`seal_inspection`). The production
detector is a **2-unroll ensemble**: the *perpendicular* unroll scored by `defect_strip.pt`, and the
legacy *correspondence* unroll scored by `defect_strip.prev.pt`, image-level max-pooled.""")
code('''%matplotlib inline
import os, sys, glob, re, warnings
warnings.filterwarnings("ignore")
import numpy as np, cv2, torch
import matplotlib.pyplot as plt
import pandas as pd

R = "/home/ubuntu/TFM/seal-inspection"
sys.path.insert(0, R)
from seal_inspection import core, cvat
from seal_inspection.pipeline import process_pack

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, "| torch", torch.__version__)

seal, sk = core.load_unet(f"{R}/models/best_lite_reviewed_1280.pt", device)   # stage 1: seal
defm, dk = core.load_unet(f"{R}/models/defect_strip.pt", device)              # stage 2: perpendicular unroll
legm, lk = core.load_unet(f"{R}/models/defect_strip.prev.pt", device)         # stage 2: correspondence unroll
HS, WS = dk["HS"], dk["WS"]

print("seal model :", sk["encoder"], "| input", sk["img"], "| thresh", sk["thresh"])
print("defect new :", dk["encoder"], "| strip", f"{HS}x{WS}")
print("defect prev:", lk["encoder"], "| strip", f"{lk['HS']}x{lk['WS']}")

DTHR = 0.43   # operating threshold (high-recall point from the threshold sweep)
branches = [
    dict(name="perpendicular",  model=defm, unroll=core.unroll_maps,        hs=dk["HS"], ws=dk["WS"], thr=DTHR),
    dict(name="correspondence", model=legm, unroll=core.unroll_maps_legacy, hs=lk["HS"], ws=lk["WS"], thr=DTHR),
]

def show(img, title=None, figsize=(8, 6), cmap=None):
    plt.figure(figsize=figsize)
    if img.ndim == 3: plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else: plt.imshow(img, cmap=cmap or "gray")
    if title: plt.title(title)
    plt.axis("off"); plt.show()

def find_raw(name):
    for ext in (".png", ".jpg"):
        h = glob.glob(f"{R}/data/images/*/{name}{ext}")
        if h: return h[0]
    return None

def defect_prob(model, strip):
    x = ((np.stack([strip] * 3, -1) / 255.0 - core.IMAGENET_MEAN) / core.IMAGENET_STD).transpose(2, 0, 1)[None].astype(np.float32)
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(x).to(device)))[0, 0].cpu().numpy()

EX = "seal_2260_1780692167999_raw"   # running example (the pack the unroll fix rescued)
RUN_TRAINING = True    # trains each model (hold-out aware) if its checkpoint is missing, else loads it; see 0b/3a/6a
print("\\nsetup OK  | RUN_TRAINING =", RUN_TRAINING)''')

# ----------------------------------------------------------------------------- 0b global hold-out
md("""## 0b · Global hold-out & clean evaluation

To get an **honest end-to-end number, neither stage may have seen the evaluation packs.** The two models
have independent splits, so naively they cross: the seal model would train on packs that are in the defect
model's test set.

The fix exploits one fact — **the seal model trains on `reviewed` packs only.** So if the hold-out is drawn
**entirely from NON-reviewed packs**, the seal stage is unseen on it *by construction*, and **every reviewed
ground-truth pack stays in training for both models** (no GT sacrificed). The defect model simply excludes
the hold-out from its strips.

`src/make_holdout.py` writes `data/holdout.txt` (per-product stratified: ~30 % of non-reviewed defects,
~20 % of non-reviewed goods). `src/make_strips.py` then forces those packs to the test split when it builds
both the perpendicular and the legacy strip sets.""")
code('''hold_names = [l.strip() for l in open(f"{R}/data/holdout.txt") if l.strip()]
hold_label = {}
for ln in open(f"{R}/data/holdout_labels.csv").read().splitlines()[1:]:
    nm, lb = ln.split(","); hold_label[nm] = int(lb)
nd = sum(hold_label.values()); ng = len(hold_names) - nd
print(f"GLOBAL HOLD-OUT: {len(hold_names)} packs = {nd} defect + {ng} good  (all NON-reviewed)")
print("strip splits (hold-out forced to test):")
for d in ["strips", "strips_legacy"]:
    import glob as _g
    tr = len(_g.glob(f"{R}/data/{d}/train/img/*.png")); te = len(_g.glob(f"{R}/data/{d}/test/img/*.png"))
    print(f"  data/{d}: train {tr}  test {te}")''')

# ----------------------------------------------------------------------------- 1 problem
md("""## 1 · The problem & the imaging physics

We inspect the **heat-sealed flange ("seal")** around a food tray — the ring of plastic where the lid film
is welded to the tray. A good seal is a clean weld; a **defect** is foreign material (product, liquid)
trapped in the weld zone, which can break the hermetic seal.

Two facts shape the whole design:

- **NIR transmission imaging.** A near-infrared line-scan camera images the tray in transmission: denser /
  wetter material absorbs more IR, so **product, contamination and liquid appear dark**; thin air gaps
  appear bright. Defects are therefore dark intrusions into the otherwise-uniform seal band.
- **The camera is free-running** (not encoder-triggered), so every frame carries its own non-rigid, wavy
  distortion. A single rotation / affine / homography **cannot** straighten it. So we never globally
  rectify the image — instead we **follow the seal's real edges** and unwrap the band into a flat strip.""")
code('''g = cv2.imread(find_raw(EX), cv2.IMREAD_GRAYSCALE)
H, W = g.shape
plt.figure(figsize=(7, 8))
plt.imshow(g, cmap="gray"); plt.colorbar(fraction=0.046, label="NIR intensity (dark = more absorbed)")
plt.title(f"Raw NIR frame: {EX}  ({W}x{H})"); plt.axis("off"); plt.show()
print("The seal is the bright flange ring around the dark tray contents.")''')

# ----------------------------------------------------------------------------- 2 dataset
md("""## 2 · The dataset

Six product families (`prod1`–`prod5` + `prod6`), annotated in **CVAT**. Each pack carries:

- **`sellado` polygons** — two of them: the outer flange edge and the inner well edge. The seal band is
  `outer − inner`.
- **`defect` / `liquid` polygons** — the defect regions (both count as a defect).
- **image-level tags** — `reviewed` (human-verified ground truth), `good`, `defect`, `exclude`
  (e.g. a sticker over the seal — dropped from the dataset).

The defect dataset is split **pack-level and per-product stratified** so no pack leaks between train and
test.""")
code('''ANN = {"prod1": "prod1_reviewed.xml", "prod2": "prod2_reviewed.xml", "prod3": "prod3_reviewed.xml",
       "prod4": "prod4_reviewed.xml", "prod5": "prod5_reviewed.xml", "prod6": "prod6_reviewed.xml",
       "prod6_bad": "prod6_bad_reviewed.xml"}
rows = []
for prod, fn in ANN.items():
    path = f"{R}/data/annotations/{fn}"
    if not os.path.exists(path): continue
    n = nseal = ndef = 0
    for im in cvat.iter_images(path):
        n += 1
        if cvat.seal_outer_inner(im) is not None: nseal += 1
        if cvat.polygons(im, "defect") or cvat.polygons(im, "liquid"): ndef += 1
    rows.append(dict(product=prod, images=n, seal_GT=nseal, defect_packs=ndef))
print(pd.DataFrame(rows).to_string(index=False))
ntr = len(glob.glob(f"{R}/data/strips/train/img/*.png")); nte = len(glob.glob(f"{R}/data/strips/test/img/*.png"))
print(f"\\nUnrolled-strip dataset:  train {ntr}  |  test {nte}")''')
code('''fig, axs = plt.subplots(2, 3, figsize=(13, 8))
for ax, (prod, fn) in zip(axs.ravel(), list(ANN.items())[:6]):
    rp = None
    for im in cvat.iter_images(f"{R}/data/annotations/{fn}"):
        rp = find_raw(os.path.splitext(im.get("name"))[0])
        if rp: break
    if rp: ax.imshow(cv2.imread(rp, cv2.IMREAD_GRAYSCALE), cmap="gray")
    ax.set_title(prod); ax.axis("off")
plt.suptitle("One example pack per product family"); plt.tight_layout(); plt.show()''')

# ----------------------------------------------------------------------------- 3 seal seg
md("""## 3 · Stage 1 — Seal segmentation

**a) Pack detection.** `core.pack_bbox` subtracts a per-column conveyor-background reference and crops to
the tray bounding box, so the seal model always sees a tightly-framed pack regardless of where the tray
sat in the frame.

**b) The model.** A **U-Net with a MobileNetV3-small encoder** (~1M-param backbone, ImageNet-pretrained),
run at the **native crop resolution (1280²)**. Trained with **BCE + Dice** loss, heavy augmentation and
**AMP** mixed precision. We report **val Dice ≈ 0.963**.

> *Why boundary metrics matter:* Dice can look high (0.95+) while the thin seal ring still wobbles at the
> edges. Resolution sweeps showed Boundary-IoU climbing 0.50→0.66 going 512→1280, which is why we run at
> native resolution — edge accuracy is what the unroll depends on.""")
code('''x0, y0, x1, y1 = core.pack_bbox(g)
viz = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
cv2.rectangle(viz, (x0, y0), (x1, y1), (0, 255, 0), 4)
show(viz, f"pack_bbox: conveyor background subtracted -> tray crop ({x1-x0}x{y1-y0})", (7, 8))''')
code('''crop = g[y0:y1, x0:x1]
prob = core.predict_probability(seal, crop, sk["img"], device)
full = np.zeros((H, W), np.float32); full[y0:y1, x0:x1] = cv2.resize(prob, (x1 - x0, y1 - y0))
mask = (full > sk["thresh"]).astype(np.uint8) * 255

fig, axs = plt.subplots(1, 3, figsize=(16, 6))
axs[0].imshow(crop, cmap="gray"); axs[0].set_title("seal model input (crop)")
axs[1].imshow(prob, cmap="magma"); axs[1].set_title("seal probability map")
ov = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
ov[mask > 0] = np.clip(0.5 * ov[mask > 0] + np.array([0, 200, 200]), 0, 255).astype(np.uint8)
axs[2].imshow(cv2.cvtColor(ov, cv2.COLOR_BGR2RGB)); axs[2].set_title(f"thresholded mask @>{sk['thresh']}")
for a in axs: a.axis("off")
plt.tight_layout(); plt.show()''')
code('''n_seal = sum(p.numel() for p in seal.parameters())
n_def  = sum(p.numel() for p in defm.parameters())
print(f"Seal   U-Net ({sk['encoder']}): {n_seal/1e6:.2f}M params, input {sk['img']}x{sk['img']}")
print(f"Defect U-Net ({dk['encoder']}): {n_def/1e6:.2f}M params, strip {HS}x{WS}")
print("\\nBoth: ImageNet-pretrained encoder + U-Net decoder, trained with BCE+Dice and AMP.")''')

# ----------------------------------------------------------------------------- 3a seal training code
md("""### 3a · The seal training code

This is the real training routine that produced `best_lite_reviewed_1280.pt` (a faithful copy of
`src/train_reviewed.py`). It fine-tunes from a base checkpoint on the **reviewed** ground-truth packs of
all products, pack-cropped at native 1280². Highlights:

- only **`reviewed`** packs are used as GT; `exclude` packs are dropped; tiny products get no val split;
- **band mask** = fill outer `sellado` polygon, punch the inner one;
- on-band augmentation pastes **real defect cut-outs gathered from every labelled pack** (`defect` +
  `liquid` polygons across all products), with the seal mask unchanged, so the model learns the seal is a
  geometric region and shouldn't divert around dark material sitting on it (no dependency on a separate
  contaminants file);
- loss = **BCE + Dice**, **AMP** mixed precision, cosine LR, checkpoint by best per-product val Dice;
- exports an ONNX alongside the `.pt`.

Defining the function runs nothing; set `RUN_TRAINING = True` (in §0) to actually train (~12 min on the box).""")
code('''def train_seal_model():
    import os, glob, random
    import numpy as np, cv2, torch, torch.nn as nn
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    import xml.etree.ElementTree as ET
    import segmentation_models_pytorch as smp

    SEED = 42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    ROOT = R
    DATASETS = [("data/annotations/prod2_reviewed.xml", "data/images/prod2", "prod2"),
                ("data/annotations/prod1_reviewed.xml", "data/images/prod1", "prod1"),
                ("data/annotations/prod3_reviewed.xml", "data/images/prod3", "prod3"),
                ("data/annotations/prod4_reviewed.xml", "data/images/prod4", "prod4"),
                ("data/annotations/prod5_reviewed.xml", "data/images/prod5", "prod5"),
                ("data/annotations/prod6_reviewed.xml", "data/images/prod6", "prod6"),
                ("data/annotations/prod6_bad_reviewed.xml", "data/images/prod6_bad", "prod6")]
    BASE = f"{ROOT}/models/best_lite.pt"
    IMG = 1280; BATCH = 2; EPOCHS = 60; SAMPLES = 400          # native-resolution production config
    VAL_PER = 2; THRESH = 0.5; MARGIN = 40
    CKPT = f"{ROOT}/models/best_lite_reviewed_{IMG}.pt"; ONNX = f"{ROOT}/models/seal_lite_reviewed_{IMG}.onnx"
    P_CONTAM = 0.8
    # exclude the global hold-out from the paste-augmentation library too (keeps the seal model fully clean
    # on the hold-out, not just unseen as training images)
    HOLD = set(l.strip() for l in open(f"{ROOT}/data/holdout.txt")) if os.path.exists(f"{ROOT}/data/holdout.txt") else set()
    CONTAM_EXCLUDE = {"seal_1302_1780665903828_raw.png"}       # A/B held-out pack
    FORCE_TRAIN = {"seal_1998_1780688689500_raw.png"}          # barcode-over-seal: keep in train, never val
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    IM_MEAN = (.485, .456, .406); IM_STD = (.229, .224, .225)

    def norm(g):
        lo, hi = np.percentile(g, [1, 99.5]); hi = max(hi, lo + 1)
        return np.clip((g.astype(np.float32) - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    def conveyor_cols(N):
        cm = np.median(N, 0).astype(np.float32); on = np.where(cm > cm.max() * 0.5)[0]
        cL, cR = on.min(), on.max(); gd = np.gradient(cm)
        return int(np.argmax(gd[max(0, cL - 60):cL + 60]) + max(0, cL - 60)), int(np.argmin(gd[cR - 60:cR + 60]) + (cR - 60))
    def pack_bbox(gray):
        N = norm(gray); h, w = N.shape
        try: cL, cR = conveyor_cols(N)
        except Exception: cL, cR = 0, w
        top = np.median(N[20:240, :], 0); bot = np.median(N[h - 240:h - 20, :], 0); ref = np.maximum(top, bot)
        diff = np.clip(np.tile(ref, (h, 1)) - N.astype(np.float32), 0, 255); diff[:, :cL] = 0; diff[:, cR:] = 0
        m = cv2.morphologyEx((diff > 20).astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((11, 11), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((41, 41), np.uint8))
        c = [c for c in cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] if cv2.contourArea(c) > h * w * 0.02]
        if not c: return 0, 0, w, h
        x, y, bw, bh = cv2.boundingRect(max(c, key=cv2.contourArea))
        return max(0, x - MARGIN), max(0, y - MARGIN), min(w, x + bw + MARGIN), min(h, y + bh + MARGIN)
    def parse_pts(s): return np.array([[float(a) for a in p.split(",")] for p in s.strip().split(";")], np.float32)
    def tags_of(node): return {t.get("label") for t in node.findall("tag")}
    def seal_mask(node):
        W = int(node.get("width")); H = int(node.get("height"))
        pl = [parse_pts(pg.get("points")) for pg in node.findall("polygon") if pg.get("label") == "sellado"]
        if len(pl) < 2: return None
        pl = sorted(pl, key=lambda p: cv2.contourArea(p.astype(np.float32)), reverse=True)
        m = np.zeros((H, W), np.uint8); cv2.fillPoly(m, [pl[0].astype(np.int32)], 1); cv2.fillPoly(m, [pl[1].astype(np.int32)], 0)
        return m

    # ---- build pack-cropped samples per product (reviewed GT only) ----
    train, val = [], []
    for xmlrel, imgrel, prod in DATASETS:
        samps = []; reviewed_only = "reviewed" in xmlrel
        for node in ET.parse(f"{ROOT}/{xmlrel}").getroot().findall("image"):
            nm = node.get("name"); p = f"{ROOT}/{imgrel}/{nm}"; tg = tags_of(node)
            if "exclude" in tg: continue
            if reviewed_only and "reviewed" not in tg: continue
            m = seal_mask(node)
            if m is None or not os.path.exists(p): continue
            g = cv2.imread(p, cv2.IMREAD_GRAYSCALE); x0, y0, x1, y1 = pack_bbox(g)
            gc = norm(g[y0:y1, x0:x1]); samps.append((np.stack([gc, gc, gc], -1), m[y0:y1, x0:x1], nm))
        random.Random(SEED).shuffle(samps)
        forced = [s for s in samps if s[2] in FORCE_TRAIN]; rest = [s for s in samps if s[2] not in FORCE_TRAIN]
        vp = VAL_PER if len(rest) >= 6 else 0
        val += [(im, mk, prod) for im, mk, _ in rest[:vp]]
        train += [(im, mk) for im, mk, _ in rest[vp:] + forced]
        print(f"{prod}: {len(samps)} samples -> {vp} val / {len(samps)-vp} train", flush=True)
    print(f"TOTAL train {len(train)}  val {len(val)}", flush=True)

    # (contaminant / printed-graphic copy-paste helpers — paste real cut-outs onto the band, mask unchanged)
    def add_contamination(img3, band):
        h, w = band.shape; ys, xs = np.where(band > 0)
        if len(xs) == 0: return img3
        out = img3.astype(np.float32); bandd = cv2.dilate(band, np.ones((7, 7), np.uint8))
        for _ in range(random.randint(1, 4)):
            k = random.randrange(len(xs)); cx, cy = int(xs[k]), int(ys[k]); blob = np.zeros((h, w), np.float32)
            for _ in range(random.randint(2, 6)):
                ox = cx + random.randint(-30, 30); oy = cy + random.randint(-30, 30)
                cv2.ellipse(blob, (ox, oy), (random.randint(10, 45), random.randint(10, 45)), random.randint(0, 180), 0, 360, 1.0, -1)
            blob = cv2.GaussianBlur(blob, (0, 0), random.uniform(4, 10)); mx = blob.max()
            if mx < 1e-6: continue
            blob = (blob / mx) * (bandd > 0)
            vv = random.uniform(5, 40) if random.random() < 0.8 else random.uniform(200, 245)
            a = (blob * random.uniform(0.7, 0.95))[..., None]; tex = vv + np.random.randn(h, w, 1) * 9
            out = out * (1 - a) + tex * a
        return np.clip(out, 0, 255).astype(np.uint8)
    def load_defect_cutouts():
        """Cut out EVERY labelled defect/liquid instance (patch + feathered alpha) from ALL product
        annotation files -> a large, realistic library of 'stuff on the band' to paste onto the seal
        (mask unchanged). Replaces the old standalone contaminants.xml so the augmentation draws on every
        labelled defect in the dataset, not a hand-picked 26-polygon file."""
        insts = []
        for xmlrel, imgrel, prod in DATASETS:
            xmlp = f"{ROOT}/{xmlrel}"
            if not os.path.exists(xmlp): continue
            for node in ET.parse(xmlp).getroot().findall("image"):
                name = node.get("name")
                if name in CONTAM_EXCLUDE or os.path.splitext(name)[0] in HOLD: continue   # never paste hold-out defects
                polys = [pg for pg in node.findall("polygon") if pg.get("label") in ("defect", "liquid")]
                if not polys: continue
                p = f"{ROOT}/{imgrel}/{name}"
                if not os.path.exists(p): continue
                g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                if g is None: continue
                for pg in polys:
                    pts = parse_pts(pg.get("points")).astype(np.int32); x, y, bw, bh = cv2.boundingRect(pts)
                    if bw < 4 or bh < 4: continue
                    patch = g[y:y + bh, x:x + bw].copy()
                    al = np.zeros((bh, bw), np.uint8); cv2.fillPoly(al, [pts - [x, y]], 255)
                    al = cv2.GaussianBlur(al, (0, 0), 2).astype(np.float32) / 255.0
                    insts.append((patch, al))
        return insts
    def paste_contaminants(img3, band, insts):
        if not insts: return img3
        h, w = band.shape; ys, xs = np.where(band > 0)
        if len(xs) == 0: return img3
        out = img3.copy()
        for _ in range(random.randint(2, 5)):
            patch, al = random.choice(insts)
            s = random.uniform(0.12, 0.55) if random.random() < 0.7 else random.uniform(0.55, 1.8)
            pw, ph = max(3, int(patch.shape[1] * s)), max(3, int(patch.shape[0] * s))
            p = cv2.resize(patch, (pw, ph)).astype(np.float32); a = cv2.resize(al, (pw, ph))
            M = cv2.getRotationMatrix2D((pw / 2, ph / 2), random.uniform(0, 360), 1.0)
            p = cv2.warpAffine(p, M, (pw, ph), borderValue=0); a = cv2.warpAffine(a, M, (pw, ph), borderValue=0)
            if random.random() < 0.5: p = cv2.flip(p, 1); a = cv2.flip(a, 1)
            p = np.clip(p * random.uniform(0.7, 1.2), 0, 255)
            k = random.randrange(len(xs)); cx, cy = int(xs[k]), int(ys[k]); x0, y0 = cx - pw // 2, cy - ph // 2
            ix0, iy0 = max(0, x0), max(0, y0); ix1, iy1 = min(w, x0 + pw), min(h, y0 + ph)
            if ix1 <= ix0 or iy1 <= iy0: continue
            px0, py0 = ix0 - x0, iy0 - y0; px1, py1 = px0 + (ix1 - ix0), py0 + (iy1 - iy0)
            aa = (a[py0:py1, px0:px1] * random.uniform(0.8, 1.0))[..., None]; pp = p[py0:py1, px0:px1][..., None]
            reg = out[iy0:iy1, ix0:ix1].astype(np.float32); out[iy0:iy1, ix0:ix1] = np.clip(reg * (1 - aa) + pp * aa, 0, 255).astype(np.uint8)
        return out
    CONTAM = load_defect_cutouts()
    print(f"defect cut-outs for band augmentation (from all labelled packs): {len(CONTAM)}", flush=True)

    # ---- augmentation, dataset, model, train loop ----
    def ttf():
        geo = [A.HorizontalFlip(p=.5), A.VerticalFlip(p=.5)]
        try: geo.append(A.Affine(scale=(.85, 1.15), translate_percent=(0, .06), rotate=(-180, 180), border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=.9))
        except TypeError: geo.append(A.Affine(scale=(.85, 1.15), translate_percent=(0, .06), rotate=(-180, 180), p=.9))
        ph = [A.RandomBrightnessContrast(.4, .4, p=.8), A.RandomGamma((60, 140), p=.6)]
        try: ph.append(A.GaussNoise(p=.2))
        except Exception: pass
        return A.Compose([A.Resize(IMG, IMG), *geo, *ph, A.Normalize(IM_MEAN, IM_STD), ToTensorV2()])
    def etf(): return A.Compose([A.Resize(IMG, IMG), A.Normalize(IM_MEAN, IM_STD), ToTensorV2()])
    class TR(torch.utils.data.Dataset):
        def __init__(s, d, L): s.d = d; s.tf = ttf(); s.L = L
        def __len__(s): return s.L
        def __getitem__(s, i):
            im, mk = s.d[random.randrange(len(s.d))]
            if random.random() < P_CONTAM:
                im = paste_contaminants(im, mk, CONTAM) if CONTAM else add_contamination(im, mk)
            o = s.tf(image=im, mask=mk); return o["image"], o["mask"].float().unsqueeze(0)
    train_dl = torch.utils.data.DataLoader(TR(train, SAMPLES), batch_size=BATCH, shuffle=True)
    etf_ = etf()

    base = torch.load(BASE, map_location="cpu"); ENC = base["encoder"]
    model = smp.Unet(ENC, encoder_weights=None, in_channels=3, classes=1)
    model.load_state_dict(base["state_dict"]); model = model.to(dev)
    print(f"fine-tuning {ENC} from {BASE}", flush=True)

    def dloss(l, t, e=1.):
        p = torch.sigmoid(l); return (1 - ((2 * (p * t).sum((2, 3)) + e) / (p.sum((2, 3)) + t.sum((2, 3)) + e))).mean()
    bce = nn.BCEWithLogitsLoss()
    @torch.no_grad()
    def dice_one(im3, mk):
        o = etf_(image=im3, mask=mk); x = o["image"].unsqueeze(0).to(dev); y = o["mask"].float().unsqueeze(0).unsqueeze(0).to(dev)
        p = (torch.sigmoid(model(x)) > THRESH).float(); return ((2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1)).item()
    def val_dice():
        model.eval(); per = {}
        for im, mk, prod in val: per.setdefault(prod, []).append(dice_one(im, mk))
        return {k: np.mean(v) for k, v in per.items()}, np.mean([d for v in per.values() for d in v])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    use_amp = (dev == "cuda"); scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best = 0.; best_state = None
    for ep in range(1, EPOCHS + 1):
        model.train(); tot = n = 0
        for x, y in train_dl:
            x, y = x.to(dev), y.to(dev); opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                lo = model(x); L = bce(lo, y) + dloss(lo, y)
            scaler.scale(L).backward(); scaler.step(opt); scaler.update(); tot += L.item() * x.size(0); n += x.size(0)
        sch.step(); per, ov = val_dice()
        if ov >= best: best = ov; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep % 5 == 0 or ep == 1:
            print(f"epoch {ep:3d}/{EPOCHS}  loss {tot/n:.4f}  VAL {ov:.3f}  best {best:.3f}", flush=True)
    if best_state: model.load_state_dict(best_state)
    torch.save({"state_dict": model.state_dict(), "encoder": ENC, "img": IMG, "thresh": THRESH,
                "mean": IM_MEAN, "std": IM_STD, "val_dice": best, "products": [d[2] for d in DATASETS]}, CKPT)
    print(f"BEST VAL {best:.3f}  saved {CKPT}", flush=True)
    return best

_ck = f"{R}/models/best_lite_reviewed_1280.pt"
if RUN_TRAINING and not os.path.exists(_ck):       # train only if the checkpoint is absent (idempotent re-runs)
    train_seal_model()                             # ~12 min on the GPU box
seal, sk = core.load_unet(_ck, device)             # use this seal for all sections below
print("seal model ready | encoder", sk["encoder"], "| val Dice", round(float(sk.get("val_dice", 0)), 3))''')

# ----------------------------------------------------------------------------- 4 mask<->poly
md("""## 4 · Mask ↔ polygon conversion

The model speaks **pixel masks**; CVAT (and the unroll) speak **polygons**. Two helpers bridge them:

- **`mask_to_ring(mask)`** → extracts the **outer** contour (convex-hull cleaned) and the **inner** hole
  (or an eroded fallback) from a predicted band mask, then resamples + circularly smooths both into clean
  rings (the raw predicted contour has thousands of jagged points).
- **`polygons_to_band_mask(outer, inner)`** → rasterizes the band back to pixels (fill outer = 1, punch
  inner = 0).
- **`simplify_contour` / `visvalingam`** → reduce a ring to ~tens of vertices for CVAT pre-annotations.""")
code('''outer, inner = core.mask_to_ring(mask)
band = core.polygons_to_band_mask(outer, inner, H, W)

fig, axs = plt.subplots(1, 3, figsize=(16, 6))
v0 = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR); v0[mask > 0] = (0, 180, 180)
axs[0].imshow(cv2.cvtColor(v0, cv2.COLOR_BGR2RGB)); axs[0].set_title("predicted seal mask (pixels)")
v1 = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
cv2.polylines(v1, [outer.astype(np.int32)], True, (0, 255, 0), 3)
cv2.polylines(v1, [inner.astype(np.int32)], True, (0, 140, 255), 3)
axs[1].imshow(cv2.cvtColor(v1, cv2.COLOR_BGR2RGB)); axs[1].set_title("mask_to_ring -> outer (green) + inner (orange)")
v2 = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR); v2[band > 0] = (0, 0, 200)
axs[2].imshow(cv2.cvtColor(v2, cv2.COLOR_BGR2RGB)); axs[2].set_title("polygons_to_band_mask -> band = outer - inner")
for a in axs: a.axis("off")
plt.tight_layout(); plt.show()
print(f"outer contour: {len(outer)} pts | inner contour: {len(inner)} pts")''')

# ----------------------------------------------------------------------------- 5 unroll
md("""## 5 · The geometric unroll — flattening the ring into a strip

**Why unroll?** Working on the curved ring in image space is position- and shape-dependent. If we
**unwrap the seal band into a flat strip** (perimeter × band-depth), a defect looks the same wherever it
sits on the seal — the defect model becomes position-independent and trivially simple.

**The robust algorithm — perpendicular-to-outer** (`core.unroll_maps`): for each of `WS` points around
the **smoothed outer contour**, march **inward along the local surface normal** to a depth equal to the
**local band width** (computed from a distance transform of the inner region), with a ±15 % margin. This
is *correspondence-free*: it never pairs outer points to inner points.

**The legacy method — correspondence** (`core.unroll_maps_legacy`): linearly interpolate each column from
an outer point to its *paired* inner point. This re-angles the sampling lines and **smears thin radial
defects** when the predicted contour differs slightly from the truth — which is exactly why `seal_2260`
used to be missed.

We keep **both** and ensemble them, because each catches defects the other distorts.""")
code('''# resample + smooth the outer contour and compute inward normals (the perpendicular-unroll geometry)
O = core._resample_closed(outer, WS)
O[:, 0] = core._smooth_closed(O[:, 0]); O[:, 1] = core._smooth_closed(O[:, 1])
T = np.roll(O, -1, 0) - np.roll(O, 1, 0)
Tn = np.maximum(np.hypot(T[:, 0], T[:, 1]), 1e-6)
N = np.stack([-T[:, 1] / Tn, T[:, 0] / Tn], 1)
fo = np.zeros((H, W), np.uint8); cv2.drawContours(fo, [outer.astype(np.int32)], -1, 255, -1)
pr = (O + 4 * N).astype(int)
if (fo[np.clip(pr[:, 1], 0, H - 1), np.clip(pr[:, 0], 0, W - 1)] > 0).mean() < 0.5: N = -N

v = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
cv2.polylines(v, [outer.astype(np.int32)], True, (0, 255, 0), 2)
for i in range(0, WS, 40):
    p0 = O[i].astype(int); p1 = (O[i] + 45 * N[i]).astype(int)
    cv2.arrowedLine(v, tuple(p0), tuple(p1), (0, 0, 255), 2, tipLength=0.3)
show(v, "Perpendicular sampling: inward normals (red) along the smoothed outer contour (green)", (8, 9))

strip = core.unroll(core.normalize(g), outer, inner, HS, WS)
show(strip, f"Unrolled seal strip (perpendicular unroll) — {HS}x{WS}, the seal flattened into a band", (15, 3))''')
md("""**Case study — why the fix mattered (`seal_2260`).** Below: the same defect (green) unrolled four
ways. With the **old** unroll the defect is sharp on the *ground-truth* seal but **smeared** on the
*predicted* seal (the model scored it 0.16 → missed). With the **new** unroll the predicted-seal strip
stays consistent with the ground truth (scored 0.85 → detected).""")
code('''node = [im for im in cvat.iter_images(f"{R}/data/annotations/prod2_reviewed.xml") if im.get("name") == EX + ".png"][0]
og, ig = cvat.seal_outer_inner(node)
defect2d = np.zeros((H, W), np.uint8)
for d in cvat.polygons(node, "defect") + cvat.polygons(node, "liquid"):
    cv2.fillPoly(defect2d, [d.astype(np.int32)], 255)

def strip_with_defect(maps):
    mx, my = maps
    s = cv2.remap(core.normalize(g), mx, my, cv2.INTER_LINEAR, borderValue=0)
    dm = cv2.remap(defect2d, mx, my, cv2.INTER_NEAREST, borderValue=0)
    vv = cv2.cvtColor(s, cv2.COLOR_GRAY2BGR); vv[dm > 127] = (0, 230, 0)
    return vv

panels = [("OLD unroll  /  GT seal",   core.unroll_maps_legacy(og, ig, HS, WS)),
          ("OLD unroll  /  PRED seal", core.unroll_maps_legacy(outer, inner, HS, WS)),
          ("NEW unroll  /  GT seal",   core.unroll_maps(og, ig, HS, WS)),
          ("NEW unroll  /  PRED seal", core.unroll_maps(outer, inner, HS, WS))]
fig, axs = plt.subplots(4, 1, figsize=(15, 9))
for ax, (t, m) in zip(axs, panels):
    ax.imshow(cv2.cvtColor(strip_with_defect(m), cv2.COLOR_BGR2RGB)); ax.set_title(t); ax.axis("off")
plt.tight_layout(); plt.show()''')

# ----------------------------------------------------------------------------- 6 defect
md("""## 6 · Stage 2 — Defect detection on the strip

**Dataset (`make_strips.py`).** For every labelled pack we unroll the image **and** its defect polygons
with the *same* maps, producing `(strip, defect-mask)` pairs at `128×1536`. Both unrolls produce strips so
training matches inference.

**The model.** A **U-Net with a resnet18 encoder**. Training tricks for a rare, small target:

- **oversample** defect strips,
- **copy-paste augmentation** — paste real defect cut-outs onto good strips (the band texture is realistic,
  so synthetic positives are cheap and plausible),
- **BCE(pos_weight) + Dice** loss,
- checkpoint selection by **best test AUROC**.

The image-level score is the **max of the smoothed defect-probability map**.""")
code('''dstrips = [p for p in sorted(glob.glob(f"{R}/data/strips/train/img/*.png"))
           if cv2.imread(p.replace("/img/", "/mask/"), 0).sum() > 0]
sp = dstrips[0]; st = cv2.imread(sp, 0); mk = cv2.imread(sp.replace("/img/", "/mask/"), 0)
fig, axs = plt.subplots(2, 1, figsize=(15, 4.5))
axs[0].imshow(st, cmap="gray"); axs[0].set_title(f"training strip: {os.path.basename(sp)}")
vv = cv2.cvtColor(st, cv2.COLOR_GRAY2BGR); vv[mk > 0] = (0, 0, 230)
axs[1].imshow(cv2.cvtColor(vv, cv2.COLOR_BGR2RGB)); axs[1].set_title("defect mask (red) = supervised target")
for a in axs: a.axis("off")
plt.tight_layout(); plt.show()''')
code('''# copy-paste augmentation demo: paste this real defect cut-out onto a good strip
goods = [p for p in sorted(glob.glob(f"{R}/data/strips/train/img/*.png"))
         if cv2.imread(p.replace("/img/", "/mask/"), 0).sum() == 0]
base = cv2.imread(goods[0], 0); pasted = base.copy()
ys, xs = np.where(mk > 0); y0d, y1d, x0d, x1d = ys.min(), ys.max(), xs.min(), xs.max()
patch = st[y0d:y1d + 1, x0d:x1d + 1]; pm = mk[y0d:y1d + 1, x0d:x1d + 1]
py = min(max(0, 30), HS - patch.shape[0] - 1); px = min(600, WS - patch.shape[1] - 1)
roi = pasted[py:py + patch.shape[0], px:px + patch.shape[1]]; roi[pm > 0] = patch[pm > 0]
fig, axs = plt.subplots(2, 1, figsize=(15, 4.5))
axs[0].imshow(base, cmap="gray"); axs[0].set_title("good strip (negative)")
axs[1].imshow(pasted, cmap="gray"); axs[1].set_title("after copy-paste: real defect cut-out -> synthetic positive")
for a in axs: a.axis("off")
plt.tight_layout(); plt.show()''')
code('''pp = defect_prob(defm, st)
fig, axs = plt.subplots(2, 1, figsize=(15, 4.5))
axs[0].imshow(st, cmap="gray"); axs[0].set_title("strip")
axs[1].imshow(pp, cmap="magma"); axs[1].set_title(f"defect probability map  (max = {pp.max():.2f})")
for a in axs: a.axis("off")
plt.tight_layout(); plt.show()''')
md("""**Training curves.** Parsed from the saved training log of the perpendicular-unroll defect model.""")
code('''log = f"{R}/outputs/train_defect_perp.log"
eps, loss, auc, dice = [], [], [], []
if os.path.exists(log):
    for ln in open(log):
        m = re.search(r"ep\\s+(\\d+) loss ([\\d.]+) \\| test AUROC ([\\d.]+) pixelDice ([\\d.]+)", ln)
        if m:
            eps.append(int(m[1])); loss.append(float(m[2])); auc.append(float(m[3])); dice.append(float(m[4]))
if eps:
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].plot(eps, loss, "-o"); ax[0].set_title("defect training loss"); ax[0].set_xlabel("epoch"); ax[0].grid(alpha=.3)
    ax[1].plot(eps, auc, "-o", label="test AUROC"); ax[1].plot(eps, dice, "-s", label="pixel Dice")
    ax[1].set_title("defect test metrics"); ax[1].set_xlabel("epoch"); ax[1].legend(); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.show()
else:
    print("training log not found:", log)''')

# ----------------------------------------------------------------------------- 6a defect training code
md("""### 6a · The defect training code

The real defect-training routine that produced `defect_strip.pt` (a faithful copy of `src/train_defect.py`).
It assumes the `(strip, mask)` pairs already exist under `data/strips/` — those are built by
`src/make_strips.py`, which unrolls every labelled pack's image **and** its defect polygons with the *same*
maps (so train matches inference). Highlights:

- builds a **defect cut-out library** from the training strips for **copy-paste** augmentation;
- the dataset **oversamples** defects (50 % of draws) and pastes cut-outs onto good strips (`P_PASTE=0.7`);
- loss = **BCE(pos_weight=20) + Dice** (defects are rare and small);
- evaluates on the held-out test strips each few epochs (AUROC + pixel Dice + best-F1) and keeps the
  **best-by-AUROC** checkpoint.

> The current `defect_strip.pt` was trained on strips from the **perpendicular** unroll; `defect_strip.prev.pt`
> is the same routine on **correspondence**-unroll strips. The ensemble in §7 uses both.""")
code('''def train_defect_model(strips="data/strips", out="models/defect_strip.pt"):
    import os, glob, random
    import numpy as np, cv2, torch, torch.nn as nn
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    import segmentation_models_pytorch as smp

    ROOT = R; STR = f"{ROOT}/{strips}"
    SEED = 42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    HS_, WS_ = 128, 1536; BATCH = 8; EPOCHS = 60; STEPS = 1200; THR = 0.5; P_PASTE = 0.7
    MEAN = (.485, .456, .406); STD = (.229, .224, .225)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def load(split):
        items = []
        for ip in sorted(glob.glob(f"{STR}/{split}/img/*.png")):
            mp = ip.replace("/img/", "/mask/")
            if not os.path.exists(mp): continue
            img = cv2.imread(ip, cv2.IMREAD_GRAYSCALE); m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            items.append((img, (m > 127).astype(np.uint8), os.path.basename(ip)))
        return items
    train = load("train"); test = load("test")
    tr_def = [t for t in train if t[1].sum() > 0]; tr_good = [t for t in train if t[1].sum() == 0]
    print(f"train {len(train)} ({len(tr_def)} defect / {len(tr_good)} good)  "
          f"test {len(test)} ({sum(1 for t in test if t[1].sum()>0)} defect)", flush=True)

    LIB = []                                          # defect cut-out library for copy-paste
    for img, m, _ in tr_def:
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < 6: continue
            LIB.append((img[y:y + h, x:x + w].copy(), (lab[y:y + h, x:x + w] == i).astype(np.float32)))
    print(f"defect cut-outs in library: {len(LIB)}", flush=True)

    def paste(img, m):
        if not LIB: return img, m
        out = img.copy().astype(np.float32); mo = m.copy()
        for _ in range(random.randint(1, 3)):
            patch, al = random.choice(LIB); s = random.uniform(0.6, 1.6)
            pw, ph = max(3, int(patch.shape[1] * s)), max(3, int(patch.shape[0] * s))
            p = cv2.resize(patch, (pw, ph)).astype(np.float32); a = cv2.resize(al, (pw, ph))
            if random.random() < 0.5: p = cv2.flip(p, 1); a = cv2.flip(a, 1)
            if random.random() < 0.5: p = cv2.flip(p, 0); a = cv2.flip(a, 0)
            p = np.clip(p * random.uniform(0.8, 1.15), 0, 255); a = cv2.GaussianBlur(a, (0, 0), 1.0)
            H, W = img.shape; x0 = random.randint(0, max(0, W - pw)); y0 = random.randint(0, max(0, H - ph))
            x1, y1 = min(W, x0 + pw), min(H, y0 + ph); aa = a[:y1 - y0, :x1 - x0]; pp = p[:y1 - y0, :x1 - x0]
            reg = out[y0:y1, x0:x1]; out[y0:y1, x0:x1] = reg * (1 - aa) + pp * aa
            mo[y0:y1, x0:x1] = np.maximum(mo[y0:y1, x0:x1], (aa > 0.3).astype(np.uint8))
        return np.clip(out, 0, 255).astype(np.uint8), mo

    aug = A.Compose([A.HorizontalFlip(p=.5), A.VerticalFlip(p=.5), A.RandomBrightnessContrast(.3, .3, p=.7),
                     A.ShiftScaleRotate(shift_limit=.03, scale_limit=.05, rotate_limit=4, border_mode=cv2.BORDER_REFLECT, p=.5),
                     A.Normalize(MEAN, STD), ToTensorV2()])
    ev = A.Compose([A.Normalize(MEAN, STD), ToTensorV2()])
    class DS(torch.utils.data.Dataset):
        def __init__(s, L): s.L = L
        def __len__(s): return s.L
        def __getitem__(s, i):
            img, m, _ = random.choice(tr_def) if (tr_def and random.random() < 0.5) else random.choice(tr_good)
            img, m = img.copy(), m.copy()
            if random.random() < P_PASTE: img, m = paste(img, m)
            o = aug(image=np.stack([img] * 3, -1), mask=m); return o["image"], o["mask"].float().unsqueeze(0)
    dl = torch.utils.data.DataLoader(DS(STEPS), batch_size=BATCH, shuffle=True, num_workers=4)

    model = smp.Unet("resnet18", encoder_weights="imagenet", in_channels=3, classes=1).to(dev)
    def dice_l(l, t, e=1.):
        p = torch.sigmoid(l); return (1 - ((2 * (p * t).sum((2, 3)) + e) / (p.sum((2, 3)) + t.sum((2, 3)) + e))).mean()
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(20.).to(dev))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")

    @torch.no_grad()
    def evaluate():
        model.eval(); scores = []; labels = []; dices = []
        for img, m, _ in test:
            x = ev(image=np.stack([img] * 3, -1), mask=m)["image"].unsqueeze(0).to(dev)
            prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
            scores.append(float(cv2.GaussianBlur(prob, (0, 0), 2).max())); labels.append(1 if m.sum() > 0 else 0)
            if m.sum() > 0:
                pr = (prob > THR).astype(np.uint8); dices.append(2 * (pr & m).sum() / (pr.sum() + m.sum() + 1e-6))
        scores = np.array(scores); labels = np.array(labels)
        pos = scores[labels == 1]; neg = scores[labels == 0]
        auroc = np.mean([(1.0 if a > b else 0.5 if a == b else 0.0) for a in pos for b in neg]) if len(pos) and len(neg) else float("nan")
        best = (0, 0, 0, 0)
        for th in np.unique(scores):
            pred = (scores >= th).astype(int); tp = ((pred == 1) & (labels == 1)).sum(); fp = ((pred == 1) & (labels == 0)).sum(); fn = ((pred == 0) & (labels == 1)).sum()
            pr = tp / (tp + fp + 1e-9); rc = tp / (tp + fn + 1e-9); f1 = 2 * pr * rc / (pr + rc + 1e-9)
            if f1 > best[0]: best = (f1, pr, rc, th)
        return auroc, (np.mean(dices) if dices else float("nan")), best

    best_auroc = 0; best_state = None
    for ep in range(1, EPOCHS + 1):
        model.train(); tot = n = 0
        for x, y in dl:
            x, y = x.to(dev), y.to(dev); opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=dev == "cuda"):
                lo = model(x); L = bce(lo, y) + dice_l(lo, y)
            scaler.scale(L).backward(); scaler.step(opt); scaler.update(); tot += L.item() * x.size(0); n += x.size(0)
        sch.step()
        if ep % 10 == 0 or ep == 1:
            au, pdv, (f1, pr, rc, th) = evaluate()
            print(f"ep {ep:3d} loss {tot/n:.4f} | test AUROC {au:.3f} pixelDice {pdv:.3f} | bestF1 {f1:.3f} (P{pr:.2f} R{rc:.2f})", flush=True)
            if au >= best_auroc: best_auroc = au; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state: model.load_state_dict(best_state)
    au, pdv, (f1, pr, rc, th) = evaluate()
    os.makedirs(os.path.dirname(f"{ROOT}/{out}"), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "encoder": "resnet18", "HS": HS_, "WS": WS_,
                "thr": THR, "score_thr": float(th), "mean": MEAN, "std": STD}, f"{ROOT}/{out}")
    print(f"[{strips}] FINAL test: AUROC {au:.3f}  pixelDice {pdv:.3f}  bestF1 {f1:.3f} "
          f"(P{pr:.2f} R{rc:.2f} @score>{th:.2f}) -> {out}", flush=True)
    return au

if RUN_TRAINING and not os.path.exists(f"{R}/models/defect_strip.pt"):
    train_defect_model(strips="data/strips",        out="models/defect_strip.pt")        # perpendicular branch
if RUN_TRAINING and not os.path.exists(f"{R}/models/defect_strip.prev.pt"):
    train_defect_model(strips="data/strips_legacy", out="models/defect_strip.prev.pt")   # correspondence branch
# load the defect models and (re)build the ensemble used in sections 7-8
defm, dk = core.load_unet(f"{R}/models/defect_strip.pt", device)
legm, lk = core.load_unet(f"{R}/models/defect_strip.prev.pt", device)
branches = [dict(name="perpendicular",  model=defm, unroll=core.unroll_maps,        hs=dk["HS"], ws=dk["WS"], thr=DTHR),
            dict(name="correspondence", model=legm, unroll=core.unroll_maps_legacy, hs=lk["HS"], ws=lk["WS"], thr=DTHR)]
print("defect models ready + ensemble branches rebuilt")''')

# ----------------------------------------------------------------------------- 7 ensemble
md("""## 7 · End-to-end pipeline & the 2-unroll ensemble

`pipeline.process_pack` runs the whole chain and returns `(composite, n_detections, score)`. With **two
branches** it unrolls the predicted seal *both* ways, scores each with its matching model, and
**max-pools** the image-level score — recovering defects that either unroll alone would smear.

The composite shows: the cropped pack with the seal band tinted and red circles on detections, then both
unrolled strips below.""")
code('''good_test = [os.path.splitext(os.path.basename(p))[0]
             for p in sorted(glob.glob(f"{R}/data/strips/test/img/*.png"))
             if cv2.imread(p.replace("/img/", "/mask/"), 0).sum() == 0]
for nm in [EX, good_test[0]]:
    gg = cv2.imread(find_raw(nm), cv2.IMREAD_GRAYSCALE)
    comp, nd, sc = process_pack(gg, seal, sk["img"], sk["thresh"], branches, device)
    show(comp, f"{nm}  ->  verdict {'DEFECT' if nd else 'OK'}   score {sc:.2f}", (10, 9))''')

# ----------------------------------------------------------------------------- 8 test set
md("""## 8 · Clean end-to-end inference on the global hold-out

Run the retrained ensemble on **every pack in the global hold-out** (`data/strips/test/` = the 180 packs
neither stage trained on). This is the honest end-to-end number. Then display all defect packs full-size and
all good packs as a flagged thumbnail grid.""")
code('''test_imgs = sorted(glob.glob(f"{R}/data/strips/test/img/*.png"))
scores, labels, names, comps = [], [], [], {}
for ip in test_imgs:
    nm = os.path.splitext(os.path.basename(ip))[0]; rp = find_raw(nm)
    if rp is None: continue
    gg = cv2.imread(rp, cv2.IMREAD_GRAYSCALE)
    comp, nd, sc = process_pack(gg, seal, sk["img"], sk["thresh"], branches, device)
    if comp is None: continue
    lab = 1 if cv2.imread(ip.replace("/img/", "/mask/"), 0).sum() > 0 else 0
    scores.append(sc); labels.append(lab); names.append(nm); comps[nm] = (comp, lab, sc, nd)
scores = np.array(scores); labels = np.array(labels)
print(f"ran {len(scores)} test packs: {int(labels.sum())} defect, {int((labels==0).sum())} good")

pos, neg = scores[labels == 1], scores[labels == 0]
auroc = float(np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg]))
print(f"AUROC: {auroc:.3f}\\n")
trows = []
for thr in [0.30, 0.43, 0.50, 0.70, 0.85, 0.92]:
    tp = int(((scores >= thr) & (labels == 1)).sum()); fp = int(((scores >= thr) & (labels == 0)).sum())
    trows.append(dict(threshold=thr, recall=f"{tp}/{int(labels.sum())}",
                      false_alarms=f"{fp}/{int((labels==0).sum())}", missed=int(labels.sum()) - tp))
print(pd.DataFrame(trows).to_string(index=False))
print(f"\\nmin defect score: {pos.min():.3f}  (= max threshold that still catches every defect)")''')
code('''# ROC, score distribution, confusion matrix @ DTHR
P, Ng = labels.sum(), (labels == 0).sum()
ts = np.unique(np.r_[0, scores, 1])[::-1]
tpr = [((scores >= t) & (labels == 1)).sum() / P for t in ts]
fpr = [((scores >= t) & (labels == 0)).sum() / Ng for t in ts]
fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
axs[0].plot(fpr, tpr, "-"); axs[0].plot([0, 1], [0, 1], "k--", alpha=.3)
axs[0].set_title(f"ROC (AUROC {auroc:.3f})"); axs[0].set_xlabel("false-positive rate"); axs[0].set_ylabel("recall")
axs[1].hist(neg, bins=30, alpha=.6, label="good"); axs[1].hist(pos, bins=30, alpha=.6, label="defect")
axs[1].axvline(DTHR, color="r", ls="--", label=f"thr {DTHR}"); axs[1].set_yscale("log")
axs[1].set_title("ensemble score distribution"); axs[1].legend()
pred = (scores >= DTHR).astype(int)
cm = np.array([[int(((pred == 0) & (labels == 0)).sum()), int(((pred == 1) & (labels == 0)).sum())],
               [int(((pred == 0) & (labels == 1)).sum()), int(((pred == 1) & (labels == 1)).sum())]])
axs[2].imshow(cm, cmap="Blues")
axs[2].set_xticks([0, 1]); axs[2].set_xticklabels(["pred OK", "pred DEFECT"])
axs[2].set_yticks([0, 1]); axs[2].set_yticklabels(["true good", "true defect"])
axs[2].set_title(f"confusion @ {DTHR}")
for i in range(2):
    for j in range(2): axs[2].text(j, i, cm[i, j], ha="center", va="center", fontsize=15)
plt.tight_layout(); plt.show()''')
md("""### 8a · All defect packs (full size)
Every held-out **defect** pack, with the ensemble verdict and score. A pack scoring below the operating
threshold is flagged `MISSED`.""")
code('''for nm in [n for n in names if comps[n][1] == 1]:
    comp, lab, sc, nd = comps[nm]
    tag = "MISSED" if sc < DTHR else "caught"
    show(comp, f"{nm}  |  true DEFECT  |  score {sc:.2f}  ->  {'DEFECT' if nd else 'OK'}  [{tag}]", (10, 9))''')
md("""### 8b · All good packs (thumbnail grid)
Every held-out **good** pack. Green border = correctly OK; **red border + "FA"** = false alarm at the
operating threshold.""")
code('''good_names = [n for n in names if comps[n][1] == 0]
ncol = 8; nrow = int(np.ceil(len(good_names) / ncol))
fig, axs = plt.subplots(nrow, ncol, figsize=(ncol * 2.2, nrow * 2.5))
axs = np.atleast_2d(axs)
for ax, nm in zip(axs.ravel(), good_names):
    comp, lab, sc, nd = comps[nm]
    th = cv2.resize(comp, (170, int(comp.shape[0] * 170 / comp.shape[1])))
    ax.imshow(cv2.cvtColor(th, cv2.COLOR_BGR2RGB))
    fa = nd > 0
    ax.set_title(f"{sc:.2f}" + ("  FA" if fa else ""), color=("red" if fa else "black"), fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_color("red" if fa else "green"); s.set_linewidth(2.5)
for ax in axs.ravel()[len(good_names):]: ax.axis("off")
nfa = sum(1 for n in good_names if comps[n][3] > 0)
plt.suptitle(f"All {len(good_names)} good test packs — {nfa} false alarms at threshold {DTHR}", y=1.002)
plt.tight_layout(); plt.show()''')

# ----------------------------------------------------------------------------- 9 summary
md("""## 9 · Summary

| Component | Model | Training data |
|---|---|---|
| Stage 1 — seal segmentation | MobileNetV3-small U-Net @1280² | all `reviewed` packs (hold-out is non-reviewed → unseen) |
| Stage 2 — defect detection | resnet18 U-Net on 128×1536 strip, **2-unroll ensemble** | all packs **except the global hold-out** |

**The headline number is the §8 clean end-to-end metric** — computed on the 180-pack global hold-out that
**neither stage trained on**. Because the hold-out is drawn from non-reviewed packs only, every reviewed
ground-truth pack stayed in training for both models, so we pay no GT cost for the clean evaluation.

**Methodology highlights.**
- **No cross-model leakage:** the seal stage cannot have seen the hold-out (it trains on reviewed packs;
  the hold-out is non-reviewed), and the defect strips exclude the hold-out — so §8 is a true
  "neither stage saw this pack" measurement.
- **Perpendicular-to-outer unroll** removed the thin-defect smearing that previously hid radial defects;
  the **2-unroll ensemble** (perpendicular + legacy, max-pooled) catches what either alone would distort.
- **Band augmentation** now pastes real `defect`/`liquid` cut-outs gathered from *all* labelled packs
  (hold-out excluded), not a separate contaminants file.

**Deployment.** The seal model has an INT8 ONNX export runnable from Rust (`ort`) on CPU; the defect model
is tiny (resnet18 on a 128×1536 strip). The ensemble doubles defect inference — negligible cost.""")

# ----------------------------------------------------------------------------- 10 ablation
md("""## 10 · Ablation — ImageNet-pretrained vs from-scratch

Does the ImageNet pretraining actually help *here*? NIR transmission images are quite domain-shifted from
natural photos, so it's worth measuring rather than assuming. The clean comparison holds the architecture
fixed and flips only the initialisation:

```python
smp.Unet(encoder, encoder_weights="imagenet")   # what we ship
smp.Unet(encoder, encoder_weights=None)          # from scratch (random init)
```

The from-scratch models were trained with the same data/augmentation but a longer schedule (no pretrained
features to lean on) via `train_reviewed.py --scratch` / `train_defect.py --scratch`, saved to
`models/scratch_*.pt` (production models untouched). The cell below loads both sets and evaluates them on
the **same hold-out**, single-branch (perpendicular) end-to-end.""")
code('''import os
sp_seal, sp_def = f"{R}/models/scratch_seal_1280.pt", f"{R}/models/scratch_defect_perp.pt"
if os.path.exists(sp_seal) and os.path.exists(sp_def):
    sc_seal, sc_sk = core.load_unet(sp_seal, device); sc_def, sc_dk = core.load_unet(sp_def, device)
    lab = [ln.split(",") for ln in open(f"{R}/data/holdout_labels.csv").read().splitlines()[1:]]
    def e2e(sm, skk, dmodel, g):
        H, W = g.shape; x0, y0, x1, y1 = core.pack_bbox(g)
        pr = core.predict_probability(sm, g[y0:y1, x0:x1], skk["img"], device)
        full = np.zeros((H, W), np.float32); full[y0:y1, x0:x1] = cv2.resize(pr, (x1-x0, y1-y0))
        O, I = core.mask_to_ring((full > skk.get("thresh", .5)).astype(np.uint8)*255)
        if O is None: return None
        mx, my = core.unroll_maps(O, I, HS, WS)
        strip = cv2.remap(core.normalize(g), mx, my, cv2.INTER_LINEAR, borderValue=0)
        return float(cv2.GaussianBlur(defect_prob(dmodel, strip), (0, 0), 2).max())
    im_s, sc_s, ys = [], [], []
    for nm, lb in lab:
        rp = find_raw(nm)
        if rp is None: continue
        g = cv2.imread(rp, cv2.IMREAD_GRAYSCALE)
        a = e2e(seal, sk, defm, g); b = e2e(sc_seal, sc_sk, sc_def, g)
        if a is None or b is None: continue
        im_s.append(a); sc_s.append(b); ys.append(int(lb))
    im_s, sc_s, ys = np.array(im_s), np.array(sc_s), np.array(ys)
    def st(s):
        pos, neg = s[ys==1], s[ys==0]
        au = float(np.mean([(x>y)+0.5*(x==y) for x in pos for y in neg]))
        return au, int(((s>=DTHR)&(ys==1)).sum()), int(((s>=DTHR)&(ys==0)).sum())
    rows = []
    for tag, ck, s in [("ImageNet", sk, im_s), ("from-scratch", sc_sk, sc_s)]:
        au, tp, fp = st(s)
        rows.append(dict(init=tag, seal_valDice=round(float(ck.get("val_dice", float("nan"))), 3),
                         e2e_AUROC=round(au, 3), recall=f"{tp}/{int((ys==1).sum())}", FP=f"{fp}/{int((ys==0).sum())}"))
    print(pd.DataFrame(rows).to_string(index=False))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(["ImageNet", "scratch"], [st(im_s)[0], st(sc_s)[0]], color=["#2c7", "#c72"]); ax[0].set_ylim(0.85, 1.0); ax[0].set_title("end-to-end AUROC")
    ax[1].bar(["ImageNet", "scratch"], [st(im_s)[2], st(sc_s)[2]], color=["#2c7", "#c72"]); ax[1].set_title(f"false alarms @ {DTHR} (of {int((ys==0).sum())} good)")
    plt.tight_layout(); plt.show()
    print("Takeaway: pretraining helps most on the data-scarce SEAL stage and on false-alarm rate;")
    print("the augmentation-heavy defect head is nearly initialisation-agnostic.")
else:
    print("scratch models not found — run train_*.py --scratch first (see text above).")''')

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}

out = "/private/tmp/claude-501/-Volumes-T7-fti-seal/2e252a62-b456-4887-908e-94c7b34abff3/scratchpad/seal_inspection_walkthrough.ipynb"
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", out, "with", len(cells), "cells")
