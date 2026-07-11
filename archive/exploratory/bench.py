import torch, time, glob, numpy as np, cv2
from predict import U, load_gray3   # reuse exact model + preprocessing

dev = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load("best.pt", map_location=dev)
H, W = ck["img_h"], ck["img_w"]
mean = np.array(ck["mean"], np.float32); std = np.array(ck["std"], np.float32)
m = U().to(dev); m.load_state_dict(ck["state_dict"]); m.eval()
files = sorted(glob.glob("prod2/*_raw.png"))
print(f"device={dev}  GPU={torch.cuda.get_device_name(0) if dev=='cuda' else '-'}  imgs={len(files)}")

def sync():
    if dev == "cuda": torch.cuda.synchronize()

# 1) pure GPU forward latency (warmup + sync)
d = torch.randn(1, 3, H, W, device=dev)
with torch.no_grad():
    for _ in range(15): m(d)
sync()
t = time.time()
with torch.no_grad():
    for _ in range(100): m(d)
sync()
print(f"pure GPU forward         : {(time.time()-t)/100*1000:6.2f} ms/pack  ({100/(time.time()-t):.0f} packs/s)")

# 2) full pipeline per pack: read + preprocess + infer + postprocess (no file write)
N = min(300, len(files)); t = time.time()
for p in files[:N]:
    orig, img3 = load_gray3(p); oh, ow = orig.shape
    x = cv2.resize(img3, (W, H)).astype(np.float32) / 255.0
    x = ((x - mean) / std).transpose(2, 0, 1)[None]
    with torch.no_grad():
        prob = torch.sigmoid(m(torch.from_numpy(x).to(dev)))[0, 0].cpu().numpy()
    mask = (cv2.resize(prob, (ow, oh)) > ck.get("thresh", 0.5)).astype(np.uint8) * 255
sync()
dt = (time.time() - t) / N * 1000
print(f"full pipeline (no write) : {dt:6.2f} ms/pack  ({1000/dt:.0f} packs/s)  over {N}")
