"""Two-stage seal-inspection pipeline running on CPU via ONNX Runtime.

seal.onnx  : U-Net (MobileNetV3) -> seal-ring probability (input 3x512x512)
defect.onnx: U-Net (ResNet18)   -> defect probability on the unrolled strip (3x128x1536)
No PyTorch required.
"""
import time
import numpy as np
import cv2
import onnxruntime as ort
import pipeline_core as C

SEAL_RES = 1280
HS, WS = 128, 1536
GREEN = (90, 200, 70); PURPLE = (150, 70, 175); RED = (230, 40, 40)


def load_session(path, threads=4):
    so = ort.SessionOptions(); so.intra_op_num_threads = threads
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def load_models(seal_path, defect_path, threads=4):
    return load_session(seal_path, threads), load_session(defect_path, threads)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _infer(sess, x):
    y = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    return _sigmoid(y[0, 0])


def _to_input(gray_u8):
    """3-channel ImageNet-normalised input (seal net and ResNet18 defect net)."""
    return ((np.stack([gray_u8] * 3, -1) / 255.0 - C.IMAGENET_MEAN) / C.IMAGENET_STD).transpose(2, 0, 1)[None].astype(np.float32)


def _to_input_defect(strip_u8, kind):
    """TinyUNet expects a single-channel (/255-0.5)/0.5 input; ResNet18 the 3-ch ImageNet one."""
    if kind == "tiny":
        return ((strip_u8 / 255.0 - 0.5) / 0.5).astype(np.float32)[None, None]
    return _to_input(strip_u8)


def run(gray, seal_sess, defect_sess, defect_kind="resnet"):
    """Full pipeline. Returns a dict with geometry + defect map + score + timing,
    or {'seal': None} if the seal ring could not be localised."""
    t0 = time.perf_counter()
    x0, y0, x1, y1 = C.pack_bbox(gray)
    crop = C.normalize(gray[y0:y1, x0:x1]); ch, cw = crop.shape
    _sh = seal_sess.get_inputs()[0].shape          # [1, 3, H, W] — use the model's own input size
    res = int(_sh[2]) if isinstance(_sh[2], int) else SEAL_RES
    im = cv2.resize(crop, (res, res))
    prob = _infer(seal_sess, _to_input(im))
    t1 = time.perf_counter()
    pm = cv2.resize(prob, (cw, ch))
    O, I = C.mask_to_ring((pm > 0.5).astype(np.uint8) * 255)
    if O is None:
        return {"seal": None, "ms": (time.perf_counter() - t0) * 1000}
    mx, my = C.unroll_maps(O, I, HS, WS)
    strip = cv2.remap(crop, mx, my, cv2.INTER_LINEAR, borderValue=0)
    t2 = time.perf_counter()
    dp = _infer(defect_sess, _to_input_defect(strip, defect_kind))
    score = float(cv2.GaussianBlur(dp, (0, 0), 2).max())
    t3 = time.perf_counter()
    return dict(seal=True, crop=crop, O=O, I=I, strip=strip, dp=dp, mx=mx, my=my, score=score,
                ms_seal=(t1 - t0) * 1000, ms_unroll=(t2 - t1) * 1000, ms_defect=(t3 - t2) * 1000,
                ms=(t3 - t0) * 1000)


def _rgb(g):
    return cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)


def _blend(img, m, col, a=0.5):
    out = img.copy(); out[m > 0] = (a * np.array(col) + (1 - a) * out[m > 0]).astype(np.uint8); return out


def overlays(res, thr):
    """Build the three display panels at the given decision threshold."""
    crop, O, I, strip, dp, mx, my = res["crop"], res["O"], res["I"], res["strip"], res["dp"], res["mx"], res["my"]
    # 1 - predicted seal ring
    p1 = _rgb(crop)
    cv2.polylines(p1, [O.astype(np.int32)], True, GREEN, 3, cv2.LINE_AA)
    cv2.polylines(p1, [I.astype(np.int32)], True, GREEN, 3, cv2.LINE_AA)
    # 2 - unrolled strip + defect overlay
    dmask = (cv2.GaussianBlur(dp, (0, 0), 2) >= thr).astype(np.uint8)
    p2 = _blend(_rgb(strip), dmask, PURPLE, 0.5)
    p2[:4, :] = GREEN; p2[-4:, :] = GREEN
    # 3 - final: project defects back onto the pack + circle them
    p3 = p1.copy(); n_det = 0
    if dmask.any():
        ys, xs = np.where(dmask > 0)
        di = np.zeros(crop.shape, np.uint8)
        iy = np.clip(my[ys, xs].astype(int), 0, crop.shape[0] - 1)
        ix = np.clip(mx[ys, xs].astype(int), 0, crop.shape[1] - 1)
        di[iy, ix] = 255; di = cv2.dilate(di, np.ones((7, 7), np.uint8))
        p3 = _blend(p3, di, PURPLE, 0.5)
        grp = cv2.dilate(di, np.ones((25, 25), np.uint8))
        ncc, _, st, ct = cv2.connectedComponentsWithStats(grp)
        for i in range(1, ncc):
            x, y, w, h, area = st[i]
            if area < 200:
                continue
            cv2.circle(p3, (int(ct[i][0]), int(ct[i][1])), int(max(w, h) * 0.55) + 10, RED, 3, cv2.LINE_AA)
            n_det += 1
    return dict(seal=p1, strip=p2, final=p3, n_det=n_det)
