#!/usr/bin/env python3
"""Load the trained lite checkpoint, export ONNX (legacy exporter), benchmark CPU."""
import os, time, numpy as np, torch
import segmentation_models_pytorch as smp

ROOT="/home/ubuntu/TFM/seal-inspection"
CKPT=f"{ROOT}/models/best_lite.pt"; ONNX=f"{ROOT}/models/seal_lite.onnx"
ck=torch.load(CKPT, map_location="cpu")
IMG=ck["img"]; ENC=ck["encoder"]
model=smp.Unet(ENC, encoder_weights=None, in_channels=3, classes=1)
model.load_state_dict(ck["state_dict"]); model.eval()
print(f"loaded {ENC}  img={IMG}  val_dice={ck['val_dice']:.3f}  ckpt={os.path.getsize(CKPT)/1e6:.1f}MB")

# ONNX export — legacy TorchScript exporter (no onnxscript dependency)
torch.onnx.export(model, torch.randn(1,3,IMG,IMG), ONNX, opset_version=17,
                  input_names=["input"], output_names=["logits"], dynamo=False)
print(f"exported ONNX: {ONNX}  ({os.path.getsize(ONNX)/1e6:.1f} MB)")

torch.set_num_threads(os.cpu_count())
def cpu_bench(sz, iters=20):
    x=torch.randn(1,3,sz,sz)
    with torch.no_grad():
        for _ in range(5): model(x)
    t=time.time()
    with torch.no_grad():
        for _ in range(iters): model(x)
    return (time.time()-t)/iters*1000
print("=== CPU torch (desktop i7-12700K; Pi5 ~8-12x slower) ===")
for sz in [IMG,256]: print(f"  torch {sz}x{sz}: {cpu_bench(sz):.1f} ms")
try:
    import onnxruntime as ort
    so=ort.SessionOptions(); so.intra_op_num_threads=os.cpu_count()
    sess=ort.InferenceSession(ONNX, sess_options=so, providers=["CPUExecutionProvider"])
    xx=np.random.randn(1,3,IMG,IMG).astype(np.float32)
    for _ in range(5): sess.run(None,{"input":xx})
    t=time.time()
    for _ in range(30): sess.run(None,{"input":xx})
    print(f"  onnxruntime {IMG}x{IMG}: {(time.time()-t)/30*1000:.1f} ms")
except Exception as e: print("ort bench failed:", e)
# GPU forward for the table
if torch.cuda.is_available():
    m=model.cuda(); x=torch.randn(1,3,IMG,IMG,device='cuda')
    with torch.no_grad():
        for _ in range(10): m(x)
    torch.cuda.synchronize(); t=time.time()
    with torch.no_grad():
        for _ in range(100): m(x)
    torch.cuda.synchronize(); print(f"  GPU {IMG}x{IMG}: {(time.time()-t)/100*1000:.2f} ms")
print("DONE")
