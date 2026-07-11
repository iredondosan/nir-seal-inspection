"""Build the demo's ONNX model set from the DEPLOYED weights (the thesis models).

- Seal: copies the training-exported ONNX (seal_lite_reviewed_{1280,512,384}.onnx),
  which are the deployed seal models (Dice 0.967 at 1280).
- Defect: exports defect_strip.pt (ResNet18, deployed) and tiny_defect.pt (TinyUNet)
  to self-contained ONNX, matching demo/pipeline.py (fixed input, raw-logit output).

Output -> demo/models/.  Run:  python deploy/export_demo_onnx.py
"""
import shutil, torch
from seal_inspection import core
from seal_inspection.tiny_unet import TinyUNet
from seal_inspection.paths import ROOT as R, MODELS
HS, WS = 128, 1536
DEMO = f"{R}/demo/models"

for src, dst in [("seal_lite_reviewed_1280.onnx", "seal.onnx"),
                 ("seal_lite_reviewed_512.onnx", "seal_512.onnx"),
                 ("seal_lite_reviewed_384.onnx", "seal_384.onnx")]:
    shutil.copyfile(f"{MODELS}/{src}", f"{DEMO}/{dst}")

m, _ = core.load_unet(f"{R}/models/defect_strip.pt", "cpu"); m.eval()
torch.onnx.export(m, torch.randn(1, 3, HS, WS), f"{DEMO}/defect.onnx",
                  input_names=["input"], output_names=["logits"], opset_version=17,
                  do_constant_folding=True, dynamo=False)
t = TinyUNet(base=16, in_ch=1)
t.load_state_dict(torch.load(f"{R}/models/tiny_defect.pt", map_location="cpu",
                             weights_only=False)["state_dict"]); t.eval()
torch.onnx.export(t, torch.randn(1, 1, HS, WS), f"{DEMO}/defect_tiny.onnx",
                  input_names=["input"], output_names=["logits"], opset_version=17,
                  do_constant_folding=True, dynamo=False)
print("demo/models/ ready: seal{,_512,_384}.onnx + defect{,_tiny}.onnx")
