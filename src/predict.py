#!/usr/bin/env python3
"""
Seal-mask inference. Loads a checkpoint from run_seal_cuda.py and predicts the
seal ring mask for one image or a whole folder.

Usage:
  python predict.py --weights best.pt --input prod2 --output preds_masks
  python predict.py --weights best.pt --input prod2 --output preds_masks --overlay
  python predict.py --weights best.pt --input some_image_raw.png --output out

Outputs per image:
  <stem>_mask.png     binary mask (0/255), full original resolution
  <stem>_overlay.png  red seal overlay on the image  (only with --overlay)
"""
import os, glob, argparse
import numpy as np, cv2, torch
import torch.nn as nn, torch.nn.functional as F, torchvision

# ---- model: layer names MUST match run_seal_cuda.py (compact D/U with .c and .f) ----
class D(nn.Module):
    def __init__(s,i,k,o):
        super().__init__()
        s.c=nn.Sequential(nn.Conv2d(i+k,o,3,padding=1,bias=False),nn.BatchNorm2d(o),nn.ReLU(True),
                          nn.Conv2d(o,o,3,padding=1,bias=False),nn.BatchNorm2d(o),nn.ReLU(True))
    def forward(s,x,sk):
        x=F.interpolate(x,size=sk.shape[-2:],mode="bilinear",align_corners=False); return s.c(torch.cat([x,sk],1))

class U(nn.Module):
    def __init__(s):
        super().__init__()
        b=torchvision.models.resnet34(weights=None)
        s.stem=nn.Sequential(b.conv1,b.bn1,b.relu); s.pool=b.maxpool
        s.l1,s.l2,s.l3,s.l4=b.layer1,b.layer2,b.layer3,b.layer4
        s.d4=D(512,256,256); s.d3=D(256,128,128); s.d2=D(128,64,64); s.d1=D(64,64,32)
        s.f=nn.Sequential(nn.Conv2d(32,32,3,padding=1,bias=False),nn.BatchNorm2d(32),nn.ReLU(True),
                          nn.Conv2d(32,1,1))
    def forward(s,x):
        e0=s.stem(x); e1=s.l1(s.pool(e0)); e2=s.l2(e1); e3=s.l3(e2); e4=s.l4(e3)
        d=s.d4(e4,e3); d=s.d3(d,e2); d=s.d2(d,e1); d=s.d1(d,e0)
        d=F.interpolate(d, scale_factor=2, mode="bilinear", align_corners=False)
        return s.f(d)

def load_gray3(path):
    img=cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None,None
    lo,hi=np.percentile(img,[1,99.5]); hi=max(hi,lo+1)
    n=np.clip((img.astype(np.float32)-lo)/(hi-lo)*255,0,255).astype(np.uint8)
    return img, np.stack([n,n,n],-1)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--input", required=True, help="image file or folder")
    ap.add_argument("--output", required=True, help="output folder")
    ap.add_argument("--overlay", action="store_true", help="also write red-overlay previews")
    ap.add_argument("--pattern", default="*_raw.png", help="glob when input is a folder")
    args=ap.parse_args()

    device="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    ck=torch.load(args.weights, map_location=device)
    H,W,THR=ck["img_h"],ck["img_w"],ck.get("thresh",0.5)
    mean=np.array(ck.get("mean",(.485,.456,.406)),np.float32); std=np.array(ck.get("std",(.229,.224,.225)),np.float32)
    model=U().to(device); model.load_state_dict(ck["state_dict"]); model.eval()
    print(f"device={device}  ckpt val_dice={ck.get('val_dice','?')}  input_size={W}x{H}  thresh={THR}")

    files=[args.input] if os.path.isfile(args.input) else sorted(glob.glob(os.path.join(args.input,args.pattern)))
    os.makedirs(args.output, exist_ok=True)
    print(f"{len(files)} image(s) -> {args.output}")
    for i,p in enumerate(files,1):
        orig,img3=load_gray3(p)
        if img3 is None: print("skip unreadable",p); continue
        oh,ow=orig.shape
        x=cv2.resize(img3,(W,H)).astype(np.float32)/255.0
        x=((x-mean)/std).transpose(2,0,1)[None]
        with torch.no_grad():
            prob=torch.sigmoid(model(torch.from_numpy(x).to(device)))[0,0].cpu().numpy()
        mask=(cv2.resize(prob,(ow,oh))>THR).astype(np.uint8)*255   # back to original res
        stem=os.path.basename(p).replace("_raw.png","").replace(".png","")
        cv2.imwrite(os.path.join(args.output,f"{stem}_mask.png"), mask)
        if args.overlay:
            lo,hi=np.percentile(orig,[1,99.5]); hi=max(hi,lo+1)
            vis=cv2.cvtColor(np.clip((orig-lo)/(hi-lo)*255,0,255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
            vis[mask>0]=np.clip(0.5*vis[mask>0]+np.array([0,0,150]),0,255).astype(np.uint8)
            cv2.imwrite(os.path.join(args.output,f"{stem}_overlay.png"), vis)
        if i%50==0 or i==len(files): print(f"  [{i}/{len(files)}]")
    print("done")

if __name__=="__main__":
    main()
