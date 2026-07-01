"""tiny_unet.py — a compact, grayscale-native U-Net for defect segmentation on the unrolled strip.

~0.9M params (vs resnet18-UNet's 14.3M). Classic encoder-decoder with skip connections:
  * 1-channel (grayscale) input — no RGB replication
  * 4 downsample levels, channels [16,32,64,128], bottleneck 128
  * DoubleConv (3x3 conv -> BN -> ReLU, x2) per stage
  * 2x2 max-pool down, 2x2 transposed-conv up, skip connections concatenated
  * 1x1 conv head -> 1 logit
"""
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
            nn.Conv2d(co, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class TinyUNet(nn.Module):
    def __init__(self, base=16, in_ch=1):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]         # 16, 32, 64, 128
        self.e1 = DoubleConv(in_ch, c[0]); self.e2 = DoubleConv(c[0], c[1])
        self.e3 = DoubleConv(c[1], c[2]); self.e4 = DoubleConv(c[2], c[3])
        self.pool = nn.MaxPool2d(2)
        self.bott = DoubleConv(c[3], c[3])
        self.u4 = nn.ConvTranspose2d(c[3], c[3], 2, 2); self.d4 = DoubleConv(c[3] + c[3], c[2])
        self.u3 = nn.ConvTranspose2d(c[2], c[2], 2, 2); self.d3 = DoubleConv(c[2] + c[2], c[1])
        self.u2 = nn.ConvTranspose2d(c[1], c[1], 2, 2); self.d2 = DoubleConv(c[1] + c[1], c[0])
        self.u1 = nn.ConvTranspose2d(c[0], c[0], 2, 2); self.d1 = DoubleConv(c[0] + c[0], c[0])
        self.head = nn.Conv2d(c[0], 1, 1)

    def forward(self, x):
        x1 = self.e1(x)
        x2 = self.e2(self.pool(x1))
        x3 = self.e3(self.pool(x2))
        x4 = self.e4(self.pool(x3))
        b = self.bott(self.pool(x4))
        y = self.d4(torch.cat([self.u4(b), x4], 1))
        y = self.d3(torch.cat([self.u3(y), x3], 1))
        y = self.d2(torch.cat([self.u2(y), x2], 1))
        y = self.d1(torch.cat([self.u1(y), x1], 1))
        return self.head(y)
