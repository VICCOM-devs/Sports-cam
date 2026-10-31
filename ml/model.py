"""AnySport detector: a compact YOLOX-style anchor-free detector in plain PyTorch (Apache-2.0 idea,
our own code, no AGPL). CSP backbone + PAFPN neck + decoupled head on strides 8/16/32, with an extra
stride-4 level when `small_objects=True` so the puck (≈6–10 px in a 1280-wide frame) gets a grid fine
enough to be found.

Export target: ONNX with output (1, N, 5 + num_classes) in the same layout the hub's detector
already decodes (xy offsets, log wh, objectness, class scores) → one decoder for both models.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, k // 2, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, c, shortcut=True):
        super().__init__()
        self.cv1 = Conv(c, c, 1)
        self.cv2 = Conv(c, c, 3)
        self.add = shortcut

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class CSP(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True):
        super().__init__()
        h = c2 // 2
        self.cv1 = Conv(c1, h, 1)
        self.cv2 = Conv(c1, h, 1)
        self.m = nn.Sequential(*[Bottleneck(h, shortcut) for _ in range(n)])
        self.cv3 = Conv(2 * h, c2, 1)

    def forward(self, x):
        return self.cv3(torch.cat([self.m(self.cv1(x)), self.cv2(x)], 1))


class SPPF(nn.Module):
    def __init__(self, c, k=5):
        super().__init__()
        self.cv1 = Conv(c, c // 2, 1)
        self.cv2 = Conv(c * 2, c, 1)
        self.m = nn.MaxPool2d(k, 1, k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x); y2 = self.m(y1); y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))


class Backbone(nn.Module):
    """Returns features at strides 4, 8, 16, 32."""

    def __init__(self, width=0.5, depth=0.33):
        super().__init__()
        c = [int(64 * width), int(128 * width), int(256 * width), int(512 * width), int(1024 * width)]
        d = [max(1, round(3 * depth)), max(1, round(9 * depth)), max(1, round(9 * depth)), max(1, round(3 * depth))]
        self.stem = nn.Sequential(Conv(3, c[0], 3, 2), Conv(c[0], c[0], 3, 1))          # /2
        self.s4 = nn.Sequential(Conv(c[0], c[1], 3, 2), CSP(c[1], c[1], d[0]))           # /4
        self.s8 = nn.Sequential(Conv(c[1], c[2], 3, 2), CSP(c[2], c[2], d[1]))           # /8
        self.s16 = nn.Sequential(Conv(c[2], c[3], 3, 2), CSP(c[3], c[3], d[2]))          # /16
        self.s32 = nn.Sequential(Conv(c[3], c[4], 3, 2), SPPF(c[4]), CSP(c[4], c[4], d[3], False))  # /32
        self.channels = c[1:]

    def forward(self, x):
        x = self.stem(x)
        p4 = self.s4(x); p8 = self.s8(p4); p16 = self.s16(p8); p32 = self.s32(p16)
        return p4, p8, p16, p32


class PAFPN(nn.Module):
    def __init__(self, chs, depth=0.33, small_objects=False):
        super().__init__()
        c4, c8, c16, c32 = chs
        n = max(1, round(3 * depth))
        self.small = small_objects
        self.lat32 = Conv(c32, c16, 1)
        self.td16 = CSP(2 * c16, c16, n, False)
        self.lat16 = Conv(c16, c8, 1)
        self.td8 = CSP(2 * c8, c8, n, False)
        if small_objects:
            self.lat8 = Conv(c8, c4, 1)
            self.td4 = CSP(2 * c4, c4, n, False)
            self.down4 = Conv(c4, c4, 3, 2)
            self.bu8 = CSP(c4 + c8, c8, n, False)
        self.down8 = Conv(c8, c8, 3, 2)
        self.bu16 = CSP(2 * c8, c16, n, False)
        self.down16 = Conv(c16, c16, 3, 2)
        self.bu32 = CSP(2 * c16, c32, n, False)
        self.out_channels = ([c4] if small_objects else []) + [c8, c16, c32]

    def forward(self, feats):
        p4, p8, p16, p32 = feats
        up = lambda t: F.interpolate(t, scale_factor=2.0, mode="nearest")
        l32 = self.lat32(p32)
        t16 = self.td16(torch.cat([up(l32), p16], 1))
        l16 = self.lat16(t16)
        t8 = self.td8(torch.cat([up(l16), p8], 1))
        outs = []
        if self.small:
            l8 = self.lat8(t8)
            t4 = self.td4(torch.cat([up(l8), p4], 1))
            outs.append(t4)
            t8 = self.bu8(torch.cat([self.down4(t4), t8], 1))
        o8 = t8
        o16 = self.bu16(torch.cat([self.down8(o8), l16], 1))
        o32 = self.bu32(torch.cat([self.down16(o16), l32], 1))
        return outs + [o8, o16, o32]


class Head(nn.Module):
    """Decoupled head per level: cls branch and reg/obj branch."""

    def __init__(self, chs, num_classes, width=0.5):
        super().__init__()
        c = int(256 * width)
        self.nc = num_classes
        self.stems = nn.ModuleList([Conv(ch, c, 1) for ch in chs])
        self.cls_convs = nn.ModuleList([nn.Sequential(Conv(c, c, 3), Conv(c, c, 3)) for _ in chs])
        self.reg_convs = nn.ModuleList([nn.Sequential(Conv(c, c, 3), Conv(c, c, 3)) for _ in chs])
        self.cls_pred = nn.ModuleList([nn.Conv2d(c, num_classes, 1) for _ in chs])
        self.reg_pred = nn.ModuleList([nn.Conv2d(c, 4, 1) for _ in chs])
        self.obj_pred = nn.ModuleList([nn.Conv2d(c, 1, 1) for _ in chs])
        prior = -math.log((1 - 0.01) / 0.01)
        for m in list(self.cls_pred) + list(self.obj_pred):
            nn.init.constant_(m.bias, prior)

    def forward(self, feats):
        outs = []
        for i, f in enumerate(feats):
            x = self.stems[i](f)
            cls = self.cls_pred[i](self.cls_convs[i](x))
            r = self.reg_convs[i](x)
            outs.append(torch.cat([self.reg_pred[i](r), self.obj_pred[i](r), cls], 1))  # (B, 5+nc, H, W)
        return outs


class Detector(nn.Module):
    SIZES = {"nano": (0.25, 0.33), "s": (0.5, 0.33), "m": (0.75, 0.67)}

    def __init__(self, num_classes=7, size="s", small_objects=True):
        super().__init__()
        width, depth = self.SIZES[size]
        self.backbone = Backbone(width, depth)
        self.neck = PAFPN(self.backbone.channels, depth, small_objects)
        self.head = Head(self.neck.out_channels, num_classes, width)
        self.strides = ([4] if small_objects else []) + [8, 16, 32]
        self.nc = num_classes

    def forward(self, x):
        return self.head(self.neck(self.backbone(x)))

    # -- decode to the hub's layout: (B, N, 5+nc) with xy in pixels, wh as exp(), obj & cls logits→sigmoid
    def decode(self, outs, apply_sigmoid=True):
        allp, grids, strides = [], [], []
        for o, s in zip(outs, self.strides):
            b, c, h, w = o.shape
            yv, xv = torch.meshgrid(torch.arange(h, device=o.device), torch.arange(w, device=o.device), indexing="ij")
            grid = torch.stack((xv, yv), 2).reshape(1, -1, 2).float()
            p = o.permute(0, 2, 3, 1).reshape(b, -1, c)
            allp.append(p); grids.append(grid); strides.append(torch.full((1, h * w, 1), float(s), device=o.device))
        p = torch.cat(allp, 1); g = torch.cat(grids, 1); st = torch.cat(strides, 1)
        xy = (p[..., :2] + g) * st
        wh = torch.exp(p[..., 2:4].clamp(max=10)) * st
        rest = p[..., 4:].sigmoid() if apply_sigmoid else p[..., 4:]
        return torch.cat([xy, wh, rest], -1), g, st


class ExportWrapper(nn.Module):
    """ONNX output exactly like the hub expects from the YOLOX decoder: raw (xy offset, log wh,
    obj logit, cls logits) with grids applied by the hub? No: simpler to ship decoded boxes."""

    def __init__(self, det: Detector):
        super().__init__()
        self.det = det

    def forward(self, x):
        dec, _, _ = self.det.decode(self.det(x))
        return dec  # (1, N, 4 + 1 + nc): cx, cy, w, h (pixels), obj, cls probs
