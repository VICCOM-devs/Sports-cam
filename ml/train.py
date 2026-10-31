"""Train the AnySport detector on COCO-format data from prepare_data.py.

    python ml/train.py --data ml/data --model yolox-s --epochs 60 --img 960 --out ml/runs/yolox-s
    python ml/train.py --data ml/data --model rtdetr-r18 --epochs 40 --out ml/runs/rtdetr

yolox-* : our plain-PyTorch detector (ml/model.py), SimOTA-lite label assignment, mosaic + flip + HSV.
rtdetr-* : Hugging Face `transformers` RT-DETR (Apache-2.0) fine-tune; needs `pip install transformers`.
Checkpoints: <out>/best.pt (by val mAP50), <out>/last.pt, <out>/log.jsonl.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classes import CLASSES  # noqa: E402
from model import Detector  # noqa: E402


# ----------------------------------------------------------------------------- data
class CocoDet(Dataset):
    def __init__(self, root: Path, split: str, img_size: int, train: bool):
        import cv2  # noqa: F401
        d = json.loads((root / f"{split}.json").read_text())
        self.root = root / split
        self.imgs = d["images"]
        anns = {}
        for a in d["annotations"]:
            anns.setdefault(a["image_id"], []).append(a)
        self.anns = anns
        self.size = img_size
        self.train = train

    def __len__(self):
        return len(self.imgs)

    def load(self, i):
        import cv2
        im = self.imgs[i]
        img = cv2.imread(str(self.root / im["file_name"]))
        boxes = np.array([[a["category_id"], *a["bbox"]] for a in self.anns.get(im["id"], [])], np.float32).reshape(-1, 5)
        if len(boxes):
            boxes[:, 3] += boxes[:, 1]; boxes[:, 4] += boxes[:, 2]   # xywh → xyxy
        return img, boxes

    def letterbox(self, img, boxes, size):
        import cv2
        h, w = img.shape[:2]
        r = min(size / h, size / w)
        nh, nw = int(h * r), int(w * r)
        img = cv2.resize(img, (nw, nh))
        canvas = np.full((size, size, 3), 114, np.uint8)
        canvas[:nh, :nw] = img
        if len(boxes):
            boxes = boxes.copy(); boxes[:, 1:] *= r
        return canvas, boxes

    def mosaic(self, i):
        import cv2
        s = self.size
        idxs = [i] + random.sample(range(len(self)), 3)
        canvas = np.full((2 * s, 2 * s, 3), 114, np.uint8)
        cx, cy = int(random.uniform(0.5 * s, 1.5 * s)), int(random.uniform(0.5 * s, 1.5 * s))
        out_boxes = []
        for k, j in enumerate(idxs):
            img, boxes = self.load(j)
            img, boxes = self.letterbox(img, boxes, s)
            h, w = img.shape[:2]
            if k == 0: x1a, y1a, x2a, y2a = max(cx - w, 0), max(cy - h, 0), cx, cy; x1b, y1b = w - (x2a - x1a), h - (y2a - y1a)
            elif k == 1: x1a, y1a, x2a, y2a = cx, max(cy - h, 0), min(cx + w, 2 * s), cy; x1b, y1b = 0, h - (y2a - y1a)
            elif k == 2: x1a, y1a, x2a, y2a = max(cx - w, 0), cy, cx, min(2 * s, cy + h); x1b, y1b = w - (x2a - x1a), 0
            else: x1a, y1a, x2a, y2a = cx, cy, min(cx + w, 2 * s), min(2 * s, cy + h); x1b, y1b = 0, 0
            canvas[y1a:y2a, x1a:x2a] = img[y1b:y1b + (y2a - y1a), x1b:x1b + (x2a - x1a)]
            if len(boxes):
                b = boxes.copy(); b[:, [1, 3]] += x1a - x1b; b[:, [2, 4]] += y1a - y1b
                out_boxes.append(b)
        boxes = np.concatenate(out_boxes, 0) if out_boxes else np.zeros((0, 5), np.float32)
        # random crop back to s×s around the centre
        ox, oy = int(random.uniform(0, s)), int(random.uniform(0, s))
        canvas = canvas[oy:oy + s, ox:ox + s]
        if len(boxes):
            boxes[:, [1, 3]] -= ox; boxes[:, [2, 4]] -= oy
            boxes[:, 1:] = boxes[:, 1:].clip(0, s)
            keep = (boxes[:, 3] - boxes[:, 1] > 2) & (boxes[:, 4] - boxes[:, 2] > 2)
            boxes = boxes[keep]
        return canvas, boxes

    def __getitem__(self, i):
        import cv2
        if self.train and random.random() < 0.8:
            img, boxes = self.mosaic(i)
        else:
            img, boxes = self.load(i)
            img, boxes = self.letterbox(img, boxes, self.size)
        if self.train:
            if random.random() < 0.5:
                img = img[:, ::-1].copy()
                if len(boxes):
                    x1 = self.size - boxes[:, 3]; x2 = self.size - boxes[:, 1]
                    boxes[:, 1], boxes[:, 3] = x1, x2
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[..., 0] = (hsv[..., 0] + random.uniform(-8, 8)) % 180
            hsv[..., 1] *= random.uniform(0.6, 1.4); hsv[..., 2] *= random.uniform(0.6, 1.4)
            img = cv2.cvtColor(hsv.clip(0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        t = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float()
        return t, torch.from_numpy(boxes)


def collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    return imgs, targets


# ----------------------------------------------------------------------------- loss (SimOTA-lite)
def box_iou(a, b):
    tl = torch.max(a[:, None, :2], b[None, :, :2]); br = torch.min(a[:, None, 2:], b[None, :, 2:])
    inter = (br - tl).clamp(min=0).prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).prod(-1); area_b = (b[:, 2:] - b[:, :2]).prod(-1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-7)


def giou_loss(p, t):
    iou = box_iou(p, t).diagonal() if p.shape == t.shape else None
    tl = torch.max(p[:, :2], t[:, :2]); br = torch.min(p[:, 2:], t[:, 2:])
    inter = (br - tl).clamp(min=0).prod(-1)
    ap = (p[:, 2:] - p[:, :2]).prod(-1); at = (t[:, 2:] - t[:, :2]).prod(-1)
    union = ap + at - inter + 1e-7
    iou = inter / union
    ctl = torch.min(p[:, :2], t[:, :2]); cbr = torch.max(p[:, 2:], t[:, 2:])
    c = (cbr - ctl).clamp(min=0).prod(-1) + 1e-7
    return 1 - (iou - (c - union) / c)


def compute_loss(model: Detector, outs, targets, nc):
    dec, grid, st = model.decode(outs, apply_sigmoid=False)      # (B, N, 5+nc) raw logits for obj/cls
    B, N, _ = dec.shape
    centers = (grid + 0.5) * st                                 # anchor point centres, (1, N, 2)
    total_obj = total_cls = total_box = 0.0
    num_fg = 0
    for b in range(B):
        p = dec[b]
        pxy, pwh = p[:, :2], p[:, 2:4]
        pbox = torch.cat([pxy - pwh / 2, pxy + pwh / 2], 1)
        obj_t = torch.zeros(N, device=p.device)
        t = targets[b].to(p.device)
        if len(t) == 0:
            total_obj = total_obj + F.binary_cross_entropy_with_logits(p[:, 4], obj_t, reduction="sum")
            continue
        tcls, tbox = t[:, 0].long(), t[:, 1:5]
        c = centers[0]
        # candidates: anchor centre inside the gt box OR within 2.5 strides of its centre
        gcx, gcy = (tbox[:, 0] + tbox[:, 2]) / 2, (tbox[:, 1] + tbox[:, 3]) / 2
        in_box = (c[None, :, 0] > tbox[:, None, 0]) & (c[None, :, 0] < tbox[:, None, 2]) & (c[None, :, 1] > tbox[:, None, 1]) & (c[None, :, 1] < tbox[:, None, 3])
        r = 2.5 * st[0, :, 0]
        in_ctr = ((c[None, :, 0] - gcx[:, None]).abs() < r[None]) & ((c[None, :, 1] - gcy[:, None]).abs() < r[None])
        cand = (in_box | in_ctr)
        cand_any = cand.any(0)
        if not cand_any.any():
            total_obj = total_obj + F.binary_cross_entropy_with_logits(p[:, 4], obj_t, reduction="sum")
            continue
        ci = cand_any.nonzero().squeeze(1)
        ious = box_iou(tbox, pbox[ci])                                          # (G, C)
        cls_p = (p[ci, 5:].sigmoid() * p[ci, 4:5].sigmoid()).sqrt()           # (C, nc)
        cls_t = F.one_hot(tcls, nc).float()
        cls_cost = F.binary_cross_entropy(cls_p[None].expand(len(t), -1, -1).clamp(1e-6, 1 - 1e-6), cls_t[:, None].expand(-1, len(ci), -1), reduction="none").sum(-1)
        cost = cls_cost + 3.0 * (-torch.log(ious + 1e-8)) + 1e5 * (~(in_box & in_ctr)[:, ci]).float()
        # dynamic k per gt
        matching = torch.zeros_like(cost, dtype=torch.bool)
        topk = min(10, len(ci))
        ks = ious.topk(topk, dim=1).values.sum(1).int().clamp(min=1)
        for g in range(len(t)):
            idx = cost[g].topk(int(ks[g]), largest=False).indices
            matching[g, idx] = True
        multi = matching.sum(0) > 1
        if multi.any():
            best = cost[:, multi].argmin(0)
            matching[:, multi] = False
            matching[best, multi.nonzero().squeeze(1)] = True
        fg_mask = matching.any(0)
        fg = ci[fg_mask]
        gt_idx = matching[:, fg_mask].float().argmax(0)
        num_fg += len(fg)
        obj_t[fg] = 1.0
        total_obj = total_obj + F.binary_cross_entropy_with_logits(p[:, 4], obj_t, reduction="sum")
        cls_target = F.one_hot(tcls[gt_idx], nc).float() * ious[gt_idx, fg_mask].unsqueeze(1).detach()
        total_cls = total_cls + F.binary_cross_entropy_with_logits(p[fg, 5:], cls_target, reduction="sum")
        total_box = total_box + giou_loss(pbox[fg], tbox[gt_idx]).sum()
    num_fg = max(num_fg, 1)
    return (5.0 * total_box + total_obj + total_cls) / num_fg, {"box": float(torch.as_tensor(total_box).detach()) / num_fg, "obj": float(torch.as_tensor(total_obj).detach()) / num_fg, "cls": float(torch.as_tensor(total_cls).detach()) / num_fg}


# ----------------------------------------------------------------------------- eval (mAP50, per class)
@torch.no_grad()
def evaluate(model, loader, nc, device, conf=0.01, iou_thr=0.5):
    import torchvision
    model.eval()
    stats = {c: {"tp": [], "conf": [], "n_gt": 0} for c in range(nc)}
    for imgs, targets in loader:
        dec, _, _ = model.decode(model(imgs.to(device)))
        for b in range(len(imgs)):
            p = dec[b]
            scores_all = p[:, 4:5] * p[:, 5:]
            score, cls = scores_all.max(1)
            keep = score > conf
            p, score, cls = p[keep], score[keep], cls[keep]
            boxes = torch.cat([p[:, :2] - p[:, 2:4] / 2, p[:, :2] + p[:, 2:4] / 2], 1)
            k = torchvision.ops.batched_nms(boxes, score, cls, 0.6)[:300]
            boxes, score, cls = boxes[k], score[k], cls[k]
            t = targets[b].to(device)
            for c in range(nc):
                gt = t[t[:, 0] == c, 1:5]
                stats[c]["n_gt"] += len(gt)
                m = cls == c
                if not m.any():
                    continue
                pb, ps = boxes[m], score[m]
                order = ps.argsort(descending=True)
                pb, ps = pb[order], ps[order]
                used = torch.zeros(len(gt), dtype=torch.bool, device=device)
                for i in range(len(pb)):
                    tp = 0
                    if len(gt):
                        ious = box_iou(pb[i:i + 1], gt)[0]
                        ious[used] = 0
                        j = int(ious.argmax())
                        if ious[j] >= iou_thr:
                            used[j] = True; tp = 1
                    stats[c]["tp"].append(tp); stats[c]["conf"].append(float(ps[i]))
    aps = {}
    for c in range(nc):
        s = stats[c]
        if s["n_gt"] == 0:
            continue
        if not s["tp"]:
            aps[CLASSES[c]] = 0.0; continue
        order = np.argsort(-np.array(s["conf"]))
        tp = np.array(s["tp"])[order]
        ctp = np.cumsum(tp); cfp = np.cumsum(1 - tp)
        rec = ctp / s["n_gt"]; prec = ctp / (ctp + cfp)
        # 101-point interpolation
        ap = 0.0
        for r in np.linspace(0, 1, 101):
            pr = prec[rec >= r]
            ap += (pr.max() if len(pr) else 0) / 101
        aps[CLASSES[c]] = float(ap)
    model.train()
    return float(np.mean(list(aps.values()))) if aps else 0.0, aps


# ----------------------------------------------------------------------------- RT-DETR path
def train_rtdetr(a, device):
    from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
    name = {"rtdetr-r18": "PekingU/rtdetr_r18vd", "rtdetr-r50": "PekingU/rtdetr_r50vd"}[a.model]
    proc = RTDetrImageProcessor.from_pretrained(name)
    model = RTDetrForObjectDetection.from_pretrained(name, num_labels=len(CLASSES), ignore_mismatched_sizes=True).to(device)
    ds = CocoDet(Path(a.data), "train", a.img, train=False)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    model.train()
    for ep in range(a.epochs):
        idx = list(range(len(ds))); random.shuffle(idx)
        for s in range(0, len(idx), a.batch):
            batch = [ds.load(i) for i in idx[s:s + a.batch]]
            images = [b[0][:, :, ::-1] for b in batch]
            labels = [{"class_labels": torch.tensor(b[1][:, 0], dtype=torch.long), "boxes": torch.tensor(np.stack([(b[1][:, 1] + b[1][:, 3]) / 2 / b[0].shape[1], (b[1][:, 2] + b[1][:, 4]) / 2 / b[0].shape[0], (b[1][:, 3] - b[1][:, 1]) / b[0].shape[1], (b[1][:, 4] - b[1][:, 2]) / b[0].shape[0]], 1) if len(b[1]) else np.zeros((0, 4)), dtype=torch.float32)} for b in batch]
            enc = proc(images=images, return_tensors="pt", size={"height": a.img, "width": a.img})
            enc = {k: v.to(device) for k, v in enc.items()}
            labels = [{k: v.to(device) for k, v in l.items()} for l in labels]
            loss = model(**enc, labels=labels).loss
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"epoch {ep + 1}/{a.epochs} loss {float(loss):.3f}")
        model.save_pretrained(out / "hf"); proc.save_pretrained(out / "hf")
    print("saved", out / "hf", "(export with: python ml/export.py --run", out, ")")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ml/data")
    ap.add_argument("--model", default="yolox-s", choices=["yolox-nano", "yolox-s", "yolox-m", "rtdetr-r18", "rtdetr-r50"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--img", type=int, default=960)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="ml/runs/run")
    ap.add_argument("--resume", default="")
    ap.add_argument("--no-small", action="store_true", help="drop the stride-4 level (faster, worse on puck)")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    if a.model.startswith("rtdetr"):
        return train_rtdetr(a, device)

    size = a.model.split("-")[1]
    model = Detector(len(CLASSES), size, small_objects=not a.no_small).to(device)
    if a.resume:
        model.load_state_dict(torch.load(a.resume, map_location=device)["model"])
    train_ds = CocoDet(Path(a.data), "train", a.img, train=True)
    val_ds = CocoDet(Path(a.data), "val", a.img, train=False)
    tl = DataLoader(train_ds, a.batch, shuffle=True, num_workers=a.workers, collate_fn=collate, drop_last=True)
    vl = DataLoader(val_ds, a.batch, shuffle=False, num_workers=a.workers, collate_fn=collate)
    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
    steps = a.epochs * len(tl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * s / max(1, steps))))
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    best = -1.0
    log = open(out / "log.jsonl", "a")
    step = 0
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); run = {"box": 0, "obj": 0, "cls": 0}; n = 0
        for imgs, targets in tl:
            imgs = imgs.to(device)
            with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                outs = model(imgs)
            loss, parts = compute_loss(model, [o.float() for o in outs], targets, len(CLASSES))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(opt); scaler.update(); sched.step(); step += 1
            for k in run: run[k] += parts[k]
            n += 1
        m50, per = evaluate(model, vl, len(CLASSES), device)
        rec = {"epoch": ep + 1, "loss": {k: v / max(n, 1) for k, v in run.items()}, "map50": m50, "per_class": per, "time_s": round(time.time() - t0)}
        print(json.dumps(rec)); log.write(json.dumps(rec) + "\n"); log.flush()
        ck = {"model": model.state_dict(), "size": size, "small_objects": not a.no_small, "classes": CLASSES, "img": a.img, "map50": m50}
        torch.save(ck, out / "last.pt")
        if m50 > best:
            best = m50; torch.save(ck, out / "best.pt")
    print(f"done. best mAP50 {best:.3f} → {out / 'best.pt'}")


if __name__ == "__main__":
    main()
