"""Export a training run to the ONNX file the hub loads.

    python ml/export.py --run ml/runs/yolox-s --out anysport-hockey.onnx [--img 960]

Output tensor: (1, N, 5 + num_classes) = cx, cy, w, h in input pixels, objectness, class probs.
A JSON sidecar (same name, .json) records classes, input size and model type for the hub.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classes import CLASSES  # noqa: E402
from model import Detector, ExportWrapper  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default="anysport-hockey.onnx")
    ap.add_argument("--img", type=int, default=0)
    a = ap.parse_args()
    run = Path(a.run)
    if (run / "hf").exists():   # RT-DETR
        from transformers import RTDetrForObjectDetection
        m = RTDetrForObjectDetection.from_pretrained(run / "hf").eval()
        img = a.img or 640

        class W(torch.nn.Module):
            def __init__(s, m): super().__init__(); s.m = m
            def forward(s, x):
                o = s.m(pixel_values=x)
                probs = o.logits.sigmoid()                     # (1, Q, nc)
                b = o.pred_boxes                               # (1, Q, 4) cxcywh normalized
                scale = torch.tensor([img, img, img, img], dtype=b.dtype)
                obj = probs.max(-1, keepdim=True).values
                return torch.cat([b * scale, obj, probs], -1)

        torch.onnx.export(W(m), torch.zeros(1, 3, img, img), a.out, input_names=["images"], output_names=["output"], opset_version=17, dynamo=False)
        meta = {"classes": CLASSES, "input": img, "type": "rtdetr", "normalize": "imagenet"}
    else:
        ck = torch.load(run / "best.pt", map_location="cpu")
        det = Detector(len(ck["classes"]), ck["size"], ck.get("small_objects", True))
        det.load_state_dict(ck["model"]); det.eval()
        img = a.img or ck.get("img", 960)
        torch.onnx.export(ExportWrapper(det), torch.zeros(1, 3, img, img), a.out, input_names=["images"], output_names=["output"], opset_version=17, dynamo=False)
        meta = {"classes": ck["classes"], "input": img, "type": "anysport-yolox", "normalize": "none", "map50": ck.get("map50")}
    Path(a.out).with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print("wrote", a.out, "and", Path(a.out).with_suffix(".json"), meta)


if __name__ == "__main__":
    main()
