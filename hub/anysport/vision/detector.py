"""Person detector on YOLOX (Apache-2.0) via ONNX Runtime.

Models are fetched from the YOLOX GitHub release on first use and verified by SHA-256 before
they are loaded. 'nano' runs at ~5 fps per camera on a laptop CPU; 's' is the accuracy pick for
a GPU hub (onnxruntime-gpu / DirectML) or for post-game processing.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import httpx
import numpy as np

log = logging.getLogger("anysport.vision.detector")

MODELS = {
    "nano": ("yolox_nano.onnx", 416, "c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d"),
    "s": ("yolox_s.onnx", 640, "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063"),
}
RELEASE = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0"
PERSON = 0  # COCO class id


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    cls: str = "player"      # player | goaltender | referee | puck | goal | faceoff_dot | center_ice

    @property
    def box(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def foot(self) -> tuple[float, float]:
        """Bottom-centre of the box: where the skates touch the ice."""
        return (self.x1 + self.x2) / 2, self.y2


def ensure_model(models_dir: Path, size: str = "nano") -> Path:
    name, _, sha = MODELS[size]
    path = models_dir / name
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == sha:
        return path
    models_dir.mkdir(parents=True, exist_ok=True)
    log.info("downloading %s", name)
    with httpx.Client(follow_redirects=True, timeout=300) as c:
        data = c.get(f"{RELEASE}/{name}").content
    if hashlib.sha256(data).hexdigest() != sha:
        raise RuntimeError(f"checksum mismatch for {name}")
    path.write_bytes(data)
    return path


CUSTOM_NAME = "custom.onnx"


class PersonDetector:
    """Generic COCO person detector (YOLOX) or an AnySport custom model (custom.onnx + custom.json).

    Custom models output decoded boxes (cx, cy, w, h in input pixels, objectness, class probs) and
    may have several classes; the generic model is decoded here from raw YOLOX outputs.
    """

    def __init__(self, model_path: Path, input_size: int, conf: float = 0.35, nms_iou: float = 0.5,
                 providers: list[str] | None = None, classes: list[str] | None = None, decoded: bool = False,
                 normalize: str = "none"):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        self.session = ort.InferenceSession(str(model_path), opts,
                                            providers=providers or ["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.size = input_size
        self.conf = conf
        self.nms_iou = nms_iou
        self.classes = classes            # None = COCO person only
        self.decoded = decoded
        self.normalize = normalize
        self.name = "custom" if classes else "coco-person"
        self._grids, self._strides = self._make_grids(input_size)

    @classmethod
    def load(cls, models_dir: Path, size: str = "nano", **kw) -> "PersonDetector":
        custom = models_dir / CUSTOM_NAME
        meta_f = custom.with_suffix(".json")
        if custom.exists() and meta_f.exists():
            meta = json.loads(meta_f.read_text())
            log.info("using custom model %s (%s)", custom, meta.get("type"))
            return cls(custom, int(meta["input"]), classes=list(meta["classes"]), decoded=True,
                       normalize=meta.get("normalize", "none"), conf=kw.get("conf", 0.3))
        name, inp, _ = MODELS[size]
        return cls(ensure_model(models_dir, size), inp, **kw)

    @staticmethod
    def _make_grids(size: int):
        grids, strides = [], []
        for s in (8, 16, 32):
            h = w = size // s
            yv, xv = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
            grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
            strides.append(np.full((h * w, 1), s))
        return np.concatenate(grids, 0).astype(np.float32), np.concatenate(strides, 0).astype(np.float32)

    def _preprocess(self, img: np.ndarray) -> tuple[np.ndarray, float]:
        h, w = img.shape[:2]
        r = min(self.size / h, self.size / w)
        resized = cv2.resize(img, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.size, self.size, 3), 114, dtype=np.uint8)
        canvas[: resized.shape[0], : resized.shape[1]] = resized
        blob = canvas.transpose(2, 0, 1)[None].astype(np.float32)  # BGR, 0..255 (YOLOX convention)
        if self.normalize == "imagenet":  # RT-DETR: RGB 0..1, mean/std
            blob = blob[:, ::-1] / 255.0
            blob = (blob - np.array([0.485, 0.456, 0.406], np.float32)[None, :, None, None]) / np.array([0.229, 0.224, 0.225], np.float32)[None, :, None, None]
        return np.ascontiguousarray(blob, dtype=np.float32), r

    def __call__(self, img_bgr: np.ndarray) -> list[Detection]:
        blob, r = self._preprocess(img_bgr)
        out = self.session.run(None, {self.input_name: blob})[0][0]
        if self.decoded:
            xy, wh, obj, cls_scores = out[:, :2], out[:, 2:4], out[:, 4], out[:, 5:]
        else:
            xy = (out[:, :2] + self._grids) * self._strides
            wh = np.exp(out[:, 2:4]) * self._strides
            obj, cls_scores = out[:, 4], out[:, 5:]
        if self.classes is None:
            cls_idx = np.full(len(out), PERSON)
            scores = obj * cls_scores[:, PERSON]
        else:
            cls_idx = cls_scores.argmax(1)
            scores = obj * cls_scores[np.arange(len(out)), cls_idx]
        keep = scores > self.conf
        if not keep.any():
            return []
        xy, wh, scores, cls_idx = xy[keep], wh[keep], scores[keep], cls_idx[keep]
        boxes = np.concatenate([xy - wh / 2, xy + wh / 2], 1) / r
        h, w = img_bgr.shape[:2]
        dets = []
        # NMS per class so a puck next to a skate is not suppressed by the player box
        for c in np.unique(cls_idx):
            m = cls_idx == c
            b_c, s_c = boxes[m], scores[m]
            idx = cv2.dnn.NMSBoxes([(float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])) for b in b_c],
                                   s_c.tolist(), self.conf, self.nms_iou)
            name = "player" if self.classes is None else self.classes[int(c)]
            for i in np.array(idx).reshape(-1):
                x1, y1, x2, y2 = b_c[i]
                dets.append(Detection(max(0.0, float(x1)), max(0.0, float(y1)), min(float(w), float(x2)),
                                      min(float(h), float(y2)), float(s_c[i]), name))
        return dets
