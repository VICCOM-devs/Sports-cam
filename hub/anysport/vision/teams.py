"""Team separation by jersey colour.

Feature: hue/saturation histogram of the torso region (upper-middle third of the box, which
avoids ice, boards, helmet and skates). Two k-means clusters = the teams; detections with very
low saturation (black/white stripes) are candidates for 'referee'. The admin confirms the
mapping cluster → home/away once per game; after that the assignment is automatic.
"""
from __future__ import annotations

import numpy as np
import cv2

BINS_H, BINS_S = 12, 4


def torso_feature(img_bgr: np.ndarray, box) -> np.ndarray | None:
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    h, w = y2 - y1, x2 - x1
    if h < 24 or w < 8:
        return None
    ty1, ty2 = y1 + int(h * 0.2), y1 + int(h * 0.55)
    tx1, tx2 = x1 + int(w * 0.2), x2 - int(w * 0.2)
    torso = img_bgr[max(0, ty1):ty2, max(0, tx1):tx2]
    if torso.size == 0:
        return None
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [BINS_H, BINS_S], [0, 180, 0, 256]).flatten()
    sat = float(hsv[..., 1].mean()) / 255
    val = float(hsv[..., 2].mean()) / 255
    hist = hist / (hist.sum() + 1e-9)
    return np.concatenate([hist, [sat, val]]).astype(np.float32)


class TeamClassifier:
    def __init__(self):
        self.centers: np.ndarray | None = None      # (2, D)
        self.labels: dict[int, str] = {0: "A", 1: "B"}  # cluster → team key; admin maps A/B → home/away
        self.mapping: dict[str, str] = {"A": "home", "B": "away"}

    def fit(self, feats: list[np.ndarray]) -> bool:
        X = np.array([f for f in feats if f is not None and f[-2] > 0.12])  # skip grey/white (refs, goalies in white)
        if len(X) < 6:
            return False
        X = X[:, :-2]
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3)
        _, labels, centers = cv2.kmeans(X.astype(np.float32), 2, None, crit, 5, cv2.KMEANS_PP_CENTERS)
        self.centers = centers
        return True

    def cluster(self, feat: np.ndarray | None) -> str | None:
        """'A' | 'B' | 'ref' | None (not fitted yet)."""
        if feat is None:
            return None
        if feat[-2] < 0.12:
            return "ref"
        if self.centers is None:
            return None
        d = np.linalg.norm(self.centers - feat[:-2], axis=1)
        return self.labels[int(np.argmin(d))]

    def classify(self, feat: np.ndarray | None) -> str | None:
        c = self.cluster(feat)
        return self.mapping.get(c, c) if c else None

    def to_dict(self) -> dict:
        return {"centers": None if self.centers is None else self.centers.tolist(), "mapping": self.mapping}

    def load(self, d: dict) -> None:
        if d.get("centers"):
            self.centers = np.array(d["centers"], np.float32)
        self.mapping = d.get("mapping", self.mapping)
