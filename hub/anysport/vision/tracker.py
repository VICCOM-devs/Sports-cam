"""Per-camera multi-object tracker: constant-velocity Kalman filter per track, Hungarian matching
on IoU, ByteTrack-style second pass with low-score detections so occluded skaters keep their id.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .detector import Detection


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between boxes a (N,4) and b (M,4), xyxy."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


@dataclass
class Track:
    id: int
    # state: cx, cy, w, h, vx, vy
    x: np.ndarray
    P: np.ndarray
    hits: int = 1
    age: int = 0
    misses: int = 0
    score: float = 0.0
    color: np.ndarray | None = None       # jersey colour feature (set by teams.py)
    team: str | None = None
    role: str = "player"                  # player | goaltender | referee (from a multi-class model)
    history: list[tuple[float, float]] = field(default_factory=list)

    @property
    def box(self) -> np.ndarray:
        cx, cy, w, h = self.x[:4]
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    @property
    def foot(self) -> tuple[float, float]:
        cx, cy, w, h = self.x[:4]
        return float(cx), float(cy + h / 2)

    @property
    def confirmed(self) -> bool:
        return self.hits >= 3


class Tracker:
    def __init__(self, dt: float = 0.2, iou_thresh: float = 0.3, max_misses: int = 10, low_score: float = 0.15):
        self.dt = dt
        self.iou_thresh = iou_thresh
        self.max_misses = max_misses
        self.low_score = low_score
        self.tracks: list[Track] = []
        self._next = 1
        self.F = np.eye(6); self.F[0, 4] = dt; self.F[1, 5] = dt
        self.Hm = np.zeros((4, 6)); self.Hm[:4, :4] = np.eye(4)
        self.Q = np.diag([1, 1, 4, 4, 25, 25]) * dt
        self.R = np.diag([4, 4, 16, 16])

    def _predict(self, t: Track) -> None:
        t.x = self.F @ t.x
        t.P = self.F @ t.P @ self.F.T + self.Q
        t.age += 1

    def _update(self, t: Track, det: Detection) -> None:
        z = np.array([(det.x1 + det.x2) / 2, (det.y1 + det.y2) / 2, det.x2 - det.x1, det.y2 - det.y1])
        y = z - self.Hm @ t.x
        S = self.Hm @ t.P @ self.Hm.T + self.R
        K = t.P @ self.Hm.T @ np.linalg.inv(S)
        t.x = t.x + K @ y
        t.P = (np.eye(6) - K @ self.Hm) @ t.P
        t.hits += 1; t.misses = 0; t.score = det.score
        t.history.append(t.foot)
        if len(t.history) > 300:
            del t.history[:-300]

    def _new(self, det: Detection) -> Track:
        x = np.array([(det.x1 + det.x2) / 2, (det.y1 + det.y2) / 2, det.x2 - det.x1, det.y2 - det.y1, 0, 0], float)
        P = np.diag([10, 10, 10, 10, 100, 100]).astype(float)
        t = Track(self._next, x, P, score=det.score)
        self._next += 1
        t.history.append(t.foot)
        return t

    def _match(self, tracks: list[Track], dets: list[Detection], thresh: float):
        if not tracks or not dets:
            return [], list(range(len(tracks))), list(range(len(dets)))
        m = iou(np.array([t.box for t in tracks]), np.array([d.box for d in dets]))
        r, c = linear_sum_assignment(-m)
        matches, ut, ud = [], set(range(len(tracks))), set(range(len(dets)))
        for i, j in zip(r, c):
            if m[i, j] >= thresh:
                matches.append((i, j)); ut.discard(i); ud.discard(j)
        return matches, sorted(ut), sorted(ud)

    def step(self, dets: list[Detection]) -> list[Track]:
        for t in self.tracks:
            self._predict(t)
        high = [d for d in dets if d.score >= 0.35]
        low = [d for d in dets if self.low_score <= d.score < 0.35]

        matches, ut, ud = self._match(self.tracks, high, self.iou_thresh)
        for i, j in matches:
            self._update(self.tracks[i], high[j])
        # second pass: leftover tracks vs low-score detections (occlusions)
        rem_tracks = [self.tracks[i] for i in ut]
        m2, ut2, _ = self._match(rem_tracks, low, 0.4)
        for i, j in m2:
            self._update(rem_tracks[i], low[j])
        for i in ut2:
            rem_tracks[i].misses += 1
        for j in ud:
            self.tracks.append(self._new(high[j]))
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
        return [t for t in self.tracks if t.confirmed and t.misses == 0]
