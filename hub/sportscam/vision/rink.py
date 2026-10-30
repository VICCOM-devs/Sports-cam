"""Rink templates and camera calibration.

Rink coordinates: metres, origin at centre ice, x along the length (towards the away goal is +x),
y across the width (+y to the left as seen from the home bench side). Landmarks are things a
parent can tap on a phone frame: faceoff dots, line ends, goal posts.

A camera is calibrated by matching ≥4 landmarks (image pixel ↔ rink metre) and fitting a
homography of the ice plane. Skate positions (bottom-centre of a detection box) are then mapped
to rink metres.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Landmark:
    id: str
    name_sv: str
    name_en: str
    x: float
    y: float


@dataclass
class RinkTemplate:
    id: str
    length: float
    width: float
    corner_radius: float
    blue_line_x: float      # distance from centre to each blue line
    goal_line_x: float      # distance from centre to each goal line
    dot_x: float            # end-zone faceoff dot distance from centre
    dot_y: float            # end-zone / neutral dot offset from centre line
    neutral_dot_x: float
    landmarks: list[Landmark] = field(default_factory=list)

    def __post_init__(self):
        if not self.landmarks:
            L = self.landmarks
            L.append(Landmark("center", "Mittpunkt", "Centre dot", 0, 0))
            for sx, side in ((-1, "home"), (1, "away")):
                sv = "hemma" if side == "home" else "borta"
                en = "home" if side == "home" else "away"
                for sy, lr in ((1, "L"), (-1, "R")):
                    L.append(Landmark(f"dot_{side}_{lr}", f"Tekningspunkt {sv} {lr}", f"Faceoff dot {en} {lr}",
                                      sx * self.dot_x, sy * self.dot_y))
                    L.append(Landmark(f"ndot_{side}_{lr}", f"Neutral tekningspunkt {sv} {lr}",
                                      f"Neutral dot {en} {lr}", sx * self.neutral_dot_x, sy * self.dot_y))
                    L.append(Landmark(f"blue_{side}_{lr}", f"Blålinje {sv}, {'vänster' if lr == 'L' else 'höger'} sarg",
                                      f"Blue line {en}, {'left' if lr == 'L' else 'right'} boards",
                                      sx * self.blue_line_x, sy * self.width / 2))
                    L.append(Landmark(f"goal_{side}_{lr}", f"Mållinje {sv}, {'vänster' if lr == 'L' else 'höger'} sarg",
                                      f"Goal line {en}, {'left' if lr == 'L' else 'right'} boards",
                                      sx * self.goal_line_x, sy * (self.width / 2 - self.corner_radius)))
                    L.append(Landmark(f"post_{side}_{lr}", f"Målstolpe {sv} {lr}", f"Goal post {en} {lr}",
                                      sx * self.goal_line_x, sy * 0.915))
            for sy, lr in ((1, "L"), (-1, "R")):
                L.append(Landmark(f"center_{lr}", f"Mittlinje, {'vänster' if lr == 'L' else 'höger'} sarg",
                                  f"Centre line, {'left' if lr == 'L' else 'right'} boards", 0, sy * self.width / 2))

    def lines(self) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        """Straight rink lines for overlay drawing (metres)."""
        w2 = self.width / 2
        out = [((0, -w2), (0, w2))]
        for sx in (-1, 1):
            out.append(((sx * self.blue_line_x, -w2), (sx * self.blue_line_x, w2)))
            g = w2 - self.corner_radius
            out.append(((sx * self.goal_line_x, -g), (sx * self.goal_line_x, g)))
        return out

    def outline(self, n: int = 12) -> list[tuple[float, float]]:
        """Rounded-rectangle boards polygon (metres)."""
        L, W, r = self.length / 2, self.width / 2, self.corner_radius
        pts = []
        for cx, cy, a0 in ((L - r, W - r, 0), (-L + r, W - r, 90), (-L + r, -W + r, 180), (L - r, -W + r, 270)):
            for i in range(n + 1):
                a = np.deg2rad(a0 + 90 * i / n)
                pts.append((cx + r * np.cos(a), cy + r * np.sin(a)))
        return pts

    def to_dict(self) -> dict:
        return {
            "id": self.id, "length": self.length, "width": self.width, "corner_radius": self.corner_radius,
            "blue_line_x": self.blue_line_x, "goal_line_x": self.goal_line_x,
            "landmarks": [vars(l) for l in self.landmarks],
            "lines": self.lines(), "outline": self.outline(),
        }


TEMPLATES = {
    # IIHF rink (Swedish standard): 60 × 30 m, corner radius 8.5 m. Goal line 4 m from the end
    # boards (±26 m), blue line 22.86 m from the end boards (±7.14 m), end-zone faceoff dots 6 m
    # in front of the goal line (±20 m) and 7 m either side of the long axis, neutral-zone dots
    # 1.5 m from the blue line (±5.64 m).
    "iihf": RinkTemplate("iihf", 60.0, 30.0, 8.5, blue_line_x=7.14, goal_line_x=26.0,
                         dot_x=20.0, dot_y=7.0, neutral_dot_x=5.64),
    # NHL rink: 200 × 85 ft = 60.96 × 25.91 m, goal line 11 ft from boards, blue line 75 ft from boards.
    "nhl": RinkTemplate("nhl", 60.96, 25.91, 8.53, blue_line_x=60.96 / 2 - 22.86, goal_line_x=60.96 / 2 - 3.35,
                        dot_x=60.96 / 2 - 3.35 - 6.1, dot_y=6.71, neutral_dot_x=60.96 / 2 - 22.86 - 1.52),
}


@dataclass
class Calibration:
    H: np.ndarray                      # 3×3 image → rink (metres)
    pairs: list[dict]                  # [{"id": landmark, "px": x, "py": y}]
    rms_error_m: float
    template: str = "iihf"

    def image_to_rink(self, px: float, py: float) -> tuple[float, float]:
        v = self.H @ np.array([px, py, 1.0])
        return float(v[0] / v[2]), float(v[1] / v[2])

    def rink_to_image(self, x: float, y: float) -> tuple[float, float]:
        v = np.linalg.inv(self.H) @ np.array([x, y, 1.0])
        return float(v[0] / v[2]), float(v[1] / v[2])

    def to_dict(self) -> dict:
        return {"H": self.H.tolist(), "pairs": self.pairs, "rms_error_m": self.rms_error_m, "template": self.template}

    @classmethod
    def from_dict(cls, d: dict) -> "Calibration":
        return cls(np.array(d["H"], dtype=np.float64), d.get("pairs", []), float(d.get("rms_error_m", 0)),
                   d.get("template", "iihf"))


def calibrate(template: RinkTemplate, pairs: list[dict]) -> Calibration:
    """Fit image→rink homography from tapped landmark pairs. Needs ≥4 non-collinear landmarks."""
    by_id = {l.id: l for l in template.landmarks}
    src, dst = [], []
    for p in pairs:
        lm = by_id.get(p["id"])
        if lm is None:
            raise ValueError(f"unknown landmark {p['id']}")
        src.append((p["px"], p["py"]))
        dst.append((lm.x, lm.y))
    if len(src) < 4:
        raise ValueError("need at least 4 landmarks")
    src_a, dst_a = np.array(src, np.float64), np.array(dst, np.float64)
    method = cv2.RANSAC if len(src) >= 6 else 0
    H, _ = cv2.findHomography(src_a, dst_a, method, 1.0)
    if H is None:
        raise ValueError("landmarks are degenerate (collinear or too close)")
    proj = cv2.perspectiveTransform(src_a.reshape(-1, 1, 2), H).reshape(-1, 2)
    rms = float(np.sqrt(np.mean(np.sum((proj - dst_a) ** 2, axis=1))))
    return Calibration(H, pairs, rms, template.id)
