"""Tiny synthetic rink dataset (YOLO format) to smoke-test the pipeline without downloads.
Draws an ice-coloured rink with coloured 'players', a black 'puck', goals and faceoff dots."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from classes import ID


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ml/synthetic")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--size", type=int, default=640)
    a = ap.parse_args()
    out = Path(a.out); (out / "images").mkdir(parents=True, exist_ok=True); (out / "labels").mkdir(exist_ok=True)
    rng = random.Random(1)
    S = a.size
    for i in range(a.n):
        img = np.full((S, S, 3), (235, 240, 245), np.uint8)
        cv2.line(img, (S // 2, 0), (S // 2, S), (60, 60, 220), 3)
        labels = []
        def add(cls, x1, y1, x2, y2):
            labels.append(f"{cls} {(x1 + x2) / 2 / S:.5f} {(y1 + y2) / 2 / S:.5f} {(x2 - x1) / S:.5f} {(y2 - y1) / S:.5f}")
        for _ in range(rng.randint(4, 9)):
            w, h = rng.randint(18, 34), rng.randint(40, 80)
            x, y = rng.randint(0, S - w), rng.randint(0, S - h)
            col = rng.choice([(200, 60, 40), (40, 40, 210)])
            cv2.rectangle(img, (x, y), (x + w, y + h), col, -1)
            cv2.circle(img, (x + w // 2, y + 8), 7, (40, 40, 40), -1)
            add(ID["player"], x, y, x + w, y + h)
        if rng.random() < 0.7:
            w, h = rng.randint(28, 40), rng.randint(50, 80); x, y = rng.randint(0, S - w), rng.randint(0, S - h)
            cv2.rectangle(img, (x, y), (x + w, y + h), (30, 30, 30), -1); cv2.rectangle(img, (x + 4, y + 4), (x + w - 4, y + h - 4), (230, 230, 230), 2)
            add(ID["referee"], x, y, x + w, y + h)
        if rng.random() < 0.7:
            w, h = rng.randint(34, 46), rng.randint(50, 76); x, y = rng.randint(0, S - w), rng.randint(0, S - h)
            cv2.rectangle(img, (x, y), (x + w, y + h), (60, 200, 220), -1)
            add(ID["goaltender"], x, y, x + w, y + h)
        px, py = rng.randint(4, S - 4), rng.randint(4, S - 4)
        cv2.circle(img, (px, py), 4, (10, 10, 10), -1); add(ID["puck"], px - 5, py - 5, px + 5, py + 5)
        for _ in range(2):
            x, y = rng.randint(0, S - 50), rng.randint(0, S - 30)
            cv2.rectangle(img, (x, y), (x + 50, y + 30), (40, 40, 200), 3); add(ID["goal"], x, y, x + 50, y + 30)
        for _ in range(3):
            x, y = rng.randint(10, S - 10), rng.randint(10, S - 10)
            cv2.circle(img, (x, y), 6, (40, 40, 200), -1); add(ID["faceoff_dot"], x - 7, y - 7, x + 7, y + 7)
        cv2.imwrite(str(out / "images" / f"syn_{i:04d}.jpg"), img)
        (out / "labels" / f"syn_{i:04d}.txt").write_text("\n".join(labels))
    print("wrote", a.n, "synthetic frames to", out)


if __name__ == "__main__":
    main()
