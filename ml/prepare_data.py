"""Download HockeyAI (MIT) + optional extra YOLO-format folders → COCO json with train/val split.

    python ml/prepare_data.py --out ml/data [--extra path/to/yolo_folder ...] [--skip-download]

Extra folders: images/*.jpg + labels/*.txt (YOLO: cls cx cy w h, normalized) using our class ids.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classes import CLASSES, HOCKEYAI_MAP, ID  # noqa: E402

HF_REPO = "SimulaMet-HOST/HockeyAI"
ZIP_NAME = "HockeyAI_Dataset.zip"


def download_hockeyai(out: Path) -> Path:
    from huggingface_hub import hf_hub_download

    out.mkdir(parents=True, exist_ok=True)
    print("downloading HockeyAI (about 400 MB)…")
    p = hf_hub_download(HF_REPO, ZIP_NAME, repo_type="dataset", local_dir=out)
    return Path(p)


def unzip(zip_path: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    return dest


def read_names(root: Path) -> dict[int, str]:
    """Class id → name from a data.yaml/classes.txt in the dataset, else HockeyAI's documented order."""
    for y in list(root.rglob("*.yaml")) + list(root.rglob("*.yml")):
        txt = y.read_text(errors="ignore")
        if "names" in txt:
            try:
                import yaml  # type: ignore
                d = yaml.safe_load(txt)
                names = d.get("names")
                if isinstance(names, dict):
                    return {int(k): str(v) for k, v in names.items()}
                if isinstance(names, list):
                    return {i: str(n) for i, n in enumerate(names)}
            except Exception:
                pass
    for t in root.rglob("classes.txt"):
        return {i: n.strip() for i, n in enumerate(t.read_text().splitlines()) if n.strip()}
    # documented order on the HockeyAI page (alphabetical); verify against data.yaml when present
    return {0: "Center Ice", 1: "Faceoff Dots", 2: "Goal Frame", 3: "Goaltender", 4: "Players", 5: "Puck", 6: "Referee"}


def iter_yolo_folder(root: Path, names: dict[int, str] | None):
    """Yield (image_path, [(our_cls, cx, cy, w, h)]) for every image with a label file."""
    imgs = {p.stem: p for p in root.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
    for lbl in root.rglob("*.txt"):
        if lbl.stem not in imgs or lbl.name == "classes.txt":
            continue
        boxes = []
        for line in lbl.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            raw = int(float(parts[0]))
            if names is None:
                cls = raw
            else:
                nm = names.get(raw, "").lower()
                ours = HOCKEYAI_MAP.get(nm)
                if ours is None:
                    continue
                cls = ID[ours]
            boxes.append((cls, *map(float, parts[1:5])))
        yield imgs[lbl.stem], boxes


def to_coco(items, out_dir: Path, split: str, start_img_id: int = 1, start_ann_id: int = 1):
    from PIL import Image

    images, anns = [], []
    img_id, ann_id = start_img_id, start_ann_id
    img_dir = out_dir / split
    img_dir.mkdir(parents=True, exist_ok=True)
    for src, boxes in items:
        with Image.open(src) as im:
            w, h = im.size
        dst = img_dir / f"{img_id:06d}{src.suffix.lower()}"
        if not dst.exists():
            dst.write_bytes(src.read_bytes())
        images.append({"id": img_id, "file_name": dst.name, "width": w, "height": h})
        for cls, cx, cy, bw, bh in boxes:
            x, y, ww, hh = (cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h
            anns.append({"id": ann_id, "image_id": img_id, "category_id": cls, "bbox": [x, y, ww, hh],
                         "area": ww * hh, "iscrowd": 0})
            ann_id += 1
        img_id += 1
    coco = {"images": images, "annotations": anns,
            "categories": [{"id": i, "name": c} for i, c in enumerate(CLASSES)]}
    (out_dir / f"{split}.json").write_text(json.dumps(coco))
    return img_id, ann_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ml/data")
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    out = Path(a.out)
    items = []
    if not a.skip_download:
        zp = download_hockeyai(out / "raw")
        root = unzip(zp, out / "raw" / "hockeyai")
        names = read_names(root)
        print("HockeyAI class names:", names)
        items += list(iter_yolo_folder(root, names))
        print(f"HockeyAI: {len(items)} labelled frames")
    for ex in a.extra:
        n0 = len(items)
        items += list(iter_yolo_folder(Path(ex), None))
        print(f"{ex}: {len(items) - n0} labelled frames")
    if not items:
        sys.exit("no data found")
    random.Random(a.seed).shuffle(items)
    n_val = max(1, int(len(items) * a.val))
    val, train = items[:n_val], items[n_val:]
    img_id, ann_id = to_coco(train, out, "train")
    to_coco(val, out, "val", img_id, ann_id)
    counts = {c: 0 for c in CLASSES}
    for _, boxes in items:
        for b in boxes:
            counts[CLASSES[b[0]]] += 1
    print(f"train {len(train)} / val {len(val)} images; boxes per class: {counts}")
    print(f"wrote {out}/train.json and {out}/val.json")


if __name__ == "__main__":
    main()
