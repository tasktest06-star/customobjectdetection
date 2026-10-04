#!/usr/bin/env python3
"""
Draw pseudo-labels on sample images for visual inspection.

Usage:
  python scripts/visualize_labels.py \
    --images data/datasets/auto_det/images/train \
    --annotations data/datasets/auto_det/annotations/instances_train.json \
    --output data/visualizations \
    --n 20
"""
import argparse
import json
import random
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

_COLORS = [
    (255, 60, 60),
    (60, 220, 60),
    (60, 60, 255),
    (255, 220, 0),
    (0, 220, 220),
    (220, 0, 220),
]


def draw_annotations(images_dir: str, ann_file: str, output_dir: str, n: int):
    with open(ann_file) as f:
        coco = json.load(f)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cat_names = {c["id"]: c["name"] for c in coco["categories"]}
    cat_colors = {
        name: _COLORS[i % len(_COLORS)]
        for i, name in enumerate(cat_names.values())
    }

    # Group annotations by image id
    img_anns: dict = {}
    for ann in coco["annotations"]:
        img_anns.setdefault(ann["image_id"], []).append(ann)

    samples = random.sample(coco["images"], min(n, len(coco["images"])))
    saved = 0
    for info in samples:
        src = Path(images_dir) / info["file_name"]
        img = cv2.imread(str(src))
        if img is None:
            continue

        for ann in img_anns.get(info["id"], []):
            name = cat_names[ann["category_id"]]
            color = cat_colors[name]
            x, y, w, h = map(int, ann["bbox"])
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            label = f"{name} {ann.get('score', 1.0):.2f}"
            cv2.putText(
                img, label, (x, max(y - 5, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
            )

        cv2.imwrite(str(out / info["file_name"]), img)
        saved += 1

    print(f"Saved {saved} visualizations to {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, help="Directory of training images")
    parser.add_argument("--annotations", required=True, help="COCO JSON annotation file")
    parser.add_argument("--output", default="data/visualizations")
    parser.add_argument("--n", type=int, default=20, help="Number of sample images")
    args = parser.parse_args()

    draw_annotations(args.images, args.annotations, args.output, args.n)


if __name__ == "__main__":
    main()
