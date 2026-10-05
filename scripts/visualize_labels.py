#!/usr/bin/env python3
"""
Draw pseudo-labels on frames and save annotated images for visual inspection.

Usage:
  python scripts/visualize_labels.py \
    --dataset data/datasets/my_run \
    --split train \
    --output visualizations/ \
    --max-images 20
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

# Distinct colors for up to 20 classes
_PALETTE = [
    (220, 20, 60), (0, 128, 255), (0, 200, 80), (255, 165, 0), (148, 0, 211),
    (255, 20, 147), (0, 206, 209), (139, 69, 19), (50, 205, 50), (255, 215, 0),
    (0, 191, 255), (255, 69, 0), (154, 205, 50), (135, 206, 235), (219, 112, 147),
    (160, 82, 45), (32, 178, 170), (255, 127, 80), (100, 149, 237), (0, 255, 127),
]


def draw_annotations(image_path: str, annotations: list, categories: dict) -> np.ndarray:
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    for ann in annotations:
        x, y, w, h = [int(v) for v in ann["bbox"]]
        cat_id = ann["category_id"]
        cat_name = categories.get(cat_id, str(cat_id))
        score = ann.get("score", 1.0)
        color = _PALETTE[(cat_id - 1) % len(_PALETTE)]
        # BGR for OpenCV
        color_bgr = (color[2], color[1], color[0])
        cv2.rectangle(img, (x, y), (x + w, y + h), color_bgr, 2)
        label = f"{cat_name} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x, y - th - 4), (x + tw + 2, y), color_bgr, -1)
        cv2.putText(img, label, (x + 1, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Dataset root (contains images/ and annotations/)")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--output", default="visualizations", help="Output directory")
    parser.add_argument("--max-images", type=int, default=20)
    args = parser.parse_args()

    ds_root = Path(args.dataset)
    ann_file = ds_root / "annotations" / f"instances_{args.split}.json"
    images_dir = ds_root / "images" / args.split
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(ann_file) as f:
        coco = json.load(f)

    categories = {c["id"]: c["name"] for c in coco["categories"]}
    anns_by_img = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    images = coco["images"]
    random.shuffle(images)
    images = images[: args.max_images]

    saved = 0
    for img_info in images:
        img_path = images_dir / img_info["file_name"]
        if not img_path.exists():
            continue
        anns = anns_by_img.get(img_info["id"], [])
        if not anns:
            continue
        drawn = draw_annotations(img_path, anns, categories)
        if drawn is None:
            continue
        out_path = out_dir / img_path.name
        cv2.imwrite(str(out_path), drawn)
        saved += 1

    print(f"Saved {saved} annotated images to {out_dir}/")


if __name__ == "__main__":
    main()
