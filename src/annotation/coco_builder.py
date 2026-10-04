"""
Build COCO JSON annotations and YOLO data.yaml from pseudo-label detections.

Output layout:
  data/datasets/<name>/
    images/train/    images/val/
    annotations/instances_train.json  instances_val.json
    data.yaml
"""
import json
import shutil
import random
import yaml
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime


class COCOBuilder:
    def __init__(
        self,
        class_names: List[str],
        output_dir: str = "data/datasets",
        train_ratio: float = 0.8,
    ):
        self.class_names = class_names
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.categories = [
            {"id": i + 1, "name": n, "supercategory": "object"}
            for i, n in enumerate(class_names)
        ]
        self.cat_map: Dict[str, int] = {c["name"]: c["id"] for c in self.categories}

    def _match_label(self, label: str) -> Optional[str]:
        """Fuzzy-match a raw Grounding DINO/OWLv2 label to a known class name."""
        label = label.lower().strip()
        for cls in self.class_names:
            if cls.lower() in label or label in cls.lower():
                return cls
        return None

    def _build_coco_dict(self, detections: List[Dict], split: str) -> Dict:
        images, annotations, ann_id = [], [], 1
        for img_id, det in enumerate(detections, 1):
            images.append(
                {
                    "id": img_id,
                    "file_name": Path(det["image_path"]).name,
                    "width": det["image_width"],
                    "height": det["image_height"],
                }
            )
            scores_list = det.get("scores", [1.0] * len(det["labels"]))
            for box, label, score in zip(
                det["boxes_xyxy"], det["labels"], scores_list
            ):
                matched = self._match_label(str(label))
                if matched is None:
                    continue
                x1, y1, x2, y2 = box.tolist()
                w, h = x2 - x1, y2 - y1
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": self.cat_map[matched],
                        "bbox": [x1, y1, w, h],
                        "area": w * h,
                        "iscrowd": 0,
                        "score": float(score),
                    }
                )
                ann_id += 1
        return {
            "info": {
                "description": f"Auto-labeled {split} set",
                "date_created": datetime.now().isoformat(),
            },
            "categories": self.categories,
            "images": images,
            "annotations": annotations,
        }

    def build_dataset(
        self, detections: List[Dict], dataset_name: str = "custom"
    ) -> str:
        """
        Build train/val split, copy images, write COCO JSON + YOLO data.yaml.
        Returns path to data.yaml.
        """
        ds_dir = self.output_dir / dataset_name
        for split in ("train", "val"):
            (ds_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (ds_dir / "annotations").mkdir(parents=True, exist_ok=True)

        random.shuffle(detections)
        n_train = int(len(detections) * self.train_ratio)
        splits = {"train": detections[:n_train], "val": detections[n_train:]}

        for split, dets in splits.items():
            for det in dets:
                src = Path(det["image_path"])
                dst = ds_dir / "images" / split / src.name
                if src.exists() and not dst.exists():
                    shutil.copy2(src, dst)
            coco = self._build_coco_dict(dets, split)
            ann_path = ds_dir / "annotations" / f"instances_{split}.json"
            ann_path.write_text(json.dumps(coco, indent=2))
            print(
                f"  {split}: {len(dets)} images, {len(coco['annotations'])} boxes"
            )

        data_yaml_content = {
            "path": str(ds_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "nc": len(self.class_names),
            "names": self.class_names,
        }
        yaml_path = ds_dir / "data.yaml"
        yaml_path.write_text(yaml.dump(data_yaml_content, default_flow_style=False))
        print(f"  Dataset written to {ds_dir}")
        return str(yaml_path)
