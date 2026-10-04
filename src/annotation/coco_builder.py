"""
Build COCO JSON annotations and data.yaml from pseudo-label detections.

COCO annotation format overview
────────────────────────────────
COCO JSON has four top-level keys:
  "info"        : metadata (description, date)
  "categories"  : list of {id, name, supercategory} — 1-indexed, id starts at 1
  "images"      : list of {id, file_name, width, height} — id starts at 1
  "annotations" : list of {id, image_id, category_id, bbox, area, iscrowd}
                  bbox format: [x_min, y_min, width, height] in pixels
                  category_id: 1-indexed (matches categories[*].id)

Output layout
─────────────
  data/datasets/<name>/
    images/train/        ← copied frame JPEGs for training split
    images/val/          ← copied frame JPEGs for validation split
    annotations/
      instances_train.json  ← COCO JSON for training split
      instances_val.json    ← COCO JSON for validation split
    data.yaml            ← dataset metadata for RTDETRTrainer._parse_data_yaml()

The data.yaml format mirrors YOLOv8's dataset format for compatibility:
  path:  /abs/path/to/dataset
  train: images/train
  val:   images/val
  nc:    <number of classes>
  names: [class1, class2, ...]
"""
import json
import shutil
import random
import yaml
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime


class COCOBuilder:
    """
    Converts a list of pseudo-label detection dicts into a COCO JSON dataset.

    Responsibilities:
      1. Split detections into train and val sets (random shuffle, 80/20 default)
      2. Copy image files to the dataset directory (images/train, images/val)
      3. Write COCO JSON annotation files for each split
      4. Write data.yaml with absolute paths for the trainer

    The same class instance can be called multiple times with different
    detections and dataset_names (once per self-training round) since all
    output is written under output_dir/dataset_name/.
    """

    def __init__(
        self,
        class_names: List[str],
        output_dir: str = "data/datasets",
        train_ratio: float = 0.8,
    ):
        """
        Args:
            class_names:  Ordered list of class name strings.
                          The order determines the COCO category_id assignment:
                          class_names[0] → category_id=1, [1] → 2, etc.
                          Must match the order used in pipeline_config.yaml classes[].
            output_dir:   Root directory for all dataset outputs.
            train_ratio:  Fraction of images for training (remainder goes to val).
                          0.8 = 80% train, 20% val.
        """
        self.class_names = class_names
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio

        # Build COCO category list — COCO IDs are 1-indexed by convention
        # supercategory: "object" is the standard catch-all for custom datasets
        self.categories = [
            {"id": i + 1, "name": n, "supercategory": "object"}
            for i, n in enumerate(class_names)
        ]
        # Lookup: class_name → category_id for fast annotation writing
        self.cat_map: Dict[str, int] = {c["name"]: c["id"] for c in self.categories}

    def _match_label(self, label: str) -> Optional[str]:
        """
        Fuzzy-match a raw GroundingDINO/OWLv2 label to a known class name.

        GroundingDINO can return decoded token strings like "raccoon." or "racoon"
        instead of the exact class name. This function normalizes the label and
        checks if any known class name is a substring of (or contains) the label.

        Returns None if the label doesn't match any known class, in which case
        the corresponding box is silently dropped from the annotation file.
        """
        label = label.lower().strip()
        for cls in self.class_names:
            if cls.lower() in label or label in cls.lower():
                return cls
        return None  # unrecognized label — will be skipped in _build_coco_dict

    def _build_coco_dict(self, detections: List[Dict], split: str) -> Dict:
        """
        Convert a list of detection dicts into a COCO JSON-compatible dict.

        Each detection dict corresponds to one image. The function assigns
        sequential IDs starting from 1 to both images and annotations.

        Box format conversion:
          Input:  boxes_xyxy = [x1, y1, x2, y2]  (xyxy pixel coords)
          Output: bbox       = [x1, y1, w, h]     (COCO top-left + dimensions)
          w = x2 - x1,  h = y2 - y1

        Boxes with unrecognized labels (not matched by _match_label) are dropped.
        """
        images, annotations, ann_id = [], [], 1

        for img_id, det in enumerate(detections, 1):  # 1-indexed image IDs
            images.append(
                {
                    "id":        img_id,
                    "file_name": Path(det["image_path"]).name,
                    "width":     det["image_width"],
                    "height":    det["image_height"],
                }
            )
            # Gracefully handle detections without a scores field (e.g. from
            # manually curated data that only has boxes and labels)
            scores_list = det.get("scores", [1.0] * len(det["labels"]))

            for box, label, score in zip(
                det["boxes_xyxy"], det["labels"], scores_list
            ):
                matched = self._match_label(str(label))
                if matched is None:
                    continue  # skip boxes with unrecognized labels

                # Convert xyxy → COCO [x, y, w, h]
                x1, y1, x2, y2 = box.tolist()
                w, h = x2 - x1, y2 - y1

                annotations.append(
                    {
                        "id":          ann_id,
                        "image_id":    img_id,
                        "category_id": self.cat_map[matched],  # 1-indexed
                        "bbox":        [x1, y1, w, h],
                        "area":        w * h,
                        "iscrowd":     0,  # 0 = individual object, 1 = crowd region
                        "score":       float(score),  # non-standard but harmless extra field
                    }
                )
                ann_id += 1

        return {
            "info": {
                "description": f"Auto-labeled {split} set",
                "date_created": datetime.now().isoformat(),
            },
            "categories":  self.categories,
            "images":      images,
            "annotations": annotations,
        }

    def build_dataset(
        self, detections: List[Dict], dataset_name: str = "custom"
    ) -> str:
        """
        Build a complete COCO dataset directory from detection dicts.

        Steps:
          1. Shuffle and split detections into train/val.
          2. Create directory structure under output_dir/dataset_name/.
          3. Copy image files from their source locations.
          4. Write COCO JSON annotation files.
          5. Write data.yaml with absolute paths.

        Args:
            detections:   Filtered detection dicts (from LabelQualityFilter).
                          Each dict must have "image_path", "boxes_xyxy",
                          "labels", "scores", "image_width", "image_height".
            dataset_name: Subdirectory name under output_dir for this dataset.

        Returns:
            Absolute path to data.yaml (str).

        Note: detections is copied before shuffling so the caller's list order
        is preserved. This is important when the same detections list is passed
        to both build_dataset() and the teacher for ensemble labeling.
        """
        ds_dir = self.output_dir / dataset_name

        # Create the full directory tree before writing any files
        for split in ("train", "val"):
            (ds_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (ds_dir / "annotations").mkdir(parents=True, exist_ok=True)

        # Copy the list to avoid mutating the caller's detections order.
        # random.shuffle() modifies in-place, so we need our own copy.
        detections = list(detections)
        random.shuffle(detections)

        n_train = int(len(detections) * self.train_ratio)
        splits = {
            "train": detections[:n_train],
            "val":   detections[n_train:],
        }

        for split, dets in splits.items():
            # Copy images into the split directory
            for det in dets:
                src = Path(det["image_path"])
                dst = ds_dir / "images" / split / src.name
                # Skip if already copied (idempotent — safe to re-run build_dataset)
                if src.exists() and not dst.exists():
                    shutil.copy2(src, dst)

            # Build and write the COCO JSON annotation file
            coco = self._build_coco_dict(dets, split)
            ann_path = ds_dir / "annotations" / f"instances_{split}.json"
            ann_path.write_text(json.dumps(coco, indent=2))
            print(
                f"  {split}: {len(dets)} images, {len(coco['annotations'])} boxes"
            )

        # Write data.yaml — RTDETRTrainer._parse_data_yaml() reads this file
        # to discover training data paths, class count, and class names
        data_yaml_content = {
            "path":  str(ds_dir.resolve()),   # absolute path for portability
            "train": "images/train",
            "val":   "images/val",
            "nc":    len(self.class_names),
            "names": self.class_names,
        }
        yaml_path = ds_dir / "data.yaml"
        yaml_path.write_text(yaml.dump(data_yaml_content, default_flow_style=False))
        print(f"  Dataset written to {ds_dir}")
        return str(yaml_path)
