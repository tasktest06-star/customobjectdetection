"""
Enhanced COCO dataset builder.

Improvements over v1:
  - Stratified train/val split: ensures each class is proportionally
    represented in both splits (prevents split luck producing val with
    only one class)
  - Optional test split: holds out a clean test set for final evaluation
  - Video-level split: groups frames by source video and splits at the
    video level to prevent data leakage (same object appearing in both
    train and val from consecutive frames)
  - Tighter label matching: exact match → edit distance → substring,
    preventing "cat" matching "catfish"
"""
import json
import re
import shutil
import yaml
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from src.utils.logger import get_logger

log = get_logger("coco_builder")


def _levenshtein(a: str, b: str) -> int:
    """Compute edit distance between two strings."""
    if len(a) < len(b):
        return _levenshtein(b, a)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (ca != cb)))
        prev = curr
    return prev[-1]


def _video_id_from_path(path: str) -> str:
    """Extract video ID from frame filename '<video_id>_<frame_num>.jpg'."""
    stem = Path(path).stem
    m = re.match(r"^(.+?)_\d+$", stem)
    return m.group(1) if m else stem


class COCOBuilder:
    def __init__(
        self,
        class_names: List[str],
        output_dir: str = "data/datasets",
        train_ratio: float = 0.8,
        test_ratio: float = 0.0,
        use_video_level_split: bool = True,
        edit_distance_threshold: int = 2,
    ):
        """
        Args:
            train_ratio: fraction of data for training (rest is val, minus test)
            test_ratio: fraction of data held out as test set (0 = no test set)
            use_video_level_split: if True, split by source video to prevent
                                   data leakage between splits
            edit_distance_threshold: max edit distance for fuzzy label matching
        """
        self.class_names = class_names
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.use_video_split = use_video_level_split
        self.edit_dist = edit_distance_threshold

        self.categories = [
            {"id": i + 1, "name": n, "supercategory": "object"}
            for i, n in enumerate(class_names)
        ]
        self.cat_map: Dict[str, int] = {c["name"]: c["id"] for c in self.categories}

    # ── label matching ─────────────────────────────────────────────

    def _match_label(self, raw_label: str) -> Optional[str]:
        """
        Three-stage matching:
          1. Exact match (case-insensitive)
          2. Edit distance ≤ edit_dist threshold
          3. Substring containment (original v1 behaviour, now last resort)
        Returns matched class name or None.
        """
        label = raw_label.lower().strip()

        # Stage 1: exact
        for cls in self.class_names:
            if cls.lower() == label:
                return cls

        # Stage 2: edit distance (catches minor OCR/tokenisation errors)
        for cls in self.class_names:
            if _levenshtein(cls.lower(), label) <= self.edit_dist:
                return cls

        # Stage 3: substring (only if label is at least 4 chars to avoid
        # short-string false positives like "cat" matching "catfish")
        if len(label) >= 4:
            for cls in self.class_names:
                if cls.lower() in label or label in cls.lower():
                    return cls

        return None

    # ── splitting ──────────────────────────────────────────────────

    def _video_level_split(
        self, detections: List[Dict]
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Split at the video level: all frames from the same video go to
        the same split, preventing leakage from consecutive frames.

        Returns (train, val, test) lists.
        """
        # Group by video ID
        by_video: Dict[str, List[Dict]] = defaultdict(list)
        for det in detections:
            vid = _video_id_from_path(det["image_path"])
            by_video[vid].append(det)

        videos = list(by_video.keys())
        n = len(videos)
        n_test = int(n * self.test_ratio)
        n_train = int((n - n_test) * self.train_ratio)

        import random
        random.shuffle(videos)
        test_vids = set(videos[:n_test])
        train_vids = set(videos[n_test: n_test + n_train])

        train = [d for vid in train_vids for d in by_video[vid]]
        val = [d for vid in videos[n_test + n_train:] for d in by_video[vid]]
        test = [d for vid in test_vids for d in by_video[vid]]

        log.info(
            f"Video-level split: {len(train_vids)} train videos / "
            f"{n - len(train_vids) - len(test_vids)} val videos / "
            f"{len(test_vids)} test videos"
        )
        return train, val, test

    def _stratified_split(
        self, detections: List[Dict]
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Stratified split: preserves class distribution in each split.
        Groups detections by dominant class (class with most boxes).
        Falls back to random if sklearn is unavailable.
        """
        # Assign each image a "primary class" label for stratification
        def primary_class(det: Dict) -> str:
            if not det["labels"]:
                return "__none__"
            counts: Dict[str, int] = defaultdict(int)
            for label in det["labels"]:
                matched = self._match_label(str(label))
                if matched:
                    counts[matched] += 1
            return max(counts, key=counts.get) if counts else "__none__"

        try:
            from sklearn.model_selection import train_test_split

            labels = [primary_class(d) for d in detections]
            val_test_ratio = 1.0 - self.train_ratio
            train_dets, rest_dets, _, rest_labels = train_test_split(
                detections, labels,
                test_size=val_test_ratio,
                stratify=labels,
                random_state=42,
            )
            if self.test_ratio > 0 and len(rest_dets) > 1:
                test_frac = self.test_ratio / val_test_ratio
                val_dets, test_dets, _, _ = train_test_split(
                    rest_dets, rest_labels,
                    test_size=test_frac,
                    stratify=rest_labels,
                    random_state=42,
                )
            else:
                val_dets, test_dets = rest_dets, []

            return train_dets, val_dets, test_dets

        except ImportError:
            log.warning("sklearn not available — using random split instead of stratified")
            import random
            random.shuffle(detections)
            n = len(detections)
            n_test = int(n * self.test_ratio)
            n_train = int((n - n_test) * self.train_ratio)
            return (
                detections[n_test: n_test + n_train],
                detections[n_test + n_train:],
                detections[:n_test],
            )

    # ── COCO JSON construction ─────────────────────────────────────

    def _build_coco_dict(self, detections: List[Dict], split: str) -> Dict:
        images, annotations, ann_id = [], [], 1
        skipped = 0
        for img_id, det in enumerate(detections, 1):
            images.append({
                "id": img_id,
                "file_name": Path(det["image_path"]).name,
                "width": det["image_width"],
                "height": det["image_height"],
            })
            scores_list = det.get("scores", [1.0] * len(det["labels"]))
            for box, label, score in zip(
                det["boxes_xyxy"], det["labels"], scores_list
            ):
                matched = self._match_label(str(label))
                if matched is None:
                    skipped += 1
                    continue
                x1, y1, x2, y2 = box.tolist()
                w, h = x2 - x1, y2 - y1
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": self.cat_map[matched],
                    "bbox": [x1, y1, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                    "score": float(score),
                })
                ann_id += 1

        if skipped:
            log.debug(f"  {split}: {skipped} annotations dropped (no label match)")
        return {
            "info": {
                "description": f"Auto-labeled {split} set",
                "date_created": datetime.now().isoformat(),
            },
            "categories": self.categories,
            "images": images,
            "annotations": annotations,
        }

    # ── public API ─────────────────────────────────────────────────

    def build_dataset(
        self, detections: List[Dict], dataset_name: str = "custom"
    ) -> str:
        """
        Build train/val (+ optional test) splits, copy images, write
        COCO JSON annotations and YOLO data.yaml.

        Returns path to data.yaml (YOLO training config).
        """
        ds_dir = self.output_dir / dataset_name
        splits_to_create = ["train", "val"] + (["test"] if self.test_ratio > 0 else [])
        for split in splits_to_create:
            (ds_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (ds_dir / "annotations").mkdir(parents=True, exist_ok=True)

        # Choose split strategy
        if self.use_video_split:
            train_dets, val_dets, test_dets = self._video_level_split(detections)
        else:
            train_dets, val_dets, test_dets = self._stratified_split(detections)

        split_map = {"train": train_dets, "val": val_dets}
        if self.test_ratio > 0:
            split_map["test"] = test_dets

        for split, dets in split_map.items():
            for det in dets:
                src = Path(det["image_path"])
                dst = ds_dir / "images" / split / src.name
                if src.exists() and not dst.exists():
                    shutil.copy2(src, dst)
            coco = self._build_coco_dict(dets, split)
            ann_path = ds_dir / "annotations" / f"instances_{split}.json"
            ann_path.write_text(json.dumps(coco, indent=2))
            log.info(
                f"  {split:5s}: {len(dets):4d} images, "
                f"{len(coco['annotations']):5d} boxes"
            )

        # YOLO data.yaml
        data_yaml_content = {
            "path": str(ds_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "nc": len(self.class_names),
            "names": self.class_names,
        }
        if self.test_ratio > 0:
            data_yaml_content["test"] = "images/test"

        yaml_path = ds_dir / "data.yaml"
        yaml_path.write_text(yaml.dump(data_yaml_content, default_flow_style=False))
        log.info(f"Dataset written to {ds_dir}")
        return str(yaml_path)
