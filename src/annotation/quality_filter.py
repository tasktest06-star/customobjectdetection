"""
Enhanced label quality filter.

Improvements over v1:
  - Vectorized NMS: uses torchvision.ops.batched_nms (10-50× faster on GPU)
    with a pure-Python fallback when torchvision is unavailable
  - Aspect ratio filter: removes extremely thin/wide boxes that are almost
    always false positives from the open-vocabulary model
  - Class imbalance checker: standalone utility that warns when one class
    dominates the dataset (can silently break training)
  - Temporal consistency filter: cross-validates detections across nearby
    frames from the same video — a box that appears in only one frame out
    of a sequence is likely a false positive
"""
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.utils.logger import get_logger

log = get_logger("quality_filter")

# ── NMS (vectorized where possible) ───────────────────────────────

try:
    import torch
    import torchvision.ops as tv_ops
    _HAS_TORCHVISION = True
except ImportError:
    _HAS_TORCHVISION = False


def compute_iou(b1: np.ndarray, b2: np.ndarray) -> float:
    """IoU between two xyxy boxes."""
    xi1, yi1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    xi2, yi2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.5,
) -> np.ndarray:
    """
    NMS with torchvision fast path (GPU-vectorized) and Python fallback.
    Returns kept indices sorted by descending score.
    """
    if len(boxes) == 0:
        return np.array([], dtype=int)

    if _HAS_TORCHVISION:
        t_boxes = torch.from_numpy(boxes.astype(np.float32))
        t_scores = torch.from_numpy(scores.astype(np.float32))
        keep = tv_ops.nms(t_boxes, t_scores, iou_threshold)
        return keep.numpy()

    # Pure-Python fallback (O(N²))
    order = scores.argsort()[::-1]
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        ious = np.array([compute_iou(boxes[i], boxes[j]) for j in order[1:]])
        order = order[1:][ious <= iou_threshold]
    return np.array(keep, dtype=int)


def batched_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: List[str],
    iou_threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Per-class NMS using torchvision batched_nms when available.
    Returns (boxes, scores, labels) after suppression.
    """
    if len(boxes) == 0:
        return boxes, scores, labels

    if _HAS_TORCHVISION:
        unique = sorted(set(labels))
        label_ids = torch.tensor([unique.index(l) for l in labels])
        t_boxes = torch.from_numpy(boxes.astype(np.float32))
        t_scores = torch.from_numpy(scores.astype(np.float32))
        keep = tv_ops.batched_nms(t_boxes, t_scores, label_ids, iou_threshold)
        keep = keep.numpy()
    else:
        # Fallback: per-class loop
        keep_all = []
        for cls in set(labels):
            m = np.array([l == cls for l in labels])
            idx = np.where(m)[0]
            k = nms(boxes[m], scores[m], iou_threshold)
            keep_all.extend(idx[k].tolist())
        keep = np.array(sorted(keep_all), dtype=int)

    return boxes[keep], scores[keep], [labels[i] for i in keep]


# ── main filter class ──────────────────────────────────────────────


class LabelQualityFilter:
    """
    Three-stage quality filter + aspect ratio check.

    Stages (applied in order):
      1. Confidence score threshold
      2. Bounding-box size / area ratio + aspect ratio check
      3. Per-class NMS (vectorized via torchvision when available)
    """

    def __init__(
        self,
        min_score: float = 0.30,
        nms_iou: float = 0.50,
        min_area_ratio: float = 0.001,
        max_area_ratio: float = 0.90,
        min_side_px: int = 20,
        max_aspect_ratio: float = 8.0,
    ):
        self.min_score = min_score
        self.nms_iou = nms_iou
        self.min_area = min_area_ratio
        self.max_area = max_area_ratio
        self.min_side = min_side_px
        self.max_aspect = max_aspect_ratio

    def filter(self, det: Dict) -> Dict:
        boxes = det["boxes_xyxy"].copy()
        scores = det["scores"].copy()
        labels = list(det["labels"])
        W, H = det["image_width"], det["image_height"]
        img_area = W * H

        if len(boxes) == 0:
            return det

        # Stage 1: confidence threshold
        mask = scores >= self.min_score
        boxes, scores = boxes[mask], scores[mask]
        labels = [l for l, m in zip(labels, mask) if m]

        # Stage 2: size / area / aspect ratio
        valid = []
        for i, b in enumerate(boxes):
            bw = b[2] - b[0]
            bh = b[3] - b[1]
            if bh <= 0 or bw <= 0:
                continue
            ratio = (bw * bh) / img_area if img_area > 0 else 0.0
            aspect = max(bw / bh, bh / bw)
            if (
                bw >= self.min_side
                and bh >= self.min_side
                and self.min_area <= ratio <= self.max_area
                and aspect <= self.max_aspect
            ):
                valid.append(i)
        boxes = boxes[valid]
        scores = scores[valid]
        labels = [labels[i] for i in valid]

        # Stage 3: per-class NMS (vectorized)
        if len(boxes) > 0:
            boxes, scores, labels = batched_nms(boxes, scores, labels, self.nms_iou)

        return {**det, "boxes_xyxy": boxes, "scores": scores, "labels": labels}

    def filter_batch(self, detections: List[Dict]) -> List[Dict]:
        out = [self.filter(d) for d in detections]
        return [d for d in out if len(d["boxes_xyxy"]) > 0]


# ── standalone utilities ───────────────────────────────────────────


def check_class_balance(
    detections: List[Dict], warn_ratio: float = 5.0
) -> Dict[str, int]:
    """
    Count annotations per class and warn if the most common class
    has more than `warn_ratio` × the count of the least common class.

    Returns {class_name: box_count} dict.
    """
    counts: Dict[str, int] = defaultdict(int)
    for det in detections:
        for label in det["labels"]:
            counts[str(label)] += 1

    if not counts:
        return {}

    max_cls = max(counts, key=counts.get)
    min_cls = min(counts, key=counts.get)
    ratio = counts[max_cls] / max(counts[min_cls], 1)

    log.info("Class distribution: " + ", ".join(
        f"{k}: {v}" for k, v in sorted(counts.items(), key=lambda x: -x[1])
    ))
    if ratio > warn_ratio:
        log.warning(
            f"Class imbalance detected: '{max_cls}' ({counts[max_cls]}) has "
            f"{ratio:.1f}× more annotations than '{min_cls}' ({counts[min_cls]}). "
            f"Consider oversampling '{min_cls}' or adding more search queries."
        )
    return dict(counts)


def temporal_consistency_filter(
    detections: List[Dict],
    min_frame_appearances: int = 2,
    iou_threshold: float = 0.4,
) -> List[Dict]:
    """
    Remove boxes that appear in only one frame from a video sequence.

    Rationale: a false positive from an open-vocabulary model tends to be
    inconsistent — it fires on one frame but not on neighbouring frames
    extracted from the same video. A true object appears consistently.

    Implementation:
      1. Group frames by video ID (extracted from filename convention
         '<video_id>_<frame_number>.jpg').
      2. For each frame, check if each box has an overlapping box
         (same class, IoU >= threshold) in at least one other frame
         from the same video.
      3. Keep the box only if it appears in >= min_frame_appearances frames.

    Args:
        detections: list of detection dicts (must have 'image_path')
        min_frame_appearances: minimum number of frames a box must appear in
        iou_threshold: IoU threshold for "same object" across frames

    Returns filtered detection list.
    """
    # Group by video ID (filename stem up to the last underscore+digits)
    def video_id(path: str) -> str:
        stem = Path(path).stem
        # e.g. "abc123_000060" → "abc123"
        m = re.match(r"^(.+?)_\d+$", stem)
        return m.group(1) if m else stem

    from pathlib import Path as _Path

    groups: Dict[str, List[Dict]] = defaultdict(list)
    for det in detections:
        vid = video_id(det["image_path"])
        groups[vid].append(det)

    result = []
    for vid, frames in groups.items():
        if len(frames) < min_frame_appearances:
            # Too few frames from this video to do consistency checking
            result.extend(frames)
            continue

        # For each frame, mark which boxes pass the consistency check
        for fi, frame in enumerate(frames):
            boxes = frame["boxes_xyxy"]
            labels = frame["labels"]
            if len(boxes) == 0:
                result.append(frame)
                continue

            kept_indices = []
            for bi, (box, label) in enumerate(zip(boxes, labels)):
                appearances = 0
                for fj, other_frame in enumerate(frames):
                    if fi == fj:
                        continue
                    for obox, olabel in zip(
                        other_frame["boxes_xyxy"], other_frame["labels"]
                    ):
                        if label == olabel and compute_iou(box, obox) >= iou_threshold:
                            appearances += 1
                            break
                if appearances >= min_frame_appearances - 1:
                    kept_indices.append(bi)

            if kept_indices:
                result.append({
                    **frame,
                    "boxes_xyxy": boxes[kept_indices],
                    "scores": frame["scores"][kept_indices],
                    "labels": [labels[i] for i in kept_indices],
                })

    before, after = len(detections), len(result)
    log.info(
        f"Temporal consistency filter: {before} → {after} annotated images "
        f"(removed {before - after} with inconsistent boxes)"
    )
    return [d for d in result if len(d["boxes_xyxy"]) > 0]
