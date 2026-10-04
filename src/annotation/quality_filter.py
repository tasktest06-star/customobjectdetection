"""NMS and quality filtering for pseudo-labels before COCO dataset generation."""
import numpy as np
from typing import List, Dict


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
    boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5
) -> np.ndarray:
    """Greedy NMS. Returns kept indices sorted by descending score."""
    if len(boxes) == 0:
        return np.array([], dtype=int)
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


class LabelQualityFilter:
    """
    Applies three successive filters to pseudo-label detections:
    1. Confidence score threshold
    2. Bounding-box size / area ratio check
    3. Per-class NMS to remove duplicates
    """

    def __init__(
        self,
        min_score: float = 0.30,
        nms_iou: float = 0.50,
        min_area_ratio: float = 0.001,
        max_area_ratio: float = 0.90,
        min_side_px: int = 20,
    ):
        self.min_score = min_score
        self.nms_iou = nms_iou
        self.min_area = min_area_ratio
        self.max_area = max_area_ratio
        self.min_side = min_side_px

    def filter(self, det: Dict) -> Dict:
        """Filter a single image detection dict. Returns same structure."""
        boxes = det["boxes_xyxy"].copy()
        scores = det["scores"].copy()
        labels = list(det["labels"])
        W, H = det["image_width"], det["image_height"]
        img_area = W * H

        if len(boxes) == 0:
            return det

        # 1. Confidence threshold
        mask = scores >= self.min_score
        boxes, scores = boxes[mask], scores[mask]
        labels = [l for l, m in zip(labels, mask) if m]

        # 2. Box size / area ratio
        valid = []
        for i, b in enumerate(boxes):
            bw, bh = b[2] - b[0], b[3] - b[1]
            ratio = (bw * bh) / img_area if img_area > 0 else 0.0
            if (
                bw >= self.min_side
                and bh >= self.min_side
                and self.min_area <= ratio <= self.max_area
            ):
                valid.append(i)
        boxes = boxes[valid]
        scores = scores[valid]
        labels = [labels[i] for i in valid]

        # 3. Per-class NMS
        if len(boxes) > 0:
            keep_all = []
            for cls in set(labels):
                m = np.array([l == cls for l in labels])
                orig_idx = np.where(m)[0]
                k = nms(boxes[m], scores[m], self.nms_iou)
                keep_all.extend(orig_idx[k].tolist())
            keep_all = sorted(keep_all)
            boxes = boxes[keep_all]
            scores = scores[keep_all]
            labels = [labels[i] for i in keep_all]

        return {**det, "boxes_xyxy": boxes, "scores": scores, "labels": labels}

    def filter_batch(self, detections: List[Dict]) -> List[Dict]:
        """Filter a list of detections; drop images with no remaining boxes."""
        out = [self.filter(d) for d in detections]
        return [d for d in out if len(d["boxes_xyxy"]) > 0]
