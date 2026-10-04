"""
NMS and quality filtering for pseudo-labels before COCO dataset generation.

Why quality filtering is necessary
────────────────────────────────────
Zero-shot detectors (GroundingDINO, OWLv2) tend to produce:
  - Duplicate boxes: slightly offset detections of the same object
  - Background detections: low-confidence boxes on textures/patterns
  - Extreme-size boxes: near-full-image or 1-pixel boxes, usually noise
  - Low-confidence boxes: below the signal/noise boundary

Training RT-DETR on unfiltered pseudo-labels degrades final mAP significantly.
This module applies three successive filters to address each category.

Filter pipeline (applied in this order)
────────────────────────────────────────
  1. Confidence threshold   — drop any box with score < min_score
  2. Box size / area ratio  — drop boxes smaller than min_side pixels or
                             with area outside [min_area_ratio, max_area_ratio]
                             (relative to image area)
  3. Per-class NMS          — for each class, greedily suppress boxes with
                              IoU > nms_iou against a higher-scoring box

Why per-class NMS, not global NMS:
  Global NMS could suppress a "raccoon" box and a "capybara" box that overlap
  (e.g., two animals standing close together). Per-class NMS only suppresses
  boxes of the same class, preserving multi-class overlapping detections.
"""
import numpy as np
from typing import List, Dict


def compute_iou(b1: np.ndarray, b2: np.ndarray) -> float:
    """
    Intersection over Union (IoU) between two axis-aligned bounding boxes.

    Both boxes must be in xyxy format: [x1, y1, x2, y2] in pixel coordinates.
    Returns a float in [0, 1]. Returns 0 if both boxes have zero area (degenerate).

    IoU = intersection_area / union_area
        = intersection_area / (area1 + area2 - intersection_area)
    """
    # Intersection rectangle: max of left edges, min of right edges
    xi1, yi1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    xi2, yi2 = min(b1[2], b2[2]), min(b1[3], b2[3])

    # Clamp to 0 in case boxes don't overlap (intersection would be negative)
    inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)

    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter  # inclusion-exclusion principle

    return inter / union if union > 0 else 0.0


def nms(
    boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5
) -> np.ndarray:
    """
    Greedy Non-Maximum Suppression (NMS).

    Algorithm:
      1. Sort boxes by score descending.
      2. Take the highest-scoring box; add it to the keep list.
      3. Remove all remaining boxes with IoU > iou_threshold against the kept box.
      4. Repeat with the next highest-scoring remaining box.

    This is "greedy" NMS — it's O(N²) in the worst case but fast in practice
    because most detection heads already prune boxes below a confidence threshold.
    Soft-NMS (decay scores instead of hard removal) would give slightly better
    recall for overlapping objects but adds complexity not needed here.

    Args:
        boxes:         np.ndarray (N, 4) in xyxy format.
        scores:        np.ndarray (N,) confidence scores.
        iou_threshold: Boxes with IoU > this against a kept box are suppressed.

    Returns:
        np.ndarray of kept indices (int dtype), sorted by descending score.
    """
    if len(boxes) == 0:
        return np.array([], dtype=int)

    # argsort() is ascending; [::-1] reverses to descending score order
    order = scores.argsort()[::-1]
    keep = []

    while len(order):
        i = order[0]    # highest-scoring remaining box
        keep.append(i)
        if len(order) == 1:
            break
        # Compute IoU of this box against all remaining boxes
        ious = np.array([compute_iou(boxes[i], boxes[j]) for j in order[1:]])
        # Keep only boxes with IoU ≤ threshold (suppress the rest)
        order = order[1:][ious <= iou_threshold]

    return np.array(keep, dtype=int)


class LabelQualityFilter:
    """
    Three-stage filter for pseudo-label detection dicts.

    Operates on single detection dicts (filter()) or lists of them (filter_batch()).
    Detection dicts use the standard schema:
      {
        "image_path"   : str,
        "boxes_xyxy"   : np.ndarray (N, 4) — xyxy pixel coords
        "scores"       : np.ndarray (N,)
        "labels"       : List[str]
        "image_width"  : int
        "image_height" : int
      }

    filter_batch() additionally drops images that have no remaining boxes after
    filtering, so the downstream COCOBuilder only receives annotated images.
    """

    def __init__(
        self,
        min_score: float = 0.30,
        nms_iou: float = 0.50,
        min_area_ratio: float = 0.001,
        max_area_ratio: float = 0.90,
        min_side_px: int = 20,
    ):
        """
        Args:
            min_score:       Drop boxes with confidence score below this value.
                             Should match pseudo_labeling.box_threshold in config
                             so the filter doesn't contradict the labeler setting.
            nms_iou:         IoU threshold for NMS. 0.5 is the standard COCO
                             definition of a duplicate detection.
            min_area_ratio:  Minimum box area as a fraction of image area.
                             0.001 = 0.1% of image — filters sub-pixel noise.
            max_area_ratio:  Maximum box area as a fraction of image area.
                             0.90 = 90% of image — filters full-image ghost detections
                             that happen when the detector confuses background context
                             for the object.
            min_side_px:     Minimum width AND height in pixels. Prevents 1×1 or
                             single-row/column boxes from entering training data.
        """
        self.min_score = min_score
        self.nms_iou = nms_iou
        self.min_area = min_area_ratio
        self.max_area = max_area_ratio
        self.min_side = min_side_px

    def filter(self, det: Dict) -> Dict:
        """
        Apply all three filters to a single image detection dict.

        Returns the same dict schema with boxes/scores/labels replaced by the
        filtered subset. Does NOT drop the image if no boxes remain — call
        filter_batch() for that behavior.
        """
        # Work on copies to avoid mutating the caller's arrays
        boxes = det["boxes_xyxy"].copy()
        scores = det["scores"].copy()
        labels = list(det["labels"])
        W, H = det["image_width"], det["image_height"]
        img_area = W * H

        if len(boxes) == 0:
            return det  # nothing to filter

        # ── Stage 1: confidence threshold ────────────────────────────────────
        # Boolean mask over all boxes; True = keep, False = discard
        mask = scores >= self.min_score
        boxes = boxes[mask]
        scores = scores[mask]
        labels = [l for l, m in zip(labels, mask) if m]

        # ── Stage 2: box size / area ratio ───────────────────────────────────
        valid = []
        for i, b in enumerate(boxes):
            bw = b[2] - b[0]   # box width in pixels
            bh = b[3] - b[1]   # box height in pixels
            ratio = (bw * bh) / img_area if img_area > 0 else 0.0
            # Keep box only if it meets all three size criteria simultaneously
            if (
                bw >= self.min_side     # wide enough
                and bh >= self.min_side # tall enough
                and self.min_area <= ratio <= self.max_area  # reasonable area fraction
            ):
                valid.append(i)
        boxes = boxes[valid]
        scores = scores[valid]
        labels = [labels[i] for i in valid]

        # ── Stage 3: per-class NMS ────────────────────────────────────────────
        if len(boxes) > 0:
            keep_all = []
            # Process each class separately so we don't suppress cross-class overlaps
            for cls in set(labels):
                # Boolean mask for this class
                m = np.array([l == cls for l in labels])
                # Original indices in the current boxes/scores/labels arrays
                orig_idx = np.where(m)[0]
                # Run NMS on just this class's boxes and scores
                k = nms(boxes[m], scores[m], self.nms_iou)
                # Map NMS output indices back to the original array indices
                keep_all.extend(orig_idx[k].tolist())

            keep_all = sorted(keep_all)  # restore spatial ordering
            boxes = boxes[keep_all]
            scores = scores[keep_all]
            labels = [labels[i] for i in keep_all]

        return {**det, "boxes_xyxy": boxes, "scores": scores, "labels": labels}

    def filter_batch(self, detections: List[Dict]) -> List[Dict]:
        """
        Filter a list of detection dicts and remove images with no boxes.

        Images where all boxes were filtered out are excluded from the returned
        list entirely. This prevents COCOBuilder from writing images with empty
        annotation lists, which would artificially increase the count of
        "annotated" images while contributing no supervision signal during training.
        """
        out = [self.filter(d) for d in detections]
        # Keep only images that have at least one box after filtering
        return [d for d in out if len(d["boxes_xyxy"]) > 0]
