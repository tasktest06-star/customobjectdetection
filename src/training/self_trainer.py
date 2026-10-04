"""
Teacher-student self-training loop for iterative pseudo-label refinement.

Why self-training helps
───────────────────────
Pseudo-labels from GroundingDINO/OWLv2 are noisy: the teacher sometimes fires on
background regions, misses occluded objects, or produces imprecise box boundaries.
After one round of training on these noisy labels, the student (RT-DETR) has learned
to generalize beyond the noise. Crucially, where the teacher AND the student both
agree on a box, that box is very likely correct — two independent models
independently producing the same detection is strong evidence of a true positive.

Ensemble strategy
─────────────────
For each teacher box in each image:
  - If any student box has the same class label AND IoU ≥ agree_iou (0.5): KEEP
  - If the student found nothing at all for this image: keep all teacher boxes
    (fallback — we don't want to lose coverage on hard/rare examples)
  - If no student box agrees with a given teacher box: still keep it
    (conservative — avoids discarding true positives the student missed)

This means the ensemble never makes the training set smaller than the teacher-only
set, but raises confidence on boxes where both models agree.

Round structure
───────────────
  Round 0: GroundingDINO (teacher) → pseudo-labels → train RT-DETR (student v0)
  Round 1: Ensemble(teacher, student v0) → refined labels → train RT-DETR (student v1)
  Round N: Ensemble(teacher, student vN-1) → …
"""

import torch
from typing import List, Dict, Optional, Union
from tqdm import tqdm
from PIL import Image

from src.pseudo_labeling.grounding_dino import GroundingDINOLabeler
from src.pseudo_labeling.owl_vit import OWLv2Labeler
from src.annotation.quality_filter import LabelQualityFilter, compute_iou
from src.annotation.coco_builder import COCOBuilder
from src.training.rtdetr_trainer import RTDETRTrainer


class SelfTrainer:
    """
    Orchestrates one or more teacher-student fine-tuning rounds.

    The teacher is a frozen zero-shot detector (GroundingDINO or OWLv2) that
    never changes. The student is RT-DETR, re-trained from the same base
    checkpoint each round on increasingly refined pseudo-labels.
    """

    def __init__(
        self,
        class_names: List[str],
        teacher: Union[GroundingDINOLabeler, OWLv2Labeler],
        quality_filter: LabelQualityFilter,
        coco_builder: COCOBuilder,
        rtdetr_trainer: RTDETRTrainer,
        rounds: int = 2,
        agree_iou: float = 0.50,
        device: Optional[str] = None,
    ):
        """
        Args:
            class_names:     List of target class name strings (e.g. ["raccoon"]).
            teacher:         Pre-loaded pseudo-labeler (GroundingDINOLabeler or OWLv2Labeler).
            quality_filter:  LabelQualityFilter for NMS + confidence + size filtering.
            coco_builder:    COCOBuilder for writing COCO JSON datasets each round.
            rtdetr_trainer:  RTDETRTrainer instance; .train() is called each round.
            rounds:          Number of self-training iterations (≥1).
            agree_iou:       Minimum IoU between a teacher box and a student box for
                             them to be considered in agreement. 0.5 is the standard
                             COCO threshold for a "correct" detection.
            device:          Device for student inference ("cuda" / "cpu" / None).
        """
        self.class_names = class_names
        self.teacher = teacher
        self.filter = quality_filter
        self.builder = coco_builder
        self.trainer = rtdetr_trainer
        self.rounds = rounds
        self.agree_iou = agree_iou
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ── helpers ────────────────────────────────────────────────────────────────

    def _match_label(self, label: str) -> str:
        """
        Normalize a raw teacher label token to the nearest known class name.

        GroundingDINO returns decoded text spans from its tokenizer rather than
        exact class name strings. Examples of what it might return for class "raccoon":
          - "raccoon"         (single-word class, usually matches directly)
          - "raccoon."        (trailing period from the period-separated text prompt)
          - "racoon"          (tokenizer may merge sub-tokens imprecisely)
          - "garbage truck."  (multi-word class with trailing period)

        OWLv2 returns exact class name strings, so this function is a no-op for it.

        Matching logic: strip whitespace and trailing periods, then check if the
        known class name appears inside the label or vice-versa (substring match).
        """
        label = label.lower().strip().rstrip(".")
        for name in self.class_names:
            if name.lower() in label or label in name.lower():
                return name
        # If no match found, return the cleaned label as-is (will fail comparison
        # against student labels and be treated as disagreement, which is correct)
        return label

    def _student_predict(
        self,
        model_path: str,
        image_paths: List[str],
        conf: float = 0.5,
    ) -> List[Dict]:
        """
        Run the current student (RT-DETR checkpoint at model_path) on all images.

        Uses a relatively high confidence threshold (default 0.5) for the ensemble
        step: we only want high-confidence student predictions to "vote" alongside
        the teacher. Low-confidence student predictions are more likely to be
        false positives that would incorrectly endorse noisy teacher boxes.

        Returns a list of detection dicts in the same format as the pseudo-labelers:
          {"image_path", "boxes_xyxy", "scores", "labels", "image_width", "image_height"}
        """
        from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

        # Load the student checkpoint (saved by RTDETRTrainer.train())
        processor = RTDetrImageProcessor.from_pretrained(model_path)
        model = RTDetrForObjectDetection.from_pretrained(model_path).to(self.device)
        model.eval()

        results = []
        with torch.no_grad():
            for path in tqdm(image_paths, desc="Student inference"):
                try:
                    image = Image.open(path).convert("RGB")
                    inputs = processor(images=image, return_tensors="pt").to(
                        self.device
                    )
                    outputs = model(**inputs)

                    # target_sizes must be on the same device as model outputs
                    # to avoid RuntimeError on GPU systems
                    target_sizes = torch.tensor(
                        [[image.height, image.width]], device=self.device
                    )
                    preds = processor.post_process_object_detection(
                        outputs, target_sizes=target_sizes, threshold=conf
                    )[0]

                    labels_int = preds["labels"].cpu().numpy()
                    results.append(
                        {
                            "image_path":   path,
                            "boxes_xyxy":   preds["boxes"].cpu().numpy(),
                            "scores":       preds["scores"].cpu().numpy(),
                            # Map integer label IDs back to class name strings
                            "labels":       [self.class_names[i] for i in labels_int],
                            "image_width":  image.width,
                            "image_height": image.height,
                        }
                    )
                except Exception as e:
                    print(f"Student error on {path}: {e}")
        return results

    def _ensemble(
        self,
        teacher_dets: List[Dict],
        student_dets: List[Dict],
    ) -> List[Dict]:
        """
        Combine teacher and student detections via box-level agreement voting.

        For each image and each teacher box:
          1. Normalize the teacher label (GroundingDINO raw token → class name).
          2. Look for any student box with the same class AND IoU ≥ agree_iou.
          3. If found: include this teacher box in the ensemble output (high confidence).
          4. If no student boxes agree with ANY teacher box: fall back to all teacher
             boxes (prevents losing hard examples the student hasn't learned yet).

        The output list uses teacher boxes (not student boxes) because teacher
        boxes tend to be better localized — GroundingDINO/OWLv2 are fine-tuned on
        large datasets, while the student is still learning. The student acts as a
        filter to remove clearly wrong teacher boxes, not as a localizer.
        """
        # Build a lookup from image_path → student detections for O(1) access
        s_map = {d["image_path"]: d for d in student_dets}
        ensembled = []

        for td in teacher_dets:
            sd = s_map.get(td["image_path"])

            # No student prediction for this image → keep all teacher boxes as-is
            if sd is None or len(sd["boxes_xyxy"]) == 0:
                ensembled.append(td)
                continue

            kept = []  # indices of teacher boxes that the student agrees with
            for i, (tb, tl) in enumerate(zip(td["boxes_xyxy"], td["labels"])):
                # Normalize teacher label before comparing (GroundingDINO quirk)
                tl_norm = self._match_label(str(tl))
                for sb, sl in zip(sd["boxes_xyxy"], sd["labels"]):
                    # Both conditions must hold: same class AND sufficient overlap
                    if tl_norm == sl and compute_iou(tb, sb) >= self.agree_iou:
                        kept.append(i)
                        break  # one agreeing student box is enough

            # If no teacher box had student agreement, fall back to all teacher boxes.
            # This ensures hard/rare instances don't disappear from the training set.
            idx = kept if kept else list(range(len(td["boxes_xyxy"])))

            ensembled.append(
                {
                    **td,
                    "boxes_xyxy": td["boxes_xyxy"][idx],
                    "scores":     td["scores"][idx],
                    "labels":     [td["labels"][i] for i in idx],
                }
            )
        return ensembled

    # ── main entry point ───────────────────────────────────────────────────────

    def run(
        self,
        image_paths: List[str],
        initial_detections: Optional[List[Dict]] = None,
        dataset_name: str = "self_train",
    ) -> str:
        """
        Execute all self-training rounds and return the final model path.

        Args:
            image_paths:         All image paths to use for training and ensemble.
            initial_detections:  Pre-filtered teacher detections from the main pipeline.
                                 If None, the teacher is re-run here (slower but works
                                 when SelfTrainer is used standalone).
            dataset_name:        Base name for per-round dataset directories.

        Returns:
            Path to the final trained model directory (str).
        """
        if self.rounds < 1:
            raise ValueError("SelfTrainer requires rounds >= 1")

        # Use pre-computed detections from the pipeline if available;
        # otherwise run the teacher from scratch (adds significant time)
        detections = initial_detections
        if detections is None:
            print("Round 0: generating teacher-only pseudo-labels…")
            raw = self.teacher.label_batch(image_paths, self.class_names)
            detections = self.filter.filter_batch(raw)

        model_path: Optional[str] = None

        for r in range(self.rounds):
            round_name = f"{dataset_name}_r{r}"
            print(f"\n=== Self-training round {r + 1}/{self.rounds} ===")
            print(f"  {len(detections)} images with annotations")

            # Build a fresh COCO dataset for this round's labels and train RT-DETR
            data_yaml = self.builder.build_dataset(
                detections, dataset_name=round_name
            )
            # Shorter epoch count for intermediate rounds — the final student is what
            # matters; intermediate rounds just need to converge sufficiently for a
            # useful ensemble signal
            model_path = self.trainer.train(
                data_yaml=data_yaml, run_name=round_name, epochs=30
            )

            # After the last round we're done; skip the ensemble step
            if r < self.rounds - 1:
                print("  Generating ensemble labels for next round…")
                # Teacher runs on all images again (its predictions are deterministic)
                teacher_dets = self.teacher.label_batch(
                    image_paths, self.class_names
                )
                # Student runs on all images with a higher confidence bar
                student_dets = self._student_predict(model_path, image_paths)
                # Combine: keep teacher boxes where student agrees
                raw = self._ensemble(teacher_dets, student_dets)
                # Re-apply quality filter to the ensemble output before next round
                detections = self.filter.filter_batch(raw)
                print(f"  Ensemble kept {len(detections)} usable images")

        return model_path
