"""
Enhanced teacher-student self-training loop.

Improvements over v1:
  - Convergence check: compares mAP50 between rounds and stops early
    when improvement falls below a threshold (saves GPU time)
  - Hard negative mining: identifies frames where the student is uncertain
    (low confidence detections) but the teacher found objects — these
    are the most informative examples for the next training round
  - Incremental teacher labeling: caches teacher predictions from round 0
    so subsequent rounds only need student inference, not a full teacher re-run
"""
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from PIL import Image

from src.pseudo_labeling.grounding_dino import GroundingDINOLabeler
from src.annotation.quality_filter import LabelQualityFilter, compute_iou
from src.annotation.coco_builder import COCOBuilder
from src.training.yolo_trainer import YOLOTrainer
from src.utils.logger import get_logger

log = get_logger("self_trainer")


class SelfTrainer:
    def __init__(
        self,
        class_names: List[str],
        teacher: GroundingDINOLabeler,
        quality_filter: LabelQualityFilter,
        coco_builder: COCOBuilder,
        yolo_trainer: YOLOTrainer,
        rounds: int = 2,
        agree_iou: float = 0.50,
        # convergence
        min_map_improvement: float = 0.02,
        # hard negative mining
        use_hard_negatives: bool = True,
        hard_negative_conf_ceiling: float = 0.40,
        hard_negative_ratio: float = 0.30,
    ):
        """
        Args:
            min_map_improvement: stop self-training if mAP50 improves less
                                  than this between consecutive rounds
            hard_negative_conf_ceiling: a student detection is considered
                                        a "hard negative" if its max confidence
                                        is below this value
            hard_negative_ratio: fraction of training images that are hard
                                  negatives (the rest are agreed-upon positives)
        """
        self.class_names = class_names
        self.teacher = teacher
        self.filter = quality_filter
        self.builder = coco_builder
        self.trainer = yolo_trainer
        self.rounds = rounds
        self.agree_iou = agree_iou
        self.min_map_improvement = min_map_improvement
        self.use_hard_negatives = use_hard_negatives
        self.hard_conf_ceiling = hard_negative_conf_ceiling
        self.hard_neg_ratio = hard_negative_ratio

        self._teacher_cache: Optional[List[Dict]] = None  # incremental labeling

    # ── student inference ──────────────────────────────────────────

    def _student_predict(
        self,
        model_path: str,
        image_paths: List[str],
        conf: float = 0.30,
    ) -> List[Dict]:
        """Run trained student model on all images."""
        from ultralytics import YOLO
        model = YOLO(model_path)
        results = []
        for path in tqdm(image_paths, desc="Student inference"):
            try:
                preds = model.predict(path, conf=conf, verbose=False)[0]
                img = Image.open(path)
                has = len(preds.boxes) > 0
                results.append({
                    "image_path": path,
                    "boxes_xyxy": preds.boxes.xyxy.cpu().numpy() if has else np.zeros((0, 4)),
                    "scores": preds.boxes.conf.cpu().numpy() if has else np.array([]),
                    "labels": [self.class_names[int(i)]
                               for i in preds.boxes.cls.cpu().numpy()] if has else [],
                    "image_width": img.width,
                    "image_height": img.height,
                })
            except Exception as e:
                log.warning(f"Student error on {path}: {e}")
        return results

    # ── convergence check ──────────────────────────────────────────

    def _get_map50(self, model_path: str, data_yaml: str) -> float:
        """Evaluate mAP50 on the current val set."""
        try:
            metrics = self.trainer.evaluate(model_path, data_yaml)
            return metrics.get("mAP50", 0.0)
        except Exception as e:
            log.warning(f"Could not evaluate mAP50: {e}")
            return 0.0

    def _converged(
        self, prev_map: Optional[float], curr_map: float
    ) -> bool:
        """True if improvement is too small to justify another round."""
        if prev_map is None:
            return False
        improvement = curr_map - prev_map
        if improvement < self.min_map_improvement:
            log.info(
                f"Convergence detected: mAP50 improved only "
                f"{improvement:.4f} (< {self.min_map_improvement}). Stopping."
            )
            return True
        return False

    # ── hard negative mining ───────────────────────────────────────

    def _mine_hard_negatives(
        self,
        teacher_dets: List[Dict],
        student_dets: List[Dict],
    ) -> List[Dict]:
        """
        Find images where:
          - The teacher detected objects (teacher has boxes)
          - The student is uncertain (max confidence < hard_conf_ceiling)

        These are the most informative training examples because the student
        hasn't yet learned to recognise these instances confidently.

        Returns list of teacher detection dicts for hard-negative images.
        """
        s_map = {d["image_path"]: d for d in student_dets}
        hard_negatives = []

        for td in teacher_dets:
            if len(td["boxes_xyxy"]) == 0:
                continue
            sd = s_map.get(td["image_path"])
            if sd is None:
                continue
            max_conf = (
                float(sd["scores"].max())
                if len(sd["scores"]) > 0
                else 0.0
            )
            if max_conf < self.hard_conf_ceiling:
                hard_negatives.append(td)

        log.info(f"Hard negative mining: {len(hard_negatives)} hard-negative images found")
        return hard_negatives

    # ── ensemble ──────────────────────────────────────────────────

    def _ensemble(
        self,
        teacher_dets: List[Dict],
        student_dets: List[Dict],
    ) -> List[Dict]:
        """
        Keep teacher boxes confirmed by student agreement.
        Falls back to teacher-only when student disagrees on everything.
        """
        s_map = {d["image_path"]: d for d in student_dets}
        ensembled = []
        for td in teacher_dets:
            sd = s_map.get(td["image_path"])
            if sd is None or len(sd["boxes_xyxy"]) == 0:
                ensembled.append(td)
                continue
            kept = []
            for i, (tb, tl) in enumerate(zip(td["boxes_xyxy"], td["labels"])):
                for sb, sl in zip(sd["boxes_xyxy"], sd["labels"]):
                    if tl == sl and compute_iou(tb, sb) >= self.agree_iou:
                        kept.append(i)
                        break
            idx = kept if kept else list(range(len(td["boxes_xyxy"])))
            ensembled.append({
                **td,
                "boxes_xyxy": td["boxes_xyxy"][idx],
                "scores": td["scores"][idx],
                "labels": [td["labels"][i] for i in idx],
            })
        return ensembled

    # ── main run ───────────────────────────────────────────────────

    def run(
        self,
        image_paths: List[str],
        initial_detections: Optional[List[Dict]] = None,
        dataset_name: str = "self_train",
    ) -> str:
        """
        Run self-training rounds with convergence checking and
        hard negative mining.

        Returns path to final trained model weights.
        """
        # Round 0: use initial detections or generate teacher pseudo-labels
        detections = initial_detections
        if detections is None:
            log.info("Round 0: generating teacher pseudo-labels (initial)…")
            raw = self.teacher.label_batch(image_paths, self.class_names)
            detections = self.filter.filter_batch(raw)
            self._teacher_cache = raw  # save for incremental re-use

        model_path: Optional[str] = None
        prev_map: Optional[float] = None

        for r in range(self.rounds):
            round_name = f"{dataset_name}_r{r}"
            log.info(f"\n{'='*50}")
            log.info(f"Self-training round {r + 1}/{self.rounds} | "
                     f"{len(detections)} annotated images")
            log.info(f"{'='*50}")

            data_yaml = self.builder.build_dataset(
                detections, dataset_name=round_name
            )
            model_path = self.trainer.train(
                data_yaml=data_yaml, run_name=round_name, epochs=30
            )

            # Convergence check
            curr_map = self._get_map50(model_path, data_yaml)
            log.info(f"Round {r + 1} mAP50: {curr_map:.4f}")
            if self._converged(prev_map, curr_map):
                break
            prev_map = curr_map

            if r < self.rounds - 1:
                # Incremental teacher labeling: reuse cache when available
                if self._teacher_cache is not None:
                    log.info("Reusing cached teacher predictions (incremental mode)")
                    teacher_dets = self._teacher_cache
                else:
                    log.info("Re-running teacher labeling…")
                    teacher_dets = self.teacher.label_batch(
                        image_paths, self.class_names
                    )
                    self._teacher_cache = teacher_dets

                student_dets = self._student_predict(model_path, image_paths)
                ensemble_dets = self._ensemble(teacher_dets, student_dets)

                # Hard negative mining: inject hardest examples into next round
                if self.use_hard_negatives:
                    hard_negs = self._mine_hard_negatives(teacher_dets, student_dets)
                    n_hard = int(len(ensemble_dets) * self.hard_neg_ratio)
                    hard_negs = hard_negs[:n_hard]
                    ensemble_dets = ensemble_dets + hard_negs
                    log.info(
                        f"Injected {len(hard_negs)} hard-negative images "
                        f"({self.hard_neg_ratio:.0%} of training set)"
                    )

                detections = self.filter.filter_batch(ensemble_dets)
                log.info(f"Next round: {len(detections)} usable images")

        return model_path
