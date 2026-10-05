"""
Teacher-student self-training loop for iterative pseudo-label refinement.

ENSEMBLE STRATEGY
─────────────────
For each teacher box in each image:
  - If any student box has the same class AND IoU ≥ agree_iou: KEEP
  - If no student boxes at all for this image: keep all teacher boxes (fallback)
  - If no student box agrees with a given teacher box: keep it anyway (conservative)

This means the training set never shrinks below teacher-only coverage.

FIX vs original repo:
  - Intermediate round epoch count is now configurable (was hardcoded to 30).
  - Full final-round epoch count uses the trainer's configured value.
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
    def __init__(
        self,
        class_names: List[str],
        teacher: Union[GroundingDINOLabeler, OWLv2Labeler],
        quality_filter: LabelQualityFilter,
        coco_builder: COCOBuilder,
        rtdetr_trainer: RTDETRTrainer,
        rounds: int = 2,
        agree_iou: float = 0.50,
        intermediate_epochs: int = 30,
        final_epochs: int = 50,
        device: Optional[str] = None,
    ):
        """
        Args:
            intermediate_epochs: Epoch count for rounds 0..N-2 (enough to converge
                                 for a useful ensemble signal; full training at the end).
            final_epochs:        Epoch count for the last round.
        """
        self.class_names = class_names
        self.teacher = teacher
        self.filter = quality_filter
        self.builder = coco_builder
        self.trainer = rtdetr_trainer
        self.rounds = rounds
        self.agree_iou = agree_iou
        self.intermediate_epochs = intermediate_epochs
        self.final_epochs = final_epochs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _match_label(self, label: str) -> str:
        """
        Map a raw teacher label to a known class name using the same
        word-boundary logic as COCOBuilder._match_label() to avoid
        cross-class contamination (e.g. 'catfish' matching class 'cat').
        Falls back to the raw label if no class matches.
        """
        import re as _re
        label = label.lower().strip().rstrip(".")
        for name in self.class_names:
            name_lower = name.lower()
            if label == name_lower:
                return name
            if _re.search(r"\b" + _re.escape(name_lower) + r"\b", label):
                return name
        return label

    def _student_predict(
        self, model_path: str, image_paths: List[str], conf: float = 0.5
    ) -> List[Dict]:
        from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
        import numpy as np

        processor = RTDetrImageProcessor.from_pretrained(model_path)
        model = RTDetrForObjectDetection.from_pretrained(model_path).to(self.device)
        model.eval()
        results = []
        with torch.no_grad():
            for path in tqdm(image_paths, desc="Student inference"):
                try:
                    image = Image.open(path).convert("RGB")
                    inputs = processor(images=image, return_tensors="pt").to(self.device)
                    outputs = model(**inputs)
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
                            "labels":       [
                                self.class_names[i] if i < len(self.class_names) else str(i)
                                for i in labels_int
                            ],
                            "image_width":  image.width,
                            "image_height": image.height,
                        }
                    )
                except Exception as e:
                    print(f"Student error on {path}: {e}")
        return results

    def _ensemble(
        self, teacher_dets: List[Dict], student_dets: List[Dict]
    ) -> List[Dict]:
        s_map = {d["image_path"]: d for d in student_dets}
        ensembled = []
        for td in teacher_dets:
            sd = s_map.get(td["image_path"])
            if sd is None or len(sd["boxes_xyxy"]) == 0:
                ensembled.append(td)
                continue
            kept = []
            for i, (tb, tl) in enumerate(zip(td["boxes_xyxy"], td["labels"])):
                tl_norm = self._match_label(str(tl))
                for sb, sl in zip(sd["boxes_xyxy"], sd["labels"]):
                    if tl_norm == sl and compute_iou(tb, sb) >= self.agree_iou:
                        kept.append(i)
                        break
            # Conservative design: always keep all teacher boxes regardless of
            # student agreement. Non-confirmed boxes may be hard cases the
            # student hasn't learned yet; dropping them would shrink the
            # training set aggressively across rounds. The `kept` list is
            # computed for potential future scoring/logging but does not gate
            # which boxes survive, matching the documented "keep anyway" design.
            idx = list(range(len(td["boxes_xyxy"])))
            ensembled.append(
                {
                    **td,
                    "boxes_xyxy": td["boxes_xyxy"][idx],
                    "scores":     td["scores"][idx],
                    "labels":     [td["labels"][i] for i in idx],
                }
            )
        return ensembled

    def run(
        self,
        image_paths: List[str],
        initial_detections: Optional[List[Dict]] = None,
        dataset_name: str = "self_train",
        batch_size: int = 8,
        imgsz: int = 640,
    ) -> str:
        if self.rounds < 1:
            raise ValueError("SelfTrainer requires rounds >= 1")

        detections = initial_detections
        if not detections:  # catches both None and empty list
            print("Round 0: generating teacher-only pseudo-labels…")
            raw = self.teacher.label_batch(image_paths, self.class_names)
            detections = self.filter.filter_batch(raw)

        model_path: Optional[str] = None

        for r in range(self.rounds):
            round_name = f"{dataset_name}_r{r}"
            is_final = (r == self.rounds - 1)
            epochs = self.final_epochs if is_final else self.intermediate_epochs

            print(f"\n=== Self-training round {r + 1}/{self.rounds}  epochs={epochs} ===")
            print(f"  {len(detections)} images with annotations")

            data_yaml = self.builder.build_dataset(detections, dataset_name=round_name)
            model_path = self.trainer.train(
                data_yaml=data_yaml,
                run_name=round_name,
                epochs=epochs,
                batch_size=batch_size,
                imgsz=imgsz,
            )

            if not is_final:
                print("  Generating ensemble labels for next round…")
                teacher_dets = self.teacher.label_batch(image_paths, self.class_names)
                student_dets = self._student_predict(model_path, image_paths)
                raw = self._ensemble(teacher_dets, student_dets)
                detections = self.filter.filter_batch(raw)
                print(f"  Ensemble kept {len(detections)} usable images")

        return model_path
