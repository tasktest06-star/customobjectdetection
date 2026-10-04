"""
Teacher-student self-training loop for iterative pseudo-label refinement.

Round 0:  GroundingDINO (teacher) → pseudo-labels → train RT-DETR (student)
Round N:  Ensemble teacher + student; keep boxes where both agree → retrain

The student model is now RT-DETR via HuggingFace transformers (Apache 2.0),
replacing the previous YOLOv8/ultralytics student (AGPL-3.0).
"""
import torch
import numpy as np
from typing import List, Dict, Optional
from tqdm import tqdm
from PIL import Image

from src.pseudo_labeling.grounding_dino import GroundingDINOLabeler
from src.annotation.quality_filter import LabelQualityFilter, compute_iou
from src.annotation.coco_builder import COCOBuilder
from src.training.rtdetr_trainer import RTDETRTrainer


class SelfTrainer:
    def __init__(
        self,
        class_names: List[str],
        teacher: GroundingDINOLabeler,
        quality_filter: LabelQualityFilter,
        coco_builder: COCOBuilder,
        rtdetr_trainer: RTDETRTrainer,
        rounds: int = 2,
        agree_iou: float = 0.50,
        device: Optional[str] = None,
    ):
        self.class_names = class_names
        self.teacher = teacher
        self.filter = quality_filter
        self.builder = coco_builder
        self.trainer = rtdetr_trainer
        self.rounds = rounds
        self.agree_iou = agree_iou
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _student_predict(
        self,
        model_path: str,
        image_paths: List[str],
        conf: float = 0.5,
    ) -> List[Dict]:
        """Run RT-DETR student inference on image_paths."""
        from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

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
                    target_sizes = torch.tensor([[image.height, image.width]])
                    preds = processor.post_process_object_detection(
                        outputs, target_sizes=target_sizes, threshold=conf
                    )[0]
                    labels_int = preds["labels"].cpu().numpy()
                    results.append(
                        {
                            "image_path": path,
                            "boxes_xyxy": preds["boxes"].cpu().numpy(),
                            "scores": preds["scores"].cpu().numpy(),
                            "labels": [self.class_names[i] for i in labels_int],
                            "image_width": image.width,
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
        For each image, keep teacher boxes that the student also detects
        (same class, IoU >= agree_iou). Falls back to all teacher boxes when
        the student finds nothing, so we don't lose coverage on hard examples.
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
            ensembled.append(
                {
                    **td,
                    "boxes_xyxy": td["boxes_xyxy"][idx],
                    "scores": td["scores"][idx],
                    "labels": [td["labels"][i] for i in idx],
                }
            )
        return ensembled

    def run(
        self,
        image_paths: List[str],
        initial_detections: Optional[List[Dict]] = None,
        dataset_name: str = "self_train",
    ) -> str:
        """Run self-training rounds. Returns path to final trained model directory."""
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
            data_yaml = self.builder.build_dataset(
                detections, dataset_name=round_name
            )
            model_path = self.trainer.train(
                data_yaml=data_yaml, run_name=round_name, epochs=30
            )

            if r < self.rounds - 1:
                print("  Generating ensemble labels for next round…")
                teacher_dets = self.teacher.label_batch(
                    image_paths, self.class_names
                )
                student_dets = self._student_predict(model_path, image_paths)
                raw = self._ensemble(teacher_dets, student_dets)
                detections = self.filter.filter_batch(raw)
                print(f"  Ensemble kept {len(detections)} usable images")

        return model_path
