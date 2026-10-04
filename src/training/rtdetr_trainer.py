"""
RT-DETR fine-tuning trainer — Apache 2.0 replacement for the YOLOv8 trainer.

Uses HuggingFace transformers RTDetrForObjectDetection (PekingU/rtdetr_r50vd).
RT-DETR was originally developed by Baidu/PaddleDetection (Apache 2.0) and is
available here without requiring the PaddlePaddle framework.

Interface mirrors YOLOTrainer: accepts data.yaml + training hyperparams, returns
the path to the best saved checkpoint directory (loadable via from_pretrained).
"""
import json
import shutil
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
from tqdm import tqdm

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    _HAS_COCO = True
except ImportError:
    _HAS_COCO = False


# ── Dataset ────────────────────────────────────────────────────────────────────

class COCODetectionDataset(Dataset):
    """
    Loads a COCO JSON annotation file and serves RT-DETR-compatible batches.

    Converts COCO [x, y, w, h] pixel boxes to normalized [cx, cy, w, h]
    expected by RTDetrForObjectDetection labels during training.
    """

    def __init__(
        self,
        image_dir: str,
        annotation_file: str,
        processor: RTDetrImageProcessor,
    ):
        self.image_dir = Path(image_dir)
        self.processor = processor

        with open(annotation_file) as f:
            coco = json.load(f)

        self.id2img: Dict[int, dict] = {img["id"]: img for img in coco["images"]}
        self.ann_by_image: Dict[int, List[dict]] = {}
        for ann in coco["annotations"]:
            self.ann_by_image.setdefault(ann["image_id"], []).append(ann)

        self.image_ids: List[int] = [img["id"] for img in coco["images"]]

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> dict:
        img_id = self.image_ids[idx]
        img_info = self.id2img[img_id]

        image = Image.open(self.image_dir / img_info["file_name"]).convert("RGB")
        W, H = image.size

        anns = self.ann_by_image.get(img_id, [])
        boxes_norm, class_labels = [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            cx = (x + w / 2) / W
            cy = (y + h / 2) / H
            boxes_norm.append([cx, cy, w / W, h / H])
            # COCO category_id is 1-indexed; RT-DETR expects 0-indexed class_labels
            class_labels.append(ann["category_id"] - 1)

        inputs = self.processor(images=image, return_tensors="pt")
        label = {
            "class_labels": torch.tensor(class_labels, dtype=torch.long),
            "boxes": (
                torch.tensor(boxes_norm, dtype=torch.float32)
                if boxes_norm
                else torch.zeros((0, 4), dtype=torch.float32)
            ),
        }
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels": label,
        }


def _collate_fn(batch: List[dict]) -> dict:
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "labels": [b["labels"] for b in batch],
    }


# ── Trainer ────────────────────────────────────────────────────────────────────

class RTDETRTrainer:
    """
    Fine-tunes RT-DETR on a custom COCO-format dataset.

    Drop-in replacement for YOLOTrainer: same train() / evaluate() / predict()
    interface, and accepts the same data.yaml produced by COCOBuilder.

    The trained model is saved as a HuggingFace checkpoint directory so it can
    be reloaded with RTDetrForObjectDetection.from_pretrained(model_path).
    """

    def __init__(
        self,
        base_model: str = "PekingU/rtdetr_r50vd",
        output_dir: str = "models",
        device: Optional[str] = None,
    ):
        self.base_model = base_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ── helpers ────────────────────────────────────────────────────────────────

    def _parse_data_yaml(self, data_yaml: str) -> dict:
        """Read data.yaml written by COCOBuilder and return resolved paths."""
        with open(data_yaml) as f:
            cfg = yaml.safe_load(f)
        ds_root = Path(cfg["path"])
        return {
            "train_images": ds_root / cfg.get("train", "images/train"),
            "val_images": ds_root / cfg.get("val", "images/val"),
            "train_ann": ds_root / "annotations" / "instances_train.json",
            "val_ann": ds_root / "annotations" / "instances_val.json",
            "class_names": cfg["names"],
            "num_classes": cfg["nc"],
        }

    def _build_loader(
        self,
        image_dir: Path,
        ann_file: Path,
        processor: RTDetrImageProcessor,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        ds = COCODetectionDataset(str(image_dir), str(ann_file), processor)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=2,
            pin_memory=(self.device != "cpu"),
            collate_fn=_collate_fn,
        )

    # ── train ──────────────────────────────────────────────────────────────────

    def train(
        self,
        data_yaml: str,
        epochs: int = 50,
        batch_size: int = 8,
        imgsz: int = 640,
        run_name: str = "run",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        **kwargs,
    ) -> str:
        """
        Fine-tune RT-DETR on a custom dataset.
        Returns path to the saved model directory (loadable with from_pretrained).
        """
        cfg = self._parse_data_yaml(data_yaml)
        num_classes = cfg["num_classes"]

        processor = RTDetrImageProcessor.from_pretrained(
            self.base_model, size={"width": imgsz, "height": imgsz}
        )
        # ignore_mismatched_sizes resizes the classification head for num_classes
        model = RTDetrForObjectDetection.from_pretrained(
            self.base_model,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        ).to(self.device)

        train_loader = self._build_loader(
            cfg["train_images"], cfg["train_ann"], processor, batch_size, shuffle=True
        )
        val_loader = self._build_loader(
            cfg["val_images"], cfg["val_ann"], processor, batch_size, shuffle=False
        )

        optimizer = AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        total_steps = epochs * len(train_loader)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            total_steps=total_steps,
            pct_start=0.1,
        )

        save_dir = self.output_dir / run_name
        best_val_loss = float("inf")
        best_dir = save_dir / "best"

        print(
            f"\nTraining RT-DETR  model={self.base_model}"
            f"  classes={num_classes}  epochs={epochs}  device={self.device}"
        )

        for epoch in range(epochs):
            # ── train pass ────────────────────────────────────────────────────
            model.train()
            train_loss = 0.0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]"):
                pixel_values = batch["pixel_values"].to(self.device)
                labels = [
                    {k: v.to(self.device) for k, v in lbl.items()}
                    for lbl in batch["labels"]
                ]
                optimizer.zero_grad()
                outputs = model(pixel_values=pixel_values, labels=labels)
                outputs.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                optimizer.step()
                scheduler.step()
                train_loss += outputs.loss.item()

            avg_train = train_loss / len(train_loader)

            # ── val pass ──────────────────────────────────────────────────────
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [val]  "):
                    pixel_values = batch["pixel_values"].to(self.device)
                    labels = [
                        {k: v.to(self.device) for k, v in lbl.items()}
                        for lbl in batch["labels"]
                    ]
                    outputs = model(pixel_values=pixel_values, labels=labels)
                    val_loss += outputs.loss.item()

            avg_val = val_loss / max(len(val_loader), 1)
            print(
                f"  Epoch {epoch+1:3d}/{epochs}"
                f"  train_loss={avg_train:.4f}"
                f"  val_loss={avg_val:.4f}"
            )

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                if best_dir.exists():
                    shutil.rmtree(best_dir)
                model.save_pretrained(str(best_dir))
                processor.save_pretrained(str(best_dir))
                print(f"  Best model saved → {best_dir}")

        print(f"\nTraining complete. Best model: {best_dir}")
        return str(best_dir)

    # ── evaluate ───────────────────────────────────────────────────────────────

    def evaluate(self, model_path: str, data_yaml: str) -> Dict[str, float]:
        """
        Run COCO mAP evaluation on the val split using pycocotools.
        Returns mAP50-95, mAP50, and size-bucketed mAP values.
        """
        if not _HAS_COCO:
            print("pycocotools not installed — install it to get mAP metrics.")
            return {}

        cfg = self._parse_data_yaml(data_yaml)
        processor = RTDetrImageProcessor.from_pretrained(model_path)
        model = RTDetrForObjectDetection.from_pretrained(model_path).to(self.device)
        model.eval()

        with open(cfg["val_ann"]) as f:
            val_meta = json.load(f)
        id2fname = {img["id"]: img["file_name"] for img in val_meta["images"]}

        coco_gt = COCO(str(cfg["val_ann"]))
        results = []

        with torch.no_grad():
            for img_id, fname in tqdm(id2fname.items(), desc="Evaluating"):
                image = Image.open(cfg["val_images"] / fname).convert("RGB")
                inputs = processor(images=image, return_tensors="pt").to(self.device)
                outputs = model(**inputs)
                target_sizes = torch.tensor([[image.height, image.width]])
                preds = processor.post_process_object_detection(
                    outputs, target_sizes=target_sizes, threshold=0.01
                )[0]
                for box, score, label in zip(
                    preds["boxes"].cpu().numpy(),
                    preds["scores"].cpu().numpy(),
                    preds["labels"].cpu().numpy(),
                ):
                    x1, y1, x2, y2 = box.tolist()
                    results.append(
                        {
                            "image_id": img_id,
                            "category_id": int(label) + 1,  # back to 1-indexed
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "score": float(score),
                        }
                    )

        if not results:
            return {"mAP50": 0.0, "mAP50-95": 0.0}

        coco_dt = coco_gt.loadRes(results)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        return {
            "mAP50-95": float(coco_eval.stats[0]),
            "mAP50": float(coco_eval.stats[1]),
            "mAP_small": float(coco_eval.stats[3]),
            "mAP_medium": float(coco_eval.stats[4]),
            "mAP_large": float(coco_eval.stats[5]),
        }

    # ── predict ────────────────────────────────────────────────────────────────

    def predict(
        self,
        model_path: str,
        source: str,
        conf: float = 0.25,
        class_names: Optional[List[str]] = None,
    ) -> List[dict]:
        """
        Run inference on a single image path or a directory of images.
        Returns a list of detection dicts matching the pseudo-labeler format.
        """
        processor = RTDetrImageProcessor.from_pretrained(model_path)
        model = RTDetrForObjectDetection.from_pretrained(model_path).to(self.device)
        model.eval()

        source_path = Path(source)
        if source_path.is_dir():
            paths = list(source_path.glob("*.jpg")) + list(source_path.glob("*.png"))
        else:
            paths = [source_path]

        all_results = []
        with torch.no_grad():
            for p in tqdm(paths, desc="RT-DETR inference"):
                image = Image.open(p).convert("RGB")
                inputs = processor(images=image, return_tensors="pt").to(self.device)
                outputs = model(**inputs)
                target_sizes = torch.tensor([[image.height, image.width]])
                preds = processor.post_process_object_detection(
                    outputs, target_sizes=target_sizes, threshold=conf
                )[0]
                labels_int = preds["labels"].cpu().numpy()
                label_names = (
                    [class_names[i] for i in labels_int]
                    if class_names
                    else [str(i) for i in labels_int]
                )
                all_results.append(
                    {
                        "image_path": str(p),
                        "boxes_xyxy": preds["boxes"].cpu().numpy(),
                        "scores": preds["scores"].cpu().numpy(),
                        "labels": label_names,
                        "image_width": image.width,
                        "image_height": image.height,
                    }
                )
        return all_results
