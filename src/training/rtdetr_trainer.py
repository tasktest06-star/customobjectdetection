"""
RT-DETR fine-tuning trainer — Apache 2.0 replacement for the YOLOv8 trainer.

RT-DETR (Real-Time Detection TRansformer) was originally developed by Baidu as
part of PaddleDetection (Apache 2.0). This implementation uses the HuggingFace
transformers port, so no PaddlePaddle dependency is required — everything stays
in the PyTorch + transformers ecosystem that the rest of this pipeline already uses.

Key design decisions:
  - Classification head is replaced for num_classes using ignore_mismatched_sizes=True,
    so the backbone (ResNet-50) weights are reused and only the head is re-initialized.
  - Loss is computed by the model itself (Hungarian bipartite matching + classification
    + L1 bbox + GIoU bbox losses). We just call outputs.loss.backward().
  - Checkpoints are saved as HuggingFace directories (config.json + safetensors),
    not .pt files, so they are loadable with from_pretrained() anywhere.
  - OneCycleLR with a short 10% warmup period prevents gradient instability at the
    start of fine-tuning when the new classification head weights are random.

Interface mirrors YOLOTrainer for drop-in compatibility:
  train(data_yaml, ...) → model_dir
  evaluate(model_path, data_yaml) → metrics_dict
  predict(model_path, source, ...) → detections_list
"""

import json
import shutil
import yaml
import torch
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
from tqdm import tqdm

# pycocotools gives standard COCO mAP metrics; gracefully degrade if missing
try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    _HAS_COCO = True
except ImportError:
    _HAS_COCO = False


# ── Dataset ────────────────────────────────────────────────────────────────────

class COCODetectionDataset(Dataset):
    """
    PyTorch Dataset that reads a COCO-format JSON annotation file and serves
    batches in the format expected by RTDetrForObjectDetection.

    COCO annotation format vs RT-DETR label format
    ───────────────────────────────────────────────
    COCO stores bboxes as [x_min, y_min, width, height] in pixel coordinates.
    RT-DETR expects labels as dicts with:
      "class_labels" : LongTensor of shape (N,)   — 0-indexed class IDs
      "boxes"        : FloatTensor of shape (N, 4) — normalized [cx, cy, w, h]

    This dataset performs the conversion in __getitem__:
      cx = (x + w/2) / img_width       # center x, normalized to [0,1]
      cy = (y + h/2) / img_height      # center y, normalized to [0,1]
      nw = w / img_width               # width,    normalized to [0,1]
      nh = h / img_height              # height,   normalized to [0,1]

    Category IDs in COCO are 1-indexed (category 1 = first class).
    RT-DETR class_labels are 0-indexed. We subtract 1 on load.
    """

    def __init__(
        self,
        image_dir: str,
        annotation_file: str,
        processor: RTDetrImageProcessor,
    ):
        """
        Args:
            image_dir:       Directory containing the image files.
            annotation_file: Path to a COCO JSON file (instances_train.json etc.).
            processor:       RTDetrImageProcessor used to resize/normalize images.
                             Must be the same processor instance used during training
                             so that image size and normalization are consistent.
        """
        self.image_dir = Path(image_dir)
        self.processor = processor

        # Load full COCO JSON into memory (typically tens of MB — fine for custom datasets)
        with open(annotation_file) as f:
            coco = json.load(f)

        # Build an id→image_info lookup so __getitem__ doesn't scan the full list
        self.id2img: Dict[int, dict] = {img["id"]: img for img in coco["images"]}

        # Group annotations by image_id for O(1) per-image lookup
        self.ann_by_image: Dict[int, List[dict]] = {}
        for ann in coco["annotations"]:
            self.ann_by_image.setdefault(ann["image_id"], []).append(ann)

        # Ordered list of image IDs — defines the dataset length and index mapping
        self.image_ids: List[int] = [img["id"] for img in coco["images"]]

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns a dict with:
          "pixel_values" : FloatTensor (3, H, W)  — normalized image
          "labels"       : dict with "class_labels" and "boxes"

        Images with no annotations return an empty boxes tensor (shape (0,4)).
        RT-DETR's Hungarian loss handles zero-annotation images correctly.
        """
        img_id = self.image_ids[idx]
        img_info = self.id2img[img_id]

        # Load image as RGB (OpenCV-saved frames are BGR but PIL reads correctly)
        image = Image.open(self.image_dir / img_info["file_name"]).convert("RGB")
        W, H = image.size  # PIL size is (width, height)

        anns = self.ann_by_image.get(img_id, [])
        boxes_norm, class_labels = [], []

        for ann in anns:
            x, y, w, h = ann["bbox"]  # COCO: top-left corner + dimensions, pixels

            # Convert to center format and normalize to [0, 1]
            cx = (x + w / 2) / W
            cy = (y + h / 2) / H
            boxes_norm.append([cx, cy, w / W, h / H])

            # COCO category_id is 1-indexed; RT-DETR expects 0-indexed class_labels
            class_labels.append(ann["category_id"] - 1)

        # Resize and normalize the image using the processor
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
            # squeeze(0): processor adds a batch dim; we remove it here so the
            # DataLoader's default collate can re-stack them into a proper batch
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels": label,
        }


def _collate_fn(batch: List[dict]) -> dict:
    """
    Custom collate function for the DataLoader.

    pixel_values can be stacked into a single tensor because the processor
    resizes all images to the same (H, W).

    labels cannot be stacked — each image has a variable number of objects —
    so they are kept as a plain Python list. RTDetrForObjectDetection.forward()
    accepts a list of label dicts, one per image.
    """
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "labels": [b["labels"] for b in batch],  # variable-length; kept as list
    }


# ── Trainer ────────────────────────────────────────────────────────────────────

class RTDETRTrainer:
    """
    Wraps RTDetrForObjectDetection for fine-tuning, evaluation, and inference.

    Drop-in replacement for the original YOLOTrainer. Accepts the same data.yaml
    produced by COCOBuilder, returns the same detection dict format, and exposes
    the same train() / evaluate() / predict() interface.

    The trained model is saved as a HuggingFace checkpoint directory:
      models/<run_name>/best/
        config.json            ← model architecture + num_labels
        model.safetensors      ← weights
        preprocessor_config.json ← image size, normalization constants

    This directory can be reloaded anywhere with:
      RTDetrForObjectDetection.from_pretrained("models/my_run/best")
    """

    def __init__(
        self,
        base_model: str = "PekingU/rtdetr_r50vd",
        output_dir: str = "models",
        device: Optional[str] = None,
    ):
        """
        Args:
            base_model:  HuggingFace model ID for the pretrained RT-DETR checkpoint.
                         All PekingU/rtdetr_* checkpoints are Apache 2.0 licensed.
                         Larger variants (rtdetr_r101vd) give better mAP at the
                         cost of memory and speed.
            output_dir:  Root directory under which run-specific subdirs are created.
            device:      "cuda", "cpu", or None (auto-detect).
        """
        self.base_model = base_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ── internal helpers ───────────────────────────────────────────────────────

    def _parse_data_yaml(self, data_yaml: str) -> dict:
        """
        Read the data.yaml written by COCOBuilder and return resolved file paths.

        data.yaml structure (produced by COCOBuilder):
          path: /abs/path/to/dataset
          train: images/train
          val:   images/val
          nc:    2
          names: [raccoon, capybara]

        Returns a dict with absolute Path objects ready for use.
        """
        with open(data_yaml) as f:
            cfg = yaml.safe_load(f)
        ds_root = Path(cfg["path"])
        return {
            "train_images": ds_root / cfg.get("train", "images/train"),
            "val_images":   ds_root / cfg.get("val",   "images/val"),
            # COCOBuilder always writes these annotation paths relative to ds_root
            "train_ann":    ds_root / "annotations" / "instances_train.json",
            "val_ann":      ds_root / "annotations" / "instances_val.json",
            "class_names":  cfg["names"],
            "num_classes":  cfg["nc"],
        }

    def _build_loader(
        self,
        image_dir: Path,
        ann_file: Path,
        processor: RTDetrImageProcessor,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        """Create a DataLoader for one split (train or val)."""
        ds = COCODetectionDataset(str(image_dir), str(ann_file), processor)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=2,           # parallel frame loading; increase on fast SSDs
            pin_memory=(self.device != "cpu"),  # speeds up CPU→GPU transfer
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
        Fine-tune RT-DETR on a COCO-format custom dataset.

        Training strategy:
          1. Load pretrained RT-DETR backbone (ResNet-50 + transformer encoder/decoder).
          2. Replace the classification head with a fresh head for num_classes classes.
             ignore_mismatched_sizes=True tells HuggingFace to skip the class head
             weights and re-initialize them randomly.
          3. Fine-tune all weights with AdamW + OneCycleLR.
             OneCycleLR ramps LR up for 10% of steps (warmup), then anneals it down.
             This is more stable than a flat LR when the class head starts from random.
          4. After every epoch, compute val loss. Save checkpoint when val loss improves.

        Args:
            data_yaml:      Path to data.yaml produced by COCOBuilder.
            epochs:         Number of full passes over the training set.
            batch_size:     Images per GPU step. Reduce to 4 if GPU OOM.
            imgsz:          Image size (square). RT-DETR default is 640.
            run_name:       Subdirectory name under output_dir.
            learning_rate:  Peak LR for OneCycleLR.
            weight_decay:   L2 regularization coefficient for AdamW.

        Returns:
            Path to the best checkpoint directory (str).
        """
        cfg = self._parse_data_yaml(data_yaml)
        num_classes = cfg["num_classes"]

        # ── build processor ─────────────────────────────────────────────────
        # The processor handles: resize to (imgsz, imgsz), pixel normalization
        # (ImageNet mean/std), and conversion to tensors. It is saved alongside
        # the model weights so inference uses the exact same preprocessing.
        processor = RTDetrImageProcessor.from_pretrained(
            self.base_model, size={"width": imgsz, "height": imgsz}
        )

        # ── load model ──────────────────────────────────────────────────────
        # num_labels tells the model how many output classes to expect.
        # ignore_mismatched_sizes=True: the pretrained head has 80 classes (COCO);
        # our custom head has num_classes classes. HuggingFace discards the old
        # head weights and initializes the new head from scratch. All other weights
        # (backbone, encoder, decoder) are loaded from the pretrained checkpoint.
        model = RTDetrForObjectDetection.from_pretrained(
            self.base_model,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        ).to(self.device)

        # ── dataloaders ─────────────────────────────────────────────────────
        train_loader = self._build_loader(
            cfg["train_images"], cfg["train_ann"], processor, batch_size, shuffle=True
        )
        val_loader = self._build_loader(
            cfg["val_images"], cfg["val_ann"], processor, batch_size, shuffle=False
        )

        # ── optimizer + scheduler ───────────────────────────────────────────
        # AdamW decouples the weight decay from the gradient update, which
        # prevents decay from being scaled by the adaptive LR — important
        # for transformers where gradients vary widely across layers.
        optimizer = AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # OneCycleLR: increases LR linearly for pct_start * total_steps steps
        # (warmup phase), then decreases it via cosine annealing. This warmup
        # prevents the randomly-initialized class head from pushing gradients
        # through the whole network at full LR before it has stabilized.
        total_steps = epochs * len(train_loader)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            total_steps=total_steps,
            pct_start=0.1,   # 10% of steps = warmup
        )

        save_dir = self.output_dir / run_name
        best_val_loss = float("inf")
        best_dir = save_dir / "best"  # checkpoint saved here when val loss improves

        print(
            f"\nTraining RT-DETR  model={self.base_model}"
            f"  classes={num_classes}  epochs={epochs}  device={self.device}"
        )

        for epoch in range(epochs):
            # ── train pass ────────────────────────────────────────────────
            model.train()
            train_loss = 0.0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]"):
                # Move inputs to the target device
                pixel_values = batch["pixel_values"].to(self.device)

                # Labels is a list of dicts (one per image); move each tensor to device
                labels = [
                    {k: v.to(self.device) for k, v in lbl.items()}
                    for lbl in batch["labels"]
                ]

                optimizer.zero_grad()

                # Forward pass: passing labels triggers loss computation internally.
                # RT-DETR loss = classification loss (focal) + L1 bbox + GIoU bbox,
                # combined via Hungarian bipartite matching between predictions and GT.
                outputs = model(pixel_values=pixel_values, labels=labels)

                outputs.loss.backward()

                # Gradient clipping prevents exploding gradients in the transformer
                # layers, which is especially important early in training.
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)

                optimizer.step()
                scheduler.step()  # OneCycleLR steps per batch, not per epoch
                train_loss += outputs.loss.item()

            avg_train = train_loss / len(train_loader)

            # ── validation pass ───────────────────────────────────────────
            model.eval()
            val_loss = 0.0

            with torch.no_grad():  # no gradient tracking needed for evaluation
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

            # Save checkpoint whenever validation loss improves.
            # We track val_loss (not train_loss) to prefer generalization.
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                if best_dir.exists():
                    shutil.rmtree(best_dir)  # remove previous best before overwriting
                # save_pretrained writes config.json + model.safetensors
                model.save_pretrained(str(best_dir))
                # processor saves preprocessor_config.json (image size, normalization)
                processor.save_pretrained(str(best_dir))
                print(f"  Best model saved → {best_dir}")

        print(f"\nTraining complete. Best model: {best_dir}")
        return str(best_dir)

    # ── evaluate ───────────────────────────────────────────────────────────────

    def evaluate(self, model_path: str, data_yaml: str) -> Dict[str, float]:
        """
        Run standard COCO object detection evaluation on the validation split.

        Evaluation procedure:
          1. Run inference on every val image at a very low threshold (0.01) so that
             the COCO evaluator sees the full precision-recall curve.
          2. Collect results in COCO detection result format:
             [{"image_id": int, "category_id": int, "bbox": [x,y,w,h], "score": float}]
          3. Pass to COCOeval, which computes the standard 12-metric summary.

        Returns:
            Dict with mAP50-95, mAP50, and size-bucketed mAP values.
            Empty dict if pycocotools is not installed.

        Note on category_id: RT-DETR outputs 0-indexed labels. COCO GT JSON
        has 1-indexed category_id. We add 1 back before calling loadRes().
        """
        if not _HAS_COCO:
            print("pycocotools not installed — install it to get mAP metrics.")
            return {}

        cfg = self._parse_data_yaml(data_yaml)
        processor = RTDetrImageProcessor.from_pretrained(model_path)
        model = RTDetrForObjectDetection.from_pretrained(model_path).to(self.device)
        model.eval()

        # Load image metadata from the val annotation file
        with open(cfg["val_ann"]) as f:
            val_meta = json.load(f)
        id2fname = {img["id"]: img["file_name"] for img in val_meta["images"]}

        # COCO ground-truth object — used by COCOeval for GT annotations
        coco_gt = COCO(str(cfg["val_ann"]))
        results = []  # will hold all per-box prediction dicts

        with torch.no_grad():
            for img_id, fname in tqdm(id2fname.items(), desc="Evaluating"):
                image = Image.open(cfg["val_images"] / fname).convert("RGB")
                inputs = processor(images=image, return_tensors="pt").to(self.device)
                outputs = model(**inputs)

                # target_sizes tells the processor how to rescale predicted boxes
                # from normalized [0,1] back to pixel coordinates.
                # MUST be on the same device as model outputs to avoid device mismatch.
                target_sizes = torch.tensor(
                    [[image.height, image.width]], device=self.device
                )
                preds = processor.post_process_object_detection(
                    outputs, target_sizes=target_sizes, threshold=0.01
                )[0]

                for box, score, label in zip(
                    preds["boxes"].cpu().numpy(),   # xyxy pixel coords
                    preds["scores"].cpu().numpy(),
                    preds["labels"].cpu().numpy(),  # 0-indexed
                ):
                    x1, y1, x2, y2 = box.tolist()
                    results.append(
                        {
                            "image_id":    img_id,
                            "category_id": int(label) + 1,  # convert back to 1-indexed
                            "bbox":        [x1, y1, x2 - x1, y2 - y1],  # COCO [x,y,w,h]
                            "score":       float(score),
                        }
                    )

        if not results:
            return {"mAP50": 0.0, "mAP50-95": 0.0}

        # loadRes() builds a COCO detection object from the results list
        coco_dt = coco_gt.loadRes(results)

        # COCOeval computes the full 12-metric suite:
        #   stats[0] = mAP @ IoU=0.50:0.95 (primary metric)
        #   stats[1] = mAP @ IoU=0.50
        #   stats[2] = mAP @ IoU=0.75
        #   stats[3] = mAP for small objects  (area < 32²)
        #   stats[4] = mAP for medium objects (32² < area < 96²)
        #   stats[5] = mAP for large objects  (area > 96²)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()  # also prints the full table to stdout

        return {
            "mAP50-95": float(coco_eval.stats[0]),
            "mAP50":    float(coco_eval.stats[1]),
            "mAP_small":  float(coco_eval.stats[3]),
            "mAP_medium": float(coco_eval.stats[4]),
            "mAP_large":  float(coco_eval.stats[5]),
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
        Run inference on a single image file or a directory of images.

        Args:
            model_path:   Path to a saved HuggingFace checkpoint directory.
            source:       Path to a single image (.jpg/.png) or a directory.
            conf:         Minimum confidence score to include a detection.
            class_names:  Optional list of class name strings to map label IDs to.
                          If None, labels are returned as string integers ("0", "1"…).

        Returns:
            List of detection dicts, one per image:
            {
              "image_path"  : str,
              "boxes_xyxy"  : np.ndarray (N, 4) — pixel coordinates
              "scores"      : np.ndarray (N,)
              "labels"      : List[str]          — class name or str(id)
              "image_width" : int,
              "image_height": int,
            }
            This format matches the pseudo-labeler output so results can be
            fed back into the quality filter or COCO builder if needed.
        """
        processor = RTDetrImageProcessor.from_pretrained(model_path)
        model = RTDetrForObjectDetection.from_pretrained(model_path).to(self.device)
        model.eval()

        # Collect image paths: handle both single file and directory inputs
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

                # Rescale predicted normalized boxes back to pixel coordinates.
                # target_sizes must be on the same device as the model outputs.
                target_sizes = torch.tensor(
                    [[image.height, image.width]], device=self.device
                )
                preds = processor.post_process_object_detection(
                    outputs, target_sizes=target_sizes, threshold=conf
                )[0]

                labels_int = preds["labels"].cpu().numpy()
                # Map integer label IDs to human-readable class names if provided
                label_names = (
                    [class_names[i] for i in labels_int]
                    if class_names
                    else [str(i) for i in labels_int]
                )

                all_results.append(
                    {
                        "image_path":   str(p),
                        "boxes_xyxy":   preds["boxes"].cpu().numpy(),
                        "scores":       preds["scores"].cpu().numpy(),
                        "labels":       label_names,
                        "image_width":  image.width,
                        "image_height": image.height,
                    }
                )
        return all_results
