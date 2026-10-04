# Label-Free Object Detection Fine-Tuning Pipeline (Apache 2.0)

Fine-tune an object detector on **any set of class names** — no annotated images, no COCO JSON, no bounding boxes required. Provide class names in plain text; the pipeline downloads training footage from YouTube, auto-generates labels using a zero-shot vision-language model, and fine-tunes RT-DETR end-to-end.

All components are **Apache 2.0** (or permissively licensed MIT/BSD). The only dependency that was AGPL-3.0 in comparable projects — `ultralytics` / YOLOv8 — is replaced here by RT-DETR via HuggingFace `transformers`.

---

## Pipeline Flowchart

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT                                   │
│          Class names (plain text, no labels needed)             │
│             e.g.  ["raccoon", "capybara"]                       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1 — Video Download          (yt-dlp, Unlicense)           │
│                                                                 │
│  For each class:                                                │
│    • Search YouTube: "<class> video", "<class> close up", etc.  │
│    • Download up to N videos (default 10 per class, ≤5 min)     │
│    • Videos saved to  data/videos/<class_name>/                 │
└───────────────────────────┬─────────────────────────────────────┘
                            │  .mp4 / .webm files
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2 — Frame Extraction        (OpenCV, Apache 2.0)          │
│                                                                 │
│  • Extract frames at target FPS (default 0.5 fps)               │
│  • Reject blurry frames via Laplacian variance threshold        │
│  • Cap at max_frames_per_video (default 100)                    │
│  • Frames saved to  data/frames/<class_name>/                   │
└───────────────────────────┬─────────────────────────────────────┘
                            │  .jpg frames
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3 — CLIP Pre-Filter         (MIT via HuggingFace)         │
│                                                                 │
│  • Encode each frame with CLIP image encoder                    │
│  • Encode class name with CLIP text encoder                     │
│    (3 prompts averaged: "a photo of a X", "X", etc.)            │
│  • Compute cosine similarity                                     │
│  • Discard frames below similarity_threshold (default 0.20)     │
│  • Cheap GPU pass — runs before the expensive detector          │
└───────────────────────────┬─────────────────────────────────────┘
                            │  filtered frames (irrelevant discarded)
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4 — Zero-Shot Pseudo-Labeling    (Apache 2.0)             │
│                                                                 │
│  Choice A — GroundingDINO  (IDEA-Research/grounding-dino-tiny)  │
│    • Text prompt: "raccoon. capybara."  (period-separated)      │
│    • Cross-modal fusion → grounded bounding boxes               │
│    • Returns boxes with text-matched labels + confidence scores │
│                                                                 │
│  Choice B — OWLv2  (google/owlv2-base-patch16-ensemble)         │
│    • Text queries: ["a photo of a raccoon", ...]                │
│    • ViT image backbone + per-query box regression              │
│    • Better for small/dense objects                             │
│                                                                 │
│  No training, no examples — text prompt → boxes at inference    │
└───────────────────────────┬─────────────────────────────────────┘
                            │  raw detections {image_path, boxes_xyxy,
                            │                  scores, labels}
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 5 — Label Quality Filtering  (numpy/scipy, BSD)           │
│                                                                 │
│  Per image:                                                     │
│  1. Confidence threshold  (drop score < box_threshold)          │
│  2. Size filter           (drop boxes too small or too large)   │
│  3. Per-class greedy NMS  (IoU threshold 0.5)                   │
│                                                                 │
│  Images with 0 remaining boxes are dropped entirely             │
└───────────────────────────┬─────────────────────────────────────┘
                            │  clean detections
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 6 — COCO Dataset Builder     (custom, Apache 2.0)         │
│                                                                 │
│  • Shuffle and split into train/val  (default 80/20)            │
│  • Copy images to  data/datasets/<name>/images/{train,val}/     │
│  • Write COCO JSON:                                             │
│      annotations/instances_train.json                           │
│      annotations/instances_val.json                             │
│  • Write data.yaml for dataset metadata                         │
│  • Fuzzy label matching (e.g. "raccoon animal" → "raccoon")     │
└───────────────────────────┬─────────────────────────────────────┘
                            │  data.yaml  +  COCO JSON files
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 7 — RT-DETR Fine-Tuning      (Apache 2.0)                 │
│                                                                 │
│  Base model: PekingU/rtdetr_r50vd  (RT-DETR R50, Apache 2.0)   │
│  Originally from Baidu/PaddleDetection — used here via HF       │
│                                                                 │
│  • Classification head resized for num_classes new classes      │
│  • Optimizer: AdamW  (lr=1e-4, weight_decay=1e-4)               │
│  • Scheduler: OneCycleLR  (10% warmup)                          │
│  • Loss: built-in Hungarian matching loss (computed internally)  │
│    = classification loss + bbox L1 loss + GIoU loss             │
│  • Best checkpoint saved by val loss → models/<name>/best/      │
│  • Checkpoint is a HuggingFace directory (from_pretrained-able) │
└───────────────────────────┬─────────────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              │  self_training.enabled?   │
              └──────┬──────────┬─────────┘
                   YES          NO
                    │            │
                    ▼            ▼
┌───────────────────────┐   ┌──────────────────────┐
│  SELF-TRAINING LOOP   │   │   TRAINED MODEL       │
│  (optional, rounds=2) │   │   models/<name>/best/ │
│                       │   └──────────────────────┘
│  Round 0: teacher     │
│   (GroundingDINO)     │
│   labels → train RT-  │
│   DETR student        │
│                       │
│  Round N: ensemble    │
│   teacher + student;  │
│   keep boxes where    │
│   both agree          │
│   (IoU ≥ 0.5)         │
│   → retrain           │
│                       │
│  Falls back to teacher│
│  when student finds   │
│  nothing (hard cases) │
└──────────┬────────────┘
           │
           ▼
   ┌──────────────────────┐
   │   TRAINED MODEL       │
   │   models/<name>/      │
   │     <name>_st_r1/best/│
   │     <name>_st_r2/best/│
   └──────────────────────┘
```

---

## Why This Approach Works

Traditional fine-tuning requires thousands of hand-labeled bounding boxes. This pipeline replaces manual labeling with three insights:

1. **Zero-shot detectors as teachers**: GroundingDINO and OWLv2 were trained on web-scale image-text pairs. They can detect objects by name without any task-specific examples — their pseudo-labels are noisy but statistically correct.

2. **Web video as free training data**: YouTube has footage of almost everything. Searching `"<class> video"` reliably surfaces content where the target class is prominently visible.

3. **Self-training refines noisy labels**: A student model trained on noisy pseudo-labels produces cleaner predictions. Ensembling teacher + student (keeping only agreed-upon boxes) iteratively improves label quality without any manual intervention.

---

## License Summary

| Component | Package | License |
|---|---|---|
| Video download | yt-dlp | Unlicense (public domain) |
| Frame extraction | opencv-python | Apache 2.0 |
| Frame relevance filter | CLIP via transformers | MIT |
| Zero-shot pseudo-labeler | GroundingDINO via transformers | Apache 2.0 |
| Zero-shot pseudo-labeler (alt) | OWLv2 via transformers | Apache 2.0 |
| Deep learning runtime | torch, torchvision | BSD 3-Clause |
| Model hub / transformer API | transformers (HuggingFace) | Apache 2.0 |
| COCO mAP evaluation | pycocotools | BSD 2-Clause |
| **Student detector (this project)** | **RT-DETR via transformers** | **Apache 2.0** |
| ~~Student detector (original)~~ | ~~ultralytics YOLOv8~~ | ~~AGPL-3.0~~ |

---

## Installation

```bash
# Python 3.9+  recommended
pip install -r requirements.txt
```

Requirements (no AGPL-3.0 packages):
```
yt-dlp>=2024.1.0
opencv-python>=4.8.0
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.38.0      # 4.38+ required for stable GroundingDINO API
Pillow>=10.0.0
numpy>=1.24.0
tqdm>=4.65.0
pyyaml>=6.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
pycocotools>=2.0.7
```

GPU recommended but not required (CPU inference is slow for GroundingDINO).

---

## Quick Start

### Run the full pipeline on custom classes

```bash
python scripts/run_pipeline.py --classes raccoon capybara --name my_run
```

This downloads videos, extracts frames, auto-labels with GroundingDINO, and fine-tunes RT-DETR. The trained model is saved to `models/my_run/best/`.

### Use a config file for more control

```bash
python scripts/run_pipeline.py --config configs/pipeline_config.yaml --name my_run
```

### Skip download (reuse already-extracted frames)

```bash
python scripts/run_pipeline.py --classes raccoon --skip-download --name my_run
```

### Evaluate mAP on the validation set

```bash
python scripts/evaluate.py \
    --model models/my_run/best \
    --data data/datasets/my_run/data.yaml
```

Output:
```
Evaluation results:
  mAP50-95    : 0.4231
  mAP50       : 0.6514
  mAP_small   : 0.1203
  mAP_medium  : 0.4387
  mAP_large   : 0.5921
```

### Inspect pseudo-labels visually

```bash
python scripts/visualize_labels.py \
    --images data/datasets/my_run/images/train \
    --annotations data/datasets/my_run/annotations/instances_train.json \
    --output data/visualizations \
    --n 20
```

### Run inference on new images

```python
from src.training.rtdetr_trainer import RTDETRTrainer

trainer = RTDETRTrainer()
results = trainer.predict(
    model_path="models/my_run/best",
    source="path/to/images/",
    conf=0.3,
    class_names=["raccoon", "capybara"],
)
for r in results:
    print(r["image_path"], r["labels"], r["scores"])
```

---

## Configuration Reference

`configs/pipeline_config.yaml`:

```yaml
# ── Classes ────────────────────────────────────────────────────
classes:
  - name: "raccoon"
    search_queries:             # optional; auto-generated if omitted
      - "raccoon backyard video"
      - "raccoon closeup wildlife"

# ── Video Download ─────────────────────────────────────────────
video_search:
  num_videos_per_class: 10     # YouTube videos to download per class
  max_duration_seconds: 300    # skip videos longer than this
  download_dir: "data/videos"

# ── Frame Extraction ───────────────────────────────────────────
frame_extraction:
  fps: 0.5                     # frames per second to extract (0.5 = 1 every 2s)
  max_frames_per_video: 100
  output_dir: "data/frames"
  min_blur_variance: 100       # Laplacian variance; lower = keep blurrier frames

# ── CLIP Pre-Filter ────────────────────────────────────────────
clip_filtering:
  enabled: true
  model: "openai/clip-vit-base-patch32"
  similarity_threshold: 0.20   # frames below this score are discarded
  batch_size: 32

# ── Pseudo-Labeling ────────────────────────────────────────────
pseudo_labeling:
  labeler: "grounding_dino"    # or "owl_vit"
  model: "IDEA-Research/grounding-dino-tiny"
  box_threshold: 0.30          # minimum box confidence to keep
  text_threshold: 0.25         # minimum text-grounding score (GroundingDINO only)
  batch_size: 8

# ── Annotation Builder ─────────────────────────────────────────
annotation:
  train_ratio: 0.8             # fraction of images used for training
  output_dir: "data/datasets"
  min_box_area_ratio: 0.001    # drop boxes smaller than 0.1% of image area
  max_box_area_ratio: 0.90     # drop boxes larger than 90% of image area

# ── RT-DETR Training ───────────────────────────────────────────
training:
  model: "PekingU/rtdetr_r50vd"  # HuggingFace model ID
  epochs: 50
  batch_size: 8                  # reduce to 4 if GPU OOM
  imgsz: 640
  learning_rate: 0.0001
  weight_decay: 0.0001
  output_dir: "models"

# ── Self-Training ──────────────────────────────────────────────
self_training:
  enabled: true
  rounds: 2                    # number of teacher-student refinement rounds
  confidence_threshold: 0.50   # student prediction threshold for ensemble
```

---

## Switching the Pseudo-Labeler

Change `pseudo_labeling.labeler` in the config:

| Value | Model | Best for |
|---|---|---|
| `"grounding_dino"` | `IDEA-Research/grounding-dino-tiny` | General use, rich text grounding |
| `"grounding_dino"` + larger model | `IDEA-Research/grounding-dino-base` | Higher accuracy, more VRAM |
| `"owl_vit"` | `google/owlv2-base-patch16-ensemble` | Small/dense objects |
| `"owl_vit"` + larger | `google/owlv2-large-patch14-ensemble` | Best accuracy |

---

## Switching the Student Model

Change `training.model` in the config to any RT-DETR checkpoint:

| Model ID | Speed | mAP COCO |
|---|---|---|
| `PekingU/rtdetr_r50vd` | Fast | ~53.1 |
| `PekingU/rtdetr_r101vd` | Medium | ~54.3 |
| `lyuwenyu/RT-DETR-R50` | Fast | ~53.1 |

All are Apache 2.0 licensed.

---

## Output Structure

After running the pipeline:

```
data/
  videos/<class>/           ← downloaded .mp4 files
  frames/<class>/           ← extracted .jpg frames
  datasets/<name>/
    images/
      train/                ← training images (copies)
      val/                  ← validation images (copies)
    annotations/
      instances_train.json  ← COCO-format training annotations
      instances_val.json    ← COCO-format validation annotations
    data.yaml               ← dataset metadata

models/<name>/
  best/                     ← best checkpoint (HuggingFace directory)
    config.json
    model.safetensors
    preprocessor_config.json
```

The checkpoint in `models/<name>/best/` is a standard HuggingFace directory. Load it anywhere:

```python
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

model = RTDetrForObjectDetection.from_pretrained("models/my_run/best")
processor = RTDetrImageProcessor.from_pretrained("models/my_run/best")
```

---

## Project Structure

```
object_detection_finetunning_apache_2/
│
├── requirements.txt
├── configs/
│   ├── pipeline_config.yaml      ← main config (edit this)
│   └── classes_example.yaml      ← example class definitions
│
├── scripts/
│   ├── run_pipeline.py           ← CLI entry point
│   ├── evaluate.py               ← COCO mAP evaluation
│   └── visualize_labels.py       ← draw pseudo-labels on images
│
└── src/
    ├── pipeline.py               ← top-level orchestrator
    │
    ├── video_pipeline/
    │   ├── downloader.py         ← yt-dlp YouTube search + download
    │   └── frame_extractor.py    ← OpenCV frame extraction + blur filter
    │
    ├── pseudo_labeling/
    │   ├── clip_filter.py        ← CLIP image-text similarity pre-filter
    │   ├── grounding_dino.py     ← GroundingDINO zero-shot labeler
    │   └── owl_vit.py            ← OWLv2 zero-shot labeler (alternative)
    │
    ├── annotation/
    │   ├── quality_filter.py     ← NMS + confidence + size filtering
    │   └── coco_builder.py       ← writes COCO JSON + data.yaml
    │
    └── training/
        ├── rtdetr_trainer.py     ← RT-DETR fine-tuning, eval, inference
        └── self_trainer.py       ← teacher-student iterative refinement
```

---

## How the Self-Training Loop Works

```
Initial pseudo-labels from GroundingDINO
           │
           ▼
    ┌─────────────┐
    │  Round 0    │
    │  Train RT-  │
    │  DETR on    │
    │  teacher    │
    │  labels     │
    └──────┬──────┘
           │ student model v0
           ▼
    ┌──────────────────────────────────────────┐
    │  Round 1                                 │
    │                                          │
    │  Teacher (GroundingDINO) runs on images  │
    │  Student v0 runs on same images          │
    │                                          │
    │  For each image, for each teacher box:   │
    │    if any student box agrees             │
    │    (same class AND IoU ≥ 0.5):           │
    │      → KEEP this box (high confidence)  │
    │    else:                                 │
    │      → keep teacher box anyway          │
    │        (don't lose hard examples)        │
    │                                          │
    │  Ensemble labels → train RT-DETR v1      │
    └──────────────────────────────────────────┘
           │ student model v1
           ▼
    ┌─────────────┐
    │  Round 2    │  (same ensemble logic)
    │  ...        │
    └─────────────┘
```

The ensemble logic is conservative: it keeps teacher boxes even when the student disagrees, so coverage on hard or rare examples is never lost.

---

## Troubleshooting

**No pseudo-labels generated**
- Lower `pseudo_labeling.box_threshold` (try 0.20)
- Lower `clip_filtering.similarity_threshold` (try 0.15) or disable CLIP filtering
- Try `labeler: "owl_vit"` as an alternative

**GPU out of memory during training**
- Reduce `training.batch_size` to 4 or 2
- Use a smaller model: `PekingU/rtdetr_r50vd` instead of `r101vd`
- Reduce `training.imgsz` to 512

**GPU out of memory during pseudo-labeling**
- Reduce `pseudo_labeling.batch_size` to 4
- Reduce `clip_filtering.batch_size` to 16

**Poor mAP after training**
- Increase `video_search.num_videos_per_class` (more training data)
- Enable self-training: `self_training.enabled: true`
- Lower `pseudo_labeling.box_threshold` to get more (noisier) pseudo-labels
- Increase `training.epochs`

**GroundingDINO import error**
- Requires `transformers>=4.38.0`: `pip install "transformers>=4.38.0"`
