# Execution Instructions

Step-by-step guide to run the auto-label object detection pipeline from a clean machine to a trained model.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Environment Setup](#2-environment-setup)
3. [Project Setup](#3-project-setup)
4. [Configure Your Classes](#4-configure-your-classes)
5. [Run the Full Pipeline](#5-run-the-full-pipeline)
6. [Run Step by Step](#6-run-step-by-step)
7. [Inspect Pseudo-Labels](#7-inspect-pseudo-labels)
8. [Evaluate the Model](#8-evaluate-the-model)
9. [Run Inference](#9-run-inference)
10. [Expected Outputs](#10-expected-outputs)
11. [Runtime Estimates](#11-runtime-estimates)
12. [Common Errors & Fixes](#12-common-errors--fixes)
13. [Re-running Parts of the Pipeline](#13-re-running-parts-of-the-pipeline)

---

## 1. Prerequisites

### System requirements

| Component | Minimum | Recommended |
|---|---|---|
| Python | 3.9 | 3.11 |
| RAM | 8 GB | 16 GB |
| Disk space | 20 GB free | 50 GB free |
| GPU VRAM | — (CPU works) | 6 GB (NVIDIA) |
| Internet | Required for download & model weights | Stable broadband |

### Required system packages

**Ubuntu / Debian**
```bash
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv git ffmpeg
```

**macOS**
```bash
brew install python ffmpeg
```

**Windows**
```
1. Install Python 3.11 from https://python.org/downloads
2. Install ffmpeg: https://ffmpeg.org/download.html
   Add ffmpeg/bin to your PATH environment variable
3. Use PowerShell or Git Bash for all commands below
```

Verify ffmpeg is on your PATH:
```bash
ffmpeg -version
# Should print: ffmpeg version 6.x ...
```

### GPU (optional but strongly recommended)

Check if CUDA is available:
```bash
nvidia-smi
# Should print GPU name and VRAM
```

If no GPU, the pipeline runs on CPU — expect 5–10× slower labeling.

---

## 2. Environment Setup

### Create a Python virtual environment

```bash
# Navigate to the project root
cd /workshop/object_detection_finetunning

# Create virtual environment
python3 -m venv .venv

# Activate it
# Linux / macOS:
source .venv/bin/activate

# Windows:
.venv\Scripts\activate
```

Your prompt should now show `(.venv)`.

### Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs (~2–5 min depending on connection):
- `yt-dlp` — YouTube downloader
- `transformers` — Grounding DINO, OWLv2, CLIP models
- `ultralytics` — YOLOv8
- `torch` + `torchvision` — deep learning backend
- `opencv-python` — frame extraction

### Verify the install

```bash
python3 -c "
import torch, transformers, ultralytics, yt_dlp, cv2
print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())
print('transformers:', transformers.__version__)
print('ultralytics:', ultralytics.__version__)
print('yt-dlp:', yt_dlp.version.__version__)
print('opencv:', cv2.__version__)
"
```

Expected output (versions may vary):
```
torch: 2.1.0 | CUDA: True
transformers: 4.36.0
ultralytics: 8.0.220
yt-dlp: 2024.01.07
opencv: 4.8.1
```

---

## 3. Project Setup

All commands below assume you are in the project root with the virtual environment active:

```bash
cd /workshop/object_detection_finetunning
source .venv/bin/activate   # Linux/macOS
```

Verify the project structure is intact:
```bash
find . -name "*.py" | sort
```

Expected output:
```
./scripts/evaluate.py
./scripts/run_pipeline.py
./scripts/visualize_labels.py
./src/__init__.py
./src/annotation/__init__.py
./src/annotation/coco_builder.py
./src/annotation/quality_filter.py
./src/pipeline.py
./src/pseudo_labeling/__init__.py
./src/pseudo_labeling/clip_filter.py
./src/pseudo_labeling/grounding_dino.py
./src/pseudo_labeling/owl_vit.py
./src/training/__init__.py
./src/training/self_trainer.py
./src/training/yolo_trainer.py
./src/video_pipeline/__init__.py
./src/video_pipeline/downloader.py
./src/video_pipeline/frame_extractor.py
```

---

## 4. Configure Your Classes

Open `configs/pipeline_config.yaml` and edit the `classes` section:

```yaml
classes:
  - name: "YOUR_CLASS_NAME"
    search_queries:
      - "your class name closeup video"
      - "your class name in the wild"
```

### Example: detecting raccoons and foxes

```yaml
classes:
  - name: "raccoon"
    search_queries:
      - "raccoon backyard night video"
      - "raccoon closeup wildlife"
      - "raccoon eating food"
  - name: "fox"
    search_queries:
      - "red fox wildlife video"
      - "fox running field"
```

### Adjusting key parameters

For a **quick test run** (fewer videos, faster training):
```yaml
video_search:
  num_videos_per_class: 3      # default: 10

frame_extraction:
  max_frames_per_video: 30     # default: 100

training:
  epochs: 10                   # default: 50
  model: "yolov8n.pt"          # nano = fastest

self_training:
  enabled: false               # disable for first run
```

For a **production run** (higher quality):
```yaml
video_search:
  num_videos_per_class: 20

frame_extraction:
  fps: 1.0
  max_frames_per_video: 200

pseudo_labeling:
  model: "IDEA-Research/grounding-dino-base"   # larger model
  box_threshold: 0.30

training:
  epochs: 100
  model: "yolov8s.pt"

self_training:
  enabled: true
  rounds: 3
```

---

## 5. Run the Full Pipeline

### Option A — Use config file (recommended)

```bash
python scripts/run_pipeline.py \
  --config configs/pipeline_config.yaml \
  --name my_first_run
```

### Option B — Override classes on the command line

```bash
python scripts/run_pipeline.py \
  --classes raccoon fox "red panda" \
  --name raccoon_fox_run
```

### Option C — Skip video download (reuse existing frames)

Useful if you already ran the download step and want to re-label or re-train:

```bash
python scripts/run_pipeline.py \
  --skip-download \
  --name my_first_run
```

### What you will see in the terminal

```
============================================================
Pipeline starting  |  classes: ['raccoon', 'fox']
============================================================

[Download] raccoon
  Downloading: raccoon backyard night video ...
  8 videos downloaded
  Extracting [raccoon]: 100%|████| 8/8
  743 frames extracted

[Download] fox
  Downloading: red fox wildlife video ...
  7 videos downloaded
  Extracting [fox]: 100%|████| 7/7
  612 frames extracted

[CLIP filter]
CLIP filter [raccoon]: 100%|████| 24/24
CLIP: kept 541/743 frames
CLIP filter [fox]: 100%|████| 20/20
CLIP: kept 489/612 frames

Frames after CLIP filter: 1030

[Pseudo-labeling] using GroundingDINOLabeler
Loading Grounding DINO: IDEA-Research/grounding-dino-tiny on cuda
Grounding DINO labeling: 100%|████| 1030/1030

[Quality filtering]
Usable annotated images: 814/1030

[Build dataset]
  train: 651 images, 1204 boxes
  val:   163 images, 298 boxes
  Dataset written to data/datasets/my_first_run

[Train YOLOv8]
Epoch   1/50: ...
...
Epoch  50/50: box_loss=1.234  cls_loss=0.567  mAP50=0.612

============================================================
Done.  Model: models/my_first_run/weights/best.pt
============================================================
```

---

## 6. Run Step by Step

If you want to run each stage separately (useful for debugging or iterating):

### Step 1 — Download videos only

```python
# run interactively or save as a script
from src.video_pipeline.downloader import YouTubeDownloader

dl = YouTubeDownloader(download_dir="data/videos")
videos = dl.search_and_download(
    class_name="raccoon",
    search_queries=["raccoon backyard video", "raccoon wildlife closeup"],
    num_videos=5,
    max_duration=300,
)
print(f"Downloaded: {videos}")
```

### Step 2 — Extract frames

```python
from src.video_pipeline.frame_extractor import FrameExtractor

extractor = FrameExtractor(
    output_dir="data/frames",
    fps=0.5,
    max_frames=100,
    min_blur_variance=100,
)
frames = extractor.extract_all(videos, class_name="raccoon")
print(f"Extracted {len(frames)} frames")
```

### Step 3 — CLIP filter

```python
from src.pseudo_labeling.clip_filter import CLIPFilter

cf = CLIPFilter(similarity_threshold=0.20)
kept_frames, scores = cf.filter_frames(frames, class_name="raccoon")
print(f"Kept {len(kept_frames)} / {len(frames)} frames")
```

### Step 4 — Generate pseudo-labels

```python
from src.pseudo_labeling.grounding_dino import GroundingDINOLabeler

labeler = GroundingDINOLabeler(
    model_id="IDEA-Research/grounding-dino-tiny",
    box_threshold=0.30,
)
raw_detections = labeler.label_batch(kept_frames, class_names=["raccoon"])
print(f"Labeled {len(raw_detections)} images")
```

### Step 5 — Quality filter

```python
from src.annotation.quality_filter import LabelQualityFilter

qf = LabelQualityFilter(min_score=0.30)
clean_detections = qf.filter_batch(raw_detections)
print(f"Clean: {len(clean_detections)} images with boxes")
```

### Step 6 — Build COCO dataset

```python
from src.annotation.coco_builder import COCOBuilder

builder = COCOBuilder(
    class_names=["raccoon"],
    output_dir="data/datasets",
    train_ratio=0.8,
)
data_yaml = builder.build_dataset(clean_detections, dataset_name="raccoon_v1")
print(f"Dataset YAML: {data_yaml}")
```

### Step 7 — Train YOLOv8

```python
from src.training.yolo_trainer import YOLOTrainer

trainer = YOLOTrainer(base_model="yolov8n.pt", output_dir="models")
model_path = trainer.train(
    data_yaml=data_yaml,
    epochs=50,
    batch_size=16,
    imgsz=640,
    run_name="raccoon_v1",
)
print(f"Model: {model_path}")
```

### Step 8 — Self-training (optional)

```python
from src.training.self_trainer import SelfTrainer

st = SelfTrainer(
    class_names=["raccoon"],
    teacher=labeler,
    quality_filter=qf,
    coco_builder=builder,
    yolo_trainer=trainer,
    rounds=2,
)
final_model = st.run(
    image_paths=kept_frames,
    initial_detections=clean_detections,
    dataset_name="raccoon_self_train",
)
print(f"Final model: {final_model}")
```

---

## 7. Inspect Pseudo-Labels

**Always do this before training.** It takes 30 seconds and immediately shows whether the pseudo-labels are useful.

```bash
python scripts/visualize_labels.py \
  --images data/datasets/my_first_run/images/train \
  --annotations data/datasets/my_first_run/annotations/instances_train.json \
  --output data/visualizations \
  --n 30
```

Open the images in `data/visualizations/`. You should see bounding boxes drawn on objects.

### What good labels look like

```
✅ Box tightly wraps the object
✅ Correct class name shown
✅ Score ≥ 0.30
✅ No box on background/sky/floor
```

### What bad labels look like — and how to fix them

| What you see | Fix |
|---|---|
| Almost no boxes | Lower `box_threshold` to `0.20`, lower `similarity_threshold` to `0.15` |
| Boxes on wrong objects | Raise `box_threshold` to `0.40`, make search queries more specific |
| Many duplicate overlapping boxes | Lower `nms_iou` to `0.30` in `quality_filter` section |
| Boxes covering 80%+ of image | Raise `min_box_area_ratio`, already handled by `max_box_area_ratio: 0.90` |
| Only 1 class showing up | Check your `search_queries` for the missing class — make them more specific |

---

## 8. Evaluate the Model

After training, evaluate on the validation set:

```bash
python scripts/evaluate.py \
  --model models/my_first_run/weights/best.pt \
  --data data/datasets/my_first_run/data.yaml
```

Output:
```
Evaluation results:
  mAP50       : 0.6120
  mAP50-95    : 0.4380
  precision   : 0.7210
  recall      : 0.5890
```

### Interpreting metrics

| Metric | What it means | Good range |
|---|---|---|
| `mAP50` | Mean average precision at IoU=0.50 | > 0.50 is usable, > 0.70 is good |
| `mAP50-95` | Averaged over IoU 0.50–0.95 (stricter) | > 0.35 is usable |
| `precision` | Of all detections made, fraction that were correct | > 0.70 |
| `recall` | Of all actual objects, fraction detected | > 0.60 |

> Note: these metrics are computed on pseudo-labeled validation data, not human-labeled ground truth. Actual real-world performance may differ.

### Compare self-training rounds

```bash
# Round 0 (base)
python scripts/evaluate.py \
  --model models/my_first_run_st_r0/weights/best.pt \
  --data data/datasets/my_first_run_st_r0/data.yaml

# Round 1
python scripts/evaluate.py \
  --model models/my_first_run_st_r1/weights/best.pt \
  --data data/datasets/my_first_run_st_r1/data.yaml
```

---

## 9. Run Inference

### On a single image

```bash
python3 -c "
from ultralytics import YOLO
model = YOLO('models/my_first_run/weights/best.pt')
results = model.predict('path/to/your/image.jpg', conf=0.25, save=True)
print('Saved to:', results[0].save_dir)
"
```

### On a directory of images

```bash
python3 -c "
from ultralytics import YOLO
model = YOLO('models/my_first_run/weights/best.pt')
model.predict('path/to/image/folder/', conf=0.25, save=True)
"
```

### On a video file

```bash
python3 -c "
from ultralytics import YOLO
model = YOLO('models/my_first_run/weights/best.pt')
model.predict('path/to/video.mp4', conf=0.25, save=True)
"
```

### On webcam (live)

```bash
python3 -c "
from ultralytics import YOLO
model = YOLO('models/my_first_run/weights/best.pt')
model.predict(source=0, conf=0.25, show=True)   # 0 = default webcam
"
```

Results are saved to `runs/detect/predict*/`.

### Using the trainer wrapper

```python
from src.training.yolo_trainer import YOLOTrainer

trainer = YOLOTrainer()
trainer.predict(
    model_path="models/my_first_run/weights/best.pt",
    source="path/to/image.jpg",
    conf=0.25,
)
```

---

## 10. Expected Outputs

After a full pipeline run with 2 classes and default settings, you will find:

```
data/
├── videos/
│   ├── raccoon/
│   │   ├── abc123.mp4           ← downloaded videos
│   │   └── def456.mp4
│   └── fox/
│       └── ghi789.mp4
│
├── frames/
│   ├── raccoon/
│   │   ├── abc123_000060.jpg    ← extracted frames
│   │   └── abc123_000180.jpg
│   └── fox/
│       └── ghi789_000060.jpg
│
└── datasets/
    └── my_first_run/
        ├── images/
        │   ├── train/           ← 80% of annotated frames
        │   └── val/             ← 20% of annotated frames
        ├── annotations/
        │   ├── instances_train.json    ← COCO format
        │   └── instances_val.json
        └── data.yaml                   ← YOLO training config

models/
└── my_first_run/
    ├── weights/
    │   ├── best.pt              ← USE THIS for inference
    │   └── last.pt
    ├── results.csv              ← per-epoch metrics
    └── confusion_matrix.png     ← visual evaluation
```

---

## 11. Runtime Estimates

These are approximate, vary by internet speed, GPU, and number of classes.

### With GPU (RTX 3060 / 6GB VRAM)

| Step | Time per class | Notes |
|---|---|---|
| Video download (10 videos) | 3–8 min | Depends on YouTube speed |
| Frame extraction | 1–2 min | CPU-bound |
| CLIP filter (500 frames) | 30 sec | Batch on GPU |
| Grounding DINO labeling (500 frames) | 5–10 min | One image at a time |
| Quality filtering | < 5 sec | Pure Python |
| Dataset build | 30 sec | File copy |
| YOLOv8 training (50 epochs) | 8–15 min | GPU |
| **Total (2 classes, no self-training)** | **~45–60 min** | |
| **Self-training (2 rounds)** | **+30–40 min** | Per round |

### Without GPU (CPU only)

| Step | Multiplier |
|---|---|
| CLIP filter | ~5× slower |
| Grounding DINO | ~8–12× slower |
| YOLOv8 training | ~10× slower |
| **Total (2 classes)** | **~5–8 hours** |

---

## 12. Common Errors & Fixes

### Installation errors

**`ERROR: No matching distribution found for torch>=2.0.0`**
```bash
# Install PyTorch manually first from pytorch.org
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

**`ModuleNotFoundError: No module named 'cv2'`**
```bash
pip install opencv-python-headless   # use this on servers without display
```

---

### Download errors

**`ERROR: Sign in to confirm you're not a bot`**

YouTube is rate-limiting. Solutions:
```bash
# Option 1: add cookies from your browser
# In downloader.py, add to ydl_opts:
"cookiesfrombrowser": ("chrome",),

# Option 2: use a VPN or wait 30 min before retrying

# Option 3: reduce num_videos_per_class to 3-5
```

**`WARNING: unable to download video`**

Some videos are geo-restricted or deleted. This is normal — yt-dlp skips them automatically. If too many are skipped, add more search queries.

---

### Labeling errors

**`RuntimeError: No usable pseudo-labels generated`**

The detector found no confident detections. Fix:
```yaml
# In pipeline_config.yaml:
pseudo_labeling:
  box_threshold: 0.20       # was 0.30 — lower threshold

clip_filtering:
  similarity_threshold: 0.15  # was 0.20 — keep more frames
  enabled: false              # temporarily disable to diagnose
```

**`CUDA out of memory`**
```yaml
clip_filtering:
  batch_size: 8              # was 32

pseudo_labeling:
  model: "IDEA-Research/grounding-dino-tiny"   # not base
```

**`KeyError: 'boxes'` during Grounding DINO**

Transformers version mismatch. Fix:
```bash
pip install transformers>=4.35.0 --upgrade
```

---

### Training errors

**`Dataset not found` during YOLOv8 training**

The path in `data.yaml` is absolute. If you moved the project folder, rebuild the dataset:
```bash
python scripts/run_pipeline.py --skip-download --name my_first_run
```

**`AssertionError: No labels found in ...`**

The COCO builder ran but produced no valid annotations. Check:
```bash
python3 -c "
import json
with open('data/datasets/my_first_run/annotations/instances_train.json') as f:
    d = json.load(f)
print('images:', len(d['images']))
print('annotations:', len(d['annotations']))
print('categories:', d['categories'])
"
```
If annotations is 0, the label matching failed — verify class names in config exactly match what Grounding DINO returns.

---

## 13. Re-running Parts of the Pipeline

### Re-label with different thresholds (keep existing frames)

```bash
python scripts/run_pipeline.py \
  --skip-download \
  --name my_first_run_v2
```

Then edit config thresholds, run again with `--name my_first_run_v3`.

### Re-train on existing dataset

```python
from src.training.yolo_trainer import YOLOTrainer

trainer = YOLOTrainer(base_model="yolov8s.pt")   # try a bigger model
trainer.train(
    data_yaml="data/datasets/my_first_run/data.yaml",
    epochs=100,
    run_name="my_first_run_retrain",
)
```

### Add a new class to an existing dataset

1. Add the new class to `configs/pipeline_config.yaml`
2. Run with `--skip-download` if existing class frames are already present (only new class will download)

```bash
python scripts/run_pipeline.py \
  --name expanded_run \
  --classes raccoon fox "new_class"
```

### Run self-training standalone on existing detections

```python
import pickle
from src.training.self_trainer import SelfTrainer
from src.pseudo_labeling.grounding_dino import GroundingDINOLabeler
from src.annotation.quality_filter import LabelQualityFilter
from src.annotation.coco_builder import COCOBuilder
from src.training.yolo_trainer import YOLOTrainer
from pathlib import Path

class_names = ["raccoon"]
image_paths = [str(p) for p in Path("data/frames/raccoon").rglob("*.jpg")]

labeler    = GroundingDINOLabeler()
qf         = LabelQualityFilter()
builder    = COCOBuilder(class_names, output_dir="data/datasets")
trainer    = YOLOTrainer()

st = SelfTrainer(class_names, labeler, qf, builder, trainer, rounds=2)
model = st.run(image_paths, dataset_name="raccoon_st")
print("Final model:", model)
```

---

## Quick Reference Card

```
# Setup
source .venv/bin/activate

# Full run (default config)
python scripts/run_pipeline.py --name run1

# Full run (custom classes)
python scripts/run_pipeline.py --classes raccoon fox --name run1

# Retrain only (reuse existing frames)
python scripts/run_pipeline.py --skip-download --name run1

# Check pseudo-labels visually
python scripts/visualize_labels.py \
  --images data/datasets/run1/images/train \
  --annotations data/datasets/run1/annotations/instances_train.json \
  --output data/visualizations

# Evaluate model
python scripts/evaluate.py \
  --model models/run1/weights/best.pt \
  --data data/datasets/run1/data.yaml

# Infer on image
python3 -c "
from ultralytics import YOLO
YOLO('models/run1/weights/best.pt').predict('image.jpg', save=True, conf=0.25)
"
```
