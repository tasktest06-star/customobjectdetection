# Execution Instructions

Step-by-step guide to set up, run, and use the Apache 2.0 object detection fine-tuning pipeline.

---

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Installation](#2-installation)
3. [Define Your Classes](#3-define-your-classes)
4. [Configure the Pipeline](#4-configure-the-pipeline)
5. [Run the Full Pipeline](#5-run-the-full-pipeline)
6. [Run Individual Stages](#6-run-individual-stages)
7. [Evaluate a Trained Model](#7-evaluate-a-trained-model)
8. [Run Inference](#8-run-inference)
9. [Visualize Pseudo-Labels](#9-visualize-pseudo-labels)
10. [Common Workflows](#10-common-workflows)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. System Requirements

| Resource | Minimum | Recommended |
|---|---|---|
| Python | 3.10 | 3.11 |
| GPU | 4 GB VRAM | 8 GB VRAM |
| RAM | 8 GB | 16 GB |
| Disk | 20 GB | 50 GB |
| OS | Linux / macOS | Linux |
| CUDA | 11.8 | 12.1 |

CPU-only mode is supported but pseudo-labeling will be 10-20x slower.

---

## 2. Installation

```bash
# Clone the repository
git clone https://github.com/tasktest06-star/customobjectdetection.git
cd customobjectdetection
git checkout apache2-rtdetr-pipeline

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate.bat       # Windows

# Install all dependencies (Apache 2.0 compliant, no ultralytics)
pip install -r requirements.txt

# Verify installation — confirm no ultralytics import
python -c "from src.pipeline import ObjectDetectionPipeline; print('OK')"
grep -r 'ultralytics' src/ scripts/ && echo "FAIL" || echo "No AGPL code: OK"
```

**Note on PyTorch:** The requirements.txt installs the CPU build of PyTorch.
For GPU acceleration, install the CUDA build first:

```bash
# CUDA 12.1 (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Verify GPU is detected:
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

---

## 3. Define Your Classes

Create a classes YAML file listing the objects you want to detect.

### Option A: Use the built-in example format

```bash
cat configs/classes_example.yaml
```

```yaml
# configs/my_classes.yaml
classes:
  - name: raccoon
    search_queries:
      - "raccoon wildlife footage"
      - "raccoon backyard close up"
      - "raccoon eating food"

  - name: capybara
    search_queries:
      - "capybara in water"
      - "capybara close up"
      - "capybara herd"
```

### Option B: Minimal format (auto-generates search queries)

```yaml
# configs/my_classes.yaml
classes:
  - name: red fox
  - name: white tailed deer
```

When `search_queries` is omitted, the pipeline generates three queries:
- `"red fox video"`
- `"red fox close up"`
- `"how to identify red fox"`

### Tips for class names

- Use specific names: `"bald eagle"` > `"eagle"` (fewer false positives)
- Multi-word classes work: `"garbage truck"`, `"mountain lion"`
- Avoid very general nouns: `"animal"`, `"bird"` (GroundingDINO will match too broadly)
- For rare classes, write more specific search queries targeting clear, isolated shots

---

## 4. Configure the Pipeline

Edit `configs/pipeline_config.yaml`. Key settings to tune:

```yaml
# ── Video settings ─────────────────────────────────────────────────────────────
video_search:
  num_videos_per_class: 10    # increase to 15-20 for rare classes
  max_duration_seconds: 300   # 5 min max; increase to 600 for documentary content
  download_dir: "data/videos"

# ── Frame extraction ───────────────────────────────────────────────────────────
frame_extraction:
  fps: 0.5                    # 1 frame every 2 seconds; increase to 1.0 for action-heavy content
  max_frames_per_video: 100   # cap per video; increase to 200 if dataset is too small
  min_blur_variance: 100.0    # decrease to 50.0 if too many frames are rejected as blurry

# ── CLIP pre-filter ────────────────────────────────────────────────────────────
clip_filtering:
  enabled: true               # set false to skip (useful for debugging)
  similarity_threshold: 0.20  # lower to keep more frames; raise to 0.25 if too noisy

# ── Pseudo-labeling ────────────────────────────────────────────────────────────
pseudo_labeling:
  labeler: "grounding_dino"   # or "owl_vit"
  model: "IDEA-Research/grounding-dino-tiny"  # or "grounding-dino-base" for better accuracy
  box_threshold: 0.30         # lower to 0.20 if too few labels are generated
  text_threshold: 0.25        # keep slightly below box_threshold

# ── Quality filter ─────────────────────────────────────────────────────────────
annotation:
  min_box_area_ratio: 0.001   # 0.1% of image area minimum
  max_box_area_ratio: 0.90    # 90% maximum (rejects full-image ghost detections)
  train_ratio: 0.8            # 80% train, 20% val

# ── Training ───────────────────────────────────────────────────────────────────
training:
  model: "PekingU/rtdetr_r50vd"      # base model; "rtdetr_r101vd" for higher accuracy
  epochs: 50                          # reduce to 20 for quick experiments
  batch_size: 8                       # reduce to 4 if GPU OOM
  imgsz: 640
  learning_rate: 0.0001
  weight_decay: 0.0001
  output_dir: "models"

# ── Self-training ──────────────────────────────────────────────────────────────
self_training:
  enabled: false              # set true to enable teacher-student refinement
  rounds: 2                   # number of iterative refinement rounds
```

---

## 5. Run the Full Pipeline

### Basic run

First, add your classes to `configs/pipeline_config.yaml` under the `classes:` key
(see Section 3). Then run:

```bash
python scripts/run_pipeline.py \
    --config configs/pipeline_config.yaml \
    --name my_detector
```

Alternatively, override classes directly on the command line (space-separated names):

```bash
python scripts/run_pipeline.py \
    --config configs/pipeline_config.yaml \
    --classes raccoon capybara \
    --name my_detector
```

This will:
1. Download ~10 YouTube videos per class to `data/videos/`
2. Extract frames to `data/frames/`
3. Filter frames with CLIP (if enabled)
4. Generate pseudo-labels with GroundingDINO
5. Build a COCO dataset at `data/datasets/my_detector/`
6. Fine-tune RT-DETR, saving the best model to `models/my_detector/best/`

### Re-run training only (skip video download)

If you already have frames from a previous run:

```bash
python scripts/run_pipeline.py \
    --config configs/pipeline_config.yaml \
    --classes configs/my_classes.yaml \
    --name my_detector_v2 \
    --skip-download
```

### Run with self-training enabled

Edit `pipeline_config.yaml`:
```yaml
self_training:
  enabled: true
  rounds: 2
```

Then run the same command. Training time roughly doubles per round.

---

## 6. Run Individual Stages

### Stage 1-2: Download and extract frames only

```python
from src.video_pipeline.downloader import YouTubeDownloader
from src.video_pipeline.frame_extractor import FrameExtractor

downloader = YouTubeDownloader("data/videos")
extractor = FrameExtractor("data/frames", fps=0.5, max_frames=100)

# Download for one class
videos = downloader.search_and_download(
    "raccoon",
    search_queries=["raccoon backyard", "raccoon wildlife"],
    num_videos=5,
    max_duration=300,
)

# Extract frames
frames = extractor.extract_all(videos, "raccoon")
print(f"Extracted {len(frames)} frames")
```

### Stage 3: CLIP filter only

```python
from src.pseudo_labeling.clip_filter import CLIPFilter

cf = CLIPFilter(similarity_threshold=0.20)
kept, scores = cf.filter_frames(frames, "raccoon")
print(f"Kept {len(kept)}/{len(frames)} frames")
```

### Stage 4: Pseudo-labeling only

```python
from src.pseudo_labeling.grounding_dino import GroundingDINOLabeler

labeler = GroundingDINOLabeler(box_threshold=0.30)
detections = labeler.label_batch(kept, ["raccoon"])
print(f"Labeled {len(detections)} images")
print(f"Example: {detections[0]['boxes_xyxy'].shape} boxes in first image")
```

### Stage 4 (alternative): Use OWLv2 instead

```python
from src.pseudo_labeling.owl_vit import OWLv2Labeler

labeler = OWLv2Labeler(
    model_id="google/owlv2-base-patch16-ensemble",
    score_threshold=0.20,
)
detections = labeler.label_batch(frames, ["raccoon"])
```

### Stage 5-6: Filter and build COCO dataset

```python
from src.annotation.quality_filter import LabelQualityFilter
from src.annotation.coco_builder import COCOBuilder

q_filter = LabelQualityFilter(min_score=0.30)
clean = q_filter.filter_batch(detections)
print(f"Usable images: {len(clean)}")

builder = COCOBuilder(["raccoon"], output_dir="data/datasets")
data_yaml = builder.build_dataset(clean, "raccoon_v1")
print(f"Dataset: {data_yaml}")
```

### Stage 7: Train RT-DETR only

```python
from src.training.rtdetr_trainer import RTDETRTrainer

trainer = RTDETRTrainer(base_model="PekingU/rtdetr_r50vd")
model_path = trainer.train(
    data_yaml="data/datasets/raccoon_v1/data.yaml",
    epochs=50,
    batch_size=8,
    run_name="raccoon_v1",
)
print(f"Model saved to: {model_path}")
```

---

## 7. Evaluate a Trained Model

```bash
python scripts/evaluate.py \
    --model models/my_detector/best \
    --data data/datasets/my_detector/data.yaml
```

Expected output:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.352
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.541
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.374
 ...
mAP50: 0.541   mAP50-95: 0.352
```

Or from Python:

```python
from src.training.rtdetr_trainer import RTDETRTrainer

trainer = RTDETRTrainer()
metrics = trainer.evaluate(
    model_path="models/my_detector/best",
    data_yaml="data/datasets/my_detector/data.yaml",
)
print(metrics)
# {"mAP50-95": 0.352, "mAP50": 0.541, ...}
```

---

## 8. Run Inference

### On a single image

```python
from src.training.rtdetr_trainer import RTDETRTrainer

trainer = RTDETRTrainer()
results = trainer.predict(
    model_path="models/my_detector/best",
    source="path/to/image.jpg",
    conf=0.25,
    class_names=["raccoon", "capybara"],
)

r = results[0]
print(f"Found {len(r['boxes_xyxy'])} objects")
for box, label, score in zip(r["boxes_xyxy"], r["labels"], r["scores"]):
    print(f"  {label}: {score:.2f}  box={box.tolist()}")
```

### On a directory of images

```python
results = trainer.predict(
    model_path="models/my_detector/best",
    source="data/datasets/my_detector/images/val",
    conf=0.25,
    class_names=["raccoon"],
)
print(f"Ran on {len(results)} images")
```

### Load model anywhere (portability)

The model is saved as a HuggingFace checkpoint directory — loadable with the
standard `from_pretrained()` API, no custom code needed:

```python
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
from PIL import Image
import torch

model_dir = "models/my_detector/best"
processor = RTDetrImageProcessor.from_pretrained(model_dir)
model = RTDetrForObjectDetection.from_pretrained(model_dir)
model.eval()

image = Image.open("test.jpg").convert("RGB")
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

target_sizes = torch.tensor([[image.height, image.width]])
results = processor.post_process_object_detection(
    outputs, target_sizes=target_sizes, threshold=0.25
)[0]
print(results["boxes"], results["scores"], results["labels"])
```

---

## 9. Visualize Pseudo-Labels

Before training, it is important to check that pseudo-labels look reasonable.
A few bad labels are acceptable; systematic false positives indicate the
box_threshold needs to be raised.

```bash
# Visualize labels for the training split
python scripts/visualize_labels.py \
    --dataset data/datasets/my_detector \
    --split train \
    --output visualizations/ \
    --max-images 20
```

This saves annotated images to `visualizations/` with bounding boxes drawn.

What to look for:
- Boxes should tightly enclose the target object
- Labels should match the object in the box
- Very large boxes (covering most of the image) indicate noise → increase box_threshold
- Very few boxes per image is OK; zero boxes per image means CLIP or the labeler
  threshold is too strict

---

## 10. Common Workflows

### Workflow A: Quick experiment (no download)

Use pre-existing images from disk instead of downloading videos:

```bash
# Put your images in data/frames/<class_name>/
mkdir -p data/frames/raccoon
cp /path/to/raccoon_images/*.jpg data/frames/raccoon/

# Run with skip-download
python scripts/run_pipeline.py \
    --config configs/pipeline_config.yaml \
    --classes configs/my_classes.yaml \
    --name test_run \
    --skip-download
```

### Workflow B: Switch labeler to OWLv2

In `pipeline_config.yaml`:
```yaml
pseudo_labeling:
  labeler: "owl_vit"
  model: "google/owlv2-base-patch16-ensemble"
  box_threshold: 0.20          # OWLv2 needs a lower threshold than GroundingDINO
```

OWLv2 tends to work better for:
- Small or densely packed objects
- Classes where GroundingDINO returns too few detections

### Workflow C: Higher accuracy model (more VRAM)

In `pipeline_config.yaml`:
```yaml
pseudo_labeling:
  model: "IDEA-Research/grounding-dino-base"  # 341M params vs 172M for tiny

training:
  model: "PekingU/rtdetr_r101vd"              # ResNet-101 backbone
  batch_size: 4                                # reduce batch size for larger model
```

### Workflow D: Enable self-training

In `pipeline_config.yaml`:
```yaml
self_training:
  enabled: true
  rounds: 2
```

Expected improvement: +3–8 mAP50 over single-round training, at ~2–3x training time cost.

### Workflow E: CPU-only (no GPU)

The pipeline will automatically use CPU if no GPU is detected.
To force CPU even when a GPU is present (for testing):

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # hide all GPUs

from src.pipeline import ObjectDetectionPipeline
pipeline = ObjectDetectionPipeline("configs/pipeline_config.yaml")
```

Expect 10-20x slower pseudo-labeling and 5-10x slower training.

---

## 11. Troubleshooting

### Out of Memory (CUDA OOM) during training

Reduce `batch_size` in `pipeline_config.yaml`:
```yaml
training:
  batch_size: 4   # try 4, then 2 if still OOM
```

Or switch to a smaller base model:
```yaml
training:
  model: "PekingU/rtdetr_r18vd"   # ResNet-18 backbone, ~30% less VRAM
```

### Too few pseudo-labels generated

Symptoms: `Usable annotated images: 0/N` or very low number.

Solutions (try in order):
1. Lower `box_threshold` to 0.20
2. Lower CLIP `similarity_threshold` to 0.15 or set `enabled: false`
3. Add more specific `search_queries` with better visibility (close-up shots)
4. Increase `num_videos_per_class`
5. Check visualizations — the labeler may be working but quality_filter is too strict

### Many false positive detections

Symptoms: boxes on background, wrong objects labeled.

Solutions:
1. Raise `box_threshold` to 0.40 or 0.50
2. Raise CLIP `similarity_threshold` to 0.25
3. Check `search_queries` — make them more specific to the visual appearance
4. Switch from `grounding_dino` to `owl_vit` (sometimes more precise)

### `RuntimeError: Expected all tensors to be on the same device`

This should not occur in the current codebase (all `target_sizes` tensors have
been fixed to use `device=self.device`). If it appears, check that you are
running the latest code from the `apache2-rtdetr-pipeline` branch.

### `ModuleNotFoundError: No module named 'pycocotools'`

```bash
pip install pycocotools>=2.0.7
```

On macOS, may need: `brew install gcc` first.
On Windows, may need Microsoft C++ Build Tools.

### `yt-dlp` download fails

YouTube changes its API frequently. Update yt-dlp:
```bash
pip install --upgrade yt-dlp
```

If a specific video fails, it may be geo-restricted or private — this is handled
gracefully by `ignoreerrors: true` in the downloader configuration.

### Models not saving to `models/`

Check that the `output_dir` in `pipeline_config.yaml` is writable and that the
training loop is reaching at least one epoch where validation loss improves.
If val loss never improves (stays at inf), check that the val split has images:
```bash
ls data/datasets/my_detector/images/val/ | wc -l
```
If the count is 0, lower `train_ratio` or add more training data.

---

## Output Structure

After a complete pipeline run:

```
models/
  my_detector/
    best/
      config.json               # RT-DETR architecture config
      model.safetensors         # trained weights
      preprocessor_config.json  # image normalization constants

data/
  videos/
    raccoon/
      abc123.mp4  def456.webm   # downloaded videos

  frames/
    raccoon/
      abc123_000060.jpg  ...    # extracted frames (0.5 fps)

  datasets/
    my_detector/
      images/train/  images/val/      # copied frames
      annotations/
        instances_train.json          # COCO format
        instances_val.json
      data.yaml                       # dataset config for RTDETRTrainer
```
