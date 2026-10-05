# Label-Free RT-DETR Fine-Tuning Pipeline

Fine-tune an RT-DETR object detector for **custom classes without any manually labeled data**.
You only provide class names. The pipeline searches YouTube for training videos, extracts frames,
uses zero-shot foundation models to auto-generate bounding-box annotations, and fine-tunes
RT-DETR end-to-end.

All components are **Apache 2.0 / MIT / BSD** licensed — no AGPL dependencies.

---

## Pipeline Flowchart

```
 You provide
 ┌──────────────┐
 │  Class names │   e.g. ["raccoon", "capybara"]
 │  (text only) │
 └──────┬───────┘
        │
        ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │  STAGE 1 — Video Search & Download                              │
 │  Tool: yt-dlp (Unlicense)                                       │
 │  • Searches YouTube: "{class} wildlife footage", etc.           │
 │  • Downloads N videos per class (default: 10, max 5 min each)  │
 │  • Output: data/videos/{class}/video_*.mp4                      │
 └──────────────────────────┬───────────────────────────────────────┘
                            │
                            ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │  STAGE 2 — Frame Extraction                                     │
 │  Tool: OpenCV (Apache 2.0)                                      │
 │  • Samples 1 frame every N seconds (default: 0.5 fps)          │
 │  • Drops blurry frames via Laplacian variance threshold         │
 │  • Output: data/frames/{class}/{video}_{frame:06d}.jpg          │
 └──────────────────────────┬───────────────────────────────────────┘
                            │
                            ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │  STAGE 3 — CLIP Pre-Filter  (optional)                          │
 │  Tool: openai/clip-vit-base-patch32 (MIT)                       │
 │  • Computes image-text cosine similarity                        │
 │  • Drops frames with score < threshold (default: 0.20)         │
 │  • Removes off-topic frames before the expensive labeler runs   │
 └──────────────────────────┬───────────────────────────────────────┘
                            │
                            ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │  STAGE 4 — Zero-Shot Pseudo-Labeling  (pick one backend)       │
 │                                                                  │
 │  ┌──────────────────┐  ┌───────────────┐  ┌──────────────────┐ │
 │  │  GroundingDINO   │  │    OWLv2      │  │  Qwen2-VL /      │ │
 │  │  (default)       │  │  (dense/small │  │  InternVL2       │ │
 │  │  Apache 2.0      │  │   objects)    │  │  (ambiguous      │ │
 │  │  IDEA-Research   │  │  Apache 2.0   │  │   classes)       │ │
 │  │                  │  │  Google       │  │  Apache 2.0      │ │
 │  └──────────────────┘  └───────────────┘  └──────────────────┘ │
 │                                                                  │
 │  Input:  class name strings  →  Output: bbox + class per frame  │
 └──────────────────────────┬───────────────────────────────────────┘
                            │
                            ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │  STAGE 5 — Quality Filter                                       │
 │  • Confidence threshold (drops low-score boxes)                 │
 │  • Min/max area ratio (drops sub-pixel noise & ghost boxes)     │
 │  • Per-class NMS (removes duplicate detections)                 │
 │  • Drops images with zero remaining boxes                       │
 └──────────────────────────┬───────────────────────────────────────┘
                            │
                            ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │  STAGE 6 — COCO Dataset Builder                                 │
 │  • Shuffles + splits 80/20 train/val                            │
 │  • Copies images with class-prefixed filenames (no collisions)  │
 │  • Writes instances_train.json / instances_val.json             │
 │  • Writes data.yaml for trainers                                │
 │  Output: data/datasets/{name}/{images,annotations}/             │
 └──────────────────────────┬───────────────────────────────────────┘
                            │
              ┌─────────────┴──────────────┐
              │  Self-training enabled?    │
              └──────┬──────────┬──────────┘
                    NO          YES
                    │           │
                    │           ▼
                    │  ┌────────────────────────────────────────┐
                    │  │  Teacher-Student Self-Training Loop    │
                    │  │  (rounds=2 default)                    │
                    │  │                                        │
                    │  │  Round 0..N-2 (intermediate_epochs):  │
                    │  │  1. Train student on teacher labels    │
                    │  │  2. Student re-annotates all frames    │
                    │  │  3. Ensemble: keep all teacher boxes   │
                    │  │     (conservative — never drops)       │
                    │  │  4. Quality filter ensemble output     │
                    │  │                                        │
                    │  │  Round N-1 (final_epochs):             │
                    │  │  Train student on final ensemble set   │
                    │  └────────────────┬───────────────────────┘
                    │                   │
                    ▼                   ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │  STAGE 7 — RT-DETR Fine-Tuning  (pick one backend)             │
 │                                                                  │
 │  ┌───────────────────────────────┐  ┌────────────────────────┐  │
 │  │  HuggingFace (backend="hf")  │  │  PaddleDetection        │  │
 │  │  PyTorch + transformers      │  │  (backend="paddle")     │  │
 │  │  Easiest setup               │  │  Native PaddlePaddle,   │  │
 │  │  PekingU/rtdetr_r50vd        │  │  multi-GPU, TRT export  │  │
 │  │  Apache 2.0                  │  │  Apache 2.0             │  │
 │  └───────────────────────────────┘  └────────────────────────┘  │
 │                                                                  │
 │  • ignore_mismatched_sizes=True: reuses COCO backbone,          │
 │    re-initialises classification head for your num_classes      │
 │  • AdamW + OneCycleLR, 10% linear warmup                        │
 │  • Saves best checkpoint by val loss                            │
 └──────────────────────────┬───────────────────────────────────────┘
                            │
                            ▼
                  ┌──────────────────┐
                  │  Trained Model   │
                  │  models/{name}/  │
                  │  best/           │
                  └──────────────────┘
```

---

## Directory Structure

```
.
├── configs/
│   ├── pipeline_config.yaml      # Main config — edit this to run your classes
│   └── classes_example.yaml      # Multi-class example
├── paddle_configs/
│   └── rtdetr_r50vd_custom.yml   # Standalone PaddleDetection config template
├── requirements.txt
├── scripts/
│   ├── run_pipeline.py           # CLI entry point
│   ├── evaluate.py               # Run COCO mAP evaluation on a trained model
│   └── visualize_labels.py       # Draw pseudo-label boxes on frames
└── src/
    ├── pipeline.py               # Orchestrator (stages 1-7)
    ├── video_pipeline/
    │   ├── downloader.py         # yt-dlp wrapper
    │   └── frame_extractor.py    # OpenCV frame sampling + blur filter
    ├── pseudo_labeling/
    │   ├── clip_filter.py        # CLIP pre-filter
    │   ├── grounding_dino.py     # GroundingDINO labeler
    │   ├── owl_vit.py            # OWLv2 labeler
    │   └── vlm_labeler.py        # Qwen2-VL / InternVL2 labelers
    ├── annotation/
    │   ├── quality_filter.py     # Confidence + NMS + area filter
    │   └── coco_builder.py       # COCO JSON + dataset layout builder
    └── training/
        ├── rtdetr_trainer.py     # HuggingFace RT-DETR trainer
        ├── paddle_rtdetr_trainer.py  # PaddleDetection subprocess wrapper
        └── self_trainer.py       # Teacher-student self-training loop
```

---

## Installation

### 1. Clone and install core dependencies

```bash
git clone https://github.com/tasktest06-star/customobjectdetection.git
cd customobjectdetection
git checkout apache2-rtdetr-pipeline-bugfixes

pip install -r requirements.txt
```

### 2. (Optional) VLM backends — only if using `vlm_qwen2` or `vlm_internvl2`

```bash
# Qwen2-VL
pip install qwen-vl-utils

# InternVL2
pip install einops timm sentencepiece
```

### 3. (Optional) PaddleDetection backend — only if `training.backend: "paddle"`

```bash
# Step A — PaddlePaddle (pick the build matching your CUDA version)
# CUDA 11.8:
pip install paddlepaddle-gpu==2.6.0 \
  -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
# CPU only:
pip install paddlepaddle==2.6.0

# Step B — PaddleDetection
git clone https://github.com/PaddlePaddle/PaddleDetection /opt/PaddleDetection
cd /opt/PaddleDetection
pip install -r requirements.txt
pip install -v -e .
cd -
```

---

## Quick Start

### Step 1 — Configure your classes

Edit `configs/pipeline_config.yaml`:

```yaml
classes:
  - name: "your_class_1"
    search_queries:
      - "your_class_1 video footage"
      - "your_class_1 closeup"
  - name: "your_class_2"
```

Everything else can stay at defaults for a first run.

### Step 2 — Run the pipeline

```bash
# Using config file:
python scripts/run_pipeline.py --config configs/pipeline_config.yaml --name my_run

# Override classes directly on the command line:
python scripts/run_pipeline.py \
  --classes raccoon capybara \
  --name raccoon_capybara

# Resume from existing frames (skip download):
python scripts/run_pipeline.py \
  --classes raccoon capybara \
  --name raccoon_capybara \
  --skip-download
```

The trained model is saved to `models/my_run/best/`.

---

## Configuration Reference

| Key | Default | Description |
|---|---|---|
| `classes[].name` | — | Class name (also used as the YouTube search seed) |
| `classes[].search_queries` | auto | Custom search terms; auto-generated if omitted |
| `video_search.num_videos_per_class` | 10 | Videos to download per class |
| `video_search.max_duration_seconds` | 300 | Skip videos longer than this |
| `frame_extraction.fps` | 0.5 | Frames per second to extract (0.5 = 1 frame every 2s) |
| `frame_extraction.max_frames_per_video` | 100 | Cap per video to prevent dominance |
| `frame_extraction.min_blur_variance` | 100.0 | Laplacian threshold; lower = keep blurrier frames |
| `clip_filtering.enabled` | true | Set false to skip CLIP pre-filter |
| `clip_filtering.similarity_threshold` | 0.20 | Lower = keep more (higher recall) |
| `pseudo_labeling.labeler` | `grounding_dino` | `grounding_dino` / `owl_vit` / `vlm_qwen2` / `vlm_internvl2` |
| `pseudo_labeling.box_threshold` | 0.30 | Minimum detection confidence |
| `annotation.train_ratio` | 0.8 | Fraction of images used for training |
| `annotation.min_box_area_ratio` | 0.001 | Drop boxes smaller than 0.1% of image area |
| `annotation.max_box_area_ratio` | 0.90 | Drop boxes larger than 90% of image area |
| `training.backend` | `hf` | `hf` (HuggingFace) or `paddle` (PaddleDetection) |
| `training.model` | `PekingU/rtdetr_r50vd` | HF checkpoint (backend=hf only) |
| `training.backbone` | `r50vd` | PaddleDetection backbone (backend=paddle only) |
| `training.paddle_det_path` | `/opt/PaddleDetection` | Path to PaddleDetection clone (backend=paddle only) |
| `training.epochs` | 50 | Total training epochs |
| `training.batch_size` | 8 | Reduce to 4 if GPU OOM |
| `training.learning_rate` | 1e-4 | AdamW base LR |
| `self_training.enabled` | false | Enable teacher-student self-training |
| `self_training.rounds` | 2 | Self-training iterations (2–3 is usually enough) |
| `self_training.intermediate_epochs` | 30 | Epochs for all rounds except the last |

---

## Labeler Backend Selection Guide

| Situation | Recommended backend |
|---|---|
| Common objects (animals, vehicles, furniture) | `grounding_dino` |
| Small or densely packed objects | `owl_vit` |
| Ambiguous names ("bat", "crane", "stool") | `vlm_qwen2` |
| Objects requiring context ("abandoned car") | `vlm_qwen2` or `vlm_internvl2` |
| Low VRAM (< 8 GB) | `grounding_dino` with `grounding-dino-tiny` |

---

## Training Backend Selection Guide

| Situation | Recommended backend |
|---|---|
| Quick experiment, single GPU | `hf` |
| Production training, multi-GPU | `paddle` |
| Need TensorRT / ONNX export | `paddle` |
| No PaddlePaddle installed | `hf` |

### Available PaddleDetection backbones

| `backbone` value | Parameters | Notes |
|---|---|---|
| `r18vd` | ~25 M | Fastest inference |
| `r34vd` | ~38 M | Good speed/accuracy balance |
| `r50vd` | ~42 M | Default, ~53 mAP on COCO |
| `r101vd` | ~60 M | Higher mAP, more VRAM |
| `hgnetv2_b4` | ~19 M | Efficient, mobile-friendly |
| `hgnetv2_b5` | ~31 M | Best accuracy/speed ratio |
| `hgnetv2_l` | ~52 M | Highest accuracy in HGNet family |

---

## Multi-GPU Training (PaddleDetection backend)

```python
from src.training.paddle_rtdetr_trainer import PaddleRTDETRTrainer

trainer = PaddleRTDETRTrainer(
    paddle_det_path="/opt/PaddleDetection",
    backbone="r50vd",
)
model_path = trainer.train_multi_gpu(
    data_yaml="data/datasets/my_run/data.yaml",
    gpu_ids=[0, 1],          # use GPUs 0 and 1
    epochs=50,
    batch_size=8,            # per-GPU batch size
)
```

Learning rate is automatically scaled by the number of GPUs.

---

## Evaluate a Trained Model

```bash
python scripts/evaluate.py \
  --model models/my_run/best \
  --data data/datasets/my_run/data.yaml \
  --backend hf
```

Reports mAP@0.5, mAP@0.5:0.95, and per-size breakdowns using pycocotools.

---

## Visualize Pseudo-Labels

Inspect the auto-generated annotations before training:

```bash
python scripts/visualize_labels.py \
  --ann data/datasets/my_run/annotations/instances_train.json \
  --images data/datasets/my_run/images/train \
  --out data/viz \
  --n 20
```

Saves annotated JPEG previews to `data/viz/`.

---

## Self-Training (Teacher-Student)

Enable in config for a free +3–8 mAP50 improvement:

```yaml
self_training:
  enabled: true
  rounds: 2
  intermediate_epochs: 30
```

**How it works:**

```
Round 1:  teacher labels → train student (30 epochs)
          student re-annotates all frames
          conservative ensemble: always keep all teacher boxes
          quality filter ensemble output

Round 2:  ensemble labels → train student (50 epochs = final_epochs)
          → final model
```

The ensemble is conservative by design: student disagreement never causes a teacher
box to be dropped. The student may simply not have learned the class yet.

---

## Extending the Pipeline

### Use a custom pseudo-labeler

Any labeler that implements `.label_batch(image_paths, class_names) -> List[dict]`
where each dict has `{image_path, boxes_xyxy, scores, labels, image_width, image_height}`
can be dropped in:

```python
pipeline = ObjectDetectionPipeline("configs/pipeline_config.yaml")
pipeline._build_components()
pipeline.labeler = MyCustomLabeler(...)
pipeline.run(dataset_name="custom_run")
```

### Export PaddleDetection model to ONNX / TensorRT

```python
trainer = PaddleRTDETRTrainer(paddle_det_path="/opt/PaddleDetection")
trainer.export_model(
    config_path="paddle_models/configs/my_run.yml",
    weights_path="paddle_models/my_run/best_model/model.pdparams",
    output_dir="exported_model",
)
```

---

## License

All source code in this repository is released under **Apache 2.0**.

| Component | License |
|---|---|
| RT-DETR (PaddleDetection) | Apache 2.0 |
| RT-DETR (HuggingFace PekingU) | Apache 2.0 |
| GroundingDINO | Apache 2.0 |
| OWLv2 | Apache 2.0 |
| Qwen2-VL | Apache 2.0 |
| InternVL2 | Apache 2.0 |
| CLIP (OpenAI) | MIT |
| yt-dlp | Unlicense |
| OpenCV | Apache 2.0 |
| HuggingFace transformers | Apache 2.0 |
| PaddlePaddle | Apache 2.0 |

No AGPL components — safe for commercial use.
