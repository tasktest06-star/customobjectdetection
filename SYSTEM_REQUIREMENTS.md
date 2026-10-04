# Minimum System Requirements

Hardware and software specifications needed to run the auto-label object detection pipeline.

---

## Quick Reference

| Tier | Use case | GPU | RAM | Storage | Est. pipeline time |
|---|---|---|---|---|---|
| **Minimum (CPU)** | Testing / small datasets | None | 8 GB | 20 GB | 5–10 hours |
| **Recommended** | Standard runs (2–5 classes) | 6 GB VRAM | 16 GB | 50 GB | 45–90 min |
| **Optimal** | Large datasets / fast iteration | 12 GB+ VRAM | 32 GB | 100 GB | 20–40 min |

---

## Minimum Configuration (CPU-only)

> The pipeline runs fully on CPU. Expect significantly longer runtimes for the CLIP filter, Grounding DINO labeling, and YOLOv8 training steps.

| Component | Minimum spec |
|---|---|
| **CPU** | 4-core x86-64 (Intel Core i5 / AMD Ryzen 5, any generation after 2016) |
| **RAM** | 8 GB |
| **Storage** | 20 GB free disk space (SSD strongly preferred over HDD) |
| **GPU** | Not required |
| **Internet** | Broadband (≥ 10 Mbps) for video download and model weight download |
| **OS** | Ubuntu 20.04+ / macOS 12+ / Windows 10 (64-bit) |
| **Python** | 3.9 or higher |

### Minimum runtime estimates (CPU, 2 classes, 10 videos each)

| Step | Estimated time |
|---|---|
| Video download | 10–20 min |
| Frame extraction | 3–5 min |
| CLIP filter (500 frames) | 8–15 min |
| Grounding DINO labeling (500 frames) | 60–90 min |
| YOLOv8 training (50 epochs) | 90–180 min |
| **Total** | **~3–5 hours** |

---

## Recommended Configuration

| Component | Recommended spec |
|---|---|
| **CPU** | 8-core (Intel Core i7/i9 or AMD Ryzen 7/9) |
| **RAM** | 16 GB |
| **GPU** | NVIDIA GPU with **6 GB VRAM** (RTX 3060 / RTX 2060 / GTX 1080) |
| **CUDA** | CUDA 11.8 or 12.x |
| **Storage** | 50 GB free SSD |
| **Internet** | ≥ 25 Mbps |
| **OS** | Ubuntu 22.04 LTS (best compatibility) / Windows 11 |
| **Python** | 3.10 or 3.11 |

### Recommended runtime estimates (GPU, 2 classes, 10 videos each)

| Step | Estimated time |
|---|---|
| Video download | 5–10 min |
| Frame extraction | 1–2 min |
| CLIP filter (500 frames) | 30–60 sec |
| Grounding DINO labeling (500 frames) | 8–15 min |
| YOLOv8 training (50 epochs) | 10–20 min |
| **Total** | **~30–60 min** |

---

## Optimal Configuration

| Component | Optimal spec |
|---|---|
| **CPU** | 12-core+ (Intel Core i9 / AMD Ryzen 9 / Threadripper) |
| **RAM** | 32 GB or more |
| **GPU** | NVIDIA RTX 3090 / RTX 4080 / A100 (**12–24 GB VRAM**) |
| **CUDA** | CUDA 12.x |
| **Storage** | 100 GB+ NVMe SSD |
| **Internet** | ≥ 50 Mbps |
| **OS** | Ubuntu 22.04 LTS |
| **Python** | 3.11 |

At this tier, multiple self-training rounds and large datasets (20+ classes, 50+ videos per class) are practical within a few hours.

---

## GPU VRAM Requirements by Model

Different model choices have different VRAM demands. Use this table to pick a configuration that fits your hardware.

### CLIP (pre-filter)

| Model | VRAM needed |
|---|---|
| `openai/clip-vit-base-patch32` (default) | ~1.5 GB |
| `openai/clip-vit-large-patch14` | ~3.5 GB |

### Grounding DINO (pseudo-labeler)

| Model | VRAM needed | Notes |
|---|---|---|
| `IDEA-Research/grounding-dino-tiny` (default) | ~2 GB | Good accuracy, fits on any modern GPU |
| `IDEA-Research/grounding-dino-base` | ~4.5 GB | Higher recall, recommended if VRAM allows |

### OWLv2 (alternative pseudo-labeler)

| Model | VRAM needed |
|---|---|
| `google/owlv2-base-patch16` | ~2 GB |
| `google/owlv2-large-patch14-ensemble` | ~5 GB |

### YOLOv8 Training

| Model size | VRAM needed (batch=16, imgsz=640) |
|---|---|
| `yolov8n.pt` (nano) | ~2 GB |
| `yolov8s.pt` (small) | ~3 GB |
| `yolov8m.pt` (medium) | ~5.5 GB |
| `yolov8l.pt` (large) | ~9 GB |
| `yolov8x.pt` (extra-large) | ~12 GB |

> **Tip:** If VRAM is tight, reduce `training.batch_size` in config from 16 to 8 or 4. Training takes longer but uses less memory.

---

## Storage Breakdown

Storage usage depends on the number of classes and videos. Below is a typical estimate for **2 classes, 10 videos each**.

| Data | Approx. size |
|---|---|
| Downloaded videos (MP4, 480p) | 2–5 GB |
| Extracted frames (JPEG) | 0.5–1.5 GB |
| CLIP score cache | < 5 MB |
| Grounding DINO result cache | 50–200 MB |
| COCO dataset (images copied) | 0.5–1.5 GB |
| YOLOv8 model weights | 50–200 MB |
| Logs and checkpoints | < 50 MB |
| **Total** | **~4–10 GB** |

For 10 classes with 20 videos each, multiply by ~10 → **40–100 GB**.

---

## Operating System Notes

### Linux (Ubuntu 22.04 — recommended)

Best compatibility with CUDA, PyTorch, and yt-dlp. All features supported.

```bash
sudo apt-get install python3 python3-pip python3-venv ffmpeg git
```

### macOS (12 Monterey or later)

Runs on Apple Silicon (M1/M2/M3) via MPS backend. YOLOv8 supports MPS.
Grounding DINO and CLIP will use CPU on older Intel Macs without Metal support.

```bash
brew install python ffmpeg git
```

GPU acceleration on Apple Silicon (set in code or via env):
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

### Windows 10/11 (64-bit)

Supported. Install WSL2 (Windows Subsystem for Linux) for best compatibility.
Native Windows also works with the following requirements:
- Install [ffmpeg](https://ffmpeg.org/download.html) and add to PATH
- Use Python from [python.org](https://python.org) (not Microsoft Store)
- Visual C++ Redistributable 2019+ must be installed

---

## Python Package Size

First-time installation downloads large model weights. Ensure sufficient disk space and a stable internet connection.

| Package | Download size (approx.) |
|---|---|
| `torch` + `torchvision` (CUDA) | ~2.5 GB |
| `torch` + `torchvision` (CPU) | ~200 MB |
| Grounding DINO weights (tiny) | ~340 MB |
| Grounding DINO weights (base) | ~900 MB |
| CLIP weights (ViT-B/32) | ~350 MB |
| YOLOv8n pretrained weights | ~6 MB |
| All other packages combined | ~500 MB |
| **Total first run (GPU setup)** | **~5–6 GB** |

Model weights are cached by HuggingFace in `~/.cache/huggingface/` after the first download.

---

## Checking Your System

Run this after installation to verify your setup:

```bash
python3 - <<'EOF'
import torch, platform, shutil

print(f"OS          : {platform.system()} {platform.release()}")
print(f"Python      : {platform.python_version()}")
print(f"PyTorch     : {torch.__version__}")
print(f"CUDA avail  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU         : {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM        : {vram:.1f} GB")

import psutil
ram = psutil.virtual_memory().total / 1e9
print(f"RAM         : {ram:.1f} GB")

disk = shutil.disk_usage(".")
free_gb = disk.free / 1e9
print(f"Free disk   : {free_gb:.1f} GB")

ffmpeg = shutil.which("ffmpeg")
print(f"ffmpeg      : {'found at ' + ffmpeg if ffmpeg else 'NOT FOUND — install ffmpeg'}")
EOF
```

Expected output (GPU machine):
```
OS          : Linux 6.x.x
Python      : 3.11.x
PyTorch     : 2.1.0
CUDA avail  : True
GPU         : NVIDIA GeForce RTX 3060
VRAM        : 12.0 GB
RAM         : 31.9 GB
Free disk   : 87.3 GB
ffmpeg      : found at /usr/bin/ffmpeg
```

---

## Cloud / Colab Alternatives

If you don't have a local GPU, the pipeline runs well on:

| Platform | Free tier GPU | Notes |
|---|---|---|
| **Google Colab** | T4 (16 GB VRAM) | Free tier has time limits; Colab Pro recommended |
| **Kaggle Notebooks** | T4 / P100 | 30 GPU hours/week free |
| **AWS EC2 `g4dn.xlarge`** | T4 (16 GB VRAM) | ~$0.52/hr on-demand, ~$0.16/hr spot |
| **Azure NC6s_v3** | V100 (16 GB VRAM) | ~$0.90/hr |
| **Google Colab Enterprise** | A100 (40/80 GB VRAM) | Paid |

For cloud runs, upload your `configs/pipeline_config.yaml` and run:
```bash
git clone https://github.com/tasktest06-star/customobjectdetection
cd customobjectdetection
pip install -r requirements.txt
python scripts/run_pipeline.py --name cloud_run
```
