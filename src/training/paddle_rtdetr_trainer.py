"""
PaddleDetection RT-DETR trainer — native PaddlePaddle implementation (Apache 2.0).

WHY USE THIS INSTEAD OF rtdetr_trainer.py (HF)?
──────────────────────────────────────────────
  1. Original implementation: RT-DETR was developed at Baidu inside PaddleDetection.
     The HF port is a faithful reimplementation but PaddleDetection remains the
     reference codebase for new RT-DETR variants (e.g. RT-DETRv2).
  2. Richer backbone options: R18, R34, R50, R101, HGNetv2 — all Apache 2.0.
  3. PaddleInference export: optimized static graph deployment, Paddle Serving,
     TensorRT via Paddle-TRT — all in one toolchain.
  4. Multi-GPU training: PaddleDetection's --fleet flag handles DDP natively.

HOW IT WORKS
──────────────
This trainer generates a PaddleDetection config YAML file that:
  - Inherits from the base RT-DETR config via _BASE_ (PaddleDetection's config system)
  - Overrides num_classes, dataset paths, training schedule, and optimizer LR
  - Sets pretrain_weights to a COCO-pretrained checkpoint URL; PaddleDetection
    automatically skips mismatched classification head weights
Then it invokes tools/train.py via subprocess.

PREREQUISITES
──────────────
  1. Install PaddlePaddle:
       GPU (CUDA 11.8): pip install paddlepaddle-gpu==2.6.0 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
       CPU only:        pip install paddlepaddle==2.6.0
  2. Clone and install PaddleDetection:
       git clone https://github.com/PaddlePaddle/PaddleDetection
       cd PaddleDetection && pip install -r requirements.txt && pip install -v -e .
  3. Pass the PaddleDetection clone path to PaddleRTDETRTrainer(paddle_det_path=...).

LICENSE: Apache 2.0 (PaddleDetection, PaddlePaddle, all model weights)
"""
import os
import json
import yaml
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Optional


# Pretrained COCO checkpoint URLs for each backbone variant
_PRETRAIN_WEIGHTS = {
    "r18vd":   "https://paddledet.bj.bcebos.com/models/rtdetr_r18vd_5x_coco.pdparams",
    "r34vd":   "https://paddledet.bj.bcebos.com/models/rtdetr_r34vd_5x_coco.pdparams",
    "r50vd":   "https://paddledet.bj.bcebos.com/models/rtdetr_r50vd_6x_coco.pdparams",
    "r101vd":  "https://paddledet.bj.bcebos.com/models/rtdetr_r101vd_6x_coco.pdparams",
    "hgnetv2_b4": "https://paddledet.bj.bcebos.com/models/rtdetr_hgnetv2_b4_6x_coco.pdparams",
    "hgnetv2_b5": "https://paddledet.bj.bcebos.com/models/rtdetr_hgnetv2_b5_6x_coco.pdparams",
    "hgnetv2_l":  "https://paddledet.bj.bcebos.com/models/rtdetrv2_hgnetv2_l_6x_coco.pdparams",
}

# Base config file names inside PaddleDetection/configs/rtdetr/
_BASE_CONFIGS = {
    "r18vd":      "rtdetr_r18vd_6x_coco.yml",
    "r34vd":      "rtdetr_r34vd_6x_coco.yml",
    "r50vd":      "rtdetr_r50vd_6x_coco.yml",
    "r101vd":     "rtdetr_r101vd_6x_coco.yml",
    "hgnetv2_b4": "rtdetr_hgnetv2_b4_6x_coco.yml",
    "hgnetv2_b5": "rtdetr_hgnetv2_b5_6x_coco.yml",
    "hgnetv2_l":  "rtdetrv2_hgnetv2_l_6x_coco.yml",
}


class PaddleRTDETRTrainer:
    """
    Wraps PaddleDetection's RT-DETR training pipeline for custom datasets.

    Interface mirrors RTDETRTrainer so either trainer can be used as a drop-in
    replacement in the pipeline config:
        training:
          backend: "paddle"        # or "hf" (HuggingFace transformers)
          paddle_det_path: "/opt/PaddleDetection"
          backbone: "r50vd"
    """

    def __init__(
        self,
        paddle_det_path: str,
        backbone: str = "r50vd",
        output_dir: str = "paddle_models",
        device: Optional[str] = None,
    ):
        """
        Args:
            paddle_det_path: Absolute path to your PaddleDetection clone.
                             Must contain tools/train.py.
            backbone:        RT-DETR backbone variant. Options:
                               "r18vd"      — ResNet-18, fastest (~46 mAP COCO)
                               "r34vd"      — ResNet-34 (~48 mAP)
                               "r50vd"      — ResNet-50, recommended (~53 mAP)
                               "r101vd"     — ResNet-101 (~54 mAP)
                               "hgnetv2_b4" — HGNetv2-B4, faster than R50 at same mAP
                               "hgnetv2_b5" — HGNetv2-B5, best efficiency/accuracy
                               "hgnetv2_l"  — RT-DETRv2, highest accuracy
            output_dir:      Root directory for generated configs and checkpoints.
            device:          "gpu" or "cpu" (PaddlePaddle convention; not torch).
                             None auto-detects: "gpu" if CUDA is available else "cpu".
        """
        self.paddle_det_path = Path(paddle_det_path)
        train_script = self.paddle_det_path / "tools" / "train.py"
        if not train_script.exists():
            raise FileNotFoundError(
                f"tools/train.py not found at {paddle_det_path}. "
                "Clone PaddleDetection:\n"
                "  git clone https://github.com/PaddlePaddle/PaddleDetection"
            )
        if backbone not in _PRETRAIN_WEIGHTS:
            raise ValueError(
                f"Unknown backbone {backbone!r}. "
                f"Valid options: {list(_PRETRAIN_WEIGHTS)}"
            )
        self.backbone = backbone
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if device is None:
            try:
                import subprocess as sp
                result = sp.run(
                    ["python", "-c", "import paddle; print(paddle.device.get_device())"],
                    capture_output=True, text=True
                )
                self._use_gpu = "gpu" in result.stdout
            except Exception:
                self._use_gpu = False
        else:
            self._use_gpu = device.lower() == "gpu"

    # ── config generation ──────────────────────────────────────────────────────

    def _generate_config(
        self,
        dataset_dir: str,
        class_names: List[str],
        run_name: str,
        epochs: int,
        batch_size: int,
        imgsz: int,
        learning_rate: float,
        weight_decay: float,
    ) -> Path:
        """
        Write a PaddleDetection YAML config for fine-tuning RT-DETR.

        The config uses PaddleDetection's _BASE_ inheritance system:
          _BASE_: [
            'configs/datasets/coco_detection.yml',
            'configs/runtime.yml',
            'configs/rtdetr/_base_/rtdetr_r50vd.yml',
            'configs/rtdetr/_base_/optimizer_6x.yml',
          ]
        and overrides only what needs to change (num_classes, dataset, schedule).

        PaddleDetection's partial weight loading:
          When pretrain_weights is set to a COCO checkpoint and num_classes ≠ 80,
          PaddleDetection automatically detects the shape mismatch in the classification
          head and re-initializes only those weights. The backbone, neck, and transformer
          encoder/decoder weights are fully transferred.
        """
        num_classes = len(class_names)
        dataset_dir_abs = str(Path(dataset_dir).resolve())
        base_cfg = _BASE_CONFIGS[self.backbone]
        pretrain_weights = _PRETRAIN_WEIGHTS[self.backbone]
        save_dir_abs = str((self.output_dir / run_name).resolve())

        # Warmup: 100 steps is enough for the re-initialized classification head
        # to reach a reasonable initialization before full-LR training begins.
        # Milestones at 80% and 95% of total epochs match the original training schedule.
        warmup_steps = min(100, epochs * 10)
        decay_epoch_1 = int(epochs * 0.8)
        decay_epoch_2 = int(epochs * 0.95)

        # PaddleDetection uses a different LR convention: base_lr is per-image LR
        # (actual LR = base_lr × batch_size). We convert back to per-step LR.
        base_lr_per_image = learning_rate / batch_size

        # Absolute paths for _BASE_ because this config is saved OUTSIDE the
        # PaddleDetection tree (under self.output_dir). Relative paths like
        # '../../configs/...' would only work if the config were nested two
        # directories deep inside PaddleDetection, which it is not.
        pd_abs = str(self.paddle_det_path.resolve())

        cfg_content = f"""\
# Auto-generated by PaddleRTDETRTrainer — do not edit manually
# Generated for backbone={self.backbone}, classes={class_names}

_BASE_: [
  '{pd_abs}/configs/datasets/coco_detection.yml',
  '{pd_abs}/configs/runtime.yml',
  '{pd_abs}/configs/rtdetr/_base_/optimizer_6x.yml',
]

# ── Custom class count ────────────────────────────────────────────────────────
# PaddleDetection handles num_classes mismatch automatically via partial loading.
num_classes: &num_classes {num_classes}

RTDETR:
  backbone:
    !{self._get_backbone_class()}
    name: {self._get_backbone_name()}
    return_idx: [1, 2, 3]
    freeze_at: -1
    freeze_norm: False
    pretrained: True
  neck:
    !HybridEncoder
    hidden_dim: 256
    use_encoder_idx: [2]
    num_encoder_layers: 1
    nhead: 8
    dim_feedforward: 1024
    dropout: 0.0
    enc_act: 'gelu'
    pe_temperature: 10000
    expansion: 1.0
    depth_mult: 1.0
    act: 'silu'
    eval_size: [{imgsz}, {imgsz}]
  head:
    !RTDETRHead
    num_classes: *num_classes
    hidden_dim: 256
    num_queries: 300
    position_embed_type: sine
    feat_strides: [8, 16, 32]
    num_levels: 3
    num_decoder_points: 4
    nhead: 8
    num_decoder_layers: 6
    dim_feedforward: 1024
    dropout: 0.0
    activation: 'relu'
    num_denoising: 100
    label_noise_ratio: 0.5
    box_noise_scale: 1.0
    learnt_init_query: False
    eval_idx: -1
    eps: 1e-2
    aux_loss: True
  post_process:
    !RTDETRPostProcess
    num_top_queries: 300
    use_focal_loss: True

# ── Pretrained weights ────────────────────────────────────────────────────────
pretrain_weights: {pretrain_weights}

# ── Training schedule ─────────────────────────────────────────────────────────
epoch: {epochs}

LearningRate:
  base_lr: {base_lr_per_image:.6f}
  schedulers:
    - !PiecewiseDecay
      gamma: 0.1
      milestones: [{decay_epoch_1}, {decay_epoch_2}]
    - !LinearWarmup
      start_factor: 0.001
      steps: {warmup_steps}

OptimizerBuilder:
  optimizer:
    type: AdamW
    weight_decay: {weight_decay}
    clip_grad_by_norm: 0.1
  regularization: null

# ── Image size ────────────────────────────────────────────────────────────────
TrainReader:
  batch_size: {batch_size}
  worker_num: 4
  collate_batch: True

EvalReader:
  batch_size: 1
  worker_num: 2

# ── Custom dataset ────────────────────────────────────────────────────────────
TrainDataset:
  !COCODataSet
    image_dir: images/train
    anno_path: annotations/instances_train.json
    dataset_dir: {dataset_dir_abs}
    data_fields: ['image', 'gt_bbox', 'gt_class', 'is_crowd']

EvalDataset:
  !COCODataSet
    image_dir: images/val
    anno_path: annotations/instances_val.json
    dataset_dir: {dataset_dir_abs}
    allow_empty: true

TestDataset:
  !ImageFolder
    anno_path: annotations/instances_val.json
    dataset_dir: {dataset_dir_abs}

# ── Checkpointing ─────────────────────────────────────────────────────────────
snapshot_epoch: 5
log_iter: 20
find_unused_parameters: True
save_dir: {save_dir_abs}
"""
        cfg_dir = self.output_dir / "configs"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = cfg_dir / f"{run_name}.yml"
        cfg_path.write_text(cfg_content)

        # Write a human-readable categories file for reference
        cats = [{"id": i + 1, "name": n} for i, n in enumerate(class_names)]
        (cfg_dir / f"{run_name}_categories.json").write_text(
            json.dumps(cats, indent=2)
        )
        print(f"PaddleDetection config: {cfg_path}")
        return cfg_path

    def _get_backbone_class(self) -> str:
        return "HGNetV2" if "hgnet" in self.backbone else "ResNet"

    def _get_backbone_name(self) -> str:
        mapping = {
            "r18vd": "ResNet18vd",
            "r34vd": "ResNet34vd",
            "r50vd": "ResNet50vd",
            "r101vd": "ResNet101vd",
            "hgnetv2_b4": "B4",
            "hgnetv2_b5": "B5",
            "hgnetv2_l": "L",
        }
        return mapping.get(self.backbone, "ResNet50vd")

    # ── training ───────────────────────────────────────────────────────────────

    def train(
        self,
        data_yaml: str,
        epochs: int = 50,
        batch_size: int = 4,
        imgsz: int = 640,
        run_name: str = "paddle_run",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        resume: bool = False,
        **kwargs,
    ) -> str:
        """
        Fine-tune RT-DETR using PaddleDetection's tools/train.py.

        Args:
            data_yaml:      Path to data.yaml from COCOBuilder.build_dataset().
            epochs:         Number of training epochs.
            batch_size:     Images per GPU per step. Start with 4; reduce to 2 if OOM.
                            PaddleDetection's effective batch size = batch_size × GPUs.
            imgsz:          Input resolution (square). Default 640 matches RT-DETR paper.
            run_name:       Subdirectory name under output_dir for checkpoints.
            learning_rate:  Total learning rate (divided by batch_size internally to
                            get per-image LR in PaddleDetection's convention).
            weight_decay:   AdamW weight decay coefficient.
            resume:         If True, resume from the latest checkpoint in run_name/.

        Returns:
            Path to the directory containing the best model checkpoint.
        """
        with open(data_yaml) as f:
            cfg = yaml.safe_load(f)
        dataset_dir = cfg["path"]
        class_names = cfg["names"]

        cfg_path = self._generate_config(
            dataset_dir=dataset_dir,
            class_names=class_names,
            run_name=run_name,
            epochs=epochs,
            batch_size=batch_size,
            imgsz=imgsz,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )

        run_dir = self.output_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        # ── build training command ──────────────────────────────────────────
        train_script = self.paddle_det_path / "tools" / "train.py"
        cmd = [
            "python", str(train_script),
            "-c", str(cfg_path),
            "--eval",
        ]
        if self._use_gpu:
            cmd += ["-o", "use_gpu=True"]
        else:
            cmd += ["-o", "use_gpu=False"]
        if resume:
            cmd += ["-r", str(run_dir / "model_final.pdparams")]

        print(f"\nLaunching PaddleDetection training:")
        print(f"  Backbone:  {self.backbone}")
        print(f"  Classes:   {class_names}")
        print(f"  Epochs:    {epochs}")
        print(f"  Batch:     {batch_size}")
        print(f"  Output:    {run_dir}")
        print(f"  GPU:       {self._use_gpu}")
        print()

        env = {**os.environ, "PYTHONPATH": str(self.paddle_det_path)}
        result = subprocess.run(cmd, cwd=str(self.paddle_det_path), env=env)

        if result.returncode != 0:
            raise RuntimeError(
                f"PaddleDetection training failed (exit code {result.returncode}).\n"
                f"Config: {cfg_path}\n"
                f"Logs above show the error."
            )

        # PaddleDetection saves best_model/ directory with best_model.pdparams
        best_dir = run_dir / "best_model"
        if best_dir.exists():
            return str(best_dir)

        # Fallback: return the last epoch checkpoint
        checkpoints = sorted(run_dir.glob("*.pdparams"))
        if checkpoints:
            return str(checkpoints[-1].parent)
        return str(run_dir)

    # ── export ─────────────────────────────────────────────────────────────────

    def export_model(
        self,
        config_path: str,
        weights_path: str,
        output_dir: str = "exported_models",
    ) -> str:
        """
        Export the trained model to PaddleInference static graph format.

        The exported model can be deployed with:
          - paddle_inference Python/C++ API (high-performance CPU/GPU)
          - Paddle Serving (REST API serving)
          - PaddleLite (edge devices, mobile)
          - TensorRT via Paddle-TRT (NVIDIA GPUs)

        Args:
            config_path:  Path to the YAML config used during training.
            weights_path: Path to the .pdparams checkpoint or best_model directory.
            output_dir:   Directory for the exported static graph files.

        Returns:
            Path to the export output directory.
        """
        export_script = self.paddle_det_path / "tools" / "export_model.py"
        cmd = [
            "python", str(export_script),
            "-c", config_path,
            "-o", f"weights={weights_path}",
            "--output_dir", output_dir,
        ]
        env = {**os.environ, "PYTHONPATH": str(self.paddle_det_path)}
        result = subprocess.run(cmd, cwd=str(self.paddle_det_path), env=env)
        if result.returncode != 0:
            raise RuntimeError(f"Model export failed (exit code {result.returncode}).")
        print(f"Exported model: {output_dir}")
        return output_dir

    # ── inference ──────────────────────────────────────────────────────────────

    def predict(
        self,
        config_path: str,
        weights_path: str,
        source: str,
        conf: float = 0.25,
        class_names: Optional[List[str]] = None,
        output_dir: str = "paddle_inference_results",
    ) -> str:
        """
        Run inference using PaddleDetection's infer.py script.

        Saves annotated images and bbox.json to output_dir.
        Returns output_dir path.

        For programmatic result access, parse the saved bbox.json.
        """
        infer_script = self.paddle_det_path / "tools" / "infer.py"
        source_path = Path(source)
        src_flag = "--infer_img" if source_path.is_file() else "--infer_dir"
        cmd = [
            "python", str(infer_script),
            "-c", config_path,
            "-o", f"weights={weights_path}",
            src_flag, str(source),
            "--output_dir", output_dir,
            "--draw_threshold", str(conf),
            "--save_results",
        ]
        if self._use_gpu:
            cmd += ["-o", "use_gpu=True"]
        env = {**os.environ, "PYTHONPATH": str(self.paddle_det_path)}
        subprocess.run(cmd, cwd=str(self.paddle_det_path), env=env)
        return output_dir

    # ── multi-GPU training ─────────────────────────────────────────────────────

    def train_multi_gpu(
        self,
        data_yaml: str,
        gpu_ids: List[int],
        epochs: int = 50,
        batch_size: int = 4,
        run_name: str = "paddle_run_multigpu",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
    ) -> str:
        """
        Launch distributed training across multiple GPUs using PaddlePaddle fleet.

        PaddleDetection supports data-parallel multi-GPU training via:
          python -m paddle.distributed.launch --gpus "0,1,2,3" tools/train.py -c config.yml

        Effective batch size = batch_size × len(gpu_ids).
        Learning rate should be scaled proportionally: lr × len(gpu_ids).

        Args:
            gpu_ids:      List of GPU IDs to use, e.g. [0, 1, 2, 3].
            batch_size:   Per-GPU batch size.
            learning_rate: Per-GPU base LR (scaled by num_GPUs automatically in config).

        Returns:
            Path to the best model checkpoint directory.
        """
        with open(data_yaml) as f:
            cfg_data = yaml.safe_load(f)
        dataset_dir = cfg_data["path"]
        class_names = cfg_data["names"]

        # Scale LR linearly with number of GPUs (linear scaling rule)
        scaled_lr = learning_rate * len(gpu_ids)

        cfg_path = self._generate_config(
            dataset_dir=dataset_dir,
            class_names=class_names,
            run_name=run_name,
            epochs=epochs,
            batch_size=batch_size,
            imgsz=640,
            learning_rate=scaled_lr,
            weight_decay=weight_decay,
        )

        run_dir = self.output_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        gpu_str = ",".join(str(g) for g in gpu_ids)
        train_script = self.paddle_det_path / "tools" / "train.py"
        cmd = [
            "python", "-m", "paddle.distributed.launch",
            "--gpus", gpu_str,
            str(train_script),
            "-c", str(cfg_path),
            "--eval",
        ]

        print(f"\nMulti-GPU training: GPUs={gpu_ids}, effective batch={batch_size * len(gpu_ids)}")
        env = {**os.environ, "PYTHONPATH": str(self.paddle_det_path)}
        result = subprocess.run(cmd, cwd=str(self.paddle_det_path), env=env)
        if result.returncode != 0:
            raise RuntimeError(f"Multi-GPU training failed (exit code {result.returncode}).")
        best_dir = run_dir / "best_model"
        return str(best_dir) if best_dir.exists() else str(run_dir)
