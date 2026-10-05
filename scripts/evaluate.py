#!/usr/bin/env python3
"""
Evaluate a trained RT-DETR model on a validation set (COCO mAP).

Usage:
  # HuggingFace model
  python scripts/evaluate.py \
    --model models/my_run/best \
    --data data/datasets/my_run/data.yaml

  # PaddleDetection model
  python scripts/evaluate.py \
    --model paddle_models/my_run/best_model \
    --data data/datasets/my_run/data.yaml \
    --backend paddle \
    --paddle-det /opt/PaddleDetection
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model path")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument(
        "--backend", choices=["hf", "paddle"], default="hf", help="Training backend"
    )
    parser.add_argument("--paddle-det", metavar="PATH", help="PaddleDetection path (backend=paddle)")
    args = parser.parse_args()

    if args.backend == "hf":
        from src.training.rtdetr_trainer import RTDETRTrainer
        trainer = RTDETRTrainer()
        metrics = trainer.evaluate(args.model, args.data)
        if metrics:
            print("\nEvaluation results:")
            for k, v in metrics.items():
                print(f"  {k:12s}: {v:.4f}")
        else:
            print("No metrics — install pycocotools: pip install pycocotools")

    elif args.backend == "paddle":
        if not args.paddle_det:
            print("--paddle-det required for paddle backend")
            sys.exit(1)
        # PaddleDetection has its own eval tool; we call it via subprocess
        import subprocess, os
        # Find the config used during training
        config_candidates = list(Path("paddle_models/configs").glob("*.yml"))
        if not config_candidates:
            print("No PaddleDetection config found in paddle_models/configs/")
            sys.exit(1)
        cfg_path = config_candidates[0]
        eval_script = Path(args.paddle_det) / "tools" / "eval.py"
        cmd = [
            "python", str(eval_script),
            "-c", str(cfg_path),
            "-o", f"weights={args.model}",
        ]
        env = {**os.environ, "PYTHONPATH": str(args.paddle_det)}
        subprocess.run(cmd, cwd=str(args.paddle_det), env=env)


if __name__ == "__main__":
    main()
