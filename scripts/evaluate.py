#!/usr/bin/env python3
"""
Evaluate a trained RT-DETR model on a validation set and print COCO mAP metrics.

Usage:
  python scripts/evaluate.py \
    --model models/auto_det/best \
    --data data/datasets/auto_det/data.yaml

The model path is a HuggingFace checkpoint directory saved by RTDETRTrainer.train().
Requires pycocotools: pip install pycocotools
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.rtdetr_trainer import RTDETRTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        help="Path to HuggingFace checkpoint directory (e.g. models/auto_det/best)",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to data.yaml produced by COCOBuilder",
    )
    args = parser.parse_args()

    trainer = RTDETRTrainer()
    metrics = trainer.evaluate(args.model, args.data)

    if metrics:
        print("\nEvaluation results:")
        for k, v in metrics.items():
            print(f"  {k:12s}: {v:.4f}")
    else:
        print("No metrics returned — check that pycocotools is installed.")


if __name__ == "__main__":
    main()
