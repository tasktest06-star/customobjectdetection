#!/usr/bin/env python3
"""
Evaluate a trained model on a validation set and print detection metrics.

Usage:
  python scripts/evaluate.py \
    --model models/auto_det/weights/best.pt \
    --data data/datasets/auto_det/data.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.yolo_trainer import YOLOTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .pt weights")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    args = parser.parse_args()

    trainer = YOLOTrainer()
    metrics = trainer.evaluate(args.model, args.data)

    print("\nEvaluation results:")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.4f}")


if __name__ == "__main__":
    main()
