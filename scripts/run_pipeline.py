#!/usr/bin/env python3
"""
CLI entry point for the label-free object detection pipeline.

Usage:
  # Run with config file
  python scripts/run_pipeline.py --config configs/pipeline_config.yaml --name my_run

  # Override classes on command line
  python scripts/run_pipeline.py --classes raccoon capybara --name my_run

  # Skip video download (reuse data/frames/)
  python scripts/run_pipeline.py --classes raccoon --skip-download --name my_run

  # Use PaddleDetection backend
  python scripts/run_pipeline.py --classes raccoon --paddle-det /opt/PaddleDetection --name my_run

  # Use VLM labeler (Qwen2-VL)
  python scripts/run_pipeline.py --classes raccoon --labeler vlm_qwen2 --name my_run
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import ObjectDetectionPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune RT-DETR without labeled data — Apache 2.0 only"
    )
    parser.add_argument(
        "--config",
        default="configs/pipeline_config.yaml",
        help="Pipeline config YAML",
    )
    parser.add_argument("--name", default="auto_det", help="Dataset / run name")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip video download; reuse frames in data/frames/",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        metavar="CLASS",
        help="Override classes, e.g. --classes raccoon fox 'garbage truck'",
    )
    parser.add_argument(
        "--labeler",
        choices=["grounding_dino", "owl_vit", "vlm_qwen2", "vlm_internvl2"],
        help="Override the pseudo-labeler backend",
    )
    parser.add_argument(
        "--paddle-det",
        metavar="PATH",
        help="Path to PaddleDetection clone. If provided, switches backend to paddle.",
    )
    parser.add_argument(
        "--backbone",
        choices=["r18vd", "r34vd", "r50vd", "r101vd", "hgnetv2_b4", "hgnetv2_b5"],
        default=None,
        help="PaddleDetection backbone (only used with --paddle-det)",
    )
    args = parser.parse_args()

    pipeline = ObjectDetectionPipeline(config_path=args.config)

    # Apply CLI overrides
    if args.classes:
        pipeline.cfg["classes"] = [{"name": c} for c in args.classes]
        pipeline.class_names = args.classes

    if args.labeler:
        pipeline.cfg["pseudo_labeling"]["labeler"] = args.labeler

    if args.paddle_det:
        pipeline.cfg["training"]["backend"] = "paddle"
        pipeline.cfg["training"]["paddle_det_path"] = args.paddle_det
        if args.backbone:
            pipeline.cfg["training"]["backbone"] = args.backbone

    # Components are built lazily in run(), so overrides above take effect
    model_path = pipeline.run(
        dataset_name=args.name,
        skip_download=args.skip_download,
    )
    print(f"\nModel saved: {model_path}")


if __name__ == "__main__":
    main()
