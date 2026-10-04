#!/usr/bin/env python3
"""
CLI entry point for the auto-labeling object detection pipeline.

Usage examples:
  # Use classes from config
  python scripts/run_pipeline.py --config configs/pipeline_config.yaml

  # Override class list on the command line
  python scripts/run_pipeline.py --classes raccoon fox --name my_run

  # Skip download, reuse frames already in data/frames/
  python scripts/run_pipeline.py --classes raccoon --skip-download
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import ObjectDetectionPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune an object detector without labeled data"
    )
    parser.add_argument(
        "--config",
        default="configs/pipeline_config.yaml",
        help="Pipeline config YAML (default: configs/pipeline_config.yaml)",
    )
    parser.add_argument(
        "--name",
        default="auto_det",
        help="Dataset / run name used for output paths",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip video download; reuse frames already in data/frames/",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        metavar="CLASS",
        help="Override classes from config, e.g. --classes raccoon fox",
    )
    args = parser.parse_args()

    pipeline = ObjectDetectionPipeline(config_path=args.config)

    if args.classes:
        pipeline.cfg["classes"] = [{"name": c} for c in args.classes]
        pipeline.class_names = args.classes
        # Rebuild labeler-dependent components with the new class list
        pipeline._build_components()

    model_path = pipeline.run(
        dataset_name=args.name,
        skip_download=args.skip_download,
    )
    print(f"\nModel saved: {model_path}")


if __name__ == "__main__":
    main()
