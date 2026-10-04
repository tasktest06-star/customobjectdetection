#!/usr/bin/env python3
"""
CLI entry point for the auto-label object detection pipeline.

Usage examples:
  # Use classes from config (full run)
  python scripts/run_pipeline.py --config configs/pipeline_config.yaml

  # Override classes on the command line
  python scripts/run_pipeline.py --classes raccoon fox --name my_run

  # Skip video download, reuse frames already in data/frames/
  python scripts/run_pipeline.py --skip-download --name my_run

  # Resume a crashed run from the last completed step
  python scripts/run_pipeline.py --resume --name my_run

  # Parallel download with 4 workers
  python scripts/run_pipeline.py --workers 4 --name my_run
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
        "--config", default="configs/pipeline_config.yaml",
        help="Pipeline config YAML",
    )
    parser.add_argument(
        "--name", default="auto_det",
        help="Dataset / run name used for output paths",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip video download; reuse frames already in data/frames/",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last completed checkpoint step",
    )
    parser.add_argument(
        "--workers", type=int, default=3,
        help="Number of parallel workers for download+extract (default: 3)",
    )
    parser.add_argument(
        "--classes", nargs="+", metavar="CLASS",
        help="Override classes from config, e.g. --classes raccoon fox",
    )
    args = parser.parse_args()

    pipeline = ObjectDetectionPipeline(config_path=args.config)

    if args.classes:
        pipeline.cfg["classes"] = [{"name": c} for c in args.classes]
        pipeline.class_names = args.classes
        pipeline._build_components()

    model_path = pipeline.run(
        dataset_name=args.name,
        skip_download=args.skip_download,
        resume=args.resume,
        max_download_workers=args.workers,
    )
    print(f"\nModel saved: {model_path}")


if __name__ == "__main__":
    main()
