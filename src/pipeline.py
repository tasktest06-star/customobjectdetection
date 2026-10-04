"""
Enhanced top-level pipeline orchestrator.

Improvements over v1:
  - Step-level checkpointing: each stage saves its output so a crash at
    step N resumes from step N-1 (use --resume flag or skip_to parameter)
  - Structured logging: replaces print() with Python logging (file + console)
  - Parallel class processing: video download and frame extraction for all
    classes run concurrently via ThreadPoolExecutor
"""
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

from src.video_pipeline.downloader import YouTubeDownloader
from src.video_pipeline.frame_extractor import FrameExtractor
from src.pseudo_labeling.clip_filter import CLIPFilter
from src.pseudo_labeling.grounding_dino import GroundingDINOLabeler
from src.pseudo_labeling.owl_vit import OWLv2Labeler
from src.annotation.quality_filter import (
    LabelQualityFilter, check_class_balance, temporal_consistency_filter
)
from src.annotation.coco_builder import COCOBuilder
from src.training.yolo_trainer import YOLOTrainer
from src.training.self_trainer import SelfTrainer
from src.utils.logger import get_logger
from src.utils.checkpoint import PipelineCheckpoint

log = get_logger("pipeline")


class ObjectDetectionPipeline:
    def __init__(self, config_path: str = "configs/pipeline_config.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        self._build_components()

    def _build_components(self):
        c = self.cfg

        self.downloader = YouTubeDownloader(c["video_search"]["download_dir"])
        self.extractor = FrameExtractor(
            output_dir=c["frame_extraction"]["output_dir"],
            fps=c["frame_extraction"]["fps"],
            max_frames=c["frame_extraction"]["max_frames_per_video"],
            min_blur_variance=c["frame_extraction"]["min_blur_variance"],
        )

        cf = c["clip_filtering"]
        self.clip: Optional[CLIPFilter] = (
            CLIPFilter(
                model_name=cf["model"],
                similarity_threshold=cf["similarity_threshold"],
                batch_size=cf["batch_size"],
            )
            if cf["enabled"]
            else None
        )

        pl = c["pseudo_labeling"]
        if pl["labeler"] == "grounding_dino":
            self.labeler: GroundingDINOLabeler | OWLv2Labeler = GroundingDINOLabeler(
                model_id=pl["model"],
                box_threshold=pl["box_threshold"],
                text_threshold=pl["text_threshold"],
            )
        else:
            self.labeler = OWLv2Labeler(score_threshold=pl["box_threshold"])

        self.class_names: List[str] = [cls["name"] for cls in c["classes"]]
        self.q_filter = LabelQualityFilter(
            min_score=pl["box_threshold"],
            min_area_ratio=c["annotation"]["min_box_area_ratio"],
            max_area_ratio=c["annotation"]["max_box_area_ratio"],
        )
        self.builder = COCOBuilder(
            class_names=self.class_names,
            output_dir=c["annotation"]["output_dir"],
            train_ratio=c["annotation"]["train_ratio"],
        )
        self.trainer = YOLOTrainer(
            base_model=c["training"]["model"],
            output_dir=c["training"]["output_dir"],
        )

    # ── parallel download helper ───────────────────────────────────

    def _download_and_extract_class(self, cls: dict, cfg: dict) -> List[str]:
        """Download + extract frames for one class. Designed for thread pool."""
        class_name = cls["name"]
        log.info(f"[{class_name}] Starting download…")
        vids = self.downloader.search_and_download(
            class_name=class_name,
            search_queries=cls.get("search_queries"),
            num_videos=cfg["video_search"]["num_videos_per_class"],
            max_duration=cfg["video_search"]["max_duration_seconds"],
        )
        log.info(f"[{class_name}] {len(vids)} videos downloaded — extracting frames…")
        frames = self.extractor.extract_all(vids, class_name)
        log.info(f"[{class_name}] {len(frames)} frames extracted")
        return frames

    # ── main run ───────────────────────────────────────────────────

    def run(
        self,
        dataset_name: str = "auto_det",
        skip_download: bool = False,
        resume: bool = False,
        max_download_workers: int = 3,
    ) -> str:
        """
        Execute the full pipeline.

        Args:
            dataset_name: name used for dataset folder and model run
            skip_download: reuse frames already in data/frames/
            resume: if True, skip already-completed steps using checkpoint
            max_download_workers: number of parallel threads for download+extract

        Returns path to trained model weights.
        """
        c = self.cfg
        ckpt = PipelineCheckpoint(run_name=dataset_name)

        if resume:
            ckpt.print_status()

        log.info("=" * 60)
        log.info(f"Pipeline start | classes: {self.class_names}")
        log.info("=" * 60)

        # ── Steps 1-2: videos → frames ──────────────────────────────
        if resume and ckpt.is_done("extract"):
            log.info("[Resume] Loading frame list from checkpoint…")
            all_frames: List[str] = ckpt.load("extract")
        elif skip_download:
            frames_dir = Path(c["frame_extraction"]["output_dir"])
            all_frames = [str(p) for p in frames_dir.rglob("*.jpg")]
            log.info(f"[Reuse] {len(all_frames)} existing frames")
        else:
            log.info(f"Downloading & extracting {len(c['classes'])} classes "
                     f"(parallel workers: {max_download_workers})…")
            all_frames = []
            with ThreadPoolExecutor(max_workers=max_download_workers) as pool:
                futures = {
                    pool.submit(self._download_and_extract_class, cls, c): cls["name"]
                    for cls in c["classes"]
                }
                for future in as_completed(futures):
                    cls_name = futures[future]
                    try:
                        frames = future.result()
                        all_frames.extend(frames)
                    except Exception as e:
                        log.error(f"[{cls_name}] download/extract failed: {e}")

            ckpt.save("extract", all_frames)

        log.info(f"Total frames: {len(all_frames)}")

        # ── Step 3: CLIP pre-filter ──────────────────────────────────
        if resume and ckpt.is_done("clip_filter"):
            log.info("[Resume] Loading CLIP-filtered frames from checkpoint…")
            all_frames = ckpt.load("clip_filter")
        elif self.clip is not None:
            log.info("[CLIP filter]")
            filtered: List[str] = []
            for cls in c["classes"]:
                tag = cls["name"].replace(" ", "_")
                cls_frames = [f for f in all_frames if tag in f]
                kept, _ = self.clip.filter_frames(cls_frames, cls["name"])
                filtered.extend(kept)
            all_frames = filtered
            ckpt.save("clip_filter", all_frames)

        log.info(f"Frames after CLIP filter: {len(all_frames)}")

        # ── Steps 4-5: pseudo-label + quality filter ─────────────────
        if resume and ckpt.is_done("quality_filter"):
            log.info("[Resume] Loading detections from checkpoint…")
            detections = ckpt.load("quality_filter")
        else:
            if resume and ckpt.is_done("pseudo_label"):
                log.info("[Resume] Loading raw detections from checkpoint…")
                raw = ckpt.load("pseudo_label")
            else:
                log.info(f"[Pseudo-labeling] {type(self.labeler).__name__}…")
                raw = self.labeler.label_batch(all_frames, self.class_names)
                ckpt.save("pseudo_label", raw)

            log.info("[Quality filter]")
            detections = self.q_filter.filter_batch(raw)

            # Class balance warning
            check_class_balance(detections)

            # Temporal consistency filter
            if c.get("temporal_filter", {}).get("enabled", False):
                detections = temporal_consistency_filter(
                    detections,
                    min_frame_appearances=c["temporal_filter"].get(
                        "min_appearances", 2
                    ),
                )

            ckpt.save("quality_filter", detections)

        log.info(f"Usable annotated images: {len(detections)}")

        if len(detections) == 0:
            raise RuntimeError(
                "No usable pseudo-labels generated. "
                "Try lowering box_threshold or similarity_threshold in config."
            )

        # ── Steps 6-7: dataset + training ──────────────────────────
        st = c["self_training"]
        if st["enabled"] and st["rounds"] > 0:
            log.info(f"[Self-training] rounds={st['rounds']}")
            model_path = SelfTrainer(
                class_names=self.class_names,
                teacher=self.labeler,
                quality_filter=self.q_filter,
                coco_builder=self.builder,
                yolo_trainer=self.trainer,
                rounds=st["rounds"],
            ).run(all_frames, detections, dataset_name=f"{dataset_name}_st")
        else:
            if resume and ckpt.is_done("build_dataset"):
                data_yaml = ckpt.load("build_dataset")
                log.info(f"[Resume] Using existing dataset: {data_yaml}")
            else:
                log.info("[Build dataset]")
                data_yaml = self.builder.build_dataset(detections, dataset_name)
                ckpt.save("build_dataset", data_yaml)

            log.info("[Train YOLOv8]")
            model_path = self.trainer.train(
                data_yaml=data_yaml,
                epochs=c["training"]["epochs"],
                batch_size=c["training"]["batch_size"],
                imgsz=c["training"]["imgsz"],
                run_name=dataset_name,
            )
            ckpt.save("train", model_path)

        log.info("=" * 60)
        log.info(f"Done.  Model: {model_path}")
        log.info("=" * 60)
        return model_path
