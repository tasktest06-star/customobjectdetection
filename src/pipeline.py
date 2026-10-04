"""
Top-level orchestrator.

Steps:
  1. Download YouTube videos per class (yt-dlp)
  2. Extract sharp frames (OpenCV)
  3. Pre-filter frames with CLIP similarity
  4. Generate pseudo-labels with Grounding DINO or OWLv2
  5. Filter pseudo-labels (confidence, NMS, size)
  6. Build COCO JSON + YOLO data.yaml
  7. Fine-tune YOLOv8  [+ optional self-training loop]
"""
import yaml
from pathlib import Path
from typing import List, Optional

from src.video_pipeline.downloader import YouTubeDownloader
from src.video_pipeline.frame_extractor import FrameExtractor
from src.pseudo_labeling.clip_filter import CLIPFilter
from src.pseudo_labeling.grounding_dino import GroundingDINOLabeler
from src.pseudo_labeling.owl_vit import OWLv2Labeler
from src.annotation.quality_filter import LabelQualityFilter
from src.annotation.coco_builder import COCOBuilder
from src.training.yolo_trainer import YOLOTrainer
from src.training.self_trainer import SelfTrainer


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

    def run(
        self,
        dataset_name: str = "auto_det",
        skip_download: bool = False,
    ) -> str:
        """
        Execute the full pipeline. Returns path to trained model weights.
        Set skip_download=True to reuse frames already in data/frames/.
        """
        c = self.cfg
        print(f"\n{'='*60}")
        print(f"Pipeline starting  |  classes: {self.class_names}")
        print(f"{'='*60}")

        # ── Steps 1-2: videos → frames ─────────────────────────────
        all_frames: List[str] = []
        if not skip_download:
            for cls in c["classes"]:
                print(f"\n[Download] {cls['name']}")
                vids = self.downloader.search_and_download(
                    class_name=cls["name"],
                    search_queries=cls.get("search_queries"),
                    num_videos=c["video_search"]["num_videos_per_class"],
                    max_duration=c["video_search"]["max_duration_seconds"],
                )
                print(f"  {len(vids)} videos downloaded")
                frames = self.extractor.extract_all(vids, cls["name"])
                print(f"  {len(frames)} frames extracted")
                all_frames.extend(frames)
        else:
            frames_dir = Path(c["frame_extraction"]["output_dir"])
            all_frames = [str(p) for p in frames_dir.rglob("*.jpg")]
            print(f"[Reuse] {len(all_frames)} existing frames")

        # ── Step 3: CLIP pre-filter ─────────────────────────────────
        if self.clip is not None:
            print("\n[CLIP filter]")
            filtered: List[str] = []
            for cls in c["classes"]:
                tag = cls["name"].replace(" ", "_")
                cls_frames = [f for f in all_frames if tag in f]
                kept, _ = self.clip.filter_frames(cls_frames, cls["name"])
                filtered.extend(kept)
            all_frames = filtered

        print(f"\nFrames after CLIP filter: {len(all_frames)}")

        # ── Steps 4-5: pseudo-label + quality filter ────────────────
        print(f"\n[Pseudo-labeling] using {type(self.labeler).__name__}")
        raw = self.labeler.label_batch(all_frames, self.class_names)
        detections = self.q_filter.filter_batch(raw)
        print(f"Usable annotated images: {len(detections)}/{len(raw)}")

        if len(detections) == 0:
            raise RuntimeError(
                "No usable pseudo-labels generated. "
                "Try lowering box_threshold or similarity_threshold in config."
            )

        # ── Steps 6-7: dataset + training ──────────────────────────
        st = c["self_training"]
        if st["enabled"] and st["rounds"] > 0:
            print(f"\n[Self-training]  rounds={st['rounds']}")
            model_path = SelfTrainer(
                class_names=self.class_names,
                teacher=self.labeler,
                quality_filter=self.q_filter,
                coco_builder=self.builder,
                yolo_trainer=self.trainer,
                rounds=st["rounds"],
            ).run(all_frames, detections, dataset_name=f"{dataset_name}_st")
        else:
            print("\n[Build dataset]")
            data_yaml = self.builder.build_dataset(detections, dataset_name)
            print("\n[Train YOLOv8]")
            model_path = self.trainer.train(
                data_yaml=data_yaml,
                epochs=c["training"]["epochs"],
                batch_size=c["training"]["batch_size"],
                imgsz=c["training"]["imgsz"],
                run_name=dataset_name,
            )

        print(f"\n{'='*60}")
        print(f"Done.  Model: {model_path}")
        print(f"{'='*60}\n")
        return model_path
