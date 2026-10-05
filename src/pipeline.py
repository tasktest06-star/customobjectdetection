"""
Top-level pipeline orchestrator.

Stages:
  1. YouTube video download              — yt-dlp (Unlicense)
  2. Frame extraction + blur filter      — OpenCV (Apache 2.0)
  3. CLIP pre-filter (optional)          — CLIP/transformers (MIT)
  4. Zero-shot pseudo-labeling           — GroundingDINO / OWLv2 / VLM (Apache 2.0)
  5. Quality filter (NMS + size)         — numpy (BSD)
  6. COCO dataset builder                — custom (Apache 2.0)
  7. RT-DETR fine-tuning                 — HF transformers OR PaddleDetection (Apache 2.0)
     [optional: teacher-student self-training loop]

Training backends:
  "hf"      → src/training/rtdetr_trainer.py       (PyTorch + HuggingFace)
  "paddle"  → src/training/paddle_rtdetr_trainer.py (PaddlePaddle + PaddleDetection)

Pseudo-labeling backends:
  "grounding_dino" → GroundingDINO via transformers (Apache 2.0)
  "owl_vit"        → OWLv2 via transformers (Apache 2.0)
  "vlm_qwen2"      → Qwen2-VL grounding (Apache 2.0)
  "vlm_internvl2"  → InternVL2 grounding (Apache 2.0)
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
from src.training.rtdetr_trainer import RTDETRTrainer
from src.training.self_trainer import SelfTrainer


def _build_labeler(cfg: dict):
    """
    Instantiate the pseudo-labeler specified in config.

    Supported labeler values:
      "grounding_dino"  → GroundingDINOLabeler (Apache 2.0, default)
      "owl_vit"         → OWLv2Labeler (Apache 2.0)
      "vlm_qwen2"       → Qwen2VLLabeler (Apache 2.0, needs qwen-vl-utils)
      "vlm_internvl2"   → InternVL2Labeler (Apache 2.0, needs einops timm)

    Models are loaded lazily here — instantiation triggers the HuggingFace download
    and GPU memory allocation. If you're running stages separately (e.g. download only),
    call pipeline.run(skip_download=True, skip_labeling=True) to avoid loading models
    you don't need.
    """
    pl = cfg["pseudo_labeling"]
    labeler_type = pl["labeler"]

    if labeler_type == "grounding_dino":
        return GroundingDINOLabeler(
            model_id=pl["model"],
            box_threshold=pl["box_threshold"],
            text_threshold=pl.get("text_threshold", max(0.01, pl["box_threshold"] - 0.05)),
        )
    elif labeler_type == "owl_vit":
        return OWLv2Labeler(
            model_id=pl["model"],
            score_threshold=pl["box_threshold"],
        )
    elif labeler_type in ("vlm_qwen2", "vlm_internvl2"):
        from src.pseudo_labeling.vlm_labeler import build_vlm_labeler
        backend = "qwen2_vl" if labeler_type == "vlm_qwen2" else "internvl2"
        return build_vlm_labeler(
            backend=backend,
            model_id=pl.get("model"),
        )
    else:
        raise ValueError(
            f"Unknown labeler: {labeler_type!r}. "
            "Valid: grounding_dino, owl_vit, vlm_qwen2, vlm_internvl2"
        )


def _build_trainer(cfg: dict):
    """
    Instantiate the fine-tuning trainer specified in config.

    training.backend:
      "hf"      → RTDETRTrainer (HuggingFace transformers, PyTorch)
      "paddle"  → PaddleRTDETRTrainer (PaddleDetection, PaddlePaddle)
                  Requires training.paddle_det_path set in config.
    """
    tr = cfg["training"]
    backend = tr.get("backend", "hf")

    if backend == "hf":
        return RTDETRTrainer(
            base_model=tr["model"],
            output_dir=tr["output_dir"],
        )
    elif backend == "paddle":
        from src.training.paddle_rtdetr_trainer import PaddleRTDETRTrainer
        paddle_path = tr.get("paddle_det_path")
        if not paddle_path:
            raise ValueError(
                "training.paddle_det_path must be set when training.backend='paddle'. "
                "Point it to your PaddleDetection clone directory."
            )
        return PaddleRTDETRTrainer(
            paddle_det_path=paddle_path,
            backbone=tr.get("backbone", "r50vd"),
            output_dir=tr["output_dir"],
        )
    else:
        raise ValueError(
            f"Unknown training backend: {backend!r}. Valid: 'hf', 'paddle'"
        )


class ObjectDetectionPipeline:
    """
    End-to-end pipeline: class names → trained RT-DETR detector.

    Supports two training backends (HuggingFace transformers or PaddleDetection)
    and four pseudo-labeling backends (GroundingDINO, OWLv2, Qwen2-VL, InternVL2).
    All components are Apache 2.0 licensed.

    Usage:
        pipeline = ObjectDetectionPipeline("configs/pipeline_config.yaml")
        model_path = pipeline.run(dataset_name="my_run")
    """

    def __init__(self, config_path: str = "configs/pipeline_config.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        self.class_names: List[str] = [cls["name"] for cls in self.cfg["classes"]]
        self._components_built = False

    def _build_components(self):
        """
        Instantiate all pipeline components from the current config.

        Called lazily on first run() call (not at __init__) to avoid loading
        GPU models when the user is only checking config or file structure.
        Re-calling after --classes override rebuilds labeler-dependent components.
        """
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
            if cf.get("enabled", True)
            else None
        )

        # Build labeler (may load a large model — deferred until here)
        self.labeler = _build_labeler(c)

        pl = c["pseudo_labeling"]
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

        # Build trainer (PaddleDetection or HF)
        self.trainer = _build_trainer(c)
        self._components_built = True

    def run(
        self,
        dataset_name: str = "auto_det",
        skip_download: bool = False,
    ) -> str:
        """
        Execute all pipeline stages and return the path to the trained model.

        Args:
            dataset_name:  Name prefix for output directories.
            skip_download: Reuse frames from data/frames/ instead of downloading.

        Returns:
            Path to the trained model (HuggingFace dir for HF backend;
            best_model dir or .pdparams path for PaddleDetection backend).
        """
        if not self._components_built:
            self._build_components()

        c = self.cfg
        print(f"\n{'='*60}")
        print(f"Pipeline  |  classes={self.class_names}")
        print(f"Backend   |  {c['training'].get('backend','hf').upper()}")
        print(f"Labeler   |  {c['pseudo_labeling']['labeler']}")
        print(f"{'='*60}")

        # ── Stages 1-2: download + extract ─────────────────────────────────
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

        # ── Stage 3: CLIP pre-filter ────────────────────────────────────────
        if self.clip is not None:
            print("\n[CLIP filter]")
            filtered: List[str] = []
            for cls in c["classes"]:
                tag = cls["name"].replace(" ", "_")
                # Match on the parent directory name (which is the class tag),
                # not a substring of the full path. Substring matching would cause
                # class "cat" to select frames from the "catfish" directory because
                # "cat" is in the string "catfish".
                cls_frames = [f for f in all_frames if Path(f).parent.name == tag]
                kept, _ = self.clip.filter_frames(cls_frames, cls["name"])
                filtered.extend(kept)
            all_frames = filtered

        print(f"\nFrames after CLIP filter: {len(all_frames)}")

        # ── Stages 4-5: pseudo-labeling + quality filter ────────────────────
        print(f"\n[Pseudo-labeling] {type(self.labeler).__name__}")
        raw = self.labeler.label_batch(all_frames, self.class_names)
        detections = self.q_filter.filter_batch(raw)
        print(f"Usable annotated images: {len(detections)}/{len(raw)}")

        if len(detections) == 0:
            raise RuntimeError(
                "No usable pseudo-labels generated. "
                "Try: lower box_threshold, disable clip_filtering, or add more search_queries."
            )

        # ── Stages 6-7: build dataset + train ──────────────────────────────
        tr = c["training"]
        st = c.get("self_training", {})
        if st.get("enabled") and st.get("rounds", 0) > 0:
            print(f"\n[Self-training]  rounds={st['rounds']}")
            # Note: SelfTrainer only supports HF RTDETRTrainer for student inference.
            # For PaddleDetection backend, disable self_training or use hf for self-training
            # then export to PaddlePaddle for deployment.
            if not isinstance(self.trainer, RTDETRTrainer):
                print(
                    "  Warning: self-training uses the HF student for inference. "
                    "Switching student to HF RTDETRTrainer for self-training rounds."
                )
                student_trainer = RTDETRTrainer(
                    base_model=tr.get("hf_model", "PekingU/rtdetr_r50vd"),
                    output_dir=tr["output_dir"],
                )
            else:
                student_trainer = self.trainer

            model_path = SelfTrainer(
                class_names=self.class_names,
                teacher=self.labeler,
                quality_filter=self.q_filter,
                coco_builder=self.builder,
                rtdetr_trainer=student_trainer,
                rounds=st["rounds"],
                intermediate_epochs=st.get("intermediate_epochs", 30),
                final_epochs=tr.get("epochs", 50),
            ).run(
                all_frames,
                detections,
                dataset_name=f"{dataset_name}_st",
                batch_size=tr.get("batch_size", 8),
                imgsz=tr.get("imgsz", 640),
            )
        else:
            print("\n[Build dataset]")
            data_yaml = self.builder.build_dataset(detections, dataset_name)
            print("\n[Train RT-DETR]")

            # Unified train() call — both backends expose the same signature
            model_path = self.trainer.train(
                data_yaml=data_yaml,
                epochs=tr.get("epochs", 50),
                batch_size=tr.get("batch_size", 8),
                imgsz=tr.get("imgsz", 640),
                learning_rate=tr.get("learning_rate", 1e-4),
                weight_decay=tr.get("weight_decay", 1e-4),
                run_name=dataset_name,
            )

        print(f"\n{'='*60}")
        print(f"Done.  Model saved: {model_path}")
        print(f"{'='*60}\n")
        return model_path
