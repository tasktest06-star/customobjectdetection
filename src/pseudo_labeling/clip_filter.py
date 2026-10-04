"""
Enhanced CLIP filter.

Improvements over v1:
  - Negative prompt support: subtract similarity to undesired concepts
    (cartoons, drawings, text images) from the positive score, reducing
    false-positive frames that pass on visual similarity alone
  - Score caching: saves {image_path: score} to a JSON file so you can
    re-threshold the entire frame set in milliseconds without re-running CLIP
  - Per-class threshold: each class can have its own similarity threshold
    (useful when some classes are visually distinctive and others are ambiguous)
"""
import json
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

from src.utils.logger import get_logger

log = get_logger("clip_filter")

# Default negative prompts applied to every class.
# These reduce frames showing non-photographic content.
DEFAULT_NEGATIVE_PROMPTS = [
    "a cartoon drawing",
    "an illustration",
    "a painting",
    "a diagram",
    "text on a screen",
    "a logo",
]


class CLIPFilter:
    """
    CLIP-based frame pre-filter with negative prompt support and score caching.

    Net score formula:
        score = sim(image, positive_avg) - alpha * sim(image, negative_avg)

    Frames with score >= threshold are kept.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        similarity_threshold: float = 0.20,
        batch_size: int = 32,
        device: Optional[str] = None,
        # negative prompt settings
        negative_prompts: Optional[List[str]] = None,
        negative_weight: float = 0.3,
        # caching
        cache_dir: Optional[str] = "data/.clip_cache",
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = similarity_threshold
        self.batch_size = batch_size
        self.negative_prompts = (
            negative_prompts
            if negative_prompts is not None
            else DEFAULT_NEGATIVE_PROMPTS
        )
        self.negative_weight = negative_weight
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"Loading CLIP: {model_name} on {self.device}")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    # ── text embedding helpers ─────────────────────────────────────

    def _mean_text_features(self, prompts: List[str]):
        """Encode a list of prompts and return their mean unit vector."""
        inputs = self.processor(
            text=prompts, return_tensors="pt", padding=True
        ).to(self.device)
        with torch.no_grad():
            feats = self.model.get_text_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            mean = feats.mean(dim=0, keepdim=True)
            return mean / mean.norm(dim=-1, keepdim=True)

    def _positive_prompts(self, class_name: str) -> List[str]:
        return [
            f"a photo of a {class_name}",
            f"a clear image of {class_name}",
            f"a real photograph of {class_name}",
            class_name,
        ]

    # ── cache helpers ──────────────────────────────────────────────

    def _cache_path(self, class_name: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{class_name.replace(' ', '_')}_scores.json"

    def _load_cache(self, class_name: str) -> Dict[str, float]:
        p = self._cache_path(class_name)
        if p and p.exists():
            with open(p) as f:
                log.debug(f"Loaded CLIP score cache for '{class_name}' from {p}")
                return json.load(f)
        return {}

    def _save_cache(self, class_name: str, scores: Dict[str, float]) -> None:
        p = self._cache_path(class_name)
        if p:
            with open(p, "w") as f:
                json.dump(scores, f, indent=2)
            log.debug(f"Saved CLIP score cache for '{class_name}' to {p}")

    # ── main filter ────────────────────────────────────────────────

    def filter_frames(
        self,
        image_paths: List[str],
        class_name: str,
        threshold_override: Optional[float] = None,
    ) -> Tuple[List[str], List[float]]:
        """
        Filter frames by net CLIP similarity.

        Args:
            threshold_override: Use this threshold instead of self.threshold
                                 for this call only (useful for per-class tuning).

        Returns:
            (kept_paths, net_scores)
        """
        threshold = threshold_override if threshold_override is not None else self.threshold

        # Load existing cache
        cached = self._load_cache(class_name)
        uncached_paths = [p for p in image_paths if p not in cached]

        # Compute text features
        pos_feats = self._mean_text_features(self._positive_prompts(class_name))
        neg_feats = (
            self._mean_text_features(self.negative_prompts)
            if self.negative_prompts
            else None
        )

        # Score uncached frames
        new_scores: Dict[str, float] = {}
        for i in tqdm(
            range(0, len(uncached_paths), self.batch_size),
            desc=f"CLIP [{class_name}]",
            disable=len(uncached_paths) == 0,
        ):
            batch_paths = uncached_paths[i : i + self.batch_size]
            imgs, valid = [], []
            for p in batch_paths:
                try:
                    imgs.append(Image.open(p).convert("RGB"))
                    valid.append(p)
                except Exception:
                    pass
            if not imgs:
                continue

            inputs = self.processor(
                images=imgs, return_tensors="pt", padding=True
            ).to(self.device)
            with torch.no_grad():
                img_feats = self.model.get_image_features(**inputs)
                img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

                pos_sim = (img_feats @ pos_feats.T).squeeze(1)
                if neg_feats is not None:
                    neg_sim = (img_feats @ neg_feats.T).squeeze(1)
                    net = pos_sim - self.negative_weight * neg_sim
                else:
                    net = pos_sim

                for path, score in zip(valid, net.cpu().numpy()):
                    new_scores[path] = float(score)

        # Merge caches and persist
        all_scores = {**cached, **new_scores}
        self._save_cache(class_name, all_scores)

        # Apply threshold
        kept = [p for p in image_paths if all_scores.get(p, -1.0) >= threshold]
        scores_out = [all_scores[p] for p in kept]

        log.info(
            f"CLIP [{class_name}]: kept {len(kept)}/{len(image_paths)} "
            f"(threshold={threshold:.2f}, neg_weight={self.negative_weight})"
        )
        return kept, scores_out
