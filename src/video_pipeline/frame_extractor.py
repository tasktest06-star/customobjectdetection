"""
Enhanced frame extractor.

Improvements over v1:
  - Scene-change detection: extract frames at actual scene boundaries
    instead of at fixed time intervals (more diversity per video)
  - Near-duplicate removal: perceptual hashing (dhash) ensures no two
    extracted frames are visually identical
  - Exposure filter: drop very dark or severely overexposed frames
    that degrade pseudo-label quality
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Set
from tqdm import tqdm

from src.utils.logger import get_logger

log = get_logger("frame_extractor")

try:
    import imagehash
    from PIL import Image as PILImage
    _HAS_IMAGEHASH = True
except ImportError:
    _HAS_IMAGEHASH = False
    log.warning("imagehash not installed — near-duplicate removal disabled. "
                "Install with: pip install imagehash")


class FrameExtractor:
    """
    Extracts sharp, diverse, well-exposed frames from video files.

    Scene-change detection algorithm:
        Compute the chi-squared distance between consecutive frame histograms.
        A distance above `scene_change_threshold` (0–1) marks a new scene.
        This is much cheaper than optical flow and works well for typical
        YouTube footage where scenes cut abruptly.

    Near-duplicate detection:
        Uses dhash (difference hash) — a 64-bit perceptual fingerprint.
        Two frames with Hamming distance ≤ `hash_distance_threshold` are
        considered duplicates; only the first is kept.
    """

    def __init__(
        self,
        output_dir: str = "data/frames",
        fps: float = 0.5,
        max_frames: int = 100,
        min_blur_variance: float = 100.0,
        # scene-change detection
        use_scene_detection: bool = True,
        scene_change_threshold: float = 0.35,
        # near-duplicate removal
        use_dedup: bool = True,
        hash_distance_threshold: int = 8,
        # exposure filter
        min_brightness: float = 30.0,
        max_brightness: float = 225.0,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.max_frames = max_frames
        self.min_blur = min_blur_variance
        self.use_scene = use_scene_detection
        self.scene_thresh = scene_change_threshold
        self.use_dedup = use_dedup and _HAS_IMAGEHASH
        self.hash_dist = hash_distance_threshold
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness

    # ── per-frame quality checks ───────────────────────────────────

    def _is_blurry(self, frame: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var() < self.min_blur

    def _is_poorly_exposed(self, frame: np.ndarray) -> bool:
        """True if the frame is too dark or severely overexposed."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean = gray.mean()
        return mean < self.min_brightness or mean > self.max_brightness

    def _is_near_duplicate(
        self, frame: np.ndarray, seen_hashes: Set
    ) -> bool:
        """True if a perceptually similar frame was already kept."""
        if not self.use_dedup:
            return False
        pil = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        h = imagehash.dhash(pil)
        for prev in seen_hashes:
            if (h - prev) <= self.hash_dist:
                return True
        seen_hashes.add(h)
        return False

    # ── scene-change detection ─────────────────────────────────────

    def _histogram(self, frame: np.ndarray) -> np.ndarray:
        """Flattened normalised BGR histogram for scene comparison."""
        hists = []
        for ch in range(3):
            h = cv2.calcHist([frame], [ch], None, [64], [0, 256])
            hists.append(h)
        hist = np.concatenate(hists).flatten().astype(np.float32)
        return hist / (hist.sum() + 1e-8)

    def _scene_changed(
        self,
        prev_hist: np.ndarray,
        curr_frame: np.ndarray,
    ) -> tuple:
        """
        Returns (changed: bool, new_hist: np.ndarray).
        Uses chi-squared distance between histograms.
        """
        curr_hist = self._histogram(curr_frame)
        diff = np.sum((prev_hist - curr_hist) ** 2 / (prev_hist + curr_hist + 1e-8))
        return diff > self.scene_thresh, curr_hist

    # ── main extraction logic ──────────────────────────────────────

    def extract_frames(self, video_path: str, class_name: str) -> List[str]:
        """
        Extract frames from one video.

        Strategy:
          1. If scene detection is enabled, sample at scene boundaries.
          2. Otherwise fall back to fixed-interval sampling (original behaviour).
          3. All frames are then checked for blur, exposure, and deduplication.
        """
        video_path = Path(video_path)
        out_dir = self.output_dir / class_name.replace(" ", "_")
        out_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            log.warning(f"Cannot open: {video_path}")
            return []

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = max(1, int(video_fps / self.fps))

        saved: List[str] = []
        seen_hashes: Set = set()
        prev_hist: np.ndarray = None
        frame_idx = 0

        while len(saved) < self.max_frames and frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            # Decide whether to consider this frame
            consider = False
            if self.use_scene:
                if prev_hist is None:
                    consider = True
                    prev_hist = self._histogram(frame)
                else:
                    changed, prev_hist = self._scene_changed(prev_hist, frame)
                    consider = changed
            else:
                consider = True

            if consider:
                if (
                    not self._is_blurry(frame)
                    and not self._is_poorly_exposed(frame)
                    and not self._is_near_duplicate(frame, seen_hashes)
                ):
                    out = out_dir / f"{video_path.stem}_{frame_idx:06d}.jpg"
                    cv2.imwrite(str(out), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    saved.append(str(out))

            frame_idx += interval

        cap.release()
        log.debug(
            f"{video_path.name}: {len(saved)} frames saved "
            f"(scene_detection={self.use_scene}, dedup={self.use_dedup})"
        )
        return saved

    def extract_all(self, video_paths: List[str], class_name: str) -> List[str]:
        """Extract frames from multiple videos."""
        all_frames: List[str] = []
        for vp in tqdm(video_paths, desc=f"Extracting [{class_name}]"):
            all_frames.extend(self.extract_frames(vp, class_name))
        log.info(f"[{class_name}] Total frames extracted: {len(all_frames)}")
        return all_frames
