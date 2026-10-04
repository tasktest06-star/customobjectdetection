"""Extract sharp, evenly-spaced frames from video files using OpenCV."""
import cv2
import numpy as np
from pathlib import Path
from typing import List
from tqdm import tqdm


class FrameExtractor:
    """Extracts non-blurry frames from video files at a target FPS."""

    def __init__(
        self,
        output_dir: str = "data/frames",
        fps: float = 0.5,
        max_frames: int = 100,
        min_blur_variance: float = 100.0,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.max_frames = max_frames
        self.min_blur = min_blur_variance

    def _is_blurry(self, frame: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var() < self.min_blur

    def extract_frames(self, video_path: str, class_name: str) -> List[str]:
        """Extract frames from one video. Returns list of saved JPEG paths."""
        video_path = Path(video_path)
        out_dir = self.output_dir / class_name.replace(" ", "_")
        out_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Cannot open: {video_path}")
            return []

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = max(1, int(video_fps / self.fps))

        saved, frame_idx, paths = 0, 0, []
        while saved < self.max_frames and frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            if not self._is_blurry(frame):
                out = out_dir / f"{video_path.stem}_{frame_idx:06d}.jpg"
                cv2.imwrite(str(out), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                paths.append(str(out))
                saved += 1
            frame_idx += interval

        cap.release()
        return paths

    def extract_all(self, video_paths: List[str], class_name: str) -> List[str]:
        """Extract frames from multiple videos."""
        all_frames: List[str] = []
        for vp in tqdm(video_paths, desc=f"Extracting [{class_name}]"):
            all_frames.extend(self.extract_frames(vp, class_name))
        return all_frames
