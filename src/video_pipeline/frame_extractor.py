"""
Extract sharp, evenly-spaced frames from video files using OpenCV.

Why frame extraction matters
─────────────────────────────
Videos contain enormous temporal redundancy: adjacent frames differ by only a
few pixels (motion blur, small movements). Running a pseudo-labeler on every
frame at full video FPS would:
  - Process thousands of near-identical frames (wasteful)
  - Produce near-identical annotations for most frames (redundant training data)
  - Include many blurry frames from camera shake or motion (noisy training data)

This extractor solves all three problems by:
  1. Sampling frames at a configurable low rate (default 0.5 fps = 1 frame / 2s)
  2. Skipping blurry frames using the Laplacian variance blur detector

Laplacian blur detection
──────────────────────────
A sharp image has high spatial frequency content: hard edges, fine details.
Applying the Laplacian operator (a second-order spatial derivative) to a
grayscale image produces high-magnitude values at sharp edges and near-zero
values in smooth regions. In a blurry image, edges are softened → lower
Laplacian magnitude → lower variance of the Laplacian map.

Threshold: var(Laplacian(gray)) < min_blur_variance → blurry.
Default min_blur_variance=100.0 works well for typical camera footage.
Increase it to be more aggressive (keep only very sharp frames).
Decrease it to accept moderately blurry frames (useful for low-quality sources).

Output format
─────────────
Frames are saved as JPEG at quality 90 (good balance of size vs. quality).
Filename pattern: {video_stem}_{frame_index:06d}.jpg
  e.g. dQw4w9WgXcQ_000300.jpg = frame 300 of video dQw4w9WgXcQ.mp4
The 6-digit zero-padded index makes filenames sort in temporal order.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List
from tqdm import tqdm


class FrameExtractor:
    """
    Extracts non-blurry frames from video files at a target FPS.

    Frames are stored in class-named subdirectories under output_dir so that
    CLIPFilter and the pseudo-labelers can process them per-class.
    """

    def __init__(
        self,
        output_dir: str = "data/frames",
        fps: float = 0.5,
        max_frames: int = 100,
        min_blur_variance: float = 100.0,
    ):
        """
        Args:
            output_dir:         Root directory for extracted frames. Class subdirs
                                are created automatically.
            fps:                Target extraction rate in frames per second.
                                0.5 fps = 1 frame every 2 seconds.
                                1.0 fps = 1 frame per second.
                                Higher values give more training data but more
                                temporal redundancy.
            max_frames:         Maximum frames to extract per video. Prevents
                                very long videos from dominating the dataset with
                                frames from a single video.
            min_blur_variance:  Minimum Laplacian variance to accept a frame.
                                Frames below this are skipped as blurry.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.max_frames = max_frames
        self.min_blur = min_blur_variance

    def _is_blurry(self, frame: np.ndarray) -> bool:
        """
        Return True if the frame is too blurry to be useful for training.

        Method: compute the variance of the Laplacian of the grayscale frame.
        The Laplacian highlights edges; a sharp image has many strong edges →
        high variance. A blurry image has soft/no edges → low variance.

        This is a fast O(H*W) operation — much cheaper than running a DNN — so
        it adds negligible time to the extraction loop.

        Args:
            frame: BGR numpy array as returned by cv2.VideoCapture.read().
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # CV_64F (64-bit float) preserves precision for the variance computation;
        # using CV_8U would truncate negative Laplacian values to 0
        return cv2.Laplacian(gray, cv2.CV_64F).var() < self.min_blur

    def extract_frames(self, video_path: str, class_name: str) -> List[str]:
        """
        Extract sharp, evenly-spaced frames from a single video file.

        Sampling strategy:
          interval = max(1, int(video_fps / target_fps))
          e.g. video at 30fps, target 0.5fps → interval = 60 (every 60th frame)

          We use cap.set(CAP_PROP_POS_FRAMES) to seek directly to each target
          frame rather than reading and discarding intermediate frames. This is
          faster but may be slightly inaccurate for some container formats (e.g.,
          B-frames in H.264). For our purposes (pseudo-label training data)
          this level of accuracy is sufficient.

        Args:
            video_path: Path to the video file (.mp4, .webm, .mkv, etc.).
            class_name: Used to name the output subdirectory.

        Returns:
            List of absolute JPEG file paths for saved frames.
        """
        video_path = Path(video_path)
        # Class subdirectory: spaces in class names → underscores in paths
        out_dir = self.output_dir / class_name.replace(" ", "_")
        out_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Cannot open: {video_path}")
            return []

        # Query video metadata: FPS and total frame count
        # CAP_PROP_FPS may return 0 for some container formats; default to 30
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Number of video frames to skip between extractions
        interval = max(1, int(video_fps / self.fps))

        saved = 0        # count of accepted (non-blurry) frames written to disk
        frame_idx = 0    # current position in the video (in frame units)
        paths = []

        while saved < self.max_frames and frame_idx < total_frames:
            # Seek directly to frame_idx — avoids decoding all intermediate frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break  # end of video or read error

            if not self._is_blurry(frame):
                # Zero-padded 6-digit frame index makes output filenames sort
                # in temporal order and avoids collisions within a class dir
                out = out_dir / f"{video_path.stem}_{frame_idx:06d}.jpg"
                # IMWRITE_JPEG_QUALITY 90 = good quality with ~3-4× size savings
                # vs quality 100; pseudolabelers are robust to JPEG compression
                cv2.imwrite(str(out), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                paths.append(str(out))
                saved += 1

            frame_idx += interval  # advance by one sampling interval

        cap.release()  # free the VideoCapture handle and file descriptor
        return paths

    def extract_all(self, video_paths: List[str], class_name: str) -> List[str]:
        """
        Extract frames from a list of videos for a single class.

        Args:
            video_paths: Paths to video files (output of YouTubeDownloader).
            class_name:  Target class name (used for subdirectory naming).

        Returns:
            Combined list of all extracted frame paths across all videos.
        """
        all_frames: List[str] = []
        for vp in tqdm(video_paths, desc=f"Extracting [{class_name}]"):
            all_frames.extend(self.extract_frames(vp, class_name))
        return all_frames
