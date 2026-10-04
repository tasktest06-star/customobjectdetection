"""
Enhanced YouTube downloader.

Improvements over v1:
  - Video ID deduplication: skips videos already on disk (survives restarts)
  - LLM-augmented queries: Claude API generates richer, diverse search queries
    with a rule-based fallback when the API is unavailable
  - Metadata saving: title, channel, description saved per video as JSON
    (useful for downstream quality filtering, e.g. skip cartoon channels)
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import yt_dlp

from src.utils.logger import get_logger

log = get_logger("downloader")


class YouTubeDownloader:
    def __init__(self, download_dir: str = "data/videos"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)

    # ── query generation ───────────────────────────────────────────

    def _rule_based_queries(self, class_name: str) -> List[str]:
        """Diverse rule-based queries — used when LLM is unavailable."""
        return [
            f"{class_name} video",
            f"{class_name} close up",
            f"{class_name} in the wild",
            f"how to identify {class_name}",
            f"{class_name} documentary footage",
        ]

    def generate_queries_with_llm(
        self, class_name: str, n_queries: int = 6
    ) -> List[str]:
        """
        Use Claude to generate diverse, visual search queries for class_name.
        Falls back to rule-based queries if the anthropic package is absent
        or the API key is not set.
        """
        try:
            import anthropic

            client = anthropic.Anthropic()
            prompt = (
                f"Generate {n_queries} diverse YouTube search queries to find "
                f"real-world video footage of '{class_name}' for an object detection "
                f"training dataset. Focus on:\n"
                f"- Different lighting conditions (day, night, indoor, outdoor)\n"
                f"- Different distances (closeup, medium shot, wide)\n"
                f"- Different contexts (wild, zoo, backyard, street)\n\n"
                f"Return ONLY the queries, one per line, no numbering or extra text."
            )
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            queries = [
                line.strip()
                for line in msg.content[0].text.strip().splitlines()
                if line.strip()
            ]
            log.info(f"LLM generated {len(queries)} queries for '{class_name}'")
            return queries[:n_queries]
        except Exception as e:
            log.warning(f"LLM query generation failed ({e}), using rule-based fallback")
            return self._rule_based_queries(class_name)

    # ── deduplication ──────────────────────────────────────────────

    def _id_cache_path(self, class_dir: Path) -> Path:
        return class_dir / ".downloaded_ids.json"

    def _load_downloaded_ids(self, class_dir: Path) -> set:
        path = self._id_cache_path(class_dir)
        if path.exists():
            with open(path) as f:
                return set(json.load(f))
        return set()

    def _save_downloaded_id(self, class_dir: Path, video_id: str) -> None:
        ids = self._load_downloaded_ids(class_dir)
        ids.add(video_id)
        with open(self._id_cache_path(class_dir), "w") as f:
            json.dump(list(ids), f)

    # ── metadata ───────────────────────────────────────────────────

    def _save_metadata(self, class_dir: Path, entry: Dict) -> None:
        """Persist title, channel, description, duration, view count."""
        if not entry:
            return
        meta = {
            "id": entry.get("id"),
            "title": entry.get("title"),
            "channel": entry.get("channel") or entry.get("uploader"),
            "description": (entry.get("description") or "")[:500],
            "duration": entry.get("duration"),
            "view_count": entry.get("view_count"),
            "upload_date": entry.get("upload_date"),
            "webpage_url": entry.get("webpage_url"),
        }
        meta_path = class_dir / f"{entry.get('id', 'unknown')}_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    # ── download ───────────────────────────────────────────────────

    def search_and_download(
        self,
        class_name: str,
        search_queries: Optional[List[str]] = None,
        num_videos: int = 10,
        max_duration: int = 300,
        use_llm_queries: bool = False,
    ) -> List[str]:
        """
        Search YouTube and download videos for a class.

        Args:
            use_llm_queries: If True, attempt to generate queries via LLM.
                             Falls back to rule-based if API unavailable.
        Returns list of local video file paths.
        """
        if search_queries is None:
            search_queries = (
                self.generate_queries_with_llm(class_name)
                if use_llm_queries
                else self._rule_based_queries(class_name)
            )

        class_dir = self.download_dir / class_name.replace(" ", "_")
        class_dir.mkdir(parents=True, exist_ok=True)
        already_downloaded = self._load_downloaded_ids(class_dir)
        log.info(f"[{class_name}] {len(already_downloaded)} videos already on disk")

        downloaded: List[str] = []
        per_query = max(1, num_videos // len(search_queries))

        for query in search_queries:
            if len(downloaded) >= num_videos:
                break
            paths = self._download_query(
                query, class_dir, per_query, max_duration, already_downloaded
            )
            downloaded.extend(paths)

        log.info(f"[{class_name}] {len(downloaded)} new videos downloaded")
        return downloaded[:num_videos]

    def _download_query(
        self,
        query: str,
        class_dir: Path,
        num_videos: int,
        max_duration: int,
        already_downloaded: set,
    ) -> List[str]:
        ydl_opts = {
            "format": "best[height<=480]/best",
            "outtmpl": str(class_dir / "%(id)s.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "match_filter": yt_dlp.utils.match_filter_func(
                f"duration <= {max_duration}"
            ),
            "ignoreerrors": True,
        }

        downloaded: List[str] = []
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(
                    f"ytsearch{num_videos}:{query}", download=True
                )
                if not (info and "entries" in info):
                    return downloaded

                for entry in info["entries"] or []:
                    if not entry:
                        continue
                    vid_id = entry.get("id", "")
                    if vid_id in already_downloaded:
                        log.debug(f"Skip duplicate: {vid_id}")
                        continue
                    fp = ydl.prepare_filename(entry)
                    if os.path.exists(fp):
                        downloaded.append(fp)
                        self._save_downloaded_id(class_dir, vid_id)
                        self._save_metadata(class_dir, entry)
                        already_downloaded.add(vid_id)
            except Exception as e:
                log.warning(f"Download error for query '{query}': {e}")

        return downloaded
