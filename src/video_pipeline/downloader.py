"""
YouTube video downloader using yt-dlp (Unlicense / public domain).

Search strategy: yt-dlp's "ytsearch{N}:{query}" format searches YouTube without
requiring an API key. Videos are filtered to ≤ max_duration seconds before
download to avoid long-form content where the target class appears only briefly.
"""
import os
from pathlib import Path
from typing import List, Optional
import yt_dlp


class YouTubeDownloader:
    def __init__(self, download_dir: str = "data/videos"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)

    def _default_queries(self, class_name: str) -> List[str]:
        return [
            f"{class_name} video",
            f"{class_name} close up",
            f"how to identify {class_name}",
        ]

    def search_and_download(
        self,
        class_name: str,
        search_queries: Optional[List[str]] = None,
        num_videos: int = 10,
        max_duration: int = 300,
    ) -> List[str]:
        if search_queries is None:
            search_queries = self._default_queries(class_name)
        class_dir = self.download_dir / class_name.replace(" ", "_")
        class_dir.mkdir(parents=True, exist_ok=True)
        downloaded: List[str] = []
        per_query = max(1, num_videos // len(search_queries))
        for query in search_queries:
            paths = self._download_query(query, class_dir, per_query, max_duration)
            downloaded.extend(paths)
            if len(downloaded) >= num_videos:
                break
        return downloaded[:num_videos]

    def _download_query(
        self, query: str, output_dir: Path, num_videos: int, max_duration: int
    ) -> List[str]:
        ydl_opts = {
            "format": "best[height<=480]/best",
            "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
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
                if info and "entries" in info:
                    for entry in info["entries"] or []:
                        if entry:
                            fp = ydl.prepare_filename(entry)
                            if os.path.exists(fp):
                                downloaded.append(fp)
            except Exception as e:
                print(f"Download error for '{query}': {e}")
        return downloaded
