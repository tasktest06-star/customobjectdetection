"""
YouTube video downloader using yt-dlp.

yt-dlp (Unlicense / public domain) is the maintained fork of youtube-dl.
It supports 1000+ sites but this pipeline uses it exclusively for YouTube
search and download.

Search format
─────────────
yt-dlp accepts a "ytsearch<N>:<query>" URL to search YouTube and return the
top N results, then download them. This avoids needing a YouTube Data API key
(which has tight daily quota limits). Example:
  ytsearch5:raccoon wildlife documentary
  → searches YouTube for "raccoon wildlife documentary" and downloads 5 videos

Video format selection
──────────────────────
"best[height<=480]/best" selects the best quality stream at or below 480p.
480p is sufficient for pseudo-labeling (GroundingDINO/OWLv2 run on 640px crops
anyway) and avoids downloading 1080p/4K files that would slow down the pipeline
and fill disk with unused detail.

Duration filter
───────────────
match_filter_func("duration <= {max_duration}") filters results to videos shorter
than max_duration seconds (default 300 = 5 minutes). This excludes long-form
lectures or full episodes where the target class appears briefly, which would
generate many irrelevant frames.
"""
import os
from pathlib import Path
from typing import List, Optional
import yt_dlp


class YouTubeDownloader:
    """
    Downloads YouTube videos by searching for class-related queries.

    For each class, generates (or uses provided) search queries and downloads
    up to num_videos videos. Videos are saved in class-named subdirectories
    under the download_dir to keep different classes organized.
    """

    def __init__(self, download_dir: str = "data/videos"):
        """
        Args:
            download_dir: Root directory for downloaded videos.
                          Subdirectories are created per class automatically.
        """
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)

    def _default_queries(self, class_name: str) -> List[str]:
        """
        Generate three default search queries for a class name.

        Using multiple queries with different phrasings increases diversity:
          - "{class} video"          → general / overview content
          - "{class} close up"       → close-range shots (better for detection)
          - "how to identify {class}" → educational content with clear subjects

        In practice, 3 queries × 3–4 videos each = 9–12 varied videos per class.
        Custom queries can be set per-class in configs/classes_example.yaml using
        the search_queries field.
        """
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
        """
        Search YouTube and download up to num_videos videos for a class.

        Videos are distributed evenly across search queries: if num_videos=9
        and there are 3 queries, each query downloads 3 videos (9 // 3 = 3).
        Downloading stops early if num_videos are collected before all queries
        are exhausted.

        Args:
            class_name:      Class name used for directory creation and as the
                             base for default search queries.
            search_queries:  List of search queries. If None, uses _default_queries().
                             Set per-class in classes_example.yaml for better results.
            num_videos:      Maximum total videos to download for this class.
            max_duration:    Reject videos longer than this many seconds.
                             Prevents downloading full movies or long lectures.

        Returns:
            List of absolute file paths to successfully downloaded video files.
        """
        if search_queries is None:
            search_queries = self._default_queries(class_name)

        # Spaces in class names would break filesystem paths — replace with underscores
        class_dir = self.download_dir / class_name.replace(" ", "_")
        class_dir.mkdir(parents=True, exist_ok=True)

        downloaded: List[str] = []
        # Distribute num_videos evenly across queries; minimum 1 per query
        per_query = max(1, num_videos // len(search_queries))

        for query in search_queries:
            paths = self._download_query(query, class_dir, per_query, max_duration)
            downloaded.extend(paths)
            # Stop as soon as we have enough videos — avoids redundant queries
            if len(downloaded) >= num_videos:
                break

        # Slice to num_videos in case rounding gave us slightly more
        return downloaded[:num_videos]

    def _download_query(
        self,
        query: str,
        output_dir: Path,
        num_videos: int,
        max_duration: int,
    ) -> List[str]:
        """
        Download up to num_videos videos matching a single search query.

        yt-dlp options:
          format:         prefer 480p or lower to save bandwidth and disk
          outtmpl:        use the video's YouTube ID as filename (unique, safe)
          quiet:          suppress yt-dlp's verbose progress output
          no_warnings:    suppress non-fatal warnings (codec, geo-restriction)
          match_filter:   filter by duration BEFORE downloading (no wasted bandwidth)
          ignoreerrors:   skip unavailable/private videos instead of crashing

        Returns list of paths for successfully downloaded files.
        """
        ydl_opts = {
            "format":       "best[height<=480]/best",
            # %(id)s = YouTube video ID (e.g., "dQw4w9WgXcQ"); unique and filesystem-safe
            "outtmpl":      str(output_dir / "%(id)s.%(ext)s"),
            "quiet":        True,
            "no_warnings":  True,
            # match_filter_func returns a filter callable that yt-dlp applies to
            # each video's metadata before deciding whether to download
            "match_filter": yt_dlp.utils.match_filter_func(
                f"duration <= {max_duration}"
            ),
            "ignoreerrors": True,  # skip private/geo-restricted videos silently
        }

        downloaded: List[str] = []
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                # ytsearch{N}:{query} = search YouTube for top N results and download
                info = ydl.extract_info(
                    f"ytsearch{num_videos}:{query}", download=True
                )
                if info and "entries" in info:
                    for entry in info["entries"] or []:
                        if entry:
                            # prepare_filename() returns the actual path yt-dlp used,
                            # accounting for the file extension chosen by format selection
                            fp = ydl.prepare_filename(entry)
                            if os.path.exists(fp):
                                downloaded.append(fp)
            except Exception as e:
                print(f"Download error for '{query}': {e}")
        return downloaded
