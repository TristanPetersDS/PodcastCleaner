"""Stage 1: Download audio from YouTube using yt-dlp."""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

from podcast_cleaner.utils import ensure_dir, is_done, mark_done, sanitize_filename, setup_logging

logger = logging.getLogger(__name__)


def get_playlist_entries(url: str) -> list[dict]:
    """Fetch playlist metadata without downloading."""
    result = subprocess.run(
        ["yt-dlp", "--flat-playlist", "--dump-json", url],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error(f"yt-dlp metadata fetch failed: {result.stderr[:500]}")
        return []

    entries = []
    for line in result.stdout.strip().split("\n"):
        if line.strip():
            entries.append(json.loads(line))
    return entries


def build_episode_dirname(title: str, playlist_index: int | None) -> str:
    """Build a directory name with optional playlist ordering prefix.

    Args:
        title: Episode title.
        playlist_index: Zero-based playlist index, or None for single videos.

    Returns:
        e.g. "01_My-Episode" or "My-Episode" (no index).
    """
    safe = sanitize_filename(title).replace(" ", "-")
    if playlist_index is not None:
        return f"{playlist_index + 1:02d}_{safe}"
    return safe


def download_single(
    url: str,
    output_dir: str,
    format_pref: str = "wav",
) -> str | None:
    """Download a single video's audio track.

    Returns path to the downloaded file, or None on failure.
    """
    output_template = str(Path(output_dir) / "%(title)s.%(ext)s")
    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format", format_pref,
        "--audio-quality", "0",
        "--no-playlist",
        "-o", output_template,
        url,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"yt-dlp failed: {result.stderr[:500]}")
        return None

    # Find the downloaded file (yt-dlp may change the extension)
    out_dir = Path(output_dir)
    audio_files = list(out_dir.glob("*.wav")) + list(out_dir.glob("*.opus")) + list(out_dir.glob("*.m4a"))
    if not audio_files:
        logger.error(f"No audio file found in {output_dir} after download")
        return None

    return str(max(audio_files, key=lambda p: p.stat().st_mtime))


def run_download(
    url: str,
    output_base: str,
    config: dict,
    stage_logger: logging.Logger | None = None,
) -> list[str]:
    """Download all episodes from a URL (playlist or single video).

    Returns list of episode directory paths.
    """
    log = stage_logger or logger
    dl_config = config.get("download", {})
    format_pref = dl_config.get("format", "wav")

    entries = get_playlist_entries(url)

    if not entries:
        # Might be a single video — try direct download
        log.info("No playlist entries found — treating as single video")
        entries = [{"id": url, "title": "download", "_url": url}]

    log.info(f"Found {len(entries)} episode(s) to download")
    downloaded_dirs = []

    for i, entry in enumerate(entries):
        title = entry.get("title", entry.get("id", f"episode_{i}"))
        is_single = len(entries) == 1 and "_url" in entry
        dirname = build_episode_dirname(title, None if is_single else i)
        episode_dir = ensure_dir(Path(output_base) / dirname)
        raw_dir = ensure_dir(episode_dir / "raw")

        log.info(f"[{i+1}/{len(entries)}] {title}")

        if is_done(episode_dir, "download"):
            log.info("  Skipping (already downloaded)")
            downloaded_dirs.append(str(episode_dir))
            continue

        video_url = entry.get("_url", f"https://www.youtube.com/watch?v={entry['id']}")
        result = download_single(video_url, str(raw_dir), format_pref)

        if result:
            mark_done(episode_dir, "download")
            downloaded_dirs.append(str(episode_dir))
            log.info(f"  Downloaded: {result}")
        else:
            log.error(f"  Failed to download: {title}")

    return downloaded_dirs
