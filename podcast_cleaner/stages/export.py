"""Stage 7: Export final audio in distribution formats (MP3, FLAC)."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from podcast_cleaner.utils import ensure_dir, is_done, mark_done

logger = logging.getLogger(__name__)


def convert_audio(
    input_path: str,
    output_path: str,
    format_name: str,
    bitrate: str = "320k",
    sample_rate: int = 48000,
) -> None:
    """Convert audio using ffmpeg."""
    cmd = ["ffmpeg", "-y", "-i", input_path, "-ar", str(sample_rate)]
    if format_name == "mp3":
        cmd.extend(["-c:a", "libmp3lame", "-b:a", bitrate])
    elif format_name == "flac":
        cmd.extend(["-c:a", "flac"])
    elif format_name == "wav":
        cmd.extend(["-c:a", "pcm_s16le"])
    cmd.append(output_path)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg export failed: {result.stderr[:500]}")


def run_export(
    episode_dir: str,
    config: dict,
    stage_logger: logging.Logger | None = None,
) -> None:
    """Export normalized audio in configured formats."""
    log = stage_logger or logger
    episode_path = Path(episode_dir)
    export_config = config.get("export", {})
    formats = export_config.get("formats", ["mp3", "flac"])
    bitrate = export_config.get("mp3_bitrate", "320k")
    sample_rate = export_config.get("sample_rate", 48000)

    if is_done(episode_path, "export"):
        log.info("Export: skipping (already done)")
        return

    # Find normalized audio
    norm_dir = episode_path / "normalized"
    audio_files = list(norm_dir.glob("*_normalized.wav")) if norm_dir.exists() else []
    if not audio_files:
        log.error(f"No normalized audio found in {episode_dir}")
        return

    final_dir = ensure_dir(episode_path / "final")
    episode_name = episode_path.name

    for audio_path in audio_files:
        for fmt in formats:
            out_name = f"{episode_name}_clean.{fmt}"
            out_path = str(final_dir / out_name)
            log.info(f"Exporting: {out_name}")
            try:
                convert_audio(str(audio_path), out_path, fmt, bitrate, sample_rate)
                size_mb = Path(out_path).stat().st_size / (1024 * 1024)
                log.info(f"  {out_name} ({size_mb:.1f} MB)")
            except RuntimeError as e:
                log.error(f"  Export failed: {e}")

    mark_done(episode_path, "export")
    log.info("Export complete")
