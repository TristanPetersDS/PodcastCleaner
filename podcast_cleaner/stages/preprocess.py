"""Stage 2: Preprocess audio — resample to 48kHz, mono, 32-bit float WAV."""

from __future__ import annotations

import logging
from pathlib import Path

from podcast_cleaner.analysis.audio_stats import compute_stats, save_stage_report
from podcast_cleaner.utils import (
    ensure_dir,
    is_done,
    mark_done,
    read_audio,
    setup_logging,
    write_audio,
)

logger = logging.getLogger(__name__)


def run_preprocess(
    episode_dir: str,
    config: dict,
    stage_logger: logging.Logger | None = None,
) -> None:
    """Resample all audio in raw/ to target sample rate, mono, float32."""
    log = stage_logger or logger
    episode_path = Path(episode_dir)
    pre_config = config.get("preprocess", {})
    target_sr = pre_config.get("sample_rate", 48000)

    if is_done(episode_path, "preprocess"):
        log.info("Preprocess: skipping (already done)")
        return

    raw_dir = episode_path / "raw"
    if not raw_dir.exists():
        log.error(f"No raw/ directory found in {episode_dir}")
        return

    audio_files = list(raw_dir.glob("*.wav")) + list(raw_dir.glob("*.mp3")) + list(raw_dir.glob("*.m4a"))
    if not audio_files:
        log.error(f"No audio files found in {raw_dir}")
        return

    out_dir = ensure_dir(episode_path / "preprocessed")

    for audio_path in audio_files:
        log.info(f"Preprocessing: {audio_path.name}")
        audio, sr = read_audio(str(audio_path), target_sr=target_sr)
        out_path = out_dir / f"{audio_path.stem}.wav"
        write_audio(str(out_path), audio, target_sr)
        duration = len(audio) / target_sr
        log.info(f"  Output: {out_path.name} ({duration:.1f}s, {target_sr}Hz, mono, float32)")

        # Analyze
        stats = compute_stats(str(out_path))
        report_path = str(episode_path / "analysis" / "audio_report.json")
        save_stage_report(report_path, "preprocessed", stats)

    mark_done(episode_path, "preprocess")
