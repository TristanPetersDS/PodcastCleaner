"""Stage 6: Loudness normalization to podcast standard (-16 LUFS)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pyloudnorm as pyln

from podcast_cleaner.analysis.audio_stats import compute_stats, save_stage_report
from podcast_cleaner.utils import (
    ensure_dir,
    is_done,
    mark_done,
    read_audio,
    write_audio,
)

logger = logging.getLogger(__name__)


def true_peak_limit(audio: np.ndarray, max_dbtp: float = -1.5) -> np.ndarray:
    """Apply simple true peak limiting.

    Scales the entire signal down so that the peak does not exceed
    *max_dbtp* dBTP.  If the signal is already below the threshold
    it is returned unchanged.
    """
    max_linear = 10 ** (max_dbtp / 20.0)
    peak = float(np.max(np.abs(audio)))
    if peak > max_linear:
        audio = audio * (max_linear / peak)
    return audio


def run_normalize(
    episode_dir: str,
    config: dict,
    stage_logger: logging.Logger | None = None,
) -> None:
    """Normalize audio loudness to podcast standard.

    Reads denoised (or separated vocal) audio, normalizes to the target
    LUFS using *pyloudnorm*, applies true-peak limiting, and writes the
    results to the ``normalized/`` sub-directory.
    """
    log = stage_logger or logger
    episode_path = Path(episode_dir)
    norm_config = config.get("normalization", {})
    target_lufs = norm_config.get("target_lufs", -16.0)
    target_tp = norm_config.get("true_peak_dbtp", -1.5)

    if is_done(episode_path, "normalize"):
        log.info("Normalize: skipping (already done)")
        return

    # Find denoised audio
    denoised_dir = episode_path / "denoised"
    audio_files = list(denoised_dir.glob("*_denoised.wav")) if denoised_dir.exists() else []
    if not audio_files:
        # Fall back to separated vocals
        sep_dir = episode_path / "separated"
        audio_files = list(sep_dir.glob("*_vocals.wav")) if sep_dir.exists() else []
    if not audio_files:
        raise FileNotFoundError(f"No audio files found to normalize in {episode_dir}")

    out_dir = ensure_dir(episode_path / "normalized")

    for audio_path in audio_files:
        log.info(f"Normalizing: {audio_path.name}")
        audio, sr = read_audio(str(audio_path))

        meter = pyln.Meter(sr)
        current_lufs = meter.integrated_loudness(audio)
        log.info(f"  Input: {current_lufs:.1f} LUFS, target: {target_lufs} LUFS")

        if np.isinf(current_lufs):
            log.warning("  Cannot measure loudness (silence?) — skipping")
            continue

        normalized = pyln.normalize.loudness(audio, current_lufs, target_lufs)
        normalized = true_peak_limit(normalized, max_dbtp=target_tp)

        stem = audio_path.stem.replace("_denoised", "").replace("_vocals", "")
        out_path = out_dir / f"{stem}_normalized.wav"
        write_audio(str(out_path), normalized, sr)

        # Verify
        final_lufs = meter.integrated_loudness(normalized)
        final_peak = 20 * np.log10(float(np.max(np.abs(normalized))) + 1e-10)
        log.info(f"  Output: {final_lufs:.1f} LUFS, peak: {final_peak:.1f} dBFS")

        # Analyze
        stats = compute_stats(str(out_path))
        report_path = str(episode_path / "analysis" / "audio_report.json")
        save_stage_report(report_path, "normalized", stats)

    mark_done(episode_path, "normalize")
