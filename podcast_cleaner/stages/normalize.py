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


def _true_peak_limit_legacy(audio: np.ndarray, max_dbtp: float = -1.5) -> np.ndarray:
    """Apply simple true peak limiting (legacy — kept for rollback).

    Scales the entire signal down so that the peak does not exceed
    *max_dbtp* dBTP.  If the signal is already below the threshold
    it is returned unchanged.

    .. deprecated::
        This function scales the ENTIRE signal based on a single peak,
        which undoes loudness normalization for high-crest-factor audio.
        Use :func:`soft_clip_peaks` instead.
    """
    max_linear = 10 ** (max_dbtp / 20.0)
    peak = float(np.max(np.abs(audio)))
    if peak > max_linear:
        audio = audio * (max_linear / peak)
    return audio


def soft_clip_peaks(audio: np.ndarray, max_dbtp: float = -1.5) -> np.ndarray:
    """Soft-clip samples exceeding the true peak threshold.

    Only samples whose absolute value exceeds the linear threshold are
    affected.  Those samples are replaced with a tanh-based soft-clip
    that smoothly compresses them toward the threshold, preserving
    signal polarity.  Samples below the threshold pass through unchanged.

    This preserves the loudness of the body of the signal while taming
    transient peaks — unlike the legacy global-scale approach.
    """
    threshold = 10 ** (max_dbtp / 20.0)
    result = audio.copy()
    above = np.abs(audio) > threshold
    if np.any(above):
        result[above] = (
            np.sign(audio[above])
            * threshold
            * np.tanh(np.abs(audio[above]) / threshold)
        )
    return result


def run_normalize(
    episode_dir: str,
    config: dict,
    stage_logger: logging.Logger | None = None,
) -> None:
    """Normalize audio loudness to podcast standard.

    Reads denoised (or separated vocal) audio, normalizes to the target
    LUFS using *pyloudnorm*, applies selective soft-clipping for peak
    control, and writes the results to the ``normalized/`` sub-directory.

    Algorithm:
      1. Apply ``pyloudnorm.normalize.loudness()`` to reach target LUFS.
      2. Soft-clip peaks exceeding the true peak threshold using
         selective tanh compression (only affects samples above threshold).
      3. Re-measure LUFS.  If more than 1 dB off target, apply iterative
         correction passes (loudness normalize + soft-clip) until within
         tolerance (up to 5 passes).
      4. Hard-clip at +/-1.0 to prevent downstream clipping.
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
    audio_files = (
        list(denoised_dir.glob("*_denoised.wav")) if denoised_dir.exists() else []
    )
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

        # Pass 1: loudness normalize then soft-clip peaks
        normalized = pyln.normalize.loudness(audio, current_lufs, target_lufs)
        normalized = soft_clip_peaks(normalized, max_dbtp=target_tp)

        # Iterative correction: soft-clipping can pull LUFS below target
        # on high-crest-factor audio. Re-normalize and re-clip until within
        # 1 dB of target (up to 5 correction passes to guarantee convergence).
        max_correction_passes = 5
        for pass_num in range(max_correction_passes):
            post_lufs = meter.integrated_loudness(normalized)
            if np.isinf(post_lufs) or abs(post_lufs - target_lufs) <= 1.0:
                break
            log.info(
                f"  Correction pass {pass_num + 1}: post-clip LUFS "
                f"{post_lufs:.1f} dB, re-normalizing to {target_lufs} LUFS"
            )
            normalized = pyln.normalize.loudness(normalized, post_lufs, target_lufs)
            normalized = soft_clip_peaks(normalized, max_dbtp=target_tp)

        # Hard-clip safety net — prevent any sample from exceeding ±1.0
        normalized = np.clip(normalized, -1.0, 1.0)

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
