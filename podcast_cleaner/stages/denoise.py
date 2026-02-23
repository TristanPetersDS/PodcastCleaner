"""Stage 4: Noise removal using DeepFilterNet3 on the vocals stem."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from podcast_cleaner.analysis.audio_stats import compute_stats, save_stage_report
from podcast_cleaner.utils import ensure_dir, is_done, mark_done, write_audio

logger = logging.getLogger(__name__)


def deepfilter_enhance(audio_path: str) -> tuple[np.ndarray, int]:
    """Run DeepFilterNet3 enhancement on an audio file.

    Returns (enhanced_audio, sample_rate).
    """
    from df.enhance import enhance, init_df, load_audio, save_audio

    model, df_state, _ = init_df()
    audio, _ = load_audio(audio_path, sr=df_state.sr())
    enhanced = enhance(model, df_state, audio)

    # Convert to numpy
    enhanced_np = enhanced.squeeze().numpy() if hasattr(enhanced, "numpy") else np.array(enhanced)
    return enhanced_np, df_state.sr()


def run_denoise(
    episode_dir: str,
    config: dict,
    stage_logger: logging.Logger | None = None,
) -> None:
    """Denoise all vocal stems in the separated/ directory."""
    log = stage_logger or logger
    episode_path = Path(episode_dir)

    if is_done(episode_path, "denoise"):
        log.info("Denoise: skipping (already done)")
        return

    # Look for vocals from separation, fall back to preprocessed
    sep_dir = episode_path / "separated"
    vocal_files = list(sep_dir.glob("*_vocals.wav")) if sep_dir.exists() else []
    if not vocal_files:
        pre_dir = episode_path / "preprocessed"
        vocal_files = list(pre_dir.glob("*.wav")) if pre_dir.exists() else []
    if not vocal_files:
        log.error(f"No input audio found for denoising in {episode_dir}")
        return

    out_dir = ensure_dir(episode_path / "denoised")
    log.info(f"Denoising {len(vocal_files)} file(s)...")

    for vocal_path in vocal_files:
        log.info(f"  Denoising: {vocal_path.name}")

        enhanced, sr = deepfilter_enhance(str(vocal_path))

        # Build output filename
        stem = vocal_path.stem.replace("_vocals", "")
        out_path = out_dir / f"{stem}_denoised.wav"
        write_audio(str(out_path), enhanced, sr)
        log.info(f"  Output: {out_path.name}")

        # Analyze
        stats = compute_stats(str(out_path))
        report_path = str(episode_path / "analysis" / "audio_report.json")
        save_stage_report(report_path, "denoised", stats)

    mark_done(episode_path, "denoise")
