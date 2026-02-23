"""Stage 3: Source separation using Demucs — isolate vocals from background."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torchaudio

from podcast_cleaner.analysis.audio_stats import compute_stats, save_stage_report
from podcast_cleaner.utils import ensure_dir, get_device, is_done, mark_done

logger = logging.getLogger(__name__)


def demucs_separate(audio_path: str, model_name: str, device: torch.device) -> dict:
    """Run Demucs separation on a single audio file.

    Returns dict with 'vocals', 'no_vocals' tensors and 'sample_rate'.
    """
    import demucs.api

    separator = demucs.api.Separator(model=model_name, device=str(device))

    try:
        origin, separated = separator.separate_audio_file(audio_path)
    except torch.cuda.OutOfMemoryError:
        logger.warning("GPU OOM — falling back to CPU")
        separator = demucs.api.Separator(model=model_name, device="cpu")
        origin, separated = separator.separate_audio_file(audio_path)

    vocals = separated.get("vocals")
    no_vocals = separated.get("no_vocals")

    # If two_stems wasn't used, reconstruct background
    if no_vocals is None:
        no_vocals = sum(v for k, v in separated.items() if k != "vocals")

    return {
        "vocals": vocals,
        "no_vocals": no_vocals,
        "sample_rate": separator.samplerate,
    }


def run_separate(
    episode_dir: str,
    config: dict,
    stage_logger: logging.Logger | None = None,
) -> None:
    """Separate vocals from background for all preprocessed audio."""
    log = stage_logger or logger
    episode_path = Path(episode_dir)
    sep_config = config.get("separation", {})

    if is_done(episode_path, "separate"):
        log.info("Separate: skipping (already done)")
        return

    pre_dir = episode_path / "preprocessed"
    if not pre_dir.exists():
        raise FileNotFoundError(f"No preprocessed/ directory in {episode_dir}")

    wav_files = list(pre_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No WAV files in {pre_dir}")

    device = get_device(sep_config.get("device", "auto"))
    model_name = sep_config.get("model", "htdemucs_ft")
    log.info(f"Separation: model={model_name}, device={device}")

    out_dir = ensure_dir(episode_path / "separated")

    target_sr = config.get("preprocess", {}).get("sample_rate", 48000)

    for wav_path in wav_files:
        log.info(f"Separating: {wav_path.name}")
        result = demucs_separate(str(wav_path), model_name, device)

        stem = wav_path.stem
        sr = result["sample_rate"]
        vocals = result["vocals"].cpu()
        no_vocals = result["no_vocals"].cpu()

        # Resample to pipeline target rate if Demucs outputs at a different rate
        if sr != target_sr:
            log.info(f"  Resampling from {sr}Hz to {target_sr}Hz")
            vocals = torchaudio.functional.resample(vocals, sr, target_sr)
            no_vocals = torchaudio.functional.resample(no_vocals, sr, target_sr)
            sr = target_sr

        vocals_path = out_dir / f"{stem}_vocals.wav"
        bg_path = out_dir / f"{stem}_background.wav"

        torchaudio.save(str(vocals_path), vocals, sr)
        torchaudio.save(str(bg_path), no_vocals, sr)

        log.info(f"  Vocals: {vocals_path.name}")
        log.info(f"  Background: {bg_path.name}")

        # Analyze vocals stem
        stats = compute_stats(str(vocals_path))
        report_path = str(episode_path / "analysis" / "audio_report.json")
        save_stage_report(report_path, "separated", stats)

    mark_done(episode_path, "separate")
