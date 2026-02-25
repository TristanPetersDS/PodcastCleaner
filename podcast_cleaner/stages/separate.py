"""Stage 3: Source separation using Demucs — isolate vocals from background."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from podcast_cleaner.analysis.audio_stats import compute_stats, save_stage_report
from podcast_cleaner.utils import ensure_dir, get_device, is_done, mark_done

logger = logging.getLogger(__name__)


def _split_audio_segments(
    audio: np.ndarray, max_segment_samples: int, overlap_samples: int
) -> list[np.ndarray]:
    """Split audio into overlapping segments for chunked processing."""
    total = len(audio)
    if total <= max_segment_samples:
        return [audio]

    segments = []
    pos = 0
    step = max_segment_samples - overlap_samples

    while pos < total:
        end = min(pos + max_segment_samples, total)
        segments.append(audio[pos:end])
        if end >= total:
            break
        pos += step

    return segments


def _crossfade_segments(
    segments: list[np.ndarray], overlap_samples: int
) -> np.ndarray:
    """Reassemble segments with Hann-windowed crossfade."""
    if len(segments) == 1:
        return segments[0]

    # Raised-cosine (Hann) crossfade: sin^2 + cos^2 = 1.0 at all points
    t = np.linspace(0.0, 0.5 * np.pi, overlap_samples, dtype=np.float32)
    fade_in = np.sin(t) ** 2
    fade_out = np.cos(t) ** 2

    result_parts = []
    for i, seg in enumerate(segments):
        if i == 0:
            tail = seg[-overlap_samples:].copy()
            tail *= fade_out
            result_parts.append(seg[:-overlap_samples])
            result_parts.append(tail)
        elif i == len(segments) - 1:
            head = seg[:overlap_samples].copy()
            head *= fade_in
            result_parts[-1] = result_parts[-1] + head
            result_parts.append(seg[overlap_samples:])
        else:
            head = seg[:overlap_samples].copy()
            head *= fade_in
            result_parts[-1] = result_parts[-1] + head
            body = seg[overlap_samples:-overlap_samples]
            result_parts.append(body)
            tail = seg[-overlap_samples:].copy()
            tail *= fade_out
            result_parts.append(tail)

    return np.concatenate(result_parts)


def demucs_separate(audio_path: str, model_name: str, device, model=None, max_segment_minutes: float = 10) -> dict:
    """Run Demucs separation on a single audio file.

    Returns dict with 'vocals', 'no_vocals' tensors and 'sample_rate'.
    If *model* is provided it is reused instead of loading a fresh copy.
    """
    import torch
    import torchaudio
    from demucs.apply import apply_model

    if model is None:
        from demucs.pretrained import get_model
        model = get_model(model_name)
        model.to(device)

    # Load audio at the model's native sample rate
    wav, sr = torchaudio.load(audio_path)
    if sr != model.samplerate:
        wav = torchaudio.functional.resample(wav, sr, model.samplerate)
    # Demucs expects (batch, channels, time) — ensure stereo
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    wav = wav.unsqueeze(0).to(device)  # (1, channels, time)

    # Clamp to avoid Demucs pad1d assertion errors with edge values
    wav = wav.clamp(-0.999, 0.999)

    try:
        sources = apply_model(model, wav, device=device, split=True, overlap=0.25, shifts=1)
    except (torch.cuda.OutOfMemoryError, AssertionError) as e:
        if isinstance(e, AssertionError):
            logger.warning("Demucs assertion error (audio may be too long for GPU memory) — retrying on CPU")
        else:
            logger.warning("GPU memory exceeded for Demucs separation — falling back to CPU (this will be slower)")
        import gc
        model.to("cpu")
        wav = wav.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        sources = apply_model(model, wav, device="cpu", split=True, overlap=0.25, shifts=0)

    # sources shape: (1, num_sources, channels, time)
    # Find vocals index from model.sources
    source_names = model.sources
    vocals_idx = source_names.index("vocals") if "vocals" in source_names else 0

    vocals = sources[0, vocals_idx]  # (channels, time)
    # Sum all non-vocal sources for background
    non_vocal_indices = [i for i in range(len(source_names)) if i != vocals_idx]
    no_vocals = sources[0, non_vocal_indices].sum(dim=0)

    # Mix to mono: average channels
    vocals = vocals.mean(dim=0, keepdim=True)  # (1, time)
    no_vocals = no_vocals.mean(dim=0, keepdim=True)  # (1, time)

    return {
        "vocals": vocals,
        "no_vocals": no_vocals,
        "sample_rate": model.samplerate,
    }


def _load_demucs_model(model_name: str, device):
    """Load a Demucs model and move it to *device*."""
    from demucs.pretrained import get_model
    model = get_model(model_name)
    model.to(device)
    return model


def run_separate(
    episode_dir: str,
    config: dict,
    stage_logger: logging.Logger | None = None,
) -> None:
    """Separate vocals from background for all preprocessed audio."""
    import torch
    import torchaudio

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

    # Load the Demucs model once for all files in this episode
    model = _load_demucs_model(model_name, device)

    for wav_path in wav_files:
        log.info(f"Separating: {wav_path.name}")
        max_seg = sep_config.get("max_segment_minutes", 10)
        result = demucs_separate(str(wav_path), model_name, device, model=model, max_segment_minutes=max_seg)

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

    # Release GPU memory for subsequent stages
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    mark_done(episode_path, "separate")
