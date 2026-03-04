"""Stage 4: Noise removal using DeepFilterNet3 on the vocals stem."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from podcast_cleaner.analysis.audio_stats import compute_stats, save_stage_report
from podcast_cleaner.utils import ensure_dir, is_done, mark_done, write_audio

logger = logging.getLogger(__name__)

# Module-level flag: once we fall back to CPU, stay on CPU for all future calls
_fallen_back_to_cpu = False


def reset_denoise_state() -> None:
    """Reset module-level state between episodes.

    Allows GPU to be retried for each new episode rather than staying
    on CPU for the entire process after one fallback.
    """
    global _fallen_back_to_cpu
    _fallen_back_to_cpu = False


def deepfilter_enhance(
    audio_path: str, model=None, df_state=None
) -> tuple[np.ndarray, int]:
    """Run DeepFilterNet3 enhancement on an audio file.

    If *model* and *df_state* are provided they are reused instead of
    loading a fresh copy.  Processes in 60-second chunks with 0.5s
    overlap to avoid OOM on long files.
    Returns (enhanced_audio, sample_rate).
    """
    global _fallen_back_to_cpu
    import gc
    import os

    import torch
    from df.enhance import enhance, init_df, load_audio

    # If we previously fell back to CPU, stay on CPU
    if _fallen_back_to_cpu:
        os.environ["DF_DEVICE"] = "cpu"

    # Clear GPU cache from previous stages
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()

    if model is None or df_state is None:
        model, df_state, _ = init_df()
    sr = df_state.sr()
    audio, _ = load_audio(audio_path, sr=sr)

    # Process in chunks to avoid OOM on long files
    chunk_seconds = 60
    overlap_seconds = 0.5
    chunk_samples = chunk_seconds * sr
    overlap_samples = int(overlap_seconds * sr)
    total_samples = audio.shape[-1]

    if total_samples <= chunk_samples:
        # Short file — process in one go
        try:
            with torch.backends.cudnn.flags(enabled=False):
                enhanced = enhance(model, df_state, audio)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if isinstance(e, torch.cuda.OutOfMemoryError):
                logger.warning(
                    "GPU memory exceeded for DeepFilterNet — falling back to CPU"
                )
            else:
                logger.warning(
                    "GPU compatibility issue (cuDNN/GRU) — retrying DeepFilterNet on CPU"
                )
            del model
            gc.collect()
            torch.cuda.empty_cache()
            os.environ["DF_DEVICE"] = "cpu"
            model, df_state, _ = init_df()
            audio, _ = load_audio(audio_path, sr=sr)
            enhanced = enhance(model, df_state, audio)
            _fallen_back_to_cpu = True
        enhanced_np = (
            enhanced.squeeze().numpy()
            if hasattr(enhanced, "numpy")
            else np.array(enhanced)
        )
        if np.any(np.isnan(enhanced_np)):
            logger.warning(
                "  NaN values detected in DeepFilterNet output — replacing with zeros (audio quality may be affected)"
            )
            enhanced_np = np.nan_to_num(enhanced_np, nan=0.0)
        return enhanced_np, sr

    # Chunked processing for long files
    logger.info(
        f"  Processing in {chunk_seconds}s chunks ({total_samples / sr:.0f}s total)"
    )
    enhanced_chunks = []
    pos = 0

    while pos < total_samples:
        end = min(pos + chunk_samples, total_samples)
        chunk = audio[..., pos:end]

        try:
            with torch.backends.cudnn.flags(enabled=False):
                enhanced_chunk = enhance(model, df_state, chunk)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if isinstance(e, torch.cuda.OutOfMemoryError):
                logger.warning(
                    "GPU memory exceeded for DeepFilterNet chunk — falling back to CPU"
                )
            else:
                logger.warning(
                    "GPU compatibility issue (cuDNN/GRU) on chunk — retrying DeepFilterNet on CPU"
                )
            del model
            gc.collect()
            torch.cuda.empty_cache()
            os.environ["DF_DEVICE"] = "cpu"
            model, df_state, _ = init_df()
            # Move FULL audio tensor to CPU so all remaining chunks are CPU
            audio = audio.cpu() if hasattr(audio, "cpu") else audio
            chunk = audio[..., pos:end]
            enhanced_chunk = enhance(model, df_state, chunk)
            _fallen_back_to_cpu = True

        chunk_np = (
            enhanced_chunk.squeeze().numpy()
            if hasattr(enhanced_chunk, "numpy")
            else np.array(enhanced_chunk)
        )

        # Sanitize NaN values that DeepFilterNet can produce
        if np.any(np.isnan(chunk_np)):
            logger.warning(
                f"  NaN values detected in DeepFilterNet output chunk at pos={pos} — replacing with zeros (audio quality may be affected)"
            )
            chunk_np = np.nan_to_num(chunk_np, nan=0.0)

        if pos == 0:
            enhanced_chunks.append(chunk_np)
        else:
            # Crossfade overlap region
            fade_len = min(overlap_samples, len(enhanced_chunks[-1]), len(chunk_np))
            if fade_len > 0:
                fade_out = np.linspace(1, 0, fade_len, dtype=np.float32)
                fade_in = np.linspace(0, 1, fade_len, dtype=np.float32)
                enhanced_chunks[-1][-fade_len:] *= fade_out
                chunk_np[:fade_len] *= fade_in
                enhanced_chunks[-1][-fade_len:] += chunk_np[:fade_len]
                enhanced_chunks.append(chunk_np[fade_len:])
            else:
                enhanced_chunks.append(chunk_np)

        pos += chunk_samples - overlap_samples

    return np.concatenate(enhanced_chunks), sr


def _load_deepfilter_model():
    """Load the DeepFilterNet3 model and return (model, df_state)."""
    from df.enhance import init_df

    model, df_state, _ = init_df()
    return model, df_state


def run_denoise(
    episode_dir: str,
    config: dict,
    stage_logger: logging.Logger | None = None,
) -> None:
    """Denoise all vocal stems in the separated/ directory."""
    log = stage_logger or logger
    reset_denoise_state()
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
        raise FileNotFoundError(f"No input audio found for denoising in {episode_dir}")

    out_dir = ensure_dir(episode_path / "denoised")
    log.info(f"Denoising {len(vocal_files)} file(s)...")

    # Load the DeepFilterNet model once for all files in this episode
    model, df_state = _load_deepfilter_model()

    for vocal_path in vocal_files:
        log.info(f"  Denoising: {vocal_path.name}")

        enhanced, sr = deepfilter_enhance(
            str(vocal_path), model=model, df_state=df_state
        )

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
