"""Shared utilities: audio I/O, logging, stage markers."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".aac"}


def get_device(preference: str = "auto"):
    """Resolve 'auto' to CUDA if available, else CPU."""
    import torch

    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


def setup_logging(log_path: str | Path, stage_name: str) -> logging.Logger:
    """Configure a logger that writes to both file and stdout.

    Uses a fixed logger name to prevent handler accumulation across episodes.
    """
    logger = logging.getLogger("podcast_cleaner.pipeline")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    # Close and remove existing file handlers
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            handler.close()
            logger.removeHandler(handler)

    # Add file handler for this episode
    fh = logging.FileHandler(str(log_path), mode="a")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Add console handler only once
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in logger.handlers):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    return logger


def read_audio(path: str | Path, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    """Read audio file, optionally resample. Returns (samples, sample_rate)."""
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)  # stereo → mono
    if target_sr and sr != target_sr:
        import torch
        import torchaudio
        waveform = torch.from_numpy(audio).unsqueeze(0)
        resampled = torchaudio.functional.resample(waveform, sr, target_sr)
        return resampled.squeeze(0).numpy(), target_sr
    return audio, sr


def write_audio(path: str | Path, audio: np.ndarray, sr: int) -> Path:
    """Write audio as 32-bit float WAV."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), audio, sr, subtype="FLOAT")
    return out


def mark_done(episode_dir: str | Path, stage_name: str) -> None:
    """Write a hidden .done marker for a completed stage."""
    marker = Path(episode_dir) / f".{stage_name}.done"
    marker.touch()


def is_done(episode_dir: str | Path, stage_name: str) -> bool:
    """Check if a stage has already completed."""
    marker = Path(episode_dir) / f".{stage_name}.done"
    return marker.exists()


def clear_done(episode_dir: str | Path, stage_name: str) -> None:
    """Remove a .done marker (for re-running a stage)."""
    marker = Path(episode_dir) / f".{stage_name}.done"
    marker.unlink(missing_ok=True)


def sanitize_filename(name: str) -> str:
    """Make a string safe for use as a filename."""
    safe = "".join(c if c.isalnum() or c in " -_." else "_" for c in name).strip()
    # Collapse repeated underscores/spaces
    while "  " in safe:
        safe = safe.replace("  ", " ")
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_. ")


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it doesn't exist, return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
