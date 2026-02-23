"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture
def tmp_audio(tmp_path: Path) -> Path:
    """Create a short synthetic audio file for testing (1 second, 48kHz, mono)."""
    sr = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # 440Hz sine wave at -20 dBFS
    amplitude = 10 ** (-20 / 20)
    audio = (amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    path = tmp_path / "test_audio.wav"
    sf.write(str(path), audio, sr, subtype="FLOAT")
    return path


@pytest.fixture
def tmp_episode_dir(tmp_path: Path, tmp_audio: Path) -> Path:
    """Create a minimal episode directory structure with a raw audio file."""
    episode_dir = tmp_path / "01_Test-Episode"
    raw_dir = episode_dir / "raw"
    raw_dir.mkdir(parents=True)
    # Copy test audio to raw/
    import shutil
    shutil.copy(str(tmp_audio), str(raw_dir / "Test-Episode.wav"))
    return episode_dir


@pytest.fixture
def sample_config(tmp_path: Path) -> dict:
    """Return a minimal config dict for testing."""
    return {
        "output_dir": str(tmp_path / "output"),
        "download": {"format": "wav", "quality": "best"},
        "preprocess": {"sample_rate": 48000, "channels": 1},
        "separation": {"model": "htdemucs_ft", "device": "cpu"},
        "denoise": {"model": "DeepFilterNet3"},
        "transcription": {"enabled": False, "model": "large-v3", "language": None, "device": "cpu"},
        "normalization": {"target_lufs": -16.0, "true_peak_dbtp": -1.5},
        "export": {"formats": ["mp3"], "mp3_bitrate": "320k", "sample_rate": 48000},
    }
