"""Tests for the preprocess stage."""

from pathlib import Path

import numpy as np
import soundfile as sf

from podcast_cleaner.stages.preprocess import run_preprocess
from podcast_cleaner.utils import is_done


def test_resamples_to_target_sr(tmp_path):
    """Input at 22050Hz should be resampled to 48000Hz."""
    # Create 22050Hz input
    sr_in = 22050
    audio = np.sin(np.linspace(0, 1, sr_in)).astype(np.float32)
    episode_dir = tmp_path / "ep"
    raw_dir = episode_dir / "raw"
    raw_dir.mkdir(parents=True)
    sf.write(str(raw_dir / "test.wav"), audio, sr_in)

    config = {"preprocess": {"sample_rate": 48000, "channels": 1}}
    run_preprocess(str(episode_dir), config)

    # Check output
    out_path = episode_dir / "preprocessed" / "test.wav"
    assert out_path.exists()
    out_audio, out_sr = sf.read(str(out_path))
    assert out_sr == 48000
    assert out_audio.ndim == 1  # mono


def test_stereo_to_mono(tmp_path):
    """Stereo input should be converted to mono."""
    sr = 48000
    stereo = np.random.randn(sr, 2).astype(np.float32) * 0.1
    episode_dir = tmp_path / "ep"
    raw_dir = episode_dir / "raw"
    raw_dir.mkdir(parents=True)
    sf.write(str(raw_dir / "test.wav"), stereo, sr)

    config = {"preprocess": {"sample_rate": 48000, "channels": 1}}
    run_preprocess(str(episode_dir), config)

    out_path = episode_dir / "preprocessed" / "test.wav"
    out_audio, _ = sf.read(str(out_path))
    assert out_audio.ndim == 1


def test_marks_done(tmp_path):
    """Preprocess should write .done marker on success."""
    sr = 48000
    audio = np.zeros(sr, dtype=np.float32)
    episode_dir = tmp_path / "ep"
    raw_dir = episode_dir / "raw"
    raw_dir.mkdir(parents=True)
    sf.write(str(raw_dir / "test.wav"), audio, sr)

    config = {"preprocess": {"sample_rate": 48000, "channels": 1}}
    run_preprocess(str(episode_dir), config)
    assert is_done(episode_dir, "preprocess")


def test_skips_if_done(tmp_path):
    """Should skip processing if .done marker exists."""
    episode_dir = tmp_path / "ep"
    episode_dir.mkdir(parents=True)
    (episode_dir / ".preprocess.done").touch()

    config = {"preprocess": {"sample_rate": 48000, "channels": 1}}
    run_preprocess(str(episode_dir), config)
    # No raw/ dir exists, but should not fail because it's skipped
    assert not (episode_dir / "preprocessed").exists()
