"""Integration test: full pipeline on synthetic audio with mocked ML models."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
import torch

from podcast_cleaner.cli import run_pipeline


@pytest.fixture
def noisy_episode(tmp_path):
    """Create a synthetic noisy episode: 5 seconds of sine + noise at 48kHz."""
    sr = 48000
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Speech-like signal: 440Hz + harmonics
    speech = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.sin(2 * np.pi * 880 * t)
    # Noise
    noise = 0.05 * np.random.randn(len(t))
    audio = (speech + noise).astype(np.float32)

    episode_dir = tmp_path / "output" / "01_Test-Episode"
    raw_dir = episode_dir / "raw"
    raw_dir.mkdir(parents=True)
    sf.write(str(raw_dir / "Test-Episode.wav"), audio, sr, subtype="FLOAT")
    return episode_dir, tmp_path


@pytest.fixture
def mock_config(tmp_path):
    return {
        "output_dir": str(tmp_path / "output"),
        "preprocess": {"sample_rate": 48000, "channels": 1},
        "separation": {"model": "htdemucs_ft", "device": "cpu"},
        "denoise": {"model": "DeepFilterNet3"},
        "transcription": {"enabled": False},
        "normalization": {"target_lufs": -16.0, "true_peak_dbtp": -1.5},
        "export": {"formats": ["mp3"], "mp3_bitrate": "320k", "sample_rate": 48000},
    }


class TestFullPipeline:
    @patch("podcast_cleaner.stages.separate._load_demucs_model")
    @patch("podcast_cleaner.stages.separate.demucs_separate")
    @patch("podcast_cleaner.stages.denoise._load_deepfilter_model")
    @patch("podcast_cleaner.stages.denoise.deepfilter_enhance")
    def test_pipeline_produces_final_output(
        self, mock_denoise, mock_load_df, mock_demucs, mock_load_demucs, noisy_episode, mock_config
    ):
        episode_dir, tmp_path = noisy_episode

        # Read the actual audio for mocks
        audio, sr = sf.read(str(episode_dir / "raw" / "Test-Episode.wav"), dtype="float32")

        # Mock model loading
        mock_load_demucs.return_value = MagicMock()
        mock_load_df.return_value = (MagicMock(), MagicMock())

        # Mock Demucs: return audio as vocals
        mock_demucs.return_value = {
            "vocals": torch.from_numpy(audio).unsqueeze(0),
            "no_vocals": torch.from_numpy(audio * 0.1).unsqueeze(0),
            "sample_rate": sr,
        }

        # Mock DeepFilterNet: return audio unchanged
        mock_denoise.return_value = (audio, sr)

        successes = run_pipeline(mock_config, [str(episode_dir)])
        assert len(successes) == 1

        # Verify output structure
        assert (episode_dir / "preprocessed").exists()
        assert (episode_dir / "separated").exists()
        assert (episode_dir / "denoised").exists()
        assert (episode_dir / "normalized").exists()
        assert (episode_dir / "final").exists()

        # Verify final output
        mp3_files = list((episode_dir / "final").glob("*.mp3"))
        assert len(mp3_files) >= 1

        # Verify analysis report exists
        assert (episode_dir / "analysis" / "audio_report.json").exists()

        # Verify processing log exists
        assert (episode_dir / "processing.log").exists()
