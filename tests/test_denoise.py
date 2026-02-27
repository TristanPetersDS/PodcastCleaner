"""Tests for the denoise stage."""

from unittest.mock import MagicMock, patch

import numpy as np
import soundfile as sf

from podcast_cleaner.stages.denoise import run_denoise
from podcast_cleaner.utils import is_done


class TestRunDenoise:
    @patch("podcast_cleaner.stages.denoise.deepfilter_enhance")
    @patch("podcast_cleaner.stages.denoise._load_deepfilter_model")
    def test_creates_denoised_file(self, mock_load_model, mock_enhance, tmp_path):
        sr = 48000
        audio = np.sin(np.linspace(0, 1, sr)).astype(np.float32)
        episode_dir = tmp_path / "ep"
        sep_dir = episode_dir / "separated"
        sep_dir.mkdir(parents=True)
        sf.write(str(sep_dir / "test_vocals.wav"), audio, sr, subtype="FLOAT")

        # Mock model loading
        mock_load_model.return_value = (MagicMock(), MagicMock())

        # Mock DeepFilterNet to return the same audio
        mock_enhance.return_value = (audio, sr)

        config = {"denoise": {"model": "DeepFilterNet3"}}
        run_denoise(str(episode_dir), config)

        out = episode_dir / "denoised" / "test_denoised.wav"
        assert out.exists()
        assert is_done(episode_dir, "denoise")

    def test_skips_if_done(self, tmp_path):
        episode_dir = tmp_path / "ep"
        episode_dir.mkdir(parents=True)
        (episode_dir / ".denoise.done").touch()

        config = {"denoise": {"model": "DeepFilterNet3"}}
        run_denoise(str(episode_dir), config)
        assert not (episode_dir / "denoised").exists()


class TestDeepFilterCPUFallback:
    @patch("podcast_cleaner.stages.denoise.deepfilter_enhance")
    @patch("podcast_cleaner.stages.denoise._load_deepfilter_model")
    def test_deepfilter_cpu_fallback(self, mock_load_model, mock_enhance, tmp_path):
        """Fallback should produce valid output."""
        sr = 48000
        audio = np.sin(np.linspace(0, 1, sr)).astype(np.float32)
        episode_dir = tmp_path / "ep"
        sep_dir = episode_dir / "separated"
        sep_dir.mkdir(parents=True)
        sf.write(str(sep_dir / "test_vocals.wav"), audio, sr, subtype="FLOAT")

        mock_load_model.return_value = (MagicMock(), MagicMock())
        mock_enhance.return_value = (audio, sr)
        config = {"denoise": {"model": "DeepFilterNet3"}}
        run_denoise(str(episode_dir), config)

        out = episode_dir / "denoised" / "test_denoised.wav"
        assert out.exists()
        out_audio, out_sr = sf.read(str(out))
        assert not np.any(np.isnan(out_audio))
        assert out_audio.shape[0] > 0
        assert out_sr == sr

    @patch("podcast_cleaner.stages.denoise.deepfilter_enhance")
    @patch("podcast_cleaner.stages.denoise._load_deepfilter_model")
    def test_deepfilter_nan_sanitization_cpu(
        self, mock_load_model, mock_enhance, tmp_path
    ):
        """Output NaN values should be sanitized."""
        sr = 48000
        audio_with_nan = np.sin(np.linspace(0, 1, sr)).astype(np.float32)
        audio_with_nan[100:200] = np.nan
        mock_load_model.return_value = (MagicMock(), MagicMock())
        mock_enhance.return_value = (audio_with_nan, sr)

        episode_dir = tmp_path / "ep"
        sep_dir = episode_dir / "separated"
        sep_dir.mkdir(parents=True)
        clean_audio = np.sin(np.linspace(0, 1, sr)).astype(np.float32)
        sf.write(str(sep_dir / "test_vocals.wav"), clean_audio, sr, subtype="FLOAT")

        config = {"denoise": {"model": "DeepFilterNet3"}}
        run_denoise(str(episode_dir), config)

        out = episode_dir / "denoised" / "test_denoised.wav"
        out_audio, _ = sf.read(str(out))
        assert not np.any(np.isnan(out_audio)), "NaN values not sanitized"
