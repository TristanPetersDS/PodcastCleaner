"""Tests for the denoise stage."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from podcast_cleaner.stages.denoise import run_denoise
from podcast_cleaner.utils import is_done


class TestRunDenoise:
    @patch("podcast_cleaner.stages.denoise.deepfilter_enhance")
    def test_creates_denoised_file(self, mock_enhance, tmp_path):
        sr = 48000
        audio = np.sin(np.linspace(0, 1, sr)).astype(np.float32)
        episode_dir = tmp_path / "ep"
        sep_dir = episode_dir / "separated"
        sep_dir.mkdir(parents=True)
        sf.write(str(sep_dir / "test_vocals.wav"), audio, sr, subtype="FLOAT")

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
