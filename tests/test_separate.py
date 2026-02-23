"""Tests for the separation stage."""

from unittest.mock import patch

import numpy as np
import soundfile as sf

from podcast_cleaner.stages.separate import run_separate
from podcast_cleaner.utils import is_done


class TestRunSeparate:
    @patch("podcast_cleaner.stages.separate.demucs_separate")
    def test_creates_output_files(self, mock_demucs, tmp_path):
        """Should create vocals and background WAVs."""
        sr = 44100
        audio = np.sin(np.linspace(0, 1, sr * 2)).astype(np.float32)
        episode_dir = tmp_path / "ep"
        pre_dir = episode_dir / "preprocessed"
        pre_dir.mkdir(parents=True)
        sf.write(str(pre_dir / "test.wav"), audio, sr)

        # Mock demucs to produce fake stems
        import torch
        mock_demucs.return_value = {
            "vocals": torch.from_numpy(audio).unsqueeze(0),
            "no_vocals": torch.from_numpy(audio * 0.1).unsqueeze(0),
            "sample_rate": sr,
        }

        config = {"separation": {"model": "htdemucs_ft", "device": "cpu"}}
        run_separate(str(episode_dir), config)

        sep_dir = episode_dir / "separated"
        assert (sep_dir / "test_vocals.wav").exists()
        assert (sep_dir / "test_background.wav").exists()
        assert is_done(episode_dir, "separate")

    def test_skips_if_done(self, tmp_path):
        episode_dir = tmp_path / "ep"
        episode_dir.mkdir(parents=True)
        (episode_dir / ".separate.done").touch()

        config = {"separation": {"model": "htdemucs_ft", "device": "cpu"}}
        run_separate(str(episode_dir), config)
        assert not (episode_dir / "separated").exists()
