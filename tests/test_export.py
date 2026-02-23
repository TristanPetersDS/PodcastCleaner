"""Tests for the export stage."""

from pathlib import Path

import numpy as np
import soundfile as sf

from podcast_cleaner.stages.export import run_export
from podcast_cleaner.utils import is_done


class TestRunExport:
    def test_creates_mp3_and_flac(self, tmp_path):
        """Should create both MP3 and FLAC outputs."""
        sr = 48000
        audio = (0.3 * np.sin(np.linspace(0, 2, sr * 2))).astype(np.float32)
        episode_dir = tmp_path / "01_Test-Episode"
        norm_dir = episode_dir / "normalized"
        norm_dir.mkdir(parents=True)
        sf.write(str(norm_dir / "test_normalized.wav"), audio, sr, subtype="FLOAT")

        config = {
            "export": {
                "formats": ["mp3", "flac"],
                "mp3_bitrate": "320k",
                "sample_rate": 48000,
            }
        }
        run_export(str(episode_dir), config)

        final_dir = episode_dir / "final"
        mp3_files = list(final_dir.glob("*.mp3"))
        flac_files = list(final_dir.glob("*.flac"))
        assert len(mp3_files) >= 1
        assert len(flac_files) >= 1
        assert is_done(episode_dir, "export")

    def test_skips_if_done(self, tmp_path):
        episode_dir = tmp_path / "ep"
        episode_dir.mkdir(parents=True)
        (episode_dir / ".export.done").touch()

        config = {"export": {"formats": ["mp3"], "mp3_bitrate": "320k", "sample_rate": 48000}}
        run_export(str(episode_dir), config)
        assert not (episode_dir / "final").exists()
