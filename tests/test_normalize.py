"""Tests for the normalization stage."""

import numpy as np
import pyloudnorm as pyln
import soundfile as sf

from podcast_cleaner.stages.normalize import run_normalize, true_peak_limit
from podcast_cleaner.utils import is_done


class TestTruePeakLimit:
    def test_limits_peak(self):
        audio = np.ones(1000, dtype=np.float32)  # peak at 0 dBFS
        limited = true_peak_limit(audio, max_dbtp=-3.0)
        peak_db = 20 * np.log10(np.max(np.abs(limited)))
        assert peak_db <= -2.9  # Allow small tolerance

    def test_does_not_modify_quiet_audio(self):
        audio = np.ones(1000, dtype=np.float32) * 0.01
        limited = true_peak_limit(audio, max_dbtp=-1.5)
        np.testing.assert_array_equal(audio, limited)


class TestRunNormalize:
    def test_normalizes_to_target_lufs(self, tmp_path):
        """Output should be close to -16 LUFS."""
        sr = 48000
        t = np.linspace(0, 2.0, sr * 2, endpoint=False)
        audio = (0.02 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)  # quiet

        episode_dir = tmp_path / "ep"
        denoised_dir = episode_dir / "denoised"
        denoised_dir.mkdir(parents=True)
        sf.write(str(denoised_dir / "test_denoised.wav"), audio, sr, subtype="FLOAT")

        config = {"normalization": {"target_lufs": -16.0, "true_peak_dbtp": -1.5}}
        run_normalize(str(episode_dir), config)

        out = episode_dir / "normalized" / "test_normalized.wav"
        assert out.exists()
        out_audio, out_sr = sf.read(str(out))
        meter = pyln.Meter(out_sr)
        lufs = meter.integrated_loudness(out_audio)
        assert -17.0 < lufs < -15.0  # Within 1 LUFS of target

    def test_marks_done(self, tmp_path):
        sr = 48000
        audio = (0.1 * np.sin(np.linspace(0, 2, sr * 2))).astype(np.float32)
        episode_dir = tmp_path / "ep"
        denoised_dir = episode_dir / "denoised"
        denoised_dir.mkdir(parents=True)
        sf.write(str(denoised_dir / "test_denoised.wav"), audio, sr, subtype="FLOAT")

        config = {"normalization": {"target_lufs": -16.0, "true_peak_dbtp": -1.5}}
        run_normalize(str(episode_dir), config)
        assert is_done(episode_dir, "normalize")

    def test_skips_if_done(self, tmp_path):
        episode_dir = tmp_path / "ep"
        episode_dir.mkdir(parents=True)
        (episode_dir / ".normalize.done").touch()

        config = {"normalization": {"target_lufs": -16.0, "true_peak_dbtp": -1.5}}
        run_normalize(str(episode_dir), config)
        assert not (episode_dir / "normalized").exists()

    def test_falls_back_to_separated_vocals(self, tmp_path):
        """When no denoised files exist, should use separated vocals."""
        sr = 48000
        t = np.linspace(0, 2.0, sr * 2, endpoint=False)
        audio = (0.05 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        episode_dir = tmp_path / "ep"
        sep_dir = episode_dir / "separated"
        sep_dir.mkdir(parents=True)
        sf.write(str(sep_dir / "test_vocals.wav"), audio, sr, subtype="FLOAT")

        config = {"normalization": {"target_lufs": -16.0, "true_peak_dbtp": -1.5}}
        run_normalize(str(episode_dir), config)

        out = episode_dir / "normalized" / "test_normalized.wav"
        assert out.exists()
        assert is_done(episode_dir, "normalize")
