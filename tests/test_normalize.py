"""Tests for the normalization stage."""

import numpy as np
import pyloudnorm as pyln
import soundfile as sf

from podcast_cleaner.stages.normalize import (
    run_normalize,
    _true_peak_limit_legacy as true_peak_limit,
)
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


class TestNormalizationV2:
    def test_high_crest_factor_normalization(self, tmp_path):
        """Audio with quiet body + loud transients should hit target LUFS."""
        sr = 48000
        duration = 3.0
        n_samples = int(sr * duration)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        audio = (0.003 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        spike_interval = sr
        for i in range(0, n_samples, spike_interval):
            spike_len = min(100, n_samples - i)
            audio[i : i + spike_len] = 0.95

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
        assert (
            -17.0 <= lufs <= -15.0
        ), f"Output LUFS {lufs:.1f} not within ±1.0 dB of -16.0"

    def test_normalization_idempotent(self, tmp_path):
        """Running normalization twice produces output within ±0.5 dB."""
        sr = 48000
        t = np.linspace(0, 2.0, sr * 2, endpoint=False)
        audio = (0.05 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        ep1 = tmp_path / "ep1"
        d1 = ep1 / "denoised"
        d1.mkdir(parents=True)
        sf.write(str(d1 / "test_denoised.wav"), audio, sr, subtype="FLOAT")
        config = {"normalization": {"target_lufs": -16.0, "true_peak_dbtp": -1.5}}
        run_normalize(str(ep1), config)
        pass1_audio, _ = sf.read(str(ep1 / "normalized" / "test_normalized.wav"))
        meter = pyln.Meter(sr)
        pass1_lufs = meter.integrated_loudness(pass1_audio)

        ep2 = tmp_path / "ep2"
        d2 = ep2 / "denoised"
        d2.mkdir(parents=True)
        sf.write(str(d2 / "test_denoised.wav"), pass1_audio, sr, subtype="FLOAT")
        run_normalize(str(ep2), config)
        pass2_audio, _ = sf.read(str(ep2 / "normalized" / "test_normalized.wav"))
        pass2_lufs = meter.integrated_loudness(pass2_audio)

        assert (
            abs(pass1_lufs - pass2_lufs) <= 0.5
        ), f"Not idempotent: pass1={pass1_lufs:.1f}, pass2={pass2_lufs:.1f}"

    def test_normalization_preserves_peak_limit(self, tmp_path):
        """Output true peak at or below -1.5 dBTP."""
        sr = 48000
        t = np.linspace(0, 2.0, sr * 2, endpoint=False)
        audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        episode_dir = tmp_path / "ep"
        denoised_dir = episode_dir / "denoised"
        denoised_dir.mkdir(parents=True)
        sf.write(str(denoised_dir / "test_denoised.wav"), audio, sr, subtype="FLOAT")
        config = {"normalization": {"target_lufs": -16.0, "true_peak_dbtp": -1.5}}
        run_normalize(str(episode_dir), config)

        out_audio, _ = sf.read(str(episode_dir / "normalized" / "test_normalized.wav"))
        peak_dbtp = 20 * np.log10(float(np.max(np.abs(out_audio))) + 1e-10)
        assert peak_dbtp <= -1.4, f"Peak {peak_dbtp:.1f} dBTP exceeds limit"

    def test_normalization_no_clipping(self, tmp_path):
        """No output sample exceeds 1.0."""
        sr = 48000
        t = np.linspace(0, 2.0, sr * 2, endpoint=False)
        audio = (0.001 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        episode_dir = tmp_path / "ep"
        denoised_dir = episode_dir / "denoised"
        denoised_dir.mkdir(parents=True)
        sf.write(str(denoised_dir / "test_denoised.wav"), audio, sr, subtype="FLOAT")
        config = {"normalization": {"target_lufs": -16.0, "true_peak_dbtp": -1.5}}
        run_normalize(str(episode_dir), config)

        out_audio, _ = sf.read(str(episode_dir / "normalized" / "test_normalized.wav"))
        assert np.max(np.abs(out_audio)) <= 1.0, "Output contains samples exceeding 1.0"
