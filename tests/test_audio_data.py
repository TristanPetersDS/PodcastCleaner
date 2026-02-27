"""Reusable audio quality regression tests.

Run with real ML models: pytest tests/test_audio_data.py -m slow -v
These tests verify that pipeline output meets quality thresholds.
Update thresholds as you tune settings.

Usage:
  1. Run the pipeline on real audio to get baseline metrics
  2. Update the thresholds below based on observed good results
  3. Re-run these tests after any pipeline change to catch regressions
"""

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from podcast_cleaner.analysis.audio_stats import compute_stats


# =============================================================================
# Quality thresholds — UPDATE THESE based on empirical results
# =============================================================================
THRESHOLDS = {
    "post_separate": {
        "snr_db_min": 15.0,  # Separation should improve SNR significantly
    },
    "post_denoise": {
        "snr_db_min": 20.0,  # Denoising should push SNR higher
    },
    "post_normalize": {
        "lufs_target": -16.0,
        "lufs_tolerance": 1.0,  # Within 1 LUFS of target
        "true_peak_max": -1.0,  # Should not exceed -1 dBTP
    },
    "general": {
        "duration_tolerance_pct": 1.0,  # Duration should not change more than 1%
    },
}
# =============================================================================


class TestAudioQualityThresholds:
    """Assert that pipeline output meets minimum quality standards.

    These thresholds are intentionally conservative and should be updated
    based on real pipeline runs. Run `podcast-cleaner analyze <episode_dir>`
    to see actual metrics before setting thresholds.
    """

    @staticmethod
    def _get_stage_audio(episode_dir: Path, stage: str) -> Path | None:
        """Find the main audio file for a given stage."""
        stage_dir = episode_dir / stage
        if not stage_dir.exists():
            return None
        wavs = list(stage_dir.glob("*.wav"))
        if not wavs:
            return None
        # Prefer vocal/denoised/normalized variants
        for w in wavs:
            if any(k in w.name for k in ("vocal", "denoised", "normalized")):
                return w
        return wavs[0]

    @pytest.mark.slow
    def test_separation_improves_snr(self):
        """After separation, SNR should be above threshold.

        Skip this test if no real separated audio is available.
        To run: process real audio first, then point EPISODE_DIR env var at it.
        """
        ep_dir = os.environ.get("EPISODE_DIR")
        if not ep_dir:
            pytest.skip("Set EPISODE_DIR env var to a processed episode directory")

        audio_path = self._get_stage_audio(Path(ep_dir), "separated")
        if audio_path is None:
            pytest.skip("No separated audio found")

        stats = compute_stats(str(audio_path))
        threshold = THRESHOLDS["post_separate"]["snr_db_min"]
        assert (
            stats["snr_db"] >= threshold
        ), f"Post-separation SNR {stats['snr_db']:.1f} dB below threshold {threshold} dB"

    @pytest.mark.slow
    def test_denoise_improves_snr(self):
        """After denoising, SNR should be above threshold."""
        ep_dir = os.environ.get("EPISODE_DIR")
        if not ep_dir:
            pytest.skip("Set EPISODE_DIR env var to a processed episode directory")

        audio_path = self._get_stage_audio(Path(ep_dir), "denoised")
        if audio_path is None:
            pytest.skip("No denoised audio found")

        stats = compute_stats(str(audio_path))
        threshold = THRESHOLDS["post_denoise"]["snr_db_min"]
        assert (
            stats["snr_db"] >= threshold
        ), f"Post-denoise SNR {stats['snr_db']:.1f} dB below threshold {threshold} dB"

    @pytest.mark.slow
    def test_normalization_hits_target(self):
        """After normalization, LUFS should be within tolerance of target."""
        ep_dir = os.environ.get("EPISODE_DIR")
        if not ep_dir:
            pytest.skip("Set EPISODE_DIR env var to a processed episode directory")

        audio_path = self._get_stage_audio(Path(ep_dir), "normalized")
        if audio_path is None:
            pytest.skip("No normalized audio found")

        stats = compute_stats(str(audio_path))
        target = THRESHOLDS["post_normalize"]["lufs_target"]
        tolerance = THRESHOLDS["post_normalize"]["lufs_tolerance"]
        assert abs(stats["lufs"] - target) <= tolerance, (
            f"Post-normalization LUFS {stats['lufs']:.1f} outside tolerance "
            f"({target - tolerance:.1f} to {target + tolerance:.1f})"
        )

        peak_max = THRESHOLDS["post_normalize"]["true_peak_max"]
        assert (
            stats["true_peak"] <= peak_max
        ), f"True peak {stats['true_peak']:.1f} dBTP exceeds maximum {peak_max} dBTP"

    @pytest.mark.slow
    def test_duration_preserved(self):
        """Pipeline should not significantly change audio duration."""
        ep_dir = os.environ.get("EPISODE_DIR")
        if not ep_dir:
            pytest.skip("Set EPISODE_DIR env var to a processed episode directory")

        raw_audio = self._get_stage_audio(Path(ep_dir), "raw")
        norm_audio = self._get_stage_audio(Path(ep_dir), "normalized")
        if raw_audio is None or norm_audio is None:
            pytest.skip("Need both raw and normalized audio")

        raw_stats = compute_stats(str(raw_audio))
        norm_stats = compute_stats(str(norm_audio))

        tolerance_pct = THRESHOLDS["general"]["duration_tolerance_pct"]
        raw_dur = raw_stats["duration"]
        norm_dur = norm_stats["duration"]
        pct_change = abs(raw_dur - norm_dur) / raw_dur * 100

        assert pct_change <= tolerance_pct, (
            f"Duration changed by {pct_change:.1f}% "
            f"(raw={raw_dur:.1f}s, normalized={norm_dur:.1f}s)"
        )


class TestSyntheticAudioQuality:
    """Quick quality checks on synthetic audio (no GPU required)."""

    def test_normalization_accuracy(self, tmp_path):
        """Normalize a known signal and verify LUFS is on target."""
        from podcast_cleaner.stages.normalize import run_normalize

        sr = 48000
        t = np.linspace(0, 3.0, sr * 3, endpoint=False)
        audio = (0.05 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        episode_dir = tmp_path / "ep"
        denoised_dir = episode_dir / "denoised"
        denoised_dir.mkdir(parents=True)
        sf.write(str(denoised_dir / "test_denoised.wav"), audio, sr, subtype="FLOAT")

        config = {"normalization": {"target_lufs": -16.0, "true_peak_dbtp": -1.5}}
        run_normalize(str(episode_dir), config)

        out = episode_dir / "normalized" / "test_normalized.wav"
        assert out.exists(), f"Normalization did not produce output at {out}"
        stats = compute_stats(str(out))
        assert abs(stats["lufs"] - (-16.0)) < 1.5

    def test_export_preserves_duration(self, tmp_path):
        """Export should not change duration."""
        from podcast_cleaner.stages.export import run_export

        sr = 48000
        duration = 2.0
        audio = (0.3 * np.sin(np.linspace(0, duration, int(sr * duration)))).astype(
            np.float32
        )

        episode_dir = tmp_path / "01_Test"
        norm_dir = episode_dir / "normalized"
        norm_dir.mkdir(parents=True)
        sf.write(str(norm_dir / "test_normalized.wav"), audio, sr, subtype="FLOAT")

        config = {
            "export": {"formats": ["mp3"], "mp3_bitrate": "320k", "sample_rate": 48000}
        }
        run_export(str(episode_dir), config)

        # Check MP3 duration
        mp3_files = list((episode_dir / "final").glob("*.mp3"))
        assert mp3_files, f"No MP3 files found in {episode_dir / 'final'}"
        mp3 = mp3_files[0]
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
                str(mp3),
            ],
            capture_output=True,
            text=True,
        )
        mp3_duration = float(result.stdout.strip())
        assert abs(mp3_duration - duration) < 0.1  # Within 100ms
