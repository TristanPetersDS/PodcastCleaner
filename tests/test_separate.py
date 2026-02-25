"""Tests for the separation stage."""

from unittest.mock import MagicMock, patch

import numpy as np
import soundfile as sf

from podcast_cleaner.stages.separate import run_separate
from podcast_cleaner.utils import is_done


class TestRunSeparate:
    @patch("podcast_cleaner.stages.separate.demucs_separate")
    @patch("podcast_cleaner.stages.separate._load_demucs_model")
    def test_creates_output_files(self, mock_load_model, mock_demucs, tmp_path):
        """Should create vocals and background WAVs."""
        sr = 44100
        audio = np.sin(np.linspace(0, 1, sr * 2)).astype(np.float32)
        episode_dir = tmp_path / "ep"
        pre_dir = episode_dir / "preprocessed"
        pre_dir.mkdir(parents=True)
        sf.write(str(pre_dir / "test.wav"), audio, sr)

        # Mock model loading
        mock_load_model.return_value = MagicMock()

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


class TestPreSegmentation:
    def test_crossfade_smoothness(self):
        """Split-then-crossfade round-trip preserves energy (discontinuity <0.5 dB).

        Simulates the real pipeline: split audio into overlapping segments,
        then reassemble with crossfade. The overlap regions share identical
        samples (as they would from splitting the same source audio), so the
        crossfade should reconstruct the original signal nearly perfectly.
        """
        from podcast_cleaner.stages.separate import _crossfade_segments, _split_audio_segments

        sr = 44100
        overlap_samples = int(30.0 * sr)  # 30s overlap
        max_segment_samples = int(10 * 60 * sr)  # 10 min chunks
        total_samples = int(25 * 60 * sr)  # 25 min — forces 3 segments

        # Create a continuous signal with varying amplitude
        rng = np.random.RandomState(42)
        original = (rng.randn(total_samples) * 0.5).astype(np.float32)

        # Split into overlapping segments, then reassemble
        segments = _split_audio_segments(original, max_segment_samples, overlap_samples)
        assert len(segments) >= 2, "Test requires multiple segments"

        result = _crossfade_segments(segments, overlap_samples)

        # Measure energy at the first crossfade boundary vs adjacent region
        crossfade_start = max_segment_samples - overlap_samples
        crossfade_mid = crossfade_start + overlap_samples // 2
        window = sr  # 1 second

        rms_before = np.sqrt(np.mean(result[crossfade_start - window : crossfade_start] ** 2))
        rms_mid = np.sqrt(np.mean(result[crossfade_mid - window // 2 : crossfade_mid + window // 2] ** 2))

        if rms_before > 1e-10 and rms_mid > 1e-10:
            energy_diff_db = abs(20 * np.log10(rms_mid / rms_before))
            assert energy_diff_db < 0.5, f"Energy discontinuity {energy_diff_db:.2f} dB"

    def test_short_audio_no_chunking(self):
        """Audio shorter than chunk size returns as single piece."""
        from podcast_cleaner.stages.separate import _split_audio_segments

        sr = 44100
        short_audio = np.sin(np.linspace(0, 10, sr * 60)).astype(np.float32)  # 1 min
        max_segment_samples = int(10 * 60 * sr)  # 10 min
        overlap_samples = int(30 * sr)

        segments = _split_audio_segments(short_audio, max_segment_samples, overlap_samples)
        assert len(segments) == 1
        np.testing.assert_array_equal(segments[0], short_audio)

    def test_chunk_boundary_exact(self):
        """Audio exactly at chunk size is processed as single chunk."""
        from podcast_cleaner.stages.separate import _split_audio_segments

        sr = 44100
        max_segment_samples = int(10 * 60 * sr)
        overlap_samples = int(30 * sr)
        exact_audio = np.ones(max_segment_samples, dtype=np.float32) * 0.5

        segments = _split_audio_segments(exact_audio, max_segment_samples, overlap_samples)
        assert len(segments) == 1
