"""Tests for the transcription stage."""

from unittest.mock import patch

import numpy as np
import soundfile as sf

from podcast_cleaner.stages.transcribe import format_srt_timestamp, run_transcribe
from podcast_cleaner.utils import is_done


class TestFormatSrtTimestamp:
    def test_basic(self):
        assert format_srt_timestamp(0.0) == "00:00:00,000"

    def test_minutes(self):
        assert format_srt_timestamp(65.5) == "00:01:05,500"

    def test_hours(self):
        assert format_srt_timestamp(3723.123) == "01:02:03,123"


class TestRunTranscribe:
    def test_skips_when_disabled(self, tmp_path):
        episode_dir = tmp_path / "ep"
        episode_dir.mkdir()
        config = {"transcription": {"enabled": False}}
        run_transcribe(str(episode_dir), config)
        assert not (episode_dir / "final" / "transcript").exists()

    def test_skips_if_done(self, tmp_path):
        episode_dir = tmp_path / "ep"
        episode_dir.mkdir()
        (episode_dir / ".transcribe.done").touch()
        config = {
            "transcription": {"enabled": True, "model": "large-v3", "device": "cpu"}
        }
        run_transcribe(str(episode_dir), config)

    @patch("podcast_cleaner.stages.transcribe.whisperx_transcribe")
    def test_creates_json_and_srt(self, mock_wx, tmp_path):
        sr = 16000
        audio = np.zeros(sr * 2, dtype=np.float32)
        episode_dir = tmp_path / "01_Test-Episode"
        denoised_dir = episode_dir / "denoised"
        denoised_dir.mkdir(parents=True)
        sf.write(str(denoised_dir / "test_denoised.wav"), audio, sr, subtype="FLOAT")

        mock_wx.return_value = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.5,
                    "text": "Hello world",
                    "speaker": "SPEAKER_00",
                },
            ],
            "language": "en",
        }

        config = {
            "transcription": {"enabled": True, "model": "large-v3", "device": "cpu"}
        }
        run_transcribe(str(episode_dir), config)

        transcript_dir = episode_dir / "final" / "transcript"
        assert (transcript_dir / "01_Test-Episode.json").exists()
        assert (transcript_dir / "01_Test-Episode.srt").exists()
        assert is_done(episode_dir, "transcribe")


class TestMissingWhisperX:
    @patch("podcast_cleaner.stages.transcribe.whisperx_transcribe")
    def test_transcribe_missing_whisperx(self, mock_wx, tmp_path, caplog):
        """When whisperx is not installed, stage completes with warning."""
        mock_wx.side_effect = ModuleNotFoundError("No module named 'whisperx'")

        sr = 16000
        audio = np.zeros(sr * 2, dtype=np.float32)
        episode_dir = tmp_path / "01_Test-Episode"
        denoised_dir = episode_dir / "denoised"
        denoised_dir.mkdir(parents=True)
        sf.write(str(denoised_dir / "test_denoised.wav"), audio, sr, subtype="FLOAT")

        config = {
            "transcription": {"enabled": True, "model": "large-v3", "device": "cpu"}
        }

        import logging

        with caplog.at_level(logging.WARNING):
            run_transcribe(str(episode_dir), config)

        assert any("whisperx" in msg.lower() for msg in caplog.messages)
        assert is_done(episode_dir, "transcribe")
