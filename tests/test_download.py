"""Tests for the download stage."""

from unittest.mock import MagicMock, patch

import pytest

from podcast_cleaner.stages.download import (
    build_episode_dirname,
    download_single,
    get_playlist_entries,
)


class TestBuildEpisodeDirname:
    def test_playlist_numbering(self):
        assert build_episode_dirname("My Episode", 0) == "01_My-Episode"
        assert build_episode_dirname("My Episode", 9) == "10_My-Episode"

    def test_sanitizes_title(self):
        result = build_episode_dirname("What's #1? The Best!", 2)
        assert result.startswith("03_")
        assert "#" not in result
        assert "?" not in result

    def test_single_video_no_index(self):
        result = build_episode_dirname("Solo Episode", None)
        assert result == "Solo-Episode"


class TestGetPlaylistEntries:
    @patch("podcast_cleaner.stages.download.subprocess.run")
    def test_parses_flat_json(self, mock_run):
        mock_run.return_value = MagicMock(
            stdout='{"id": "abc123", "title": "Episode 1"}\n{"id": "def456", "title": "Episode 2"}\n',
            returncode=0,
        )
        entries = get_playlist_entries("https://youtube.com/playlist?list=PLtest")
        assert len(entries) == 2
        assert entries[0]["id"] == "abc123"
        assert entries[1]["title"] == "Episode 2"

    @patch("podcast_cleaner.stages.download.subprocess.run")
    def test_empty_playlist(self, mock_run):
        mock_run.return_value = MagicMock(stdout="", returncode=0)
        entries = get_playlist_entries("https://youtube.com/playlist?list=PLempty")
        assert entries == []


class TestDownloadSingle:
    @patch("podcast_cleaner.stages.download.subprocess.run")
    def test_constructs_correct_command(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        # Create a fake WAV so the glob finds something
        fake_wav = tmp_path / "test.wav"
        fake_wav.touch()

        result = download_single("https://youtube.com/watch?v=abc", str(tmp_path))
        assert result is not None
        # Verify yt-dlp was called with audio extraction flags
        call_args = mock_run.call_args[0][0]
        assert "yt-dlp" in call_args
        assert "-x" in call_args
        assert "--audio-format" in call_args
