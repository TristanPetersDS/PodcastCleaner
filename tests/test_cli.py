"""Tests for the CLI interface."""

from unittest.mock import patch, MagicMock

import pytest
from click.testing import CliRunner

from podcast_cleaner.cli import main


@pytest.fixture
def runner():
    return CliRunner()


class TestRunCommand:
    def test_requires_url_or_input(self, runner):
        """Should fail if neither --url nor --input/--input-dir is given."""
        result = runner.invoke(main, ["run"])
        assert result.exit_code != 0

    def test_url_flag_accepted(self, runner):
        """--url flag should be recognized (even if download would fail)."""
        with patch("podcast_cleaner.cli.run_pipeline") as mock_run, \
             patch("podcast_cleaner.stages.download.run_download", return_value=[]) as mock_dl, \
             patch("podcast_cleaner.utils.setup_logging", return_value=MagicMock()):
            mock_run.return_value = []
            result = runner.invoke(main, ["run", "--url", "https://youtube.com/watch?v=test"])
            assert result.exit_code == 0

    def test_input_dir_flag_accepted(self, runner, tmp_path):
        """--input-dir flag should be recognized."""
        input_dir = tmp_path / "episodes"
        input_dir.mkdir()
        with patch("podcast_cleaner.cli.run_pipeline") as mock_run:
            mock_run.return_value = []
            result = runner.invoke(main, ["run", "--input-dir", str(input_dir)])
            assert result.exit_code == 0


class TestAnalyzeCommand:
    def test_analyze_requires_path(self, runner):
        result = runner.invoke(main, ["analyze"])
        assert result.exit_code != 0


class TestStageCommand:
    def test_stage_requires_name_and_dir(self, runner):
        result = runner.invoke(main, ["stage"])
        assert result.exit_code != 0
