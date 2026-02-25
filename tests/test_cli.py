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


class TestCleanupIntermediates:
    def test_cleanup_resume_conflict(self):
        """--cleanup-intermediates and --resume should conflict."""
        runner = CliRunner()
        result = runner.invoke(main, ["run", "--input", __file__, "--cleanup-intermediates", "--resume"])
        assert result.exit_code != 0
        assert "Cannot use --cleanup-intermediates with --resume" in result.output

    def test_cleanup_intermediates_flag_exists(self):
        """The --cleanup-intermediates flag should appear in help."""
        runner = CliRunner()
        result = runner.invoke(main, ["run", "--help"])
        assert "--cleanup-intermediates" in result.output


class TestStageCommand:
    def test_stage_requires_name_and_dir(self, runner):
        result = runner.invoke(main, ["stage"])
        assert result.exit_code != 0


class TestCheckCommand:
    def test_check_command_output(self, runner):
        """check command should report system dependencies."""
        result = runner.invoke(main, ["check"])
        assert result.exit_code == 0
        assert "Python" in result.output
        assert "ffmpeg" in result.output


class TestVerboseQuietFlags:
    def test_verbose_flag_exists(self, runner):
        """--verbose flag should appear in help."""
        result = runner.invoke(main, ["run", "--help"])
        assert "--verbose" in result.output or "-v" in result.output

    def test_quiet_flag_exists(self, runner):
        """--quiet flag should appear in help."""
        result = runner.invoke(main, ["run", "--help"])
        assert "--quiet" in result.output or "-q" in result.output

    def test_verbose_quiet_conflict(self, runner):
        """--verbose and --quiet should conflict."""
        result = runner.invoke(main, ["run", "--input", __file__, "--verbose", "--quiet"])
        assert result.exit_code != 0


class TestRichConsole:
    def test_rich_console_fallback_no_tty(self):
        """Rich should use plain text when not a TTY."""
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "-c",
             "from podcast_cleaner.display import console; "
             "assert not console.is_terminal"],
            capture_output=True, text=True, timeout=30,
        )
        # When piped (capture_output), console should not be a terminal
        assert result.returncode == 0, f"Failed: {result.stderr}"
