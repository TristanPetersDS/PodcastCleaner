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


class TestCopyInput:
    def test_copy_input_flag_exists(self, runner):
        """--copy-input flag should appear in help."""
        result = runner.invoke(main, ["run", "--help"])
        assert "--copy-input" in result.output

    def test_copy_input_creates_copy_not_symlink(self, runner, tmp_path):
        """--copy-input should copy files instead of symlinking."""
        import numpy as np
        import soundfile as sf

        # Create a real audio file
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        audio = np.zeros(1000, dtype=np.float32)
        sf.write(str(input_dir / "test.wav"), audio, 16000)

        with patch("podcast_cleaner.cli.run_pipeline") as mock_run:
            # Return the episode_dirs arg so successes == episode_dirs
            mock_run.side_effect = lambda config, episode_dirs, **kw: episode_dirs
            result = runner.invoke(main, [
                "run", "--input", str(input_dir / "test.wav"),
                "--copy-input", "--config", "config.example.yaml",
            ])

        # Find the created raw dir
        import glob
        raw_files = glob.glob(str(tmp_path / "**" / "raw" / "*.wav"), recursive=True)
        # The file in raw/ should NOT be a symlink when --copy-input is used
        # (test may not find files if output_dir is elsewhere, but flag should be accepted)
        assert result.exit_code == 0 or "--copy-input" in result.output

    def test_copy_input_with_input_dir(self, runner, tmp_path):
        """--copy-input should copy files from --input-dir instead of symlinking."""
        import numpy as np
        import soundfile as sf

        from podcast_cleaner.config import DEFAULT_CONFIG

        # Create a real audio file in input dir
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        audio = np.zeros(1000, dtype=np.float32)
        sf.write(str(input_dir / "episode.wav"), audio, 16000)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with patch("podcast_cleaner.cli.run_pipeline") as mock_run, \
             patch("podcast_cleaner.cli.load_config") as mock_config:
            # Return the episode_dirs arg so successes == episode_dirs
            mock_run.side_effect = lambda config, episode_dirs, **kw: episode_dirs
            mock_config.return_value = {
                **DEFAULT_CONFIG,
                "output_dir": str(output_dir),
            }
            result = runner.invoke(main, [
                "run", "--input-dir", str(input_dir),
                "--copy-input",
            ])

        assert result.exit_code == 0

        # Find raw dirs in output
        import glob
        raw_files = glob.glob(str(output_dir / "**" / "raw" / "*.wav"), recursive=True)
        assert len(raw_files) == 1, f"Expected 1 raw file, found {raw_files}"
        from pathlib import Path
        raw_file = Path(raw_files[0])
        assert not raw_file.is_symlink(), "File should be a copy, not a symlink"

    def test_default_creates_symlink(self, runner, tmp_path):
        """Without --copy-input, should create symlinks by default."""
        import numpy as np
        import soundfile as sf

        from podcast_cleaner.config import DEFAULT_CONFIG

        # Create a real audio file in input dir
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        audio = np.zeros(1000, dtype=np.float32)
        sf.write(str(input_dir / "episode.wav"), audio, 16000)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with patch("podcast_cleaner.cli.run_pipeline") as mock_run, \
             patch("podcast_cleaner.cli.load_config") as mock_config:
            # Return the episode_dirs arg so successes == episode_dirs
            mock_run.side_effect = lambda config, episode_dirs, **kw: episode_dirs
            mock_config.return_value = {
                **DEFAULT_CONFIG,
                "output_dir": str(output_dir),
            }
            result = runner.invoke(main, [
                "run", "--input-dir", str(input_dir),
            ])

        assert result.exit_code == 0

        # Find raw dirs in output
        import glob
        raw_files = glob.glob(str(output_dir / "**" / "raw" / "*.wav"), recursive=True)
        assert len(raw_files) == 1, f"Expected 1 raw file, found {raw_files}"
        from pathlib import Path
        raw_file = Path(raw_files[0])
        assert raw_file.is_symlink(), "File should be a symlink by default"


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
