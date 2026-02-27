"""Tests for shared utilities."""

import numpy as np

from podcast_cleaner.utils import (
    clear_done,
    ensure_dir,
    is_done,
    mark_done,
    read_audio,
    sanitize_filename,
    write_audio,
)


class TestSanitizeFilename:
    def test_basic(self):
        assert sanitize_filename("Hello World") == "Hello World"

    def test_special_chars(self):
        result = sanitize_filename("Episode #5: What's Next?")
        assert "#" not in result
        assert "?" not in result
        assert "'" not in result

    def test_collapses_underscores(self):
        result = sanitize_filename("a///b///c")
        assert "___" not in result

    def test_empty_string(self):
        assert sanitize_filename("") == ""

    def test_unicode(self):
        result = sanitize_filename("Épisode Café")
        assert isinstance(result, str)


class TestDoneMarkers:
    def test_mark_and_check(self, tmp_path):
        mark_done(tmp_path, "preprocess")
        assert is_done(tmp_path, "preprocess")

    def test_not_done_by_default(self, tmp_path):
        assert not is_done(tmp_path, "preprocess")

    def test_clear_done(self, tmp_path):
        mark_done(tmp_path, "preprocess")
        clear_done(tmp_path, "preprocess")
        assert not is_done(tmp_path, "preprocess")

    def test_clear_nonexistent(self, tmp_path):
        clear_done(tmp_path, "preprocess")  # Should not raise


class TestAudioIO:
    def test_write_and_read(self, tmp_path):
        sr = 16000
        audio = np.sin(np.linspace(0, 1, sr)).astype(np.float32)
        path = tmp_path / "test.wav"
        write_audio(path, audio, sr)
        loaded, loaded_sr = read_audio(path)
        assert loaded_sr == sr
        assert len(loaded) == len(audio)
        np.testing.assert_allclose(loaded, audio, atol=1e-5)

    def test_read_stereo_to_mono(self, tmp_path):
        import soundfile as sf

        sr = 16000
        stereo = np.random.randn(sr, 2).astype(np.float32) * 0.1
        path = tmp_path / "stereo.wav"
        sf.write(str(path), stereo, sr)
        mono, _ = read_audio(path)
        assert mono.ndim == 1
        assert len(mono) == sr


class TestEnsureDir:
    def test_creates_nested(self, tmp_path):
        p = ensure_dir(tmp_path / "a" / "b" / "c")
        assert p.is_dir()

    def test_idempotent(self, tmp_path):
        p = ensure_dir(tmp_path / "x")
        ensure_dir(tmp_path / "x")  # Should not raise
        assert p.is_dir()


class TestLazyImports:
    def test_cli_startup_fast(self):
        """Importing podcast_cleaner.cli should NOT load torch."""
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import podcast_cleaner.cli; import sys; "
                "assert 'torch' not in sys.modules, 'torch was eagerly imported'",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"torch eagerly imported: {result.stderr}"
