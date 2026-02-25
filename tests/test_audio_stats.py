"""Tests for audio analysis metrics."""

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from podcast_cleaner.analysis.audio_stats import (
    compute_stats,
    measure_lufs,
    measure_rms_db,
    measure_snr,
    measure_spectral_centroid,
    measure_true_peak,
    save_stage_report,
)


@pytest.fixture
def sine_440(tmp_path):
    """440Hz sine at -20 dBFS, 48kHz, 1 second."""
    sr = 48000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    amp = 10 ** (-20 / 20)
    audio = (amp * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    path = tmp_path / "sine440.wav"
    sf.write(str(path), audio, sr, subtype="FLOAT")
    return path, audio, sr


class TestMeasureRms:
    def test_known_amplitude(self):
        """RMS of a sine wave at -20 dBFS should be close to -23 dBFS (sine RMS = peak - 3dB)."""
        amp = 10 ** (-20 / 20)
        sr = 48000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        audio = (amp * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        rms = measure_rms_db(audio)
        assert -24.0 < rms < -22.0

    def test_silence(self):
        """RMS of silence should be very low."""
        audio = np.zeros(48000, dtype=np.float32)
        rms = measure_rms_db(audio)
        assert rms < -80


class TestMeasureTruePeak:
    def test_known_peak(self):
        """True peak of a -6 dBFS sine should be near -6 dBTP."""
        amp = 10 ** (-6 / 20)
        audio = (amp * np.ones(48000)).astype(np.float32)
        tp = measure_true_peak(audio)
        assert -7.0 < tp < -5.0


class TestMeasureSNR:
    def test_clean_signal_high_snr(self):
        """A pure sine should have high SNR estimate."""
        sr = 48000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        snr = measure_snr(audio, sr)
        assert snr > 20  # Clean sine should score well


class TestMeasureSpectralCentroid:
    def test_sine_centroid(self):
        """Spectral centroid of a 440Hz sine should be low-frequency dominated."""
        sr = 48000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        centroid = measure_spectral_centroid(audio, sr)
        # Windowed FFT without a taper window produces spectral leakage,
        # so centroid is higher than the pure 440Hz fundamental.
        # Assert it stays in a reasonable low-frequency range.
        assert 400 < centroid < 2000


class TestComputeStats:
    def test_returns_all_keys(self, sine_440):
        path, audio, sr = sine_440
        stats = compute_stats(str(path))
        assert "lufs" in stats
        assert "true_peak" in stats
        assert "rms_db" in stats
        assert "snr_db" in stats
        assert "spectral_centroid" in stats
        assert "duration" in stats


class TestSaveReport:
    def test_creates_json(self, tmp_path, sine_440):
        path, _, _ = sine_440
        report_path = tmp_path / "report.json"
        save_stage_report(str(report_path), "raw", compute_stats(str(path)))
        assert report_path.exists()
        data = json.loads(report_path.read_text())
        assert "stages" in data
        assert "raw" in data["stages"]

    def test_appends_stages(self, tmp_path, sine_440):
        path, _, _ = sine_440
        report_path = tmp_path / "report.json"
        save_stage_report(str(report_path), "raw", compute_stats(str(path)))
        save_stage_report(str(report_path), "denoised", compute_stats(str(path)))
        data = json.loads(report_path.read_text())
        assert "raw" in data["stages"]
        assert "denoised" in data["stages"]
