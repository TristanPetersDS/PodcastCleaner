"""Audio quality metrics: LUFS, true peak, SNR, spectral centroid, RMS."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf


def measure_lufs(audio: np.ndarray, sr: int) -> float:
    """Measure integrated loudness in LUFS."""
    meter = pyln.Meter(sr)
    lufs = meter.integrated_loudness(audio)
    return round(float(lufs), 1) if not np.isinf(lufs) else -100.0


def measure_true_peak(audio: np.ndarray) -> float:
    """Measure true peak in dBTP."""
    peak = float(np.max(np.abs(audio)))
    if peak < 1e-10:
        return -100.0
    return round(20 * np.log10(peak), 1)


def measure_rms_db(audio: np.ndarray) -> float:
    """Measure RMS level in dBFS."""
    rms = float(np.sqrt(np.mean(audio**2)))
    if rms < 1e-10:
        return -100.0
    return round(20 * np.log10(rms), 1)


def measure_snr(audio: np.ndarray, sr: int) -> float:
    """Estimate SNR by comparing signal energy to noise floor.

    Uses a simple approach: compute RMS over short frames, assume the
    quietest 10% of frames represent the noise floor.  When frame
    energies are very uniform (coefficient of variation < 5%), the
    signal is considered clean and a high SNR is returned.
    """
    frame_size = int(0.02 * sr)  # 20ms frames
    num_frames = len(audio) // frame_size
    if num_frames < 10:
        return 0.0

    frame_rms = np.array(
        [
            np.sqrt(np.mean(audio[i * frame_size : (i + 1) * frame_size] ** 2))
            for i in range(num_frames)
        ]
    )

    # Filter out silent frames
    active = frame_rms[frame_rms > 1e-8]
    if len(active) < 5:
        return 0.0

    # If frame energies are very uniform (e.g. a pure tone or
    # constant-level signal), there is effectively no noise floor
    # variation -- treat as a very clean signal.
    mean_rms = float(np.mean(active))
    std_rms = float(np.std(active))
    if mean_rms > 1e-8 and (std_rms / mean_rms) < 0.05:
        return 60.0

    # Noise floor: bottom 10% of active frames
    sorted_rms = np.sort(active)
    noise_floor = np.mean(sorted_rms[: max(1, len(sorted_rms) // 10)])
    signal_level = np.mean(sorted_rms[len(sorted_rms) // 2 :])  # top 50%

    if noise_floor < 1e-10:
        return 60.0  # Very clean

    snr = 20 * np.log10(signal_level / noise_floor)
    return round(float(snr), 1)


def measure_spectral_centroid(audio: np.ndarray, sr: int) -> float:
    """Compute spectral centroid (brightness measure) in Hz.

    Uses windowed FFTs averaged over the signal to avoid allocating a
    massive array for the full-signal FFT on long recordings.
    """
    win_size = min(8192, len(audio))
    hop = win_size // 2
    freqs = np.fft.rfftfreq(win_size, d=1.0 / sr)
    weighted_sum = 0.0
    energy_sum = 0.0

    for start in range(0, len(audio) - win_size + 1, hop):
        spectrum = np.abs(np.fft.rfft(audio[start : start + win_size]))
        energy = np.sum(spectrum)
        if energy > 1e-10:
            weighted_sum += float(np.sum(freqs * spectrum))
            energy_sum += float(energy)

    if energy_sum < 1e-10:
        return 0.0
    return round(weighted_sum / energy_sum, 1)


def compute_stats(
    audio_path: str, audio_data: tuple[np.ndarray, int] | None = None
) -> dict:
    """Compute all audio quality metrics for a file.

    Args:
        audio_path: Path to the audio file (used if audio_data is None)
        audio_data: Optional (audio_array, sample_rate) tuple to skip disk read
    """
    if audio_data is not None:
        audio, sr = audio_data
    else:
        audio, sr = sf.read(audio_path, dtype="float32")

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    return {
        "lufs": measure_lufs(audio, sr),
        "true_peak": measure_true_peak(audio),
        "rms_db": measure_rms_db(audio),
        "snr_db": measure_snr(audio, sr),
        "spectral_centroid": measure_spectral_centroid(audio, sr),
        "duration": round(len(audio) / sr, 2),
    }


def save_stage_report(
    report_path: str,
    stage_name: str,
    stats: dict,
) -> None:
    """Append stage metrics to the analysis report JSON file."""
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        with open(path) as f:
            report = json.load(f)
    else:
        report = {"stages": {}}

    report["stages"][stage_name] = stats

    with open(path, "w") as f:
        json.dump(report, f, indent=2)
