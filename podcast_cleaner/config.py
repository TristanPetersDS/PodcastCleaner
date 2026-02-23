"""Configuration loading and validation."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG = {
    "output_dir": "./output",
    "download": {"format": "wav", "quality": "best"},
    "preprocess": {"sample_rate": 48000, "channels": 1},
    "separation": {"model": "htdemucs_ft", "device": "auto"},
    "denoise": {"model": "DeepFilterNet3"},
    "transcription": {"enabled": True, "model": "large-v3", "language": None, "device": "auto"},
    "normalization": {"target_lufs": -16.0, "true_peak_dbtp": -1.5},
    "export": {"formats": ["mp3", "flac"], "mp3_bitrate": "320k", "sample_rate": 48000},
}


def load_config(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load pipeline configuration from YAML, merged with defaults."""
    config = dict(DEFAULT_CONFIG)
    path = Path(config_path)
    if path.exists():
        with open(path) as f:
            user_config = yaml.safe_load(f) or {}
        config = _deep_merge(config, user_config)
    return config


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
