"""Configuration loading and validation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

KNOWN_TOP_LEVEL_KEYS = {
    "output_dir", "download", "preprocess", "separation",
    "denoise", "transcription", "normalization", "export",
}


DEFAULT_CONFIG = {
    "output_dir": "./output",
    "download": {"format": "wav", "quality": "best"},
    "preprocess": {"sample_rate": 48000, "channels": 1},
    "separation": {"model": "htdemucs_ft", "device": "auto", "max_segment_minutes": 10},
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
    elif Path("config.example.yaml").exists():
        with open("config.example.yaml") as f:
            user_config = yaml.safe_load(f) or {}
        config = _deep_merge(config, user_config)
    validate_config(config)
    return config


def validate_config(config: dict) -> list[str]:
    """Validate config and return list of warnings. Also logs them."""
    warnings: list[str] = []

    # Check unknown top-level keys
    for key in config:
        if key not in KNOWN_TOP_LEVEL_KEYS:
            msg = f"Unknown config key '{key}' — possible typo? Known keys: {', '.join(sorted(KNOWN_TOP_LEVEL_KEYS))}"
            warnings.append(msg)
            logger.warning(msg)

    # Validate normalization
    norm = config.get("normalization", {})
    if isinstance(norm, dict):
        lufs = norm.get("target_lufs")
        if lufs is not None and not isinstance(lufs, str) and lufs >= 0:
            msg = f"normalization.target_lufs should be negative (got {lufs})"
            warnings.append(msg)
            logger.warning(msg)
        peak = norm.get("true_peak_dbtp")
        if peak is not None and not isinstance(peak, str) and peak >= 0:
            msg = f"normalization.true_peak_dbtp should be negative (got {peak})"
            warnings.append(msg)
            logger.warning(msg)

    # Validate preprocess
    pre = config.get("preprocess", {})
    if isinstance(pre, dict):
        sr = pre.get("sample_rate")
        if sr is not None and isinstance(sr, (int, float)) and sr <= 0:
            msg = f"preprocess.sample_rate must be positive (got {sr})"
            warnings.append(msg)
            logger.warning(msg)

    # Validate separation
    sep = config.get("separation", {})
    if isinstance(sep, dict):
        max_seg = sep.get("max_segment_minutes")
        if max_seg is not None and isinstance(max_seg, (int, float)) and max_seg <= 0:
            msg = f"separation.max_segment_minutes must be positive (got {max_seg})"
            warnings.append(msg)
            logger.warning(msg)

    return warnings


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
