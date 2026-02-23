"""Tests for config loading and merging."""

from pathlib import Path

import yaml

from podcast_cleaner.config import DEFAULT_CONFIG, load_config, _deep_merge


def test_load_config_returns_defaults_when_no_file(tmp_path):
    """When config file doesn't exist, returns defaults."""
    config = load_config(tmp_path / "nonexistent.yaml")
    assert config["output_dir"] == "./output"
    assert config["separation"]["model"] == "htdemucs_ft"
    assert config["normalization"]["target_lufs"] == -16.0


def test_load_config_merges_with_defaults(tmp_path):
    """User config overrides only the keys it sets; defaults remain for the rest."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump({"output_dir": "/custom/output", "separation": {"device": "cpu"}}))
    config = load_config(config_path)
    assert config["output_dir"] == "/custom/output"
    assert config["separation"]["device"] == "cpu"
    assert config["separation"]["model"] == "htdemucs_ft"  # default preserved
    assert config["normalization"]["target_lufs"] == -16.0  # default preserved


def test_deep_merge_nested():
    """Nested dicts are merged recursively, not replaced."""
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    override = {"a": {"b": 10}}
    result = _deep_merge(base, override)
    assert result == {"a": {"b": 10, "c": 2}, "d": 3}


def test_deep_merge_new_keys():
    """Override can add new keys."""
    base = {"a": 1}
    override = {"b": 2}
    result = _deep_merge(base, override)
    assert result == {"a": 1, "b": 2}
