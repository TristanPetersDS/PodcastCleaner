"""Tests for config loading and merging."""

import logging
from pathlib import Path

import yaml

from podcast_cleaner.config import DEFAULT_CONFIG, load_config, validate_config, _deep_merge


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


def test_config_fallback_to_example(tmp_path, monkeypatch):
    """When config.yaml missing, loads config.example.yaml."""
    monkeypatch.chdir(tmp_path)
    example = tmp_path / "config.example.yaml"
    example.write_text("output_dir: ./example_output\n")
    from podcast_cleaner.config import load_config
    config = load_config("config.yaml")
    assert config["output_dir"] == "./example_output"


class TestConfigValidation:
    def test_warns_unknown_top_level_key(self, caplog):
        """Unknown top-level keys should produce warnings."""
        config = dict(DEFAULT_CONFIG)
        config["normalisation"] = {"target_lufs": -16.0}  # British spelling typo
        with caplog.at_level(logging.WARNING):
            validate_config(config)
        assert any("normalisation" in msg for msg in caplog.messages)

    def test_valid_config_no_warnings(self, caplog):
        """Valid config should produce no warnings."""
        with caplog.at_level(logging.WARNING):
            validate_config(dict(DEFAULT_CONFIG))
        config_warnings = [m for m in caplog.messages if "config" in m.lower() or "unknown" in m.lower()]
        assert len(config_warnings) == 0

    def test_validates_target_lufs_negative(self, caplog):
        """target_lufs must be negative."""
        config = dict(DEFAULT_CONFIG)
        config["normalization"] = {"target_lufs": 5.0, "true_peak_dbtp": -1.5}
        with caplog.at_level(logging.WARNING):
            validate_config(config)
        assert any("target_lufs" in msg for msg in caplog.messages)

    def test_validates_sample_rate_positive(self, caplog):
        """sample_rate must be positive."""
        config = dict(DEFAULT_CONFIG)
        config["preprocess"] = {"sample_rate": -1, "channels": 1}
        with caplog.at_level(logging.WARNING):
            validate_config(config)
        assert any("sample_rate" in msg for msg in caplog.messages)

    def test_load_config_calls_validate(self, tmp_path, caplog):
        """load_config should call validate_config."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump({"bogus_key": True}))
        with caplog.at_level(logging.WARNING):
            load_config(config_path)
        assert any("bogus_key" in msg for msg in caplog.messages)
