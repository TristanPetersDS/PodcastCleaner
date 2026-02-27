# CLAUDE.md

## Project Overview

PodcastCleaner is a CLI tool that downloads and cleans podcast audio through a 7-stage pipeline: download (yt-dlp) → preprocess (48kHz mono) → separate (Demucs) → denoise (DeepFilterNet3) → transcribe (WhisperX, optional) → normalize (pyloudnorm) → export (ffmpeg). All processing is local, no API calls.

## Commands

```bash
# Run tests (fast, mocked ML) — 94 tests
.venv/bin/pytest -v -m "not slow"

# Run single test
.venv/bin/pytest tests/test_normalize.py::TestRunNormalize::test_marks_done -v

# Run quality regression tests (needs real processed audio)
EPISODE_DIR=output/01_Episode .venv/bin/pytest tests/test_audio_data.py -m slow -v

# Install in dev mode
pip install -e '.[dev]'

# Run CLI
.venv/bin/podcast-cleaner run --url "https://youtube.com/watch?v=ID"
.venv/bin/podcast-cleaner run --input-dir ./episodes --verbose
.venv/bin/podcast-cleaner run --input file.wav --copy-input --cleanup-intermediates
.venv/bin/podcast-cleaner analyze output/01_Episode
.venv/bin/podcast-cleaner stage normalize output/01_Episode
.venv/bin/podcast-cleaner check

# Docker
docker-compose up --build  # GPU-accelerated
./setup.sh                 # Local setup script (creates venv, installs deps)

# Pre-commit hooks
pre-commit install
pre-commit run --all-files
```

## Architecture

Entry point: `podcast_cleaner/cli.py` — Click CLI with `run`, `analyze`, `stage`, `check` commands. `run_pipeline()` orchestrates all stages sequentially with `PipelineTracker` progress reporting.

**Pipeline stages** (`podcast_cleaner/stages/`): Each stage is a module with a `run_<stage>(episode_dir, config, stage_logger=None)` function. Stages use `.done` marker files for resumability (`is_done`/`mark_done`/`clear_done` from `utils.py`). When `--resume` is not passed, markers are cleared before each stage.

**Stage pattern**: check is_done → find input files (with fallback to earlier stage) → process → compute_stats/save_stage_report → mark_done. Missing input raises `FileNotFoundError` (caught by pipeline's try/except per episode).

**Config**: `config.py` loads `config.yaml` merged with `DEFAULT_CONFIG`, then validates via `validate_config()`. Validation checks unknown top-level keys, and validates types/ranges (target_lufs negative, sample_rate positive, etc.). Key sections: download, preprocess, separation, denoise, transcription, normalization, export.

**Display**: `display.py` exports a shared `Console` instance with TTY auto-detection (`force_terminal=False` when not a TTY). All CLI output uses `console.print()` except the `check` command (uses `click.echo` for CliRunner test compatibility).

**Progress**: `tracker.py` has `PipelineTracker` dataclass tracking episode/stage progress. Methods: `start_episode()`, `start_stage()`, `complete_stage()`, `complete_episode()`, `fail_episode()`, `print_summary()`.

**Analysis**: `analysis/audio_stats.py` has `compute_stats()` returning dict with `lufs`, `true_peak`, `rms_db`, `snr_db`, `spectral_centroid`, `duration`. Reports accumulate per-stage in `analysis/audio_report.json`.

**Utils**: `utils.py` has `read_audio`/`write_audio` (soundfile-based, auto mono conversion), `get_device` (CUDA/CPU auto-detection), `setup_logging` (file + console handler), `sanitize_filename`, `ensure_dir`.

## Key Conventions

- **Python 3.10+**, type hints throughout, `from __future__ import annotations` in all modules
- **ML models are lazy-imported** inside stage functions to keep CLI startup fast (~0.5s)
- **Model caching**: Demucs and DeepFilterNet models loaded once per stage via `_load_demucs_model()` / `_load_deepfilter_model()` helpers, reused across files in the same episode. WhisperX loads per-file due to memory constraints.
- **GPU memory**: `gc.collect()` + `torch.cuda.empty_cache()` between ML stages. Demucs and DeepFilterNet have automatic CPU fallback on OOM.
- **DeepFilterNet** runs with `torch.backends.cudnn.flags(enabled=False)` due to cuDNN/GRU compatibility. Long audio chunked in 60s windows with 0.5s crossfade overlap.
- **Demucs** uses `demucs.pretrained.get_model` + `demucs.apply.apply_model` (not `demucs.api` which is unreleased on PyPI). Output is resampled from 44100Hz to pipeline target (48kHz).
- **yt-dlp**: Uses venv binary via `_ytdlp_cmd()` helper (system yt-dlp may be outdated). Requires `deno` JS runtime (~/.deno/bin) and browser cookies (`cookies_from_browser` in config.yaml). `_ytdlp_env()` adds deno to PATH; `_ytdlp_extra_args()` adds cookie/remote-component flags.
- **DeepFilterNet**: Can produce NaN values in output chunks — sanitized with `np.nan_to_num`. Always check denoised output for NaN before downstream stages.
- **Demucs**: Input audio must be clamped to ±0.999 to avoid `pad1d` assertion errors in `hdemucs.py`. `htdemucs_ft` returns a `BagOfModels` wrapper (no `.segment` attribute). Pass `split=True` to `apply_model` for chunked processing.
- **Memory**: `del` large audio arrays before `compute_stats()` calls. Spectral centroid uses windowed FFT (8192 samples) to avoid massive allocations on long files.
- **Cross-filesystem input**: `_link_or_copy()` in cli.py auto-detects cross-filesystem (st_dev mismatch) and falls back to copy. `--copy-input` flag forces copy (useful for Docker).
- **Tests**: All ML calls mocked in unit/integration tests. Real ffmpeg used in export tests. `@pytest.mark.slow` for tests requiring real processed audio + `EPISODE_DIR` env var.

## CLI Flags (run command)

| Flag | Description |
|------|-------------|
| `--url` | YouTube playlist or video URL |
| `--input-dir` | Directory of local audio files (preserves existing `NN_` prefixes) |
| `--input` | Single local audio file |
| `--config` | Path to config.yaml (default: config.yaml) |
| `--skip` | Skip a stage (repeatable) |
| `--resume` | Resume from last completed stage |
| `--cleanup-intermediates` | Delete intermediate stage dirs after success (conflicts with --resume) |
| `--copy-input` | Copy input files instead of symlinking (for Docker/cross-filesystem) |
| `--verbose` / `-v` | Show detailed logging output (conflicts with --quiet) |
| `--quiet` / `-q` | Only show progress and errors (conflicts with --verbose) |

## File Layout

```
podcast_cleaner/
  cli.py              # CLI entry point, run_pipeline orchestrator, _link_or_copy
  config.py           # load_config, validate_config, DEFAULT_CONFIG, _deep_merge
  display.py          # Shared Rich Console with TTY auto-detection
  tracker.py          # PipelineTracker dataclass for progress reporting
  utils.py            # Audio I/O, markers, device, logging, sanitize
  stages/
    __init__.py       # STAGE_ORDER list
    download.py       # yt-dlp: get_playlist_entries, download_single, run_download
    preprocess.py     # Resample/mono: run_preprocess
    separate.py       # Demucs: demucs_separate, _load_demucs_model, run_separate
    denoise.py        # DeepFilterNet3: deepfilter_enhance, _load_deepfilter_model, run_denoise
    transcribe.py     # WhisperX: whisperx_transcribe, run_transcribe, segments_to_srt
    normalize.py      # pyloudnorm: true_peak_limit, run_normalize
    export.py         # ffmpeg: convert_audio, run_export
  analysis/
    audio_stats.py    # measure_lufs, measure_snr, compute_stats, save_stage_report
tests/
  conftest.py         # Fixtures: tmp_audio, tmp_episode_dir, sample_config
  test_config.py      # Config loading/merging/validation (10 tests)
  test_utils.py       # Sanitize, markers, audio I/O, ensure_dir (13 tests)
  test_audio_stats.py # Analysis functions (8 tests)
  test_download.py    # yt-dlp mocked (6 tests)
  test_preprocess.py  # Resample/mono/skip (4 tests)
  test_separate.py    # Demucs mocked (2 tests)
  test_denoise.py     # DeepFilterNet mocked (2 tests)
  test_normalize.py   # Loudness/peak/fallback (6 tests)
  test_export.py      # Real ffmpeg (2 tests)
  test_transcribe.py  # WhisperX mocked (6 tests)
  test_cli.py         # CLI args, flags, Rich, check, copy-input (16 tests)
  test_tracker.py     # PipelineTracker lifecycle/summary (4 tests)
  test_integration.py # Full pipeline mocked end-to-end (1 test)
  test_audio_data.py  # Quality regression: 4 slow + 2 synthetic
```

## Infrastructure

- **Dockerfile**: Multi-stage build on nvidia/cuda:12.1.0, pre-bakes Demucs + DeepFilterNet models
- **docker-compose.yml**: GPU passthrough, volume mounts for output/config/cookies
- **setup.sh**: Local setup script — detects Python 3.10+, creates venv, installs deps, copies example config, runs check
- **.github/workflows/ci.yml**: GitHub Actions CI — matrix testing Python 3.10-3.12, installs ffmpeg, runs `pytest -m "not slow"`
- **.pre-commit-config.yaml**: ruff (lint + format), trailing-whitespace, end-of-file-fixer, check-yaml, check-added-large-files

## Runtime Data

Projects output to `output/` (configurable). Each episode gets a subdirectory with: `raw/`, `preprocessed/`, `separated/`, `denoised/`, `normalized/`, `final/`, `analysis/`, `processing.log`, and `.done` marker files.
