# CLAUDE.md

## Project Overview

PodcastCleaner is a CLI tool that downloads and cleans podcast audio through a 7-stage pipeline: download (yt-dlp) → preprocess (48kHz mono) → separate (Demucs) → denoise (DeepFilterNet3) → transcribe (WhisperX, optional) → normalize (pyloudnorm) → export (ffmpeg). All processing is local, no API calls.

## Commands

```bash
# Run tests (fast, mocked ML)
.venv/bin/pytest -v -m "not slow"

# Run single test
.venv/bin/pytest tests/test_normalize.py::TestRunNormalize::test_marks_done -v

# Run quality regression tests (needs real processed audio)
EPISODE_DIR=output/01_Episode .venv/bin/pytest tests/test_audio_data.py -m slow -v

# Install in dev mode
pip install -e '.[dev]'

# Run CLI
.venv/bin/podcast-cleaner run --url "https://youtube.com/watch?v=ID"
.venv/bin/podcast-cleaner analyze output/01_Episode
.venv/bin/podcast-cleaner stage normalize output/01_Episode
```

## Architecture

Entry point: `podcast_cleaner/cli.py` — Click CLI with `run`, `analyze`, `stage` commands. `run_pipeline()` orchestrates all stages sequentially.

**Pipeline stages** (`podcast_cleaner/stages/`): Each stage is a module with a `run_<stage>(episode_dir, config, stage_logger=None)` function. Stages use `.done` marker files for resumability (`is_done`/`mark_done`/`clear_done` from `utils.py`). When `--resume` is not passed, markers are cleared before each stage.

**Stage pattern**: check is_done → find input files (with fallback to earlier stage) → process → compute_stats/save_stage_report → mark_done. Missing input raises `FileNotFoundError` (caught by pipeline's try/except per episode).

**Config**: `config.py` loads `config.yaml` merged with `DEFAULT_CONFIG`. YAML defaults → runtime overrides. Key sections: download, preprocess, separation, denoise, transcription, normalization, export.

**Analysis**: `analysis/audio_stats.py` has `compute_stats()` returning dict with `lufs`, `true_peak`, `rms_db`, `snr_db`, `spectral_centroid`, `duration`. Reports accumulate per-stage in `analysis/audio_report.json`.

**Utils**: `utils.py` has `read_audio`/`write_audio` (soundfile-based, auto mono conversion), `get_device` (CUDA/CPU auto-detection), `setup_logging` (file + console handler), `sanitize_filename`, `ensure_dir`.

## Key Conventions

- **Python 3.10+**, type hints throughout, `from __future__ import annotations` in all modules
- **ML models are lazy-imported** inside stage functions to keep CLI startup fast (~0.5s)
- **GPU memory**: `gc.collect()` + `torch.cuda.empty_cache()` between ML stages. Demucs and DeepFilterNet have automatic CPU fallback on OOM.
- **DeepFilterNet** runs with `torch.backends.cudnn.flags(enabled=False)` due to cuDNN/GRU compatibility. Long audio chunked in 60s windows with 0.5s crossfade overlap.
- **Demucs** uses `demucs.pretrained.get_model` + `demucs.apply.apply_model` (not `demucs.api` which is unreleased on PyPI). Output is resampled from 44100Hz to pipeline target (48kHz).
- **yt-dlp**: Uses venv binary via `_ytdlp_cmd()` helper (system yt-dlp may be outdated). Requires `deno` JS runtime (~/.deno/bin) and browser cookies (`cookies_from_browser` in config.yaml). `_ytdlp_env()` adds deno to PATH; `_ytdlp_extra_args()` adds cookie/remote-component flags.
- **DeepFilterNet**: Can produce NaN values in output chunks — sanitized with `np.nan_to_num`. Always check denoised output for NaN before downstream stages.
- **Demucs**: Input audio must be clamped to ±0.999 to avoid `pad1d` assertion errors in `hdemucs.py`. `htdemucs_ft` returns a `BagOfModels` wrapper (no `.segment` attribute). Pass `split=True` to `apply_model` for chunked processing.
- **Memory**: `del` large audio arrays before `compute_stats()` calls. Spectral centroid uses windowed FFT (8192 samples) to avoid massive allocations on long files.
- **Tests**: All ML calls mocked in unit/integration tests. Real ffmpeg used in export tests. `@pytest.mark.slow` for tests requiring real processed audio + `EPISODE_DIR` env var.

## File Layout

```
podcast_cleaner/
  cli.py              # CLI entry point, run_pipeline orchestrator
  config.py           # load_config, DEFAULT_CONFIG, _deep_merge
  utils.py            # Audio I/O, markers, device, logging, sanitize
  stages/
    __init__.py       # STAGE_ORDER list
    download.py       # yt-dlp: get_playlist_entries, download_single, run_download
    preprocess.py     # Resample/mono: run_preprocess
    separate.py       # Demucs: demucs_separate, run_separate
    denoise.py        # DeepFilterNet3: deepfilter_enhance, run_denoise
    transcribe.py     # WhisperX: whisperx_transcribe, run_transcribe, segments_to_srt
    normalize.py      # pyloudnorm: true_peak_limit, run_normalize
    export.py         # ffmpeg: convert_audio, run_export
  analysis/
    audio_stats.py    # measure_lufs, measure_snr, compute_stats, save_stage_report
tests/
  conftest.py         # Fixtures: tmp_audio, tmp_episode_dir, sample_config
  test_config.py      # Config loading/merging (4 tests)
  test_utils.py       # Sanitize, markers, audio I/O, ensure_dir (13 tests)
  test_audio_stats.py # Analysis functions (8 tests)
  test_download.py    # yt-dlp mocked (6 tests)
  test_preprocess.py  # Resample/mono/skip (4 tests)
  test_separate.py    # Demucs mocked (2 tests)
  test_denoise.py     # DeepFilterNet mocked (2 tests)
  test_normalize.py   # Loudness/peak/fallback (6 tests)
  test_export.py      # Real ffmpeg (2 tests)
  test_transcribe.py  # WhisperX mocked (6 tests)
  test_cli.py         # Click CLI args (5 tests)
  test_integration.py # Full pipeline mocked end-to-end (1 test)
  test_audio_data.py  # Quality regression: 4 slow + 2 synthetic
```

## Runtime Data

Projects output to `output/` (configurable). Each episode gets a subdirectory with: `raw/`, `preprocessed/`, `separated/`, `denoised/`, `normalized/`, `final/`, `analysis/`, `processing.log`, and `.done` marker files.
