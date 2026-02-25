# PodcastCleaner

CLI tool for downloading and cleaning podcast audio. Downloads from YouTube (playlists or single videos), separates speech from background noise, removes residual noise, normalizes loudness, and exports in distribution formats.

## Pipeline

Audio flows through 7 stages. Each stage writes to its own subdirectory and is independently resumable via `.done` marker files.

```
download → preprocess → separate → denoise → transcribe → normalize → export
```

| Stage | What it does | Model/Tool |
|---|---|---|
| **download** | Grab audio from YouTube | yt-dlp |
| **preprocess** | Resample to 48kHz mono float32 | librosa/soundfile |
| **separate** | Isolate vocals from background | Demucs htdemucs_ft |
| **denoise** | Remove residual noise from vocals | DeepFilterNet3 |
| **transcribe** | Word-level transcription (optional) | WhisperX large-v3 |
| **normalize** | Loudness normalize to -16 LUFS | pyloudnorm |
| **export** | Convert to MP3/FLAC | ffmpeg |

## Quick Start

```bash
# Clone and install
git clone <repo-url> && cd PodcastCleaner
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'

# Optional: install transcription support
pip install -e '.[transcribe]'
```

### Requirements

- Python 3.10+
- ffmpeg (`sudo apt install ffmpeg` or `brew install ffmpeg`)
- CUDA GPU recommended (12GB+ VRAM). CPU fallback is automatic but slow.
- Node.js (for yt-dlp YouTube extraction)

### Run on a YouTube video

```bash
podcast-cleaner run --url "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Run on a YouTube playlist

```bash
podcast-cleaner run --url "https://www.youtube.com/playlist?list=PLAYLIST_ID"
```

### Run on local audio files

```bash
# Single file
podcast-cleaner run --input /path/to/episode.wav

# Directory of files
podcast-cleaner run --input-dir /path/to/episodes/
```

### Resume a failed run

```bash
podcast-cleaner run --url "..." --resume
```

Without `--resume`, stages re-run from scratch. With `--resume`, completed stages are skipped.

### Skip stages

```bash
podcast-cleaner run --url "..." --skip transcribe --skip denoise
```

### Run a single stage

```bash
podcast-cleaner stage normalize output/01_My-Episode
```

### Analyze quality metrics

```bash
podcast-cleaner analyze output/01_My-Episode
```

Output:
```
raw              LUFS= -30.1  SNR= 31.4dB  Peak= -5.3dBTP  Duration=990.0s
preprocessed     LUFS= -30.1  SNR= 31.4dB  Peak= -5.3dBTP  Duration=990.0s
separated        LUFS= -30.1  SNR= 34.6dB  Peak= -5.4dBTP  Duration=990.0s
denoised         LUFS= -30.3  SNR= 54.2dB  Peak= -5.6dBTP  Duration=990.0s
normalized       LUFS= -26.2  SNR= 54.2dB  Peak= -1.5dBTP  Duration=990.0s
```

## Configuration

Edit `config.yaml` to tune the pipeline:

```yaml
output_dir: "./output"

download:
  format: "wav"

preprocess:
  sample_rate: 48000

separation:
  model: "htdemucs_ft"   # Demucs model variant
  device: "auto"          # auto, cuda, cpu

denoise:
  model: "DeepFilterNet3"

transcription:
  enabled: false          # Set true and install whisperx to enable
  model: "large-v3"

normalization:
  target_lufs: -16.0      # Podcast standard loudness
  true_peak_dbtp: -1.5    # Maximum true peak (prevents clipping)

export:
  formats: ["mp3", "flac"]
  mp3_bitrate: "320k"
  sample_rate: 48000
```

## Output Structure

```
output/
  01_Episode-Title/
    raw/                    # Downloaded audio
    preprocessed/           # 48kHz mono float32
    separated/              # vocals.wav + background.wav
    denoised/               # Noise-removed vocals
    normalized/             # Loudness-normalized audio
    final/                  # MP3, FLAC exports
      transcript/           # JSON + SRT (if transcription enabled)
    analysis/
      audio_report.json     # Quality metrics per stage
    processing.log          # Full pipeline log
    .download.done          # Stage completion markers
    .preprocess.done
    ...
```

## Testing

```bash
# Run all tests (fast, no GPU needed)
pytest -v -m "not slow"

# Run a specific test file
pytest tests/test_normalize.py -v

# Run quality regression tests against real processed audio
EPISODE_DIR=output/01_My-Episode pytest tests/test_audio_data.py -m slow -v
```

61 unit/integration tests cover all stages with mocked ML models. 4 additional slow tests validate quality thresholds against real processed audio.

## Architecture

```
podcast_cleaner/
  cli.py                    # Click CLI: run, analyze, stage commands
  config.py                 # YAML config loading with defaults
  utils.py                  # Audio I/O, done markers, device detection
  stages/
    __init__.py             # STAGE_ORDER definition
    download.py             # yt-dlp wrapper
    preprocess.py           # Resample + mono conversion
    separate.py             # Demucs vocal isolation
    denoise.py              # DeepFilterNet3 noise removal (chunked)
    transcribe.py           # WhisperX transcription + SRT
    normalize.py            # pyloudnorm + true peak limiting
    export.py               # ffmpeg format conversion
  analysis/
    audio_stats.py          # LUFS, SNR, RMS, spectral centroid, true peak
```

Each stage follows the same pattern:
1. Check `is_done()` marker — skip if already completed
2. Find input files (with fallback to previous stage)
3. Process audio
4. Compute and save analysis metrics
5. Write `mark_done()` marker

ML models are lazy-imported to keep CLI startup fast. GPU memory is released between stages via `gc.collect()` + `torch.cuda.empty_cache()`.

## Known Behavior

- **Normalization may not reach -16 LUFS** if the source audio has loud transients. The true peak limiter (-1.5 dBTP) prevents clipping, which can cap the overall loudness below target. This is intentional — clean audio over loud audio.
- **DeepFilterNet runs with cuDNN disabled** due to a compatibility issue with some torch/cuDNN versions. Performance impact is minimal.
- **Long audio is processed in 60s chunks** by DeepFilterNet to avoid GPU OOM on files >5 minutes.
- **Demucs outputs at 44100Hz** internally. The pipeline automatically resamples back to 48kHz after separation.
- **Demucs processing time** is roughly 2x real-time on a 12GB GPU. Episodes longer than ~35 minutes may fall back to CPU, which is significantly slower.
- **Very noisy recordings** (outdoor, ambient) may produce near-silence after denoising. The pipeline works best on studio/indoor podcast audio with SNR above ~20 dB.
- **yt-dlp requires a JavaScript runtime** (deno) and browser cookies for YouTube authentication. Configure `cookies_from_browser` in `config.yaml`.
