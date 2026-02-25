"""Stage 5: Transcription with WhisperX (optional)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from podcast_cleaner.utils import ensure_dir, get_device, is_done, mark_done

logger = logging.getLogger(__name__)


def format_srt_timestamp(seconds: float) -> str:
    """Format seconds as SRT timestamp HH:MM:SS,mmm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds % 1) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def segments_to_srt(segments: list[dict]) -> str:
    """Convert transcript segments to SRT subtitle format."""
    lines: list[str] = []
    for i, seg in enumerate(segments, 1):
        start = format_srt_timestamp(seg["start"])
        end = format_srt_timestamp(seg["end"])
        speaker = seg.get("speaker", "")
        text = seg.get("text", "").strip()
        label = f"[{speaker}] " if speaker else ""
        lines.extend([str(i), f"{start} --> {end}", f"{label}{text}", ""])
    return "\n".join(lines)


def whisperx_transcribe(audio_path: str, model_size: str, device: str) -> dict:
    """Run WhisperX transcription + alignment on an audio file.

    Returns dict with 'segments' and 'language'.
    """
    import gc

    import torch
    import whisperx

    # Free GPU memory from prior stages before loading whisper model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    compute_type = "float16" if "cuda" in device else "int8"
    model = whisperx.load_model(model_size, device, compute_type=compute_type)
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=16)

    detected_lang = result.get("language", "en")

    # Free transcription model before loading alignment model
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Align word-level timestamps
    model_a, metadata = whisperx.load_align_model(language_code=detected_lang, device=device)
    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    # Free alignment model
    del model_a
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "segments": result.get("segments", []),
        "language": detected_lang,
    }


def run_transcribe(
    episode_dir: str,
    config: dict,
    stage_logger: logging.Logger | None = None,
) -> None:
    """Transcribe denoised audio with WhisperX."""
    log = stage_logger or logger
    episode_path = Path(episode_dir)
    tx_config = config.get("transcription", {})

    if not tx_config.get("enabled", True):
        log.info("Transcription disabled -- skipping")
        return

    if is_done(episode_path, "transcribe"):
        log.info("Transcribe: skipping (already done)")
        return

    # Find audio to transcribe (denoised preferred)
    denoised_dir = episode_path / "denoised"
    audio_files = list(denoised_dir.glob("*_denoised.wav")) if denoised_dir.exists() else []
    if not audio_files:
        sep_dir = episode_path / "separated"
        audio_files = list(sep_dir.glob("*_vocals.wav")) if sep_dir.exists() else []
    if not audio_files:
        raise FileNotFoundError(f"No audio files found for transcription in {episode_dir}")

    device = str(get_device(tx_config.get("device", "auto")))
    model_size = tx_config.get("model", "large-v3")
    log.info(f"Transcription: model={model_size}, device={device}")

    episode_name = episode_path.name
    transcript_dir = ensure_dir(episode_path / "final" / "transcript")

    for audio_path in audio_files:
        log.info(f"Transcribing: {audio_path.name}")

        try:
            result = whisperx_transcribe(str(audio_path), model_size, device)
        except ModuleNotFoundError:
            log.warning(
                "whisperx is not installed. Install with: "
                "pip install podcast-cleaner[transcribe]. Skipping transcription."
            )
            mark_done(episode_path, "transcribe")
            return

        segments = result.get("segments", [])

        # Build transcript data
        transcript_segments = []
        for seg in segments:
            transcript_segments.append({
                "start": round(seg.get("start", 0), 3),
                "end": round(seg.get("end", 0), 3),
                "speaker": seg.get("speaker", ""),
                "text": seg.get("text", "").strip(),
            })

        transcript = {
            "episode": episode_name,
            "language": result.get("language", "en"),
            "segments": transcript_segments,
        }

        # Save JSON
        json_path = transcript_dir / f"{episode_name}.json"
        with open(json_path, "w") as f:
            json.dump(transcript, f, indent=2, ensure_ascii=False)
        log.info(f"  JSON: {json_path.name}")

        # Save SRT
        srt_path = transcript_dir / f"{episode_name}.srt"
        with open(srt_path, "w") as f:
            f.write(segments_to_srt(transcript_segments))
        log.info(f"  SRT: {srt_path.name}")

    mark_done(episode_path, "transcribe")
