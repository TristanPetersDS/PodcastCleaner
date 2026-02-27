"""Click CLI entry point for podcast-cleaner."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import click

from podcast_cleaner.config import load_config
from podcast_cleaner.display import console
from podcast_cleaner.stages import STAGE_ORDER
from podcast_cleaner.tracker import PipelineTracker
from podcast_cleaner.utils import AUDIO_EXTENSIONS

_EPISODE_NUMBER_RE = re.compile(r"^(\d{2,3})_(.+)$")


def _parse_episode_number(stem: str) -> tuple[int, str] | None:
    """Extract episode number and title from a filename stem like '03_My-Episode'.

    Returns (number, title) if the stem starts with a 2-3 digit prefix,
    or None if no prefix is found.
    """
    m = _EPISODE_NUMBER_RE.match(stem)
    if m:
        return int(m.group(1)), m.group(2)
    return None


def _link_or_copy(source: Path, dest: Path, force_copy: bool = False) -> None:
    """Symlink source to dest, or copy if cross-filesystem or forced."""
    if dest.exists():
        return
    if force_copy:
        import shutil

        shutil.copy2(str(source), str(dest))
        return
    # Auto-detect cross-filesystem
    try:
        if source.stat().st_dev != dest.parent.stat().st_dev:
            import shutil

            shutil.copy2(str(source), str(dest))
            return
    except OSError:
        pass
    # Default: symlink
    try:
        dest.symlink_to(source.resolve())
    except OSError:
        # Fallback to copy (e.g., Windows without privileges)
        import shutil

        shutil.copy2(str(source), str(dest))


def run_pipeline(
    config: dict,
    episode_dirs: list[str],
    skip_stages: tuple[str, ...] = (),
    resume: bool = False,
    verbose: bool = False,
    quiet: bool = False,
) -> list[str]:
    """Run the full pipeline on a list of episode directories.

    Returns list of successfully processed episode dirs.
    """
    import logging

    from podcast_cleaner.stages.denoise import run_denoise
    from podcast_cleaner.stages.export import run_export
    from podcast_cleaner.stages.normalize import run_normalize
    from podcast_cleaner.stages.preprocess import run_preprocess
    from podcast_cleaner.stages.separate import run_separate
    from podcast_cleaner.stages.transcribe import run_transcribe
    from podcast_cleaner.utils import clear_done, setup_logging

    stage_runners = {
        "preprocess": run_preprocess,
        "separate": run_separate,
        "denoise": run_denoise,
        "transcribe": run_transcribe,
        "normalize": run_normalize,
        "export": run_export,
    }

    tracker = PipelineTracker(console, total_episodes=len(episode_dirs))
    successes: list[str] = []

    for episode_dir in episode_dirs:
        ep_name = Path(episode_dir).name
        stage_logger = setup_logging(Path(episode_dir) / "processing.log", ep_name)

        # Configure console output level based on verbosity flags
        if quiet:
            # Suppress console handler -- only log to file
            for handler in stage_logger.handlers:
                if hasattr(handler, "stream") and handler.stream in (
                    sys.stdout,
                    sys.stderr,
                ):
                    handler.setLevel(logging.ERROR)
        elif verbose:
            # Show all log messages on console
            stage_logger.setLevel(logging.DEBUG)
            for handler in stage_logger.handlers:
                if hasattr(handler, "stream") and handler.stream in (
                    sys.stdout,
                    sys.stderr,
                ):
                    handler.setLevel(logging.DEBUG)

        stage_logger.info(f"=== Processing: {ep_name} ===")

        active_stages = [
            s for s in STAGE_ORDER if s != "download" and s not in skip_stages
        ]
        tracker.start_episode(ep_name, len(active_stages))

        try:
            for stage_name in STAGE_ORDER:
                if stage_name == "download":
                    continue  # Already handled
                if stage_name in skip_stages:
                    stage_logger.info(f"Skipping stage: {stage_name}")
                    continue

                # When not resuming, clear done markers to force re-run
                if not resume:
                    clear_done(episode_dir, stage_name)

                stage_logger.info(f"--- Stage: {stage_name} ---")
                tracker.start_stage(stage_name)
                runner = stage_runners.get(stage_name)
                if runner:
                    runner(episode_dir, config, stage_logger=stage_logger)
                tracker.complete_stage()

            stage_logger.info(f"=== Complete: {ep_name} ===")
            tracker.complete_episode()
            successes.append(episode_dir)

        except Exception as e:
            stage_logger.error(f"Pipeline failed for {ep_name}: {e}", exc_info=True)
            tracker.fail_episode(str(e))

    tracker.print_summary()
    return successes


@click.group()
def main():
    """PodcastCleaner -- download and clean podcast audio."""
    pass


@main.command()
@click.option("--url", default=None, help="YouTube playlist or video URL")
@click.option(
    "--input-dir",
    default=None,
    type=click.Path(exists=True),
    help="Directory of local audio files",
)
@click.option(
    "--input",
    "input_file",
    default=None,
    type=click.Path(exists=True),
    help="Single local audio file",
)
@click.option(
    "--config", "config_path", default="config.yaml", help="Path to config.yaml"
)
@click.option("--skip", multiple=True, help="Skip a stage (can repeat)")
@click.option("--resume", is_flag=True, help="Resume from last completed stage")
@click.option(
    "--cleanup-intermediates",
    is_flag=True,
    help="Delete intermediate stage dirs after success",
)
@click.option(
    "--copy-input",
    is_flag=True,
    help="Copy input files instead of symlinking (for Docker/cross-filesystem)",
)
@click.option("--verbose", "-v", is_flag=True, help="Show detailed logging output")
@click.option("--quiet", "-q", is_flag=True, help="Only show progress and errors")
def run(
    url,
    input_dir,
    input_file,
    config_path,
    skip,
    resume,
    cleanup_intermediates,
    copy_input,
    verbose,
    quiet,
):
    """Run the full audio cleaning pipeline."""
    if verbose and quiet:
        raise click.UsageError("Cannot use --verbose with --quiet")
    if cleanup_intermediates and resume:
        raise click.UsageError("Cannot use --cleanup-intermediates with --resume")

    config = load_config(config_path)
    output_base = config["output_dir"]

    episode_dirs: list[str] = []

    if url:
        from podcast_cleaner.stages.download import run_download
        from podcast_cleaner.utils import setup_logging

        Path(output_base).mkdir(parents=True, exist_ok=True)
        dl_logger = setup_logging(Path(output_base) / "download.log", "download")
        episode_dirs = run_download(url, output_base, config, stage_logger=dl_logger)

    elif input_dir:
        from podcast_cleaner.stages.download import build_episode_dirname
        from podcast_cleaner.utils import ensure_dir

        input_path = Path(input_dir).resolve()
        audio_files = sorted(
            f for f in input_path.iterdir() if f.suffix.lower() in AUDIO_EXTENSIONS
        )
        # Sequential counter for files without an existing NN_ prefix
        next_seq = 0
        for f in audio_files:
            parsed = _parse_episode_number(f.stem)
            if parsed is not None:
                ep_num, title = parsed
                dirname = build_episode_dirname(title, ep_num - 1)
            else:
                dirname = build_episode_dirname(f.stem, next_seq)
                next_seq += 1
            ep_dir = ensure_dir(Path(output_base) / dirname)
            raw_dir = ensure_dir(ep_dir / "raw")
            dest = raw_dir / f.name
            _link_or_copy(f, dest, force_copy=copy_input)
            episode_dirs.append(str(ep_dir))

    elif input_file:
        from podcast_cleaner.stages.download import build_episode_dirname
        from podcast_cleaner.utils import ensure_dir

        f = Path(input_file).resolve()
        dirname = build_episode_dirname(f.stem, None)
        ep_dir = ensure_dir(Path(output_base) / dirname)
        raw_dir = ensure_dir(ep_dir / "raw")
        dest = raw_dir / f.name
        _link_or_copy(f, dest, force_copy=copy_input)
        episode_dirs.append(str(ep_dir))

    else:
        raise click.UsageError("Provide --url, --input-dir, or --input")

    if not episode_dirs:
        console.print("No episodes to process.")
        return

    console.print(f"Processing {len(episode_dirs)} episode(s)...")
    successes = run_pipeline(
        config,
        episode_dirs,
        skip_stages=skip,
        resume=resume,
        verbose=verbose,
        quiet=quiet,
    )

    if cleanup_intermediates:
        import shutil

        intermediate_dirs = ["preprocessed", "separated", "denoised", "normalized"]
        for ep_dir in successes:
            for dirname in intermediate_dirs:
                d = Path(ep_dir) / dirname
                if d.exists():
                    shutil.rmtree(d)

    if len(successes) < len(episode_dirs):
        sys.exit(1)


@main.command()
@click.argument("episode_dir", type=click.Path(exists=True))
@click.option(
    "--config", "config_path", default="config.yaml", help="Path to config.yaml"
)
def analyze(episode_dir, config_path):
    """Analyze audio quality for an episode directory."""
    from podcast_cleaner.analysis.audio_stats import compute_stats, save_stage_report

    episode_path = Path(episode_dir)
    report_path = episode_path / "analysis" / "audio_report.json"

    # Scan all stage directories for audio
    stage_dirs = {
        "raw": episode_path / "raw",
        "preprocessed": episode_path / "preprocessed",
        "separated": episode_path / "separated",
        "denoised": episode_path / "denoised",
        "normalized": episode_path / "normalized",
    }

    for stage_name, stage_dir in stage_dirs.items():
        if not stage_dir.exists():
            continue
        wavs = list(stage_dir.glob("*.wav"))
        if not wavs:
            continue
        # Use the first vocal/main file
        target = wavs[0]
        for w in wavs:
            if "vocal" in w.name or "denoised" in w.name or "normalized" in w.name:
                target = w
                break
        stats = compute_stats(str(target))
        save_stage_report(str(report_path), stage_name, stats)
        console.print(
            f"{stage_name:15s}  LUFS={stats['lufs']:6.1f}  "
            f"SNR={stats['snr_db']:5.1f}dB  "
            f"Peak={stats['true_peak']:5.1f}dBTP  "
            f"Duration={stats['duration']:.1f}s"
        )

    console.print(f"\nFull report: {report_path}")


@main.command()
@click.argument("stage_name")
@click.argument("episode_dir", type=click.Path(exists=True))
@click.option(
    "--config", "config_path", default="config.yaml", help="Path to config.yaml"
)
def stage(stage_name, episode_dir, config_path):
    """Run a single pipeline stage on an episode directory."""
    from podcast_cleaner.stages.denoise import run_denoise
    from podcast_cleaner.stages.export import run_export
    from podcast_cleaner.stages.normalize import run_normalize
    from podcast_cleaner.stages.preprocess import run_preprocess
    from podcast_cleaner.stages.separate import run_separate
    from podcast_cleaner.stages.transcribe import run_transcribe
    from podcast_cleaner.utils import clear_done

    config = load_config(config_path)
    runners = {
        "preprocess": run_preprocess,
        "separate": run_separate,
        "denoise": run_denoise,
        "transcribe": run_transcribe,
        "normalize": run_normalize,
        "export": run_export,
    }

    if stage_name not in runners:
        raise click.UsageError(
            f"Unknown stage: {stage_name}. Available: {', '.join(runners)}"
        )

    # Clear done marker so it re-runs
    clear_done(episode_dir, stage_name)
    runners[stage_name](episode_dir, config)
    console.print(f"Stage '{stage_name}' complete for {episode_dir}")


@main.command()
def check():
    """Check system dependencies and capabilities."""
    import platform
    import shutil
    import subprocess

    click.echo(f"Python: {platform.python_version()}")
    click.echo(f"Platform: {platform.platform()}")

    # Check ffmpeg
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"], capture_output=True, text=True, timeout=10
            )
            version_line = result.stdout.split("\n")[0] if result.stdout else "unknown"
            click.echo(f"ffmpeg: {version_line}")
        except Exception:
            click.echo(f"ffmpeg: found at {ffmpeg_path} (version unknown)")
    else:
        click.echo("ffmpeg: NOT FOUND")

    # Check GPU/CUDA
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            click.echo(f"CUDA: available ({gpu_name})")
        else:
            click.echo("CUDA: not available (CPU only)")
    except ImportError:
        click.echo("PyTorch: NOT INSTALLED")

    # Check yt-dlp
    ytdlp_path = shutil.which("yt-dlp")
    click.echo(f"yt-dlp: {'found' if ytdlp_path else 'NOT FOUND'}")

    # Check ML libraries
    for lib_name, import_name in [
        ("Demucs", "demucs"),
        ("DeepFilterNet", "df"),
        ("WhisperX", "whisperx"),
        ("pyloudnorm", "pyloudnorm"),
        ("Rich", "rich"),
    ]:
        try:
            __import__(import_name)
            click.echo(f"{lib_name}: installed")
        except ImportError:
            click.echo(f"{lib_name}: NOT INSTALLED")
