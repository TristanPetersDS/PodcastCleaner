"""Click CLI entry point for podcast-cleaner."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from podcast_cleaner.config import load_config
from podcast_cleaner.stages import STAGE_ORDER


def run_pipeline(
    config: dict,
    episode_dirs: list[str],
    skip_stages: tuple[str, ...] = (),
    resume: bool = False,
) -> list[str]:
    """Run the full pipeline on a list of episode directories.

    Returns list of successfully processed episode dirs.
    """
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

    successes: list[str] = []
    failures: list[tuple[str, str]] = []

    for episode_dir in episode_dirs:
        ep_name = Path(episode_dir).name
        stage_logger = setup_logging(Path(episode_dir) / "processing.log", ep_name)
        stage_logger.info(f"=== Processing: {ep_name} ===")

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
                runner = stage_runners.get(stage_name)
                if runner:
                    runner(episode_dir, config, stage_logger=stage_logger)

            stage_logger.info(f"=== Complete: {ep_name} ===")
            successes.append(episode_dir)

        except Exception as e:
            stage_logger.error(f"Pipeline failed for {ep_name}: {e}", exc_info=True)
            failures.append((ep_name, str(e)))

    # Summary
    click.echo(f"\n{'=' * 40}")
    click.echo(f"Processed: {len(successes)} succeeded, {len(failures)} failed")
    for name, err in failures:
        click.echo(f"  FAILED: {name} — {err}")

    return successes


@click.group()
def main():
    """PodcastCleaner -- download and clean podcast audio."""
    pass


@main.command()
@click.option("--url", default=None, help="YouTube playlist or video URL")
@click.option("--input-dir", default=None, type=click.Path(exists=True), help="Directory of local audio files")
@click.option("--input", "input_file", default=None, type=click.Path(exists=True), help="Single local audio file")
@click.option("--config", "config_path", default="config.yaml", help="Path to config.yaml")
@click.option("--skip", multiple=True, help="Skip a stage (can repeat)")
@click.option("--resume", is_flag=True, help="Resume from last completed stage")
def run(url, input_dir, input_file, config_path, skip, resume):
    """Run the full audio cleaning pipeline."""
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
        import shutil

        from podcast_cleaner.stages.download import build_episode_dirname
        from podcast_cleaner.utils import ensure_dir

        input_path = Path(input_dir)
        audio_exts = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".aac"}
        for i, f in enumerate(sorted(input_path.iterdir())):
            if f.suffix.lower() in audio_exts:
                dirname = build_episode_dirname(f.stem, i)
                ep_dir = ensure_dir(Path(output_base) / dirname)
                raw_dir = ensure_dir(ep_dir / "raw")
                shutil.copy2(str(f), str(raw_dir / f.name))
                episode_dirs.append(str(ep_dir))

    elif input_file:
        import shutil

        from podcast_cleaner.stages.download import build_episode_dirname
        from podcast_cleaner.utils import ensure_dir

        f = Path(input_file)
        dirname = build_episode_dirname(f.stem, None)
        ep_dir = ensure_dir(Path(output_base) / dirname)
        raw_dir = ensure_dir(ep_dir / "raw")
        shutil.copy2(str(f), str(raw_dir / f.name))
        episode_dirs.append(str(ep_dir))

    else:
        raise click.UsageError("Provide --url, --input-dir, or --input")

    if not episode_dirs:
        click.echo("No episodes to process.")
        return

    click.echo(f"Processing {len(episode_dirs)} episode(s)...")
    successes = run_pipeline(config, episode_dirs, skip_stages=skip, resume=resume)

    if len(successes) < len(episode_dirs):
        sys.exit(1)


@main.command()
@click.argument("episode_dir", type=click.Path(exists=True))
@click.option("--config", "config_path", default="config.yaml", help="Path to config.yaml")
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
        click.echo(
            f"{stage_name:15s}  LUFS={stats['lufs']:6.1f}  "
            f"SNR={stats['snr_db']:5.1f}dB  "
            f"Peak={stats['true_peak']:5.1f}dBTP  "
            f"Duration={stats['duration']:.1f}s"
        )

    click.echo(f"\nFull report: {report_path}")


@main.command()
@click.argument("stage_name")
@click.argument("episode_dir", type=click.Path(exists=True))
@click.option("--config", "config_path", default="config.yaml", help="Path to config.yaml")
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
        raise click.UsageError(f"Unknown stage: {stage_name}. Available: {', '.join(runners)}")

    # Clear done marker so it re-runs
    clear_done(episode_dir, stage_name)
    runners[stage_name](episode_dir, config)
    click.echo(f"Stage '{stage_name}' complete for {episode_dir}")
