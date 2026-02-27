"""Pipeline progress tracker using Rich."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from rich.console import Console

logger = logging.getLogger(__name__)


@dataclass
class PipelineTracker:
    """Track progress of the audio processing pipeline."""

    console: Console
    total_episodes: int
    _completed: int = field(default=0, init=False)
    _failed: int = field(default=0, init=False)
    _current_episode: str = field(default="", init=False)
    _current_stage: str = field(default="", init=False)
    _failures: list[tuple[str, str]] = field(default_factory=list, init=False)

    def start_episode(self, name: str, total_stages: int) -> None:
        """Begin tracking a new episode."""
        self._current_episode = name
        self._current_stage = ""
        episode_num = self._completed + self._failed + 1
        total = max(self.total_episodes, 1)
        self.console.print(f"[bold]Episode {episode_num}/{total}:[/bold] {name}")

    def start_stage(self, stage_name: str) -> None:
        """Begin tracking a stage within the current episode."""
        self._current_stage = stage_name
        self.console.print(f"  {stage_name}...", end="")

    def complete_stage(self) -> None:
        """Mark the current stage as complete."""
        self.console.print(" done")
        self._current_stage = ""

    def complete_episode(self) -> None:
        """Mark the current episode as successfully complete."""
        self._completed += 1
        self._current_episode = ""

    def fail_episode(self, error: str) -> None:
        """Mark the current episode as failed."""
        self._failed += 1
        self._failures.append((self._current_episode, error))
        self.console.print(f"  [red]FAILED:[/red] {error}")
        self._current_episode = ""
        self._current_stage = ""

    def get_summary(self) -> dict:
        """Return a summary of the pipeline run."""
        return {
            "succeeded": self._completed,
            "failed": self._failed,
            "failures": list(self._failures),
        }

    def print_summary(self) -> None:
        """Print a final summary of the pipeline run."""
        self.console.print(f"\n{'=' * 40}")
        self.console.print(
            f"Processed: {self._completed} succeeded, {self._failed} failed"
        )
        for name, err in self._failures:
            self.console.print(f"  [red]FAILED:[/red] {name} — {err}")
