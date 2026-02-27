"""Tests for the pipeline progress tracker."""

from podcast_cleaner.tracker import PipelineTracker
from podcast_cleaner.display import console


class TestPipelineTracker:
    def test_tracker_zero_episodes(self):
        """Tracker with total_episodes=0 should not divide by zero."""
        tracker = PipelineTracker(console, total_episodes=0)
        # Should not raise
        tracker.start_episode("test", 5)
        tracker.complete_episode()

    def test_tracker_fail_episode_recovery(self):
        """After fail_episode(), display should remain usable."""
        tracker = PipelineTracker(console, total_episodes=2)
        tracker.start_episode("ep1", 5)
        tracker.fail_episode("test error")
        # Next episode should work fine
        tracker.start_episode("ep2", 5)
        tracker.start_stage("preprocess")
        tracker.complete_stage()
        tracker.complete_episode()

    def test_tracker_stage_lifecycle(self):
        """Normal lifecycle: start episode -> start/complete stages -> complete episode."""
        tracker = PipelineTracker(console, total_episodes=1)
        tracker.start_episode("ep1", 3)
        for stage in ["preprocess", "separate", "denoise"]:
            tracker.start_stage(stage)
            tracker.complete_stage()
        tracker.complete_episode()

    def test_tracker_summary(self):
        """Summary should report successes and failures."""
        tracker = PipelineTracker(console, total_episodes=3)

        tracker.start_episode("ep1", 2)
        tracker.start_stage("preprocess")
        tracker.complete_stage()
        tracker.complete_episode()

        tracker.start_episode("ep2", 2)
        tracker.fail_episode("some error")

        tracker.start_episode("ep3", 2)
        tracker.start_stage("preprocess")
        tracker.complete_stage()
        tracker.complete_episode()

        summary = tracker.get_summary()
        assert summary["succeeded"] == 2
        assert summary["failed"] == 1
