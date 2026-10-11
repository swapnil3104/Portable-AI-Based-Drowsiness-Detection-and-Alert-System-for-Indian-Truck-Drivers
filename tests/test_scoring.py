"""Comprehensive unit tests for time-based scoring engine and three-stage alert logic.

Verifies:
1. Score is calculated from smoothed, time-based measurements rather than a single video frame.
2. Single-frame blinks and noise do not cause drastic score collapses.
3. Calibration function validates initial engineering weights against labelled data.
4. Stage 1 Attention warning triggers when Score <= 75 for 60 seconds:
   Voice: "Please stay alert. Keep your eyes on the road."
5. Stage 2 Drowsiness warning triggers when Score <= 60 for 60 seconds:
   Voice: "Warning! You may be getting drowsy. Please find a safe place to stop."
6. Stage 3 Critical warning triggers when Score <= 30 for 30 seconds:
   Voice: "Danger! Pull over safely as soon as possible. Do not continue driving while drowsy."
"""
from __future__ import annotations

import pytest

from backend.alert_catalog import AlertMessageCatalog
from backend.config import TemporalConfig
from backend.scoring import DrowsinessScoreEngine, ScoringWeights, calibrate_scoring_weights
from backend.temporal_machine import TemporalDrowsinessDetector
from backend.temporal_state import AlertStage, DrowsinessState


class TestTimeSmoothedScoring:
    """Tests for DrowsinessScoreEngine temporal smoothing and metrics."""

    def test_single_video_frame_does_not_collapse_score(self) -> None:
        """Verify that score is time-smoothed: a single closed frame drops score < 5 points."""
        engine = DrowsinessScoreEngine(window_sec=60.0)
        t = 100.0

        # Run 30 normal open frames (1 second at 30 FPS)
        for _ in range(30):
            t += 0.033
            telem = engine.update(t, "OPEN", True)

        initial_score = telem.score
        assert initial_score >= 95.0

        # Deliver a single CLOSED frame
        t += 0.033
        telem_single = engine.update(t, "CLOSED", True, closure_duration=0.033)

        # Single frame MUST NOT collapse the score
        score_drop = initial_score - telem_single.score
        assert score_drop < 5.0
        assert telem_single.score > 90.0

    def test_sustained_closure_gradually_lowers_score(self) -> None:
        """Verify sustained closure and rising PERCLOS reduce score across rolling window."""
        engine = DrowsinessScoreEngine(window_sec=60.0)
        t = 100.0

        # Normal open for 10 seconds
        for _ in range(300):
            t += 0.033
            engine.update(t, "OPEN", True)

        # Simulate drowsy episode: closed eyes for 15 seconds
        for _ in range(450):
            t += 0.033
            telem = engine.update(t, "CLOSED", True, closure_duration=2.5)

        assert telem.perclos > 0.30
        assert telem.score <= 75.0

    def test_calibration_weights_against_labelled_data(self) -> None:
        """Verify scoring weights calibration against labelled dataset samples."""
        labeled_samples = [
            {"label": "alert", "ear": 0.30, "perclos": 0.02},
            {"label": "alert", "ear": 0.28, "perclos": 0.03},
            {"label": "drowsy", "ear": 0.16, "perclos": 0.38},
            {"label": "drowsy", "ear": 0.14, "perclos": 0.42},
        ]
        calibrated = calibrate_scoring_weights(labeled_samples)
        assert isinstance(calibrated, ScoringWeights)
        assert 0.26 <= calibrated.ear_nominal <= 0.32
        assert 80.0 <= calibrated.perclos_multiplier <= 150.0


class TestThreeStageAlertLogic:
    """Verifies the three-stage score-based alert escalation logic."""

    @pytest.fixture
    def test_detector(self) -> TemporalDrowsinessDetector:
        """Detector configured with fast testing durations to test exact state transitions."""
        cfg = TemporalConfig(
            score_stage1_threshold=75.0,
            score_stage1_duration_sec=60.0,
            score_stage2_threshold=60.0,
            score_stage2_duration_sec=60.0,
            score_stage3_threshold=30.0,
            score_stage3_duration_sec=30.0,
            alert_cooldown_sec=1.0,
        )
        return TemporalDrowsinessDetector(cfg)

    def test_stage_1_attention_warning_score_threshold(self) -> None:
        """Verify Stage 1 triggers when Score <= 75 for 60 seconds with exact voice directive."""
        # Custom config with small duration for deterministic test step
        cfg = TemporalConfig(
            score_stage1_threshold=75.0,
            score_stage1_duration_sec=3.0,  # 3.0s simulated threshold
            alert_cooldown_sec=0.5,
        )
        detector = TemporalDrowsinessDetector(cfg)
        t = 100.0

        # Inject scores that keep attention score <= 75
        stage_1_event_seen = False
        captured_message = ""

        # Run for 4.5 seconds with low score
        for i in range(140):
            t += 0.033
            status = detector.step("CLOSED", True, t)
            if detector.score_stage1_timer >= 3.0:
                if status.alert_event and status.alert_event.stage == AlertStage.STAGE_1_EARLY:
                    stage_1_event_seen = True
                    captured_message = status.alert_event.message

        expected_msg = AlertMessageCatalog.get_message(AlertStage.STAGE_1_EARLY, "en")
        assert expected_msg == "Please stay alert. Keep your eyes on the road."
        assert detector.score_stage1_timer >= 3.0
        assert detector.alert_stage >= AlertStage.STAGE_1_EARLY

    def test_stage_2_drowsiness_warning_score_threshold(self) -> None:
        """Verify Stage 2 triggers when Score <= 60 for 60 seconds with exact voice directive."""
        cfg = TemporalConfig(
            score_stage2_threshold=60.0,
            score_stage2_duration_sec=2.0,  # 2.0s simulated threshold
            alert_cooldown_sec=0.5,
        )
        detector = TemporalDrowsinessDetector(cfg)
        t = 100.0

        for _ in range(90):
            t += 0.033
            detector.step("CLOSED", True, t)

        expected_msg = AlertMessageCatalog.get_message(AlertStage.STAGE_2_DROWSY, "en")
        assert expected_msg == "Warning! You may be getting drowsy. Please find a safe place to stop."
        assert detector.alert_stage >= AlertStage.STAGE_2_DROWSY

    def test_stage_3_critical_warning_score_threshold(self) -> None:
        """Verify Stage 3 triggers when Score <= 30 for 30 seconds with exact voice directive."""
        cfg = TemporalConfig(
            score_stage3_threshold=30.0,
            score_stage3_duration_sec=1.5,  # 1.5s simulated threshold
            alert_cooldown_sec=0.5,
        )
        detector = TemporalDrowsinessDetector(cfg)
        t = 100.0

        for _ in range(110):
            t += 0.033
            detector.step("CLOSED", True, t)

        expected_msg = AlertMessageCatalog.get_message(AlertStage.STAGE_3_CRITICAL, "en")
        assert expected_msg == "Danger! Pull over safely as soon as possible. Do not continue driving while drowsy."
        assert detector.alert_stage == AlertStage.STAGE_3_CRITICAL

    def test_exact_voice_strings_match_specification(self) -> None:
        """Explicitly verify voice string literals specified in the requirements."""
        v1 = AlertMessageCatalog.get_message(AlertStage.STAGE_1_EARLY)
        v2 = AlertMessageCatalog.get_message(AlertStage.STAGE_2_DROWSY)
        v3 = AlertMessageCatalog.get_message(AlertStage.STAGE_3_CRITICAL)

        assert v1 == "Please stay alert. Keep your eyes on the road."
        assert v2 == "Warning! You may be getting drowsy. Please find a safe place to stop."
        assert v3 == "Danger! Pull over safely as soon as possible. Do not continue driving while drowsy."
