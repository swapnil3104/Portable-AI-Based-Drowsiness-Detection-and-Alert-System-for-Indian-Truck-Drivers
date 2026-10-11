"""Unit tests for Phase 5 — Temporal Drowsiness Logic.

Tests state transitions, thresholds, debouncing, blinking vs sustained closure,
alert escalation, deduplication, unreliable observations, recovery, and sequence validation.
"""
from __future__ import annotations

import pytest

from backend.config import TemporalConfig
from backend.temporal_debouncer import EyeStateDebouncer
from backend.temporal_machine import TemporalDrowsinessDetector
from backend.temporal_state import (
    AlertStage,
    DrowsinessState,
    TemporalEventType,
)
from backend.temporal_validator import (
    SequenceStep,
    TemporalSequenceValidator,
    generate_blinking_sequence,
    generate_sustained_closure_sequence,
)


@pytest.fixture
def default_detector() -> TemporalDrowsinessDetector:
    """Instantiate detector with default configurable thresholds."""
    config = TemporalConfig(
        blink_max_sec=0.45,
        sustained_early_sec=0.8,
        sustained_drowsy_sec=1.8,
        sustained_critical_sec=3.0,
        debounce_window_sec=0.12,
        recovery_time_sec=1.5,
        repeated_window_sec=20.0,
        repeated_event_threshold=3,
        alert_cooldown_sec=2.0,
    )
    return TemporalDrowsinessDetector(config)


def test_normal_open_eyes_remains_in_normal_monitoring(default_detector: TemporalDrowsinessDetector) -> None:
    """Verify Task 8: Continuous open eyes stay in NORMAL monitoring state (Stage 0)."""
    t = 100.0
    for _ in range(30):
        t += 0.033
        status = default_detector.step("OPEN", True, t)

    assert status.state == DrowsinessState.NORMAL
    assert status.alert_stage == AlertStage.STAGE_0_NORMAL
    assert status.closure_duration_sec == 0.0
    assert status.continuous_open_sec > 0.9
    assert status.alert_event is None
    assert status.is_observation_reliable


def test_single_blink_handled_separately_from_sustained_closure(default_detector: TemporalDrowsinessDetector) -> None:
    """Verify Task 6: Normal physiological blink (200ms) increments blink count and triggers NO alert."""
    t = 100.0
    # 0.5s open
    for _ in range(15):
        t += 0.033
        default_detector.step("OPEN", True, t)

    # 0.2s closed (physiological blink)
    for _ in range(6):
        t += 0.033
        status = default_detector.step("CLOSED", True, t)
        assert status.alert_stage == AlertStage.STAGE_0_NORMAL

    # Re-opened
    for _ in range(10):
        t += 0.033
        status = default_detector.step("OPEN", True, t)

    assert status.blink_count == 1
    assert 0.15 <= status.last_blink_duration_sec <= 0.35
    assert status.state == DrowsinessState.NORMAL
    assert status.alert_stage == AlertStage.STAGE_0_NORMAL
    assert status.alert_event is None


def test_sustained_closure_escalates_through_all_alert_stages(default_detector: TemporalDrowsinessDetector) -> None:
    """Verify Task 1, 2, 8: Sustained eye closure escalates NORMAL -> EARLY -> DROWSY -> CRITICAL."""
    t = 100.0
    default_detector.step("OPEN", True, t)

    # Step through sustained closure for 3.5 seconds
    stages_seen = set()
    events_seen = []

    for _ in range(105):  # ~3.5 seconds at ~30 FPS
        t += 0.033
        status = default_detector.step("CLOSED", True, t)
        stages_seen.add(status.alert_stage)
        if status.alert_event is not None:
            events_seen.append(status.alert_event)

    assert AlertStage.STAGE_1_EARLY in stages_seen
    assert AlertStage.STAGE_2_DROWSY in stages_seen
    assert AlertStage.STAGE_3_CRITICAL in stages_seen
    assert status.state == DrowsinessState.CRITICAL
    assert status.alert_stage == AlertStage.STAGE_3_CRITICAL
    assert status.closure_duration_sec >= 3.0

    # Verify escalation events emitted
    event_stages = [e.stage for e in events_seen]
    assert AlertStage.STAGE_1_EARLY in event_stages
    assert AlertStage.STAGE_2_DROWSY in event_stages
    assert AlertStage.STAGE_3_CRITICAL in event_stages


def test_alert_deduplication_prevents_repeated_alerts_within_cooldown(
    default_detector: TemporalDrowsinessDetector,
) -> None:
    """Verify Task 7: Alerts are not spammed every frame while sustained in the same stage."""
    t = 100.0
    alert_events = []

    # Run into Stage 1 (0.8s threshold) and stay in Stage 1 for 0.5s (before Stage 2 at 1.8s)
    for _ in range(40):  # ~1.3s total
        t += 0.033
        status = default_detector.step("CLOSED", True, t)
        if status.alert_event is not None:
            alert_events.append(status.alert_event)

    # Exactly 1 event should have been emitted for Stage 1 escalation, NOT 15 duplicate events
    stage_1_events = [e for e in alert_events if e.stage == AlertStage.STAGE_1_EARLY]
    assert len(stage_1_events) == 1


def test_debouncer_ignores_transient_single_frame_flicker() -> None:
    """Verify Task 3: A single-frame flicker (< debounce window) does not flip smoothed state."""
    debouncer = EyeStateDebouncer(debounce_window_sec=0.12, blink_max_sec=0.45)
    t = 10.0

    # Establish initial OPEN state
    state, _ = debouncer.update("OPEN", t)
    assert state == "OPEN"

    # Single transient CLOSED frame (33ms)
    t += 0.033
    state, _ = debouncer.update("CLOSED", t)
    # Debounce filter holds previous smoothed state
    assert state == "OPEN"

    # Immediately returns to OPEN
    t += 0.033
    state, _ = debouncer.update("OPEN", t)
    assert state == "OPEN"


def test_recovery_requires_continuous_sustained_open_eyes(default_detector: TemporalDrowsinessDetector) -> None:
    """Verify Task 8: Transitioning out of an alert requires sustained open eyes (1.5s)."""
    t = 100.0
    # Push into Early Warning (1.0s closed)
    for _ in range(30):
        t += 0.033
        default_detector.step("CLOSED", True, t)
    assert default_detector.alert_stage == AlertStage.STAGE_1_EARLY

    # Eyes reopen for 0.5s (< recovery_time_sec=1.5s)
    for _ in range(15):
        t += 0.033
        status = default_detector.step("OPEN", True, t)

    assert status.state == DrowsinessState.RECOVERY
    assert status.alert_stage == AlertStage.STAGE_1_EARLY  # alert stage held during recovery

    # Eyes continue open past 1.5s
    for _ in range(35):
        t += 0.033
        status = default_detector.step("OPEN", True, t)

    assert status.state == DrowsinessState.NORMAL
    assert status.alert_stage == AlertStage.STAGE_0_NORMAL
    assert status.alert_event is not None
    assert status.alert_event.event_type == TemporalEventType.RECOVERY_COMPLETE


def test_unreliable_observation_never_escalates_and_marks_uncertain(
    default_detector: TemporalDrowsinessDetector,
) -> None:
    """Verify Task 5 & Safety: Missing face or degraded camera NEVER causes runaway alert escalation."""
    t = 100.0
    # Run normal
    default_detector.step("OPEN", True, t)

    # Face lost / camera failure for 3.0 seconds
    for _ in range(90):
        t += 0.033
        status = default_detector.step("UNCERTAIN", False, t)

    assert status.state == DrowsinessState.UNCERTAIN
    assert not status.is_observation_reliable
    # Critical Safety: Alert stage MUST NOT escalate to Critical on camera failure!
    assert status.alert_stage == AlertStage.STAGE_0_NORMAL
    assert status.closure_duration_sec == 0.0


def test_repeated_micro_closures_escalate_to_drowsiness_warning(
    default_detector: TemporalDrowsinessDetector,
) -> None:
    """Verify Task 2: Repeated closure episodes within rolling window escalate to DROWSY (Stage 2)."""
    t = 100.0

    # Helper to simulate a 0.9s closure (triggers Early Warning episode) followed by 2.0s open
    def simulate_closure_episode(start_t: float) -> float:
        curr = start_t
        for _ in range(30):  # 0.99s closed
            curr += 0.033
            default_detector.step("CLOSED", True, curr)
        for _ in range(65):  # 2.14s open (recovers)
            curr += 0.033
            default_detector.step("OPEN", True, curr)
        return curr

    # Episode 1 and 2
    t = simulate_closure_episode(t)
    t = simulate_closure_episode(t)

    # Episode 3 starts (3rd event in < 20s window)
    for _ in range(30):
        t += 0.033
        status = default_detector.step("CLOSED", True, t)

    assert status.recent_events_count >= 3
    assert status.alert_stage == AlertStage.STAGE_2_DROWSY
    assert status.state == DrowsinessState.DROWSY


def test_sequence_validator_on_attentive_blinking_sequence() -> None:
    """Verify Task 10: Attentive driving sequence with periodic blinks has 0 false alarms."""
    validator = TemporalSequenceValidator()
    steps = generate_blinking_sequence(duration_sec=10.0, fps=30.0, blink_interval_sec=2.5, blink_duration_sec=0.2)
    report, _ = validator.run_sequence("attentive_driver", steps)

    assert report.passed
    assert report.max_alert_stage == 0  # 0 false alarms!
    assert report.detected_blinks >= 3  # correctly counted blinks


def test_sequence_validator_on_sustained_closure_sequence() -> None:
    """Verify Task 10: Validation on held-out micro-sleep sequence transitions properly."""
    validator = TemporalSequenceValidator()
    steps = generate_sustained_closure_sequence(pre_open_sec=1.0, closure_sec=3.2, post_open_sec=2.0)
    report, _ = validator.run_sequence("microsleep_sequence", steps)

    assert report.passed
    assert report.max_alert_stage == 3  # Reached critical alert
    assert report.alert_events_count >= 3
