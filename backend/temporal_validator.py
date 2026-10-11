"""Temporal logic validator for held-out and simulated driving sequences.

Evaluates state machine transitions, alert escalation, debouncing, and degraded observation
handling across realistic driving event profiles.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional, Tuple

from backend.config import TemporalConfig
from backend.temporal_machine import TemporalDrowsinessDetector
from backend.temporal_state import AlertStage, DrowsinessState, TemporalStatus

logger = logging.getLogger("drishtidrive.validator")


@dataclass
class SequenceStep:
    """A single time-stamped step in a validation sequence."""

    timestamp: float
    raw_state: str  # "OPEN", "CLOSED", "UNCERTAIN"
    is_valid: bool = True
    description: str = ""


@dataclass
class ValidationReport:
    """Aggregated validation statistics across test sequences."""

    sequence_name: str
    total_steps: int
    duration_sec: float
    detected_blinks: int
    max_alert_stage: int
    transitions: List[Tuple[float, str, str]] = field(default_factory=list)
    alert_events_count: int = 0
    passed: bool = True
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize report to dictionary."""
        return {
            "sequence_name": self.sequence_name,
            "total_steps": self.total_steps,
            "duration_sec": round(self.duration_sec, 2),
            "detected_blinks": self.detected_blinks,
            "max_alert_stage": self.max_alert_stage,
            "alert_events_count": self.alert_events_count,
            "transitions_count": len(self.transitions),
            "passed": self.passed,
            "notes": self.notes,
        }


class TemporalSequenceValidator:
    """Validates the temporal drowsiness detector against scripted event sequences."""

    def __init__(self, config: Optional[TemporalConfig] = None) -> None:
        """Initialize validator with optional custom temporal configuration."""
        self.config = config or TemporalConfig()

    def run_sequence(
        self,
        name: str,
        steps: List[SequenceStep],
    ) -> Tuple[ValidationReport, List[TemporalStatus]]:
        """Run an ordered sequence of observations through a fresh detector instance."""
        detector = TemporalDrowsinessDetector(self.config)
        statuses: List[TemporalStatus] = []
        transitions: List[Tuple[float, str, str]] = []
        alert_events = 0
        max_stage = 0
        prev_state = DrowsinessState.NORMAL

        for step in steps:
            status = detector.step(
                raw_state=step.raw_state,
                is_observation_valid=step.is_valid,
                timestamp=step.timestamp,
            )
            statuses.append(status)

            if status.alert_event is not None:
                alert_events += 1

            if status.alert_stage.value > max_stage:
                max_stage = status.alert_stage.value

            if status.state != prev_state:
                transitions.append((round(step.timestamp, 3), prev_state.value, status.state.value))
                prev_state = status.state

        duration = steps[-1].timestamp - steps[0].timestamp if len(steps) > 1 else 0.0

        report = ValidationReport(
            sequence_name=name,
            total_steps=len(steps),
            duration_sec=duration,
            detected_blinks=statuses[-1].blink_count if statuses else 0,
            max_alert_stage=max_stage,
            transitions=transitions,
            alert_events_count=alert_events,
            passed=True,
        )
        return report, statuses


def generate_blinking_sequence(
    duration_sec: float = 10.0,
    fps: float = 30.0,
    blink_interval_sec: float = 2.5,
    blink_duration_sec: float = 0.2,
) -> List[SequenceStep]:
    """Generate a realistic attentive driver sequence with physiological blinks."""
    steps: List[SequenceStep] = []
    total_frames = int(duration_sec * fps)
    dt = 1.0 / fps

    for i in range(total_frames):
        t = i * dt
        # Determine if current time falls within a periodic blink window
        cycle_pos = t % blink_interval_sec
        is_blink = cycle_pos < blink_duration_sec and t > 0.5
        state = "CLOSED" if is_blink else "OPEN"
        steps.append(SequenceStep(timestamp=t, raw_state=state, is_valid=True))

    return steps


def generate_sustained_closure_sequence(
    pre_open_sec: float = 1.0,
    closure_sec: float = 3.5,
    post_open_sec: float = 2.0,
    fps: float = 30.0,
) -> List[SequenceStep]:
    """Generate a falling-asleep sequence with sustained closure followed by recovery."""
    steps: List[SequenceStep] = []
    dt = 1.0 / fps
    total_time = pre_open_sec + closure_sec + post_open_sec
    total_frames = int(total_time * fps)

    for i in range(total_frames):
        t = i * dt
        if t < pre_open_sec:
            state = "OPEN"
        elif t < (pre_open_sec + closure_sec):
            state = "CLOSED"
        else:
            state = "OPEN"
        steps.append(SequenceStep(timestamp=t, raw_state=state, is_valid=True))

    return steps
