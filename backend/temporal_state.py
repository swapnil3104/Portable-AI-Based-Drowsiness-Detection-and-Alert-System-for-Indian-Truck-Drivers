"""State definitions and data structures for temporal drowsiness logic.

Explicitly categorizes driving awareness states, alert stages, and event telemetry.
Eye closure is treated as an indicator of possible drowsiness, not definitive proof.
Thresholds are experimental engineering heuristics and are not medically validated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class DrowsinessState(str, Enum):
    """Explicit states of the driver monitoring state machine."""

    NORMAL = "NORMAL"
    EARLY_WARNING = "EARLY_WARNING"
    DROWSY = "DROWSY"
    CRITICAL = "CRITICAL"
    RECOVERY = "RECOVERY"
    UNCERTAIN = "UNCERTAIN"


class AlertStage(int, Enum):
    """Discrete alert escalation stages for audio-visual feedback."""

    STAGE_0_NORMAL = 0
    STAGE_1_EARLY = 1
    STAGE_2_DROWSY = 2
    STAGE_3_CRITICAL = 3


class TemporalEventType(str, Enum):
    """Types of temporal events emitted by the state machine."""

    NONE = "NONE"
    BLINK = "BLINK"
    SUSTAINED_CLOSURE = "SUSTAINED_CLOSURE"
    REPEATED_CLOSURES = "REPEATED_CLOSURES"
    RECOVERY_COMPLETE = "RECOVERY_COMPLETE"
    OBSERVATION_LOST = "OBSERVATION_LOST"
    OBSERVATION_RESTORED = "OBSERVATION_RESTORED"


@dataclass
class TemporalAlertEvent:
    """An alert escalation event emitted by the state machine.

    Designed to be ingested by the alert manager without duplicate triggers.
    """

    stage: AlertStage
    event_type: TemporalEventType
    message: str
    timestamp: float
    duration_sec: float
    repeat_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize alert event to dictionary."""
        return {
            "stage": self.stage.value,
            "event_type": self.event_type.value,
            "message": self.message,
            "timestamp": round(self.timestamp, 4),
            "duration_sec": round(self.duration_sec, 3),
            "repeat_count": self.repeat_count,
        }


@dataclass
class TemporalStatus:
    """Comprehensive state machine telemetry for the current processing step."""

    state: DrowsinessState
    alert_stage: AlertStage
    closure_duration_sec: float
    continuous_open_sec: float
    blink_count: int
    last_blink_duration_sec: float
    recent_events_count: int
    is_observation_reliable: bool
    alert_event: Optional[TemporalAlertEvent] = None
    status_message: str = ""
    attention_score: float = 100.0
    perclos: float = 0.0
    score_stage1_timer: float = 0.0
    score_stage2_timer: float = 0.0
    score_stage3_timer: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize status to dictionary."""
        return {
            "state": self.state.value,
            "alert_stage": self.alert_stage.value,
            "closure_duration_sec": round(self.closure_duration_sec, 3),
            "continuous_open_sec": round(self.continuous_open_sec, 3),
            "blink_count": self.blink_count,
            "last_blink_duration_sec": round(self.last_blink_duration_sec, 3),
            "recent_events_count": self.recent_events_count,
            "is_observation_reliable": self.is_observation_reliable,
            "alert_event": self.alert_event.to_dict() if self.alert_event else None,
            "status_message": self.status_message,
            "attention_score": round(self.attention_score, 1),
            "perclos": round(self.perclos, 3),
            "score_stage1_timer": round(self.score_stage1_timer, 2),
            "score_stage2_timer": round(self.score_stage2_timer, 2),
            "score_stage3_timer": round(self.score_stage3_timer, 2),
        }
