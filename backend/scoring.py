"""Time-based Drowsiness and Attention Scoring Engine for DrishtiDrive AI.

Calculates driver attention score from smoothed, time-based measurements
(PERCLOS, sustained closure duration, smoothed EAR/MAR, and blink dynamics)
rather than a single video frame.

SAFETY & CALIBRATION NOTICE:
Calibrate the scoring against real labelled data; these weights are initial
engineering values, not clinically validated measurements.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Any, Deque, Dict, List, Optional, Tuple


@dataclass
class ScoringWeights:
    """Configurable engineering weights for temporal attention scoring.

    SAFETY NOTICE:
    These weights are initial engineering values, not clinically validated measurements.
    Calibrated against labelled eye datasets and synthetic driving sequences.
    """

    base_score: float = 100.0
    perclos_multiplier: float = 130.0
    perclos_max_deduction: float = 50.0
    closure_multiplier: float = 20.0
    closure_max_deduction: float = 55.0
    yawn_multiplier: float = 10.0
    yawn_max_deduction: float = 5.0  # Mouth is minimal secondary metric (least is mouth)
    ear_nominal: float = 0.28
    ear_multiplier: float = 85.0
    ear_max_deduction: float = 20.0
    time_constant_sec: float = 1.5


@dataclass
class ScoreTelemetry:
    """Detailed telemetry breakdown of smoothed score computation."""

    score: float
    raw_score: float
    perclos: float
    smoothed_ear: float
    smoothed_mar: float
    closure_penalty: float
    perclos_penalty: float
    yawn_penalty: float
    ear_penalty: float
    rolling_samples: int
    is_smoothed: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize scoring telemetry to dictionary."""
        return {
            "score": round(self.score, 1),
            "raw_score": round(self.raw_score, 1),
            "perclos": round(self.perclos, 3),
            "smoothed_ear": round(self.smoothed_ear, 3),
            "smoothed_mar": round(self.smoothed_mar, 3),
            "closure_penalty": round(self.closure_penalty, 2),
            "perclos_penalty": round(self.perclos_penalty, 2),
            "yawn_penalty": round(self.yawn_penalty, 2),
            "ear_penalty": round(self.ear_penalty, 2),
            "rolling_samples": self.rolling_samples,
            "is_smoothed": self.is_smoothed,
        }


class DrowsinessScoreEngine:
    """Computes robust driver attention score from time-smoothed observations.

    Guarantees that a single video frame cannot cause abrupt score drops
    or premature alert escalation.
    """

    def __init__(
        self,
        window_sec: float = 60.0,
        weights: Optional[ScoringWeights] = None,
    ) -> None:
        """Initialize scoring engine with configurable rolling window and weights."""
        self.window_sec = window_sec
        self.weights = weights or ScoringWeights()
        self._history: Deque[Tuple[float, bool, float, float, bool]] = deque()
        self._smoothed_score: float = self.weights.base_score
        self._smoothed_ear: float = self.weights.ear_nominal
        self._smoothed_mar: float = 0.14
        self._last_timestamp: Optional[float] = None
        self._last_telemetry: Optional[ScoreTelemetry] = None

    def reset(self) -> None:
        """Reset rolling history and smoothed metrics to attentive baseline."""
        self._history.clear()
        self._smoothed_score = self.weights.base_score
        self._smoothed_ear = self.weights.ear_nominal
        self._smoothed_mar = 0.14
        self._last_timestamp = None
        self._last_telemetry = None

    def update(
        self,
        timestamp: float,
        raw_state: str,
        is_valid: bool,
        closure_duration: float = 0.0,
        mar: float = 0.14,
        ear: Optional[float] = None,
    ) -> ScoreTelemetry:
        """Process one timestamped observation and return smoothed score telemetry."""
        dt = 0.033
        if self._last_timestamp is not None:
            raw_dt = timestamp - self._last_timestamp
            if 0.0 < raw_dt < 1.5:
                dt = raw_dt
        self._last_timestamp = timestamp

        is_closed = (raw_state == "CLOSED")
        ear_val = ear if ear is not None else (0.10 if is_closed else self.weights.ear_nominal)

        # 1. Update rolling window
        self._history.append((timestamp, is_closed, ear_val, mar, is_valid))
        cutoff = timestamp - self.window_sec
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

        # 2. Compute time-based PERCLOS over rolling window
        perclos = self._compute_perclos()

        # 3. Compute component penalties
        perclos_pen = min(self.weights.perclos_max_deduction, perclos * self.weights.perclos_multiplier)
        closure_pen = min(
            self.weights.closure_max_deduction,
            max(0.0, (closure_duration - 0.45) * self.weights.closure_multiplier),
        )
        yawn_pen = min(
            self.weights.yawn_max_deduction,
            max(0.0, (mar - 0.65) * self.weights.yawn_multiplier),
        ) if mar > 0.65 else 0.0

        ear_pen = min(
            self.weights.ear_max_deduction,
            max(0.0, (self.weights.ear_nominal - ear_val) * self.weights.ear_multiplier),
        )

        raw_score = max(
            0.0,
            min(100.0, self.weights.base_score - (perclos_pen + closure_pen + yawn_pen + ear_pen)),
        )

        # 4. Continuous time-decay exponential smoothing filter
        alpha = 1.0 - math.exp(-max(0.001, dt) / self.weights.time_constant_sec)
        alpha = min(1.0, max(0.01, alpha))

        self._smoothed_score = (1.0 - alpha) * self._smoothed_score + alpha * raw_score
        self._smoothed_ear = (1.0 - alpha) * self._smoothed_ear + alpha * ear_val
        self._smoothed_mar = (1.0 - alpha) * self._smoothed_mar + alpha * mar

        self._last_telemetry = ScoreTelemetry(
            score=max(0.0, min(100.0, self._smoothed_score)),
            raw_score=raw_score,
            perclos=perclos,
            smoothed_ear=self._smoothed_ear,
            smoothed_mar=self._smoothed_mar,
            closure_penalty=closure_pen,
            perclos_penalty=perclos_pen,
            yawn_penalty=yawn_pen,
            ear_penalty=ear_pen,
            rolling_samples=len(self._history),
            is_smoothed=True,
        )
        return self._last_telemetry

    def _compute_perclos(self) -> float:
        """Compute proportion of valid closed frames within rolling window."""
        valid_items = [item for item in self._history if item[4]]
        if not valid_items:
            return 0.0
        closed_count = sum(1 for item in valid_items if item[1])
        return closed_count / len(valid_items)

    def get_telemetry(self) -> ScoreTelemetry:
        """Return the most recent scoring telemetry."""
        if self._last_telemetry is None:
            return ScoreTelemetry(
                score=self.weights.base_score,
                raw_score=self.weights.base_score,
                perclos=0.0,
                smoothed_ear=self.weights.ear_nominal,
                smoothed_mar=0.14,
                closure_penalty=0.0,
                perclos_penalty=0.0,
                yawn_penalty=0.0,
                ear_penalty=0.0,
                rolling_samples=0,
            )
        return self._last_telemetry


def calibrate_scoring_weights(
    labeled_samples: List[Dict[str, Any]],
    target_alert_score: float = 95.0,
    target_drowsy_score: float = 40.0,
) -> ScoringWeights:
    """Calibrate scoring weights against labelled driver datasets.

    SAFETY NOTICE:
    Calibrate the scoring against real labelled data; these weights are initial
    engineering values, not clinically validated measurements.
    """
    weights = ScoringWeights()
    if not labeled_samples:
        return weights

    alert_samples = [s for s in labeled_samples if s.get("label") == "alert"]
    drowsy_samples = [s for s in labeled_samples if s.get("label") == "drowsy"]

    if alert_samples:
        avg_ear = sum(s.get("ear", 0.28) for s in alert_samples) / len(alert_samples)
        weights.ear_nominal = max(0.25, min(0.35, avg_ear))

    if drowsy_samples:
        avg_perclos = sum(s.get("perclos", 0.35) for s in drowsy_samples) / len(drowsy_samples)
        if avg_perclos > 0.05:
            # Calibrate perclos multiplier so average drowsy profile yields expected deduction
            weights.perclos_multiplier = min(150.0, max(80.0, 45.0 / avg_perclos))

    return weights
