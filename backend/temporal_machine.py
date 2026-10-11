"""Configurable, time-based drowsiness state machine for DrishtiDrive AI.

Enforces deterministic transitions across monitoring, warnings, recovery, and degradation.
Uses real timestamps rather than assuming a fixed camera FPS.
Calculates attention score from smoothed, time-based measurements.
Executes three-stage alert escalation based on sustained score thresholds.

SAFETY NOTICE:
Eye closure and attention score are indicators of possible drowsiness, not definitive proof.
Thresholds are initial engineering values, not clinically validated measurements.
"""
from __future__ import annotations

import collections
import logging
from typing import Deque, Optional, Tuple

from backend.config import TemporalConfig
from backend.scoring import DrowsinessScoreEngine
from backend.temporal_debouncer import EyeStateDebouncer
from backend.temporal_state import (
    AlertStage,
    DrowsinessState,
    TemporalAlertEvent,
    TemporalEventType,
    TemporalStatus,
)

logger = logging.getLogger("drishtidrive.temporal")


class TemporalDrowsinessDetector:
    """Deterministic time-based state machine for eye closure and drowsiness risk."""

    def __init__(self, config: Optional[TemporalConfig] = None) -> None:
        """Initialize state machine with configurable thresholds, debouncer, and score engine."""
        self.config = config or TemporalConfig()
        self.debouncer = EyeStateDebouncer(
            debounce_window_sec=self.config.debounce_window_sec,
            blink_max_sec=self.config.blink_max_sec,
        )
        self.score_engine = DrowsinessScoreEngine(
            window_sec=self.config.score_window_sec,
        )

        self.state: DrowsinessState = DrowsinessState.NORMAL
        self.alert_stage: AlertStage = AlertStage.STAGE_0_NORMAL

        self.closure_duration_sec: float = 0.0
        self.continuous_open_sec: float = 0.0
        self.uncertain_duration_sec: float = 0.0

        self.blink_count: int = 0
        self.last_blink_duration: float = 0.0
        self.yawn_duration_sec: float = 0.0

        self.score_stage1_timer: float = 0.0
        self.score_stage2_timer: float = 0.0
        self.score_stage3_timer: float = 0.0
        self.attention_score: float = 100.0
        self.perclos: float = 0.0

        self._recent_closure_timestamps: Deque[float] = collections.deque()
        self._last_timestamp: Optional[float] = None
        self._last_alert_time: float = -999.0
        self._active_closure_episode_recorded: bool = False

    def reset(self) -> None:
        """Reset all timers, score engine, event buffers, and return to NORMAL monitoring."""
        self.debouncer.reset()
        self.score_engine.reset()
        self.state = DrowsinessState.NORMAL
        self.alert_stage = AlertStage.STAGE_0_NORMAL
        self.closure_duration_sec = 0.0
        self.continuous_open_sec = 0.0
        self.uncertain_duration_sec = 0.0
        self.yawn_duration_sec = 0.0
        self.score_stage1_timer = 0.0
        self.score_stage2_timer = 0.0
        self.score_stage3_timer = 0.0
        self.attention_score = 100.0
        self.perclos = 0.0
        self.blink_count = 0
        self.last_blink_duration = 0.0
        self._recent_closure_timestamps.clear()
        self._last_timestamp = None
        self._last_alert_time = -999.0
        self._active_closure_episode_recorded = False

    def step(
        self,
        raw_state: str,
        is_observation_valid: bool,
        timestamp: float,
        mar: float = 0.14,
        ear: Optional[float] = None,
    ) -> TemporalStatus:
        """Process one frame observation through the temporal state machine."""
        dt = self._calculate_dt(timestamp)
        self._last_timestamp = timestamp
        self._prune_recent_events(timestamp)

        # Update scoring engine with smoothed time-based measurements
        score_telem = self.score_engine.update(
            timestamp=timestamp,
            raw_state=raw_state,
            is_valid=is_observation_valid,
            closure_duration=self.closure_duration_sec,
            mar=mar,
            ear=ear,
        )
        self.attention_score = score_telem.score
        self.perclos = score_telem.perclos

        # 1. Handle Invalid/Unreliable Observations Safely
        if not is_observation_valid or raw_state == "UNCERTAIN":
            return self._handle_unreliable_observation(dt, timestamp)

        self.uncertain_duration_sec = 0.0

        # Update score threshold duration timers
        self._update_score_timers(dt)

        # 2. Smooth Prediction and Handle Physiological Blinks
        smoothed_state, blink_duration = self.debouncer.update(raw_state, timestamp)
        if blink_duration is not None:
            self.blink_count += 1
            self.last_blink_duration = blink_duration
            self.closure_duration_sec = 0.0
            self._active_closure_episode_recorded = False

        # 3. Track Yawn Dynamics (Secondary telemetry only; eyes are primary focus)
        if mar >= 0.65:
            self.yawn_duration_sec += dt
        else:
            self.yawn_duration_sec = max(0.0, self.yawn_duration_sec - dt * 2.0)

        alert_event: Optional[TemporalAlertEvent] = None
        status_msg = ""

        if smoothed_state == "CLOSED":
            alert_event, status_msg = self._process_closed_state(dt, timestamp)
        else:
            alert_event, status_msg = self._process_open_state(dt, timestamp)

        # 4. Check Three-Stage Score Alert Logic (Primary: Eye Closure, PERCLOS, EAR)
        score_event, score_msg = self._check_score_alert_logic(timestamp)
        if score_event is not None:
            alert_event = score_event
            status_msg = score_msg

        return TemporalStatus(
            state=self.state,
            alert_stage=self.alert_stage,
            closure_duration_sec=self.closure_duration_sec,
            continuous_open_sec=self.continuous_open_sec,
            blink_count=self.blink_count,
            last_blink_duration_sec=self.last_blink_duration,
            recent_events_count=len(self._recent_closure_timestamps),
            is_observation_reliable=True,
            alert_event=alert_event,
            status_message=status_msg,
            attention_score=self.attention_score,
            perclos=self.perclos,
            score_stage1_timer=self.score_stage1_timer,
            score_stage2_timer=self.score_stage2_timer,
            score_stage3_timer=self.score_stage3_timer,
        )

    def _update_score_timers(self, dt: float) -> None:
        """Advance duration timers for score-based alert stages."""
        if self.attention_score <= self.config.score_stage3_threshold:
            self.score_stage3_timer += dt
        else:
            self.score_stage3_timer = max(0.0, self.score_stage3_timer - dt * 0.5)

        if self.attention_score <= self.config.score_stage2_threshold:
            self.score_stage2_timer += dt
        else:
            self.score_stage2_timer = max(0.0, self.score_stage2_timer - dt * 0.5)

        if self.attention_score <= self.config.score_stage1_threshold:
            self.score_stage1_timer += dt
        else:
            self.score_stage1_timer = max(0.0, self.score_stage1_timer - dt * 0.5)

    def _check_score_alert_logic(
        self,
        timestamp: float,
    ) -> Tuple[Optional[TemporalAlertEvent], str]:
        """Evaluate Three-Stage Score alert logic: Stage 1 <=75 (60s), Stage 2 <=60 (60s), Stage 3 <=30 (30s)."""
        score_target = AlertStage.STAGE_0_NORMAL
        score_state = DrowsinessState.NORMAL
        score_msg = ""

        if self.score_stage3_timer >= self.config.score_stage3_duration_sec:
            score_target = AlertStage.STAGE_3_CRITICAL
            score_state = DrowsinessState.CRITICAL
            score_msg = "Danger! Pull over safely as soon as possible. Do not continue driving while drowsy."
        elif self.score_stage2_timer >= self.config.score_stage2_duration_sec:
            score_target = AlertStage.STAGE_2_DROWSY
            score_state = DrowsinessState.DROWSY
            score_msg = "Warning! You may be getting drowsy. Please find a safe place to stop."
        elif self.score_stage1_timer >= self.config.score_stage1_duration_sec:
            score_target = AlertStage.STAGE_1_EARLY
            score_state = DrowsinessState.EARLY_WARNING
            score_msg = "Please stay alert. Keep your eyes on the road."

        if score_target > self.alert_stage:
            self.alert_stage = score_target
            self.state = score_state
            is_cooldown_expired = (timestamp - self._last_alert_time) >= self.config.alert_cooldown_sec
            if is_cooldown_expired:
                self._last_alert_time = timestamp
                event = TemporalAlertEvent(
                    stage=score_target,
                    event_type=TemporalEventType.SUSTAINED_CLOSURE,
                    message=score_msg,
                    timestamp=timestamp,
                    duration_sec=max(self.closure_duration_sec, 30.0 if score_target == AlertStage.STAGE_3_CRITICAL else 60.0),
                    repeat_count=len(self._recent_closure_timestamps),
                )
                return event, score_msg
        return None, ""

    def _calculate_dt(self, timestamp: float) -> float:
        """Compute delta time, protecting against backward jumps and large camera gaps."""
        if self._last_timestamp is None:
            return 0.0
        raw_dt = timestamp - self._last_timestamp
        if raw_dt <= 0.0:
            return 0.0
        if raw_dt > 1.5:
            logger.warning("Large timestamp jump detected (%.2fs); capping elapsed time.", raw_dt)
            return 0.1
        return raw_dt

    def _prune_recent_events(self, current_time: float) -> None:
        """Discard closure event timestamps older than the configured rolling window."""
        cutoff = current_time - self.config.repeated_window_sec
        while self._recent_closure_timestamps and self._recent_closure_timestamps[0] < cutoff:
            self._recent_closure_timestamps.popleft()

    def _handle_unreliable_observation(self, dt: float, timestamp: float) -> TemporalStatus:
        """Safely degrade state when face or eyes cannot be reliably tracked."""
        self.uncertain_duration_sec += dt
        self.state = DrowsinessState.UNCERTAIN

        if self.uncertain_duration_sec >= self.config.uncertain_reset_sec:
            self.closure_duration_sec = 0.0
            self.alert_stage = AlertStage.STAGE_0_NORMAL
            self._active_closure_episode_recorded = False
            msg = "Observation lost > 2.0s: State reset to UNCERTAIN (safety hold, no escalation)."
        else:
            msg = f"Observation degraded ({self.uncertain_duration_sec:.2f}s); timers paused."

        return TemporalStatus(
            state=DrowsinessState.UNCERTAIN,
            alert_stage=self.alert_stage,
            closure_duration_sec=self.closure_duration_sec,
            continuous_open_sec=0.0,
            blink_count=self.blink_count,
            last_blink_duration_sec=self.last_blink_duration,
            recent_events_count=len(self._recent_closure_timestamps),
            is_observation_reliable=False,
            alert_event=None,
            status_message=msg,
            attention_score=self.attention_score,
            perclos=self.perclos,
            score_stage1_timer=self.score_stage1_timer,
            score_stage2_timer=self.score_stage2_timer,
            score_stage3_timer=self.score_stage3_timer,
        )

    def _process_closed_state(
        self,
        dt: float,
        timestamp: float,
    ) -> Tuple[Optional[TemporalAlertEvent], str]:
        """Update closure timers, track repeated closures, and escalate alert stages."""
        self.continuous_open_sec = 0.0
        self.closure_duration_sec += dt

        target_stage = AlertStage.STAGE_0_NORMAL
        target_state = DrowsinessState.NORMAL
        event_type = TemporalEventType.NONE
        message = ""

        if self.closure_duration_sec >= self.config.sustained_critical_sec:
            target_stage = AlertStage.STAGE_3_CRITICAL
            target_state = DrowsinessState.CRITICAL
            event_type = TemporalEventType.SUSTAINED_CLOSURE
            message = "Danger! Pull over safely as soon as possible. Do not continue driving while drowsy."
        elif self.closure_duration_sec >= self.config.sustained_drowsy_sec:
            target_stage = AlertStage.STAGE_2_DROWSY
            target_state = DrowsinessState.DROWSY
            event_type = TemporalEventType.SUSTAINED_CLOSURE
            message = "Warning! You may be getting drowsy. Please find a safe place to stop."
        elif self.closure_duration_sec >= self.config.sustained_early_sec:
            target_stage = AlertStage.STAGE_1_EARLY
            target_state = DrowsinessState.EARLY_WARNING
            event_type = TemporalEventType.SUSTAINED_CLOSURE
            message = "Please stay alert. Keep your eyes on the road."

            if not self._active_closure_episode_recorded:
                self._recent_closure_timestamps.append(timestamp)
                self._active_closure_episode_recorded = True

        if (
            len(self._recent_closure_timestamps) >= self.config.repeated_event_threshold
            and target_stage < AlertStage.STAGE_2_DROWSY
        ):
            target_stage = AlertStage.STAGE_2_DROWSY
            target_state = DrowsinessState.DROWSY
            event_type = TemporalEventType.REPEATED_CLOSURES
            message = "Warning! You may be getting drowsy. Please find a safe place to stop."

        alert_event: Optional[TemporalAlertEvent] = None
        if target_stage > AlertStage.STAGE_0_NORMAL:
            is_escalation = target_stage > self.alert_stage
            is_cooldown_expired = (timestamp - self._last_alert_time) >= self.config.alert_cooldown_sec

            if is_escalation or (is_cooldown_expired and target_stage == self.alert_stage):
                alert_event = TemporalAlertEvent(
                    stage=target_stage,
                    event_type=event_type,
                    message=message,
                    timestamp=timestamp,
                    duration_sec=self.closure_duration_sec,
                    repeat_count=len(self._recent_closure_timestamps),
                )
                self._last_alert_time = timestamp

            self.state = target_state
            self.alert_stage = target_stage
        else:
            self.state = DrowsinessState.NORMAL

        return alert_event, message or f"Eyes closed ({self.closure_duration_sec:.2f}s)"

    def _process_open_state(
        self,
        dt: float,
        timestamp: float,
    ) -> Tuple[Optional[TemporalAlertEvent], str]:
        """Update open timers and manage confirmed driver recovery."""
        self.closure_duration_sec = 0.0
        self.continuous_open_sec += dt
        self._active_closure_episode_recorded = False

        alert_event: Optional[TemporalAlertEvent] = None
        msg: str

        if self.alert_stage > AlertStage.STAGE_0_NORMAL or self.state in (
            DrowsinessState.EARLY_WARNING,
            DrowsinessState.DROWSY,
            DrowsinessState.CRITICAL,
            DrowsinessState.RECOVERY,
        ):
            if self.continuous_open_sec < self.config.recovery_time_sec:
                self.state = DrowsinessState.RECOVERY
                msg = f"Recovery in progress: Eyes open {self.continuous_open_sec:.2f}s / {self.config.recovery_time_sec:.2f}s."
            else:
                self.state = DrowsinessState.NORMAL
                self.alert_stage = AlertStage.STAGE_0_NORMAL
                self.score_stage1_timer = 0.0
                self.score_stage2_timer = 0.0
                self.score_stage3_timer = 0.0
                alert_event = TemporalAlertEvent(
                    stage=AlertStage.STAGE_0_NORMAL,
                    event_type=TemporalEventType.RECOVERY_COMPLETE,
                    message="Driver recovery confirmed: Eyes open continuously.",
                    timestamp=timestamp,
                    duration_sec=self.continuous_open_sec,
                )
                msg = "Recovery confirmed: Returned to normal monitoring."
        else:
            self.state = DrowsinessState.NORMAL
            self.alert_stage = AlertStage.STAGE_0_NORMAL
            msg = "Normal monitoring: Driver attentive."

        return alert_event, msg
