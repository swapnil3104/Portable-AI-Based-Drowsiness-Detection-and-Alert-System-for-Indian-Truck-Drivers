"""Independent Alert Manager for audio playback, alert stages, and visual notifications.

Coordinates escalation, recovery, deduplication, volume control, and non-blocking
background delivery for DrishtiDrive AI.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
import queue
import threading
import time
from typing import Any, Dict, Optional

from backend.alert_catalog import AlertMessageCatalog
from backend.alert_visual import VisualAlertNotification, get_visual_notification
from backend.audio_backend import BaseAudioBackend, LocalAudioBackend
from backend.config import AlertConfig
from backend.temporal_state import AlertStage, TemporalStatus

logger = logging.getLogger("drishtidrive.alert")


@dataclass
class AlertTask:
    """An alert request to be dispatched by the background audio worker."""

    stage: AlertStage
    reason: str
    language: str
    volume: float
    timestamp: float
    is_test: bool = False


class AlertManager:
    """Thread-safe, non-blocking manager for multi-stage audio and visual alerts."""

    def __init__(
        self,
        config: Optional[AlertConfig] = None,
        audio_backend: Optional[BaseAudioBackend] = None,
    ) -> None:
        """Initialize alert manager with configuration and audio backend."""
        self.config = config or AlertConfig()
        self.audio_backend = audio_backend or LocalAudioBackend(
            use_tts=self.config.use_tts,
            initial_volume=self.config.volume,
        )

        self.current_stage: AlertStage = AlertStage.STAGE_0_NORMAL
        self.language: str = self.config.language
        self.volume: float = max(0.0, min(1.0, self.config.volume))
        self.is_muted: bool = not self.config.enabled

        self._last_alert_time: float = -999.0
        self._last_stage_time: Dict[AlertStage, float] = {
            AlertStage.STAGE_1_EARLY: -999.0,
            AlertStage.STAGE_2_DROWSY: -999.0,
            AlertStage.STAGE_3_CRITICAL: -999.0,
        }

        # Bounded single-item queue prevents queue buildup and latency accumulation
        self._queue: queue.Queue[AlertTask] = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(
            target=self._audio_worker_loop,
            name="DrishtiDriveAlertWorker",
            daemon=True,
        )
        self._worker_thread.start()

        self._active_visual = get_visual_notification(self.current_stage, self.language)
        self.total_alerts_emitted: Dict[int, int] = {1: 0, 2: 0, 3: 0}

    def set_volume(self, volume: float) -> None:
        """Update audio output volume (clamped between 0.0 and 1.0)."""
        self.volume = max(0.0, min(1.0, volume))
        self.config.volume = self.volume

    def set_language(self, language: str) -> None:
        """Set spoken alert language if supported ('en', 'hi', 'mr')."""
        if AlertMessageCatalog.is_language_supported(language):
            self.language = language.lower()
            self.config.language = self.language
            self._active_visual = get_visual_notification(self.current_stage, self.language)

    def mute(self, muted: bool = True) -> None:
        """Mute or unmute audio alert playback."""
        self.is_muted = muted
        if muted:
            self.audio_backend.stop()

    def handle_temporal_status(self, temp_status: TemporalStatus) -> VisualAlertNotification:
        """Process real-time temporal status from Phase 5 logic in microseconds."""
        target_stage = temp_status.alert_stage

        # Handle recovery back to normal
        if target_stage == AlertStage.STAGE_0_NORMAL:
            if self.current_stage != AlertStage.STAGE_0_NORMAL:
                self.current_stage = AlertStage.STAGE_0_NORMAL
                self._active_visual = get_visual_notification(
                    AlertStage.STAGE_0_NORMAL,
                    self.language,
                )
            return self._active_visual

        # Check for alert trigger or escalation
        if temp_status.alert_event is not None:
            self.trigger_alert(
                stage=temp_status.alert_event.stage,
                reason=temp_status.alert_event.message,
            )

        return self._active_visual

    def trigger_alert(self, stage: AlertStage, reason: str = "") -> bool:
        """Evaluate deduplication and cooldown rules before enqueuing an alert."""
        if stage <= AlertStage.STAGE_0_NORMAL:
            return False

        now = time.time()
        is_escalation = stage.value > self.current_stage.value
        time_since_stage = now - self._last_stage_time.get(stage, -999.0)

        # Enforce cooldown unless escalating to a higher severity stage
        if not is_escalation and time_since_stage < self.config.cooldown_sec:
            return False

        # Update stage state and visual notification
        self.current_stage = stage
        self._last_stage_time[stage] = now
        self._last_alert_time = now
        self.total_alerts_emitted[stage.value] = self.total_alerts_emitted.get(stage.value, 0) + 1
        self._active_visual = get_visual_notification(stage, self.language, timestamp=now)

        if self.is_muted or not self.config.enabled:
            return True

        task = AlertTask(
            stage=stage,
            reason=reason,
            language=self.language,
            volume=self.volume,
            timestamp=now,
        )
        self._enqueue_task(task, is_escalation)
        return True

    def trigger_manual_test(self, stage: AlertStage) -> bool:
        """Perform a manual audio test for diagnostic calibration."""
        if stage <= AlertStage.STAGE_0_NORMAL:
            return False

        now = time.time()
        self.current_stage = stage
        self._active_visual = get_visual_notification(stage, self.language, timestamp=now)

        task = AlertTask(
            stage=stage,
            reason="Manual audio calibration test",
            language=self.language,
            volume=self.volume,
            timestamp=now,
            is_test=True,
        )
        self._enqueue_task(task, force_preemption=True)
        return True

    def _enqueue_task(self, task: AlertTask, force_preemption: bool = False) -> None:
        """Enqueue task into bounded worker queue without blocking the caller."""
        if force_preemption:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self._queue.put_nowait(task)
        except queue.Full:
            # Drop redundant alert; prevents queue backlog
            pass

    def _audio_worker_loop(self) -> None:
        """Dedicated background thread worker executing sequential tone and speech playback."""
        while not self._stop_event.is_set():
            try:
                task = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                # 1. Play attention tone for the stage
                self.audio_backend.play_tone(task.stage, task.volume)

                # 2. Speak the standardized voice message
                message = AlertMessageCatalog.get_message(task.stage, task.language)
                self.audio_backend.speak(
                    text=message,
                    stage=task.stage,
                    language=task.language,
                    volume=task.volume,
                )
            except Exception as exc:
                logger.warning("Audio worker exception handled safely: %s", exc)
            finally:
                self._queue.task_done()

    def get_status(self) -> Dict[str, Any]:
        """Return comprehensive status of the alert manager and audio backend."""
        backend_stat = self.audio_backend.get_status().to_dict()
        return {
            "current_stage": self.current_stage.value,
            "language": self.language,
            "volume": self.volume,
            "is_muted": self.is_muted,
            "last_alert_time": round(self._last_alert_time, 3),
            "total_alerts_emitted": self.total_alerts_emitted,
            "visual_notification": self._active_visual.to_dict(),
            "audio_backend": backend_stat,
        }

    def get_visual_notification(self) -> VisualAlertNotification:
        """Return current visual alert banner state."""
        return self._active_visual

    def stop(self) -> None:
        """Stop worker thread and cancel pending alerts."""
        self._stop_event.set()
        self.audio_backend.stop()
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
