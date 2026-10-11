"""Comprehensive unit and integration tests for Phase 6 alert subsystem.

Verifies:
- All three alert stages produce the exact specified sound and voice message.
- Escalation, recovery, cancellation, and cooldown behaviors.
- Speech overlap prevention and queue backlog protection.
- Multi-language catalog support (English, Hindi, Marathi).
- Graceful degradation when audio devices or speech engines fail.
- Non-blocking execution (<1.0 ms on main thread).
"""
from __future__ import annotations

import time
import numpy as np
import pytest

from backend.alert_catalog import ALERT_MESSAGES, AlertMessageCatalog
from backend.alert_manager import AlertManager, AlertTask
from backend.alert_sounds import ensure_alert_tones, synthesize_wav_tone
from backend.alert_visual import get_visual_notification, render_alert_overlay
from backend.audio_backend import MockAudioBackend
from backend.config import AlertConfig
from backend.temporal_state import (
    AlertStage,
    DrowsinessState,
    TemporalAlertEvent,
    TemporalEventType,
    TemporalStatus,
)


class TestAlertCatalogAndSounds:
    """Verifies alert messages and tone synthesis."""

    def test_stage_messages_english(self) -> None:
        """Verify exact specified voice messages for Stages 1, 2, and 3 in English."""
        msg1 = AlertMessageCatalog.get_message(AlertStage.STAGE_1_EARLY, "en")
        msg2 = AlertMessageCatalog.get_message(AlertStage.STAGE_2_DROWSY, "en")
        msg3 = AlertMessageCatalog.get_message(AlertStage.STAGE_3_CRITICAL, "en")

        assert msg1 == "Please stay alert. Keep your eyes on the road."
        assert msg2 == "Warning! You may be getting drowsy. Please find a safe place to stop."
        assert msg3 == "Danger! Pull over safely as soon as possible. Do not continue driving while drowsy."

    def test_multilingual_support(self) -> None:
        """Verify Hindi and Marathi message availability and fallback."""
        assert AlertMessageCatalog.is_language_supported("en")
        assert AlertMessageCatalog.is_language_supported("hi")
        assert AlertMessageCatalog.is_language_supported("mr")

        hi_msg = AlertMessageCatalog.get_message(AlertStage.STAGE_1_EARLY, "hi")
        mr_msg = AlertMessageCatalog.get_message(AlertStage.STAGE_1_EARLY, "mr")
        assert "सतर्क" in hi_msg
        assert "सावध" in mr_msg

        # Fallback to English for unknown language
        fallback_msg = AlertMessageCatalog.get_message(AlertStage.STAGE_1_EARLY, "de")
        assert fallback_msg == "Please stay alert. Keep your eyes on the road."

    def test_tone_synthesis(self, tmp_path) -> None:
        """Verify WAV tone generation produces valid non-empty audio files."""
        out_tones = ensure_alert_tones(tmp_path)
        assert len(out_tones) == 3
        for stage, path in out_tones.items():
            assert path.exists()
            assert path.stat().st_size > 1000

    def test_prerecorded_voice_mp3_assets(self) -> None:
        """Verify all three required voice MP3 files exist in backend/assets."""
        from pathlib import Path
        assets_dir = Path("backend/assets")
        mp3_files = {
            AlertStage.STAGE_1_EARLY: "Stage1_Voice.mp3",
            AlertStage.STAGE_2_DROWSY: "Stage2_voice.mp3",
            AlertStage.STAGE_3_CRITICAL: "Stage3_voice.mp3",
        }
        for stage, filename in mp3_files.items():
            p = assets_dir / filename
            assert p.exists(), f"Missing required voice MP3: {p}"
            assert p.stat().st_size > 10000, f"MP3 file too small or empty: {p}"


class TestAlertManagerLogic:
    """Verifies AlertManager lifecycle, deduplication, escalation, and cooldown."""

    def test_all_three_stages_dispatch(self) -> None:
        """Verify each stage triggers corresponding tone and exact voice message."""
        mock_backend = MockAudioBackend()
        manager = AlertManager(audio_backend=mock_backend)

        try:
            # Stage 1: Early Warning
            manager.trigger_alert(AlertStage.STAGE_1_EARLY, "Stage 1 test")
            time.sleep(0.05)
            assert len(mock_backend.played_tones) == 1
            assert mock_backend.played_tones[0][0] == AlertStage.STAGE_1_EARLY
            assert len(mock_backend.spoken_messages) == 1
            assert mock_backend.spoken_messages[0][0] == "Please stay alert. Keep your eyes on the road."

            # Stage 2: Drowsiness Warning
            manager.trigger_alert(AlertStage.STAGE_2_DROWSY, "Stage 2 test")
            time.sleep(0.05)
            assert len(mock_backend.played_tones) == 2
            assert mock_backend.played_tones[1][0] == AlertStage.STAGE_2_DROWSY
            assert len(mock_backend.spoken_messages) == 2
            assert "Warning! You may be getting drowsy." in mock_backend.spoken_messages[1][0]

            # Stage 3: Critical Warning
            manager.trigger_alert(AlertStage.STAGE_3_CRITICAL, "Stage 3 test")
            time.sleep(0.05)
            assert len(mock_backend.played_tones) == 3
            assert mock_backend.played_tones[2][0] == AlertStage.STAGE_3_CRITICAL
            assert len(mock_backend.spoken_messages) == 3
            assert "Danger! Pull over safely" in mock_backend.spoken_messages[2][0]
        finally:
            manager.stop()

    def test_cooldown_and_deduplication(self) -> None:
        """Verify repeated calls within cooldown are suppressed without escalation."""
        mock_backend = MockAudioBackend()
        cfg = AlertConfig(cooldown_sec=5.0)
        manager = AlertManager(config=cfg, audio_backend=mock_backend)

        try:
            # First trigger succeeds
            res1 = manager.trigger_alert(AlertStage.STAGE_1_EARLY)
            assert res1 is True

            # Immediate repeat within cooldown is suppressed
            res2 = manager.trigger_alert(AlertStage.STAGE_1_EARLY)
            assert res2 is False

            # Escalation to higher stage bypasses cooldown
            res3 = manager.trigger_alert(AlertStage.STAGE_2_DROWSY)
            assert res3 is True
            assert manager.current_stage == AlertStage.STAGE_2_DROWSY
        finally:
            manager.stop()

    def test_recovery_handling(self) -> None:
        """Verify temporal recovery transitions stage back to NORMAL."""
        mock_backend = MockAudioBackend()
        manager = AlertManager(audio_backend=mock_backend)

        try:
            # Trigger warning stage
            manager.trigger_alert(AlertStage.STAGE_2_DROWSY)
            assert manager.current_stage == AlertStage.STAGE_2_DROWSY

            # Recovery event
            recovery_status = TemporalStatus(
                state=DrowsinessState.NORMAL,
                alert_stage=AlertStage.STAGE_0_NORMAL,
                closure_duration_sec=0.0,
                continuous_open_sec=2.0,
                blink_count=1,
                last_blink_duration_sec=0.2,
                recent_events_count=0,
                is_observation_reliable=True,
                status_message="Recovery confirmed",
            )
            visual = manager.handle_temporal_status(recovery_status)
            assert manager.current_stage == AlertStage.STAGE_0_NORMAL
            assert visual.is_active is False
            assert "ATTENTIVE" in visual.title or "NORMAL" in visual.badge_text
        finally:
            manager.stop()

    def test_manual_audio_test_and_volume(self) -> None:
        """Verify manual audio test dispatch and volume adjustments."""
        mock_backend = MockAudioBackend()
        manager = AlertManager(audio_backend=mock_backend)

        try:
            manager.set_volume(0.35)
            assert manager.volume == 0.35

            res = manager.trigger_manual_test(AlertStage.STAGE_2_DROWSY)
            assert res is True
            time.sleep(0.05)
            assert len(mock_backend.played_tones) == 1
            assert mock_backend.played_tones[0][1] == pytest.approx(0.35, abs=0.01)
        finally:
            manager.stop()

    def test_queue_backlog_protection(self) -> None:
        """Verify that rapid repeated triggers do not build an unbounded queue."""
        mock_backend = MockAudioBackend()
        manager = AlertManager(audio_backend=mock_backend)

        try:
            for _ in range(20):
                manager.trigger_alert(AlertStage.STAGE_1_EARLY)
            # Queue maxsize is strictly 1
            assert manager._queue.qsize() <= 1
        finally:
            manager.stop()

    def test_graceful_handling_of_device_failure(self) -> None:
        """Verify missing audio device does not crash or raise exceptions."""
        failing_backend = MockAudioBackend(simulate_device_failure=True)
        manager = AlertManager(audio_backend=failing_backend)

        try:
            # Triggering alert when device is unavailable must succeed gracefully without crash
            success = manager.trigger_alert(AlertStage.STAGE_3_CRITICAL)
            assert success is True
            time.sleep(0.05)
            status = manager.get_status()
            assert status["audio_backend"]["device_available"] is False
        finally:
            manager.stop()

    def test_non_blocking_execution_latency(self) -> None:
        """Verify handle_temporal_status executes in <1.0 ms on caller thread."""
        mock_backend = MockAudioBackend()
        manager = AlertManager(audio_backend=mock_backend)

        try:
            alert_event = TemporalAlertEvent(
                stage=AlertStage.STAGE_2_DROWSY,
                event_type=TemporalEventType.SUSTAINED_CLOSURE,
                message="Test alert",
                timestamp=time.time(),
                duration_sec=2.0,
            )
            temp_status = TemporalStatus(
                state=DrowsinessState.DROWSY,
                alert_stage=AlertStage.STAGE_2_DROWSY,
                closure_duration_sec=2.0,
                continuous_open_sec=0.0,
                blink_count=0,
                last_blink_duration_sec=0.0,
                recent_events_count=1,
                is_observation_reliable=True,
                alert_event=alert_event,
            )

            t0 = time.perf_counter()
            manager.handle_temporal_status(temp_status)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            # Caller latency must be well under 1.0 ms (camera frame loop unblocked)
            assert elapsed_ms < 1.0
        finally:
            manager.stop()


class TestVisualAlerts:
    """Verifies visual banner creation and overlay rendering."""

    def test_visual_notification_content(self) -> None:
        """Verify visual notifications carry stage-appropriate colors and directives."""
        vis1 = get_visual_notification(AlertStage.STAGE_1_EARLY)
        vis2 = get_visual_notification(AlertStage.STAGE_2_DROWSY)
        vis3 = get_visual_notification(AlertStage.STAGE_3_CRITICAL)

        assert vis1.is_active is True
        assert vis2.is_active is True
        assert vis3.is_active is True
        assert "ATTENTION" in vis1.badge_text
        assert "WARNING" in vis2.badge_text
        assert "DANGER" in vis3.badge_text

    def test_render_alert_overlay(self) -> None:
        """Verify alert overlay renders on standard frame without modifying dimensions."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        vis = get_visual_notification(AlertStage.STAGE_3_CRITICAL)

        out_frame = render_alert_overlay(frame, vis, audio_status_text="OK")
        assert out_frame.shape == frame.shape
        # Top banner should have non-zero pixels rendered
        assert out_frame[:60, :, :].sum() > 0
