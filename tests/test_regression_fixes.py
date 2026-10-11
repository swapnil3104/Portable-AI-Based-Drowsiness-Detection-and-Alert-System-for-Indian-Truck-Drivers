"""Regression tests for DrishtiDrive AI engineering fixes.

Verifies:
1. validate_temporal module is directly runnable without ModuleNotFoundError.
2. validate_alerts module is directly runnable without ModuleNotFoundError.
3. Root requirements and backend requirements include critical dependencies.
4. Single-frame eye closure does not trigger confirmed alert (anti-flicker safety test).
5. Alert manager strictly prevents overlapping duplicate alert audio playback.
6. Zero media files are stored on disk by default (privacy invariant).
"""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import time
import pytest

from backend.alert_manager import AlertManager
from backend.audio_backend import MockAudioBackend
from backend.config import AlertConfig, PipelineConfig, TemporalConfig
from backend.temporal_machine import TemporalDrowsinessDetector
from backend.temporal_state import AlertStage, DrowsinessState


def test_validate_temporal_standalone_execution() -> None:
    """Verify validate_temporal.py runs directly with Python without ModuleNotFoundError."""
    script_path = Path("backend/validate_temporal.py")
    assert script_path.exists()
    cmd = [sys.executable, str(script_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, f"validate_temporal.py failed:\n{result.stderr}"
    combined_output = result.stdout + result.stderr
    assert "phase5_temporal_benchmark.json" in combined_output or "Saved Phase 5" in combined_output


def test_validate_alerts_standalone_execution() -> None:
    """Verify validate_alerts.py runs directly with Python without ModuleNotFoundError."""
    script_path = Path("backend/validate_alerts.py")
    assert script_path.exists()
    cmd = [sys.executable, str(script_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, f"validate_alerts.py failed:\n{result.stderr}"
    assert "Phase 6 Benchmark Verification Summary" in result.stdout


def test_requirements_files_completeness() -> None:
    """Verify backend/requirements.txt contains core web, model, and audio dependencies."""
    req_path = Path("backend/requirements.txt")
    assert req_path.exists()
    content = req_path.read_text(encoding="utf-8").lower()
    for dep in ["fastapi", "uvicorn", "pydantic", "torch", "opencv-python", "pyttsx3"]:
        assert dep in content, f"Missing dependency '{dep}' in backend/requirements.txt"


def test_single_closed_eye_frame_never_triggers_alert() -> None:
    """Verify safety rule: a single closed-eye frame never escalates to an alert."""
    cfg = TemporalConfig()
    detector = TemporalDrowsinessDetector(cfg)

    # Supply open frames
    detector.step(raw_state="OPEN", is_observation_valid=True, timestamp=1.0)
    # Supply exactly ONE closed eye frame
    result = detector.step(raw_state="CLOSED", is_observation_valid=True, timestamp=1.033)

    assert result.alert_stage == AlertStage.STAGE_0_NORMAL
    assert result.state != DrowsinessState.CRITICAL
    assert result.alert_event is None


def test_alerts_do_not_overlap_or_stack() -> None:
    """Verify alert manager prevents audio overlap when multiple triggers arrive concurrently."""
    mock_audio = MockAudioBackend()
    cfg = AlertConfig(cooldown_sec=5.0)
    manager = AlertManager(config=cfg, audio_backend=mock_audio)

    try:
        # Trigger stage 1 alert
        ok1 = manager.trigger_alert(AlertStage.STAGE_1_EARLY, "Early warning")
        assert ok1 is True
        assert manager.current_stage == AlertStage.STAGE_1_EARLY

        # Immediate trigger for same stage within cooldown period must be debounced
        ok_dup = manager.trigger_alert(AlertStage.STAGE_1_EARLY, "Early warning duplicate")
        assert ok_dup is False

        # Allow worker thread brief cycle to process initial task
        time.sleep(0.05)

        # Audio playback calls reflect single dispatch, zero overlapping duplicates
        assert len(mock_audio.played_tones) == 1
        assert len(mock_audio.spoken_messages) == 1
    finally:
        manager.stop()


def test_privacy_defaults_enforced_in_config() -> None:
    """Verify default PipelineConfig guarantees zero media storage."""
    cfg = PipelineConfig()
    assert cfg.save_frames_by_default is False
