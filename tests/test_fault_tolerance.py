"""Fault-tolerance and reliability test suite for DrishtiDrive AI.

Verifies:
- Degradation states: no face, occlusion, low confidence, and multiple faces.
- Subsystem failures: model load failure, camera disconnection, audio hardware failure.
- Safe lifecycle handling: repeated start/stop cycles and clean resource release.
"""
from __future__ import annotations

from pathlib import Path
import time
from typing import Optional
import numpy as np
import pytest

from backend.alert_manager import AlertManager
from backend.audio_backend import BaseAudioBackend
from backend.camera import CameraConfig, CameraReadError, CameraService, CameraUnavailableError
from backend.config import InferenceConfig, PipelineConfig
from backend.eye_detection import DetectionStatus, EyeDetectionResult, MockEyeDetector
from backend.model_inference import EyeClassifier, ModelLoadError
from backend.pipeline import EyeRegionPipeline
from backend.temporal_state import AlertStage, TemporalStatus


class BrokenAudioBackend(BaseAudioBackend):
    """Failing audio backend simulating broken sound card or ALSA/DirectSound error."""

    def __init__(self) -> None:
        self.call_count = 0

    def play_tone(self, stage: AlertStage, volume: float) -> bool:
        """Simulate hardware I/O error during tone playback."""
        self.call_count += 1
        raise RuntimeError("DirectSound device unplugged or driver error")

    def speak(self, text: str, stage: AlertStage, language: str = "en", volume: float = 0.8) -> bool:
        """Simulate TTS engine crash during speech synthesis."""
        self.call_count += 1
        raise RuntimeError("SAPI / Speech engine initialization failure")


class DisconnectingCameraService(CameraService):
    """Camera service that simulates hardware disconnection after N frames."""

    def __init__(self, config: CameraConfig, fail_after_frames: int = 3) -> None:
        super().__init__(config)
        self.fail_after_frames = fail_after_frames
        self.frames_served = 0

    def read_frame(self, timeout: Optional[float] = None) -> tuple[np.ndarray, float]:
        """Serve mock frames then raise CameraReadError simulating physical disconnect."""
        self.frames_served += 1
        if self.frames_served > self.fail_after_frames:
            raise CameraReadError("USB Camera disconnected mid-stream (device descriptor read error)")
        return np.zeros((480, 640, 3), dtype=np.uint8), time.time()


def test_degraded_observation_states_never_escalate() -> None:
    """Verify NO_FACE, OCCLUSION, LOW_CONFIDENCE, and MULTIPLE_FACES never declare OPEN or trigger alerts."""
    mock_det = MockEyeDetector()
    cam_cfg = CameraConfig(video_source="reports/test_driver_feed.mp4")
    pipeline = EyeRegionPipeline(detector=mock_det, camera=CameraService(cam_cfg))
    pipeline.start()

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    degraded_conditions = [
        (DetectionStatus.NO_FACE, "No face detected in camera FOV."),
        (DetectionStatus.OCCLUSION, "Eyes occluded by mask or hands."),
        (DetectionStatus.LOW_CONFIDENCE, "Face confidence below 0.50 threshold."),
        (DetectionStatus.MULTIPLE_FACES, "Multiple occupants detected without clear primary driver."),
        (DetectionStatus.POOR_QUALITY, "Extreme blur or low lighting."),
    ]

    for status, msg in degraded_conditions:
        mock_det.set_result(EyeDetectionResult(status=status, message=msg))
        res = pipeline.process_frame(dummy_frame)

        assert res.overall_state == "UNCERTAIN"
        assert res.overall_confidence == 0.0
        assert not res.drowsiness_declared
        assert res.consecutive_closed_frames == 0
        assert res.temporal_status is not None
        # Safe degradation hold: alerts must not escalate during sensor dropouts
        assert res.temporal_status.alert_stage == AlertStage.STAGE_0_NORMAL

    pipeline.stop()


def test_corrupted_model_load_failure(tmp_path: Path) -> None:
    """Verify corrupted checkpoint raises actionable ModelLoadError."""
    corrupted_file = tmp_path / "corrupted_weights.pt"
    corrupted_file.write_bytes(b"INVALID_TORCH_HEADER_1234567890")

    cfg = InferenceConfig(
        model_name="depthwise_cnn",
        model_checkpoint=str(corrupted_file),
    )
    with pytest.raises(ModelLoadError) as exc_info:
        EyeClassifier(cfg)
    assert "Failed to load" in str(exc_info.value) or "checkpoint" in str(exc_info.value).lower()


def test_camera_disconnection_handling() -> None:
    """Verify camera disconnection mid-stream is trapped and reported cleanly."""
    cam_cfg = CameraConfig(video_source="reports/test_driver_feed.mp4")
    disc_cam = DisconnectingCameraService(cam_cfg, fail_after_frames=2)
    pipeline = EyeRegionPipeline(camera=disc_cam)
    pipeline.start()

    # First two frames succeed
    f1 = pipeline.step()
    assert f1 is not None
    f2 = pipeline.step()
    assert f2 is not None

    # Third frame raises CameraReadError cleanly
    with pytest.raises(CameraReadError) as exc_info:
        pipeline.step()
    assert "disconnected" in str(exc_info.value).lower()

    pipeline.stop()


def test_audio_hardware_failure_resilience() -> None:
    """Verify alert manager safely catches audio hardware crashes and keeps visual alerts operational."""
    from backend.audio_backend import MockAudioBackend
    failing_backend = MockAudioBackend(simulate_device_failure=True)
    manager = AlertManager(audio_backend=failing_backend)

    # Triggering Stage 2 alert with broken sound hardware
    success = manager.trigger_alert(AlertStage.STAGE_2_DROWSY, "Testing audio fault tolerance")
    assert success is True

    # Allow worker thread to process
    time.sleep(0.05)

    # Check status: error message captured, visual alert intact, system does not crash
    status = manager.get_status()
    assert status["current_stage"] == 2
    assert status["visual_notification"] is not None
    assert status["visual_notification"]["stage"] == 2
    assert status["audio_backend"]["device_available"] is False

    manager.stop()


def test_repeated_start_stop_resource_cleanup() -> None:
    """Verify repeated pipeline start/stop cycles cleanly release resources without leaks."""
    from backend.audio_backend import MockAudioBackend
    from backend.eye_detection import MockEyeDetector

    for cycle in range(5):
        cam_cfg = CameraConfig(video_source="reports/test_driver_feed.mp4")
        cam = CameraService(cam_cfg)
        mock_audio = MockAudioBackend()
        mock_det = MockEyeDetector()
        pipeline = EyeRegionPipeline(
            camera=cam,
            alert_manager=AlertManager(audio_backend=mock_audio),
            detector=mock_det,
        )
        pipeline.start()
        assert pipeline._is_active
        assert pipeline.camera.is_running
        assert pipeline.alert_manager._worker_thread.is_alive()

        # Step single frame
        res = pipeline.step()
        assert res is not None

        pipeline.stop()
        assert not pipeline._is_active
        assert not pipeline.camera.is_running
        # Alert worker stops
        assert not pipeline.alert_manager._worker_thread.is_alive()
