"""Unit and integration tests for camera capture service and lifecycle management."""
from __future__ import annotations

from pathlib import Path
import tempfile
import time
import cv2
import numpy as np
import pytest

from backend.camera import (
    CameraConfig,
    CameraReadError,
    CameraService,
    CameraUnavailableError,
)


@pytest.fixture
def sample_video_path() -> Path:
    """Create a temporary dummy video file with 15 frames for deterministic testing."""
    temp_dir = Path(tempfile.mkdtemp())
    video_file = temp_dir / "test_feed.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_file), fourcc, 30.0, (320, 240))

    for i in range(15):
        frame = np.full((240, 320, 3), (i * 15) % 255, dtype=np.uint8)
        out.write(frame)
    out.release()

    yield video_file

    if video_file.exists():
        video_file.unlink()
    if temp_dir.exists():
        temp_dir.rmdir()


def test_camera_unavailable_error() -> None:
    """Verify CameraUnavailableError is raised when video source does not exist."""
    config = CameraConfig(video_source="non_existent_camera_stream_xyz.mp4")
    service = CameraService(config)

    with pytest.raises(CameraUnavailableError):
        service.start()
    assert not service.is_running


def test_camera_lifecycle_with_video(sample_video_path: Path) -> None:
    """Verify video source opens, buffers frames, reports FPS, and stops cleanly."""
    config = CameraConfig(video_source=str(sample_video_path), target_fps=30)
    service = CameraService(config)

    service.start()
    assert service.is_running

    # Read frame
    frame, timestamp = service.read_frame(timeout=2.0)
    assert frame is not None
    assert frame.shape == (240, 320, 3)
    assert timestamp > 0.0

    # Test latest cached frame
    cached_frame, cached_ts = service.get_latest_frame()
    assert cached_frame is not None

    service.stop()
    assert not service.is_running

    # Reading after stop should fail
    with pytest.raises(CameraReadError):
        service.read_frame()


def test_repeated_start_stop_cycles(sample_video_path: Path) -> None:
    """Verify camera handles rapid repeated start and stop cycles without resource leaks."""
    config = CameraConfig(video_source=str(sample_video_path))
    service = CameraService(config)

    for cycle in range(3):
        service.start()
        assert service.is_running
        time.sleep(0.05)
        service.stop()
        assert not service.is_running


def test_camera_context_manager(sample_video_path: Path) -> None:
    """Verify context manager enters and safely releases camera on exit."""
    config = CameraConfig(video_source=str(sample_video_path))

    with CameraService(config) as service:
        assert service.is_running
        frame, _ = service.read_frame(timeout=1.5)
        assert frame is not None

    assert not service.is_running
