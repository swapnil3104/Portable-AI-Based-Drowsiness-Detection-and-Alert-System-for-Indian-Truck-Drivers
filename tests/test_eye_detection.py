"""Unit tests for face and eye detection, degraded conditions, and edge cases."""
from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
import pytest

from backend.config import DetectionConfig
from backend.eye_detection import (
    DetectionStatus,
    EyeDetectionResult,
    MockEyeDetector,
    PoseEyeDetector,
)


@pytest.fixture
def real_driver_image() -> np.ndarray:
    """Load an actual driver frame from dataset."""
    img_path = Path("dataset/Dataset 2/test/images/221023-091302_jpg.rf.17ebfd180b322cd4cc0bb71466308510.jpg")
    if not img_path.exists():
        # Fallback to any image in dataset
        images = list(Path("dataset/Dataset 2").glob("**/*.jpg"))
        assert len(images) > 0, "No dataset images found for testing."
        img_path = images[0]

    img = cv2.imread(str(img_path))
    assert img is not None, f"Failed to load image at {img_path}"
    return img


def test_real_driver_detection(real_driver_image: np.ndarray) -> None:
    """Verify pose detector extracts face and eye regions on real driver image."""
    config = DetectionConfig(allow_multiple_faces=True)
    detector = PoseEyeDetector(config)

    result = detector.detect(real_driver_image)
    assert result.status == DetectionStatus.SUCCESS
    assert result.is_valid
    assert result.face_box is not None
    assert result.left_eye_crop is not None or result.right_eye_crop is not None
    assert result.face_confidence > 0.5


def test_no_face_detection() -> None:
    """Verify sharp non-face frame passes quality check and is explicitly flagged as NO_FACE."""
    detector = PoseEyeDetector()
    # Create sharp checkerboard pattern with high contrast (passes quality check, has no face)
    sharp_bg = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(0, 480, 40):
        for j in range(0, 640, 40):
            if (i // 40 + j // 40) % 2 == 0:
                sharp_bg[i : i + 40, j : j + 40] = 200

    result = detector.detect(sharp_bg)
    assert result.status == DetectionStatus.NO_FACE
    assert not result.is_valid
    assert result.left_eye_crop is None
    assert result.right_eye_crop is None



def test_poor_quality_dark_frame() -> None:
    """Verify very dark frame is rejected with POOR_QUALITY status."""
    detector = PoseEyeDetector()
    dark_frame = np.full((480, 640, 3), 5, dtype=np.uint8)

    result = detector.detect(dark_frame)
    assert result.status == DetectionStatus.POOR_QUALITY
    assert not result.is_valid
    assert "dark" in result.message.lower()


def test_poor_quality_blurry_frame() -> None:
    """Verify extremely blurry / flat frame is rejected with POOR_QUALITY status."""
    detector = PoseEyeDetector(DetectionConfig(min_brightness=10.0, blur_threshold=15.0))
    # Gaussian blur image creates low Laplacian variance
    flat_frame = np.full((480, 640, 3), 150, dtype=np.uint8)

    result = detector.detect(flat_frame)
    assert result.status == DetectionStatus.POOR_QUALITY
    assert not result.is_valid
    assert "blurry" in result.message.lower()


def test_multiple_faces_rejection(real_driver_image: np.ndarray) -> None:
    """Verify multiple faces are flagged as MULTIPLE_FACES when allow_multiple_faces=False."""
    # Place two copies of the driver side-by-side
    h, w = real_driver_image.shape[:2]
    double_frame = np.zeros((h, w * 2, 3), dtype=np.uint8)
    double_frame[:, :w] = real_driver_image
    double_frame[:, w:] = real_driver_image

    config = DetectionConfig(allow_multiple_faces=False)
    detector = PoseEyeDetector(config)

    result = detector.detect(double_frame)
    assert result.status == DetectionStatus.MULTIPLE_FACES
    assert not result.is_valid
    assert "multiple" in result.message.lower()


def test_mock_detector_fault_injection() -> None:
    """Verify MockEyeDetector allows deterministic simulation of occlusions and failures."""
    mock = MockEyeDetector()
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)

    default_res = mock.detect(dummy)
    assert default_res.status == DetectionStatus.SUCCESS
    assert default_res.is_valid

    # Inject occlusion
    mock.set_result(
        EyeDetectionResult(
            status=DetectionStatus.OCCLUSION,
            message="Simulated occlusion fault.",
        )
    )
    occ_res = mock.detect(dummy)
    assert occ_res.status == DetectionStatus.OCCLUSION
    assert not occ_res.is_valid
