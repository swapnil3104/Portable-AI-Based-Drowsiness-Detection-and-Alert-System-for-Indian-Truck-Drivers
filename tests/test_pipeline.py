"""Integration tests for the end-to-end OpenCV Camera and Eye-Region Pipeline."""
from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
import pytest

from backend.config import PipelineConfig
from backend.eye_detection import (
    DetectionStatus,
    EyeDetectionResult,
    MockEyeDetector,
    PoseEyeDetector,
)
from backend.model_inference import EyeClassifier
from backend.pipeline import EyeRegionPipeline, PipelineFrameResult
from backend.preprocessing import EyePreprocessor


@pytest.fixture
def test_pipeline() -> EyeRegionPipeline:
    """Instantiate pipeline with real preprocessor, real classifier, and mock detector."""
    config = PipelineConfig()
    mock_detector = MockEyeDetector()
    preprocessor = EyePreprocessor(config.preprocessing)
    classifier = EyeClassifier(config.inference)
    pipeline = EyeRegionPipeline(
        config=config,
        detector=mock_detector,
        preprocessor=preprocessor,
        classifier=classifier,
    )
    return pipeline


def find_crop_for_state(pipeline: EyeRegionPipeline, state: str) -> np.ndarray:
    """Find a crop from available datasets that pipeline classifies as state."""
    for base in ["data/dataset2_eye_state/test", "data/eye_state/test"]:
        folder = Path(base) / state.lower()
        if not folder.exists():
            continue
        for f in folder.glob("*.*"):
            img = cv2.imread(str(f))
            if img is not None:
                tensor = pipeline.preprocessor.preprocess_crop(img)
                pred = pipeline.classifier.predict_tensor(tensor)
                if pred.state == state:
                    return img
    return np.zeros((96, 96, 3), dtype=np.uint8)


def test_single_closed_frame_guard(test_pipeline: EyeRegionPipeline) -> None:
    """Verify Task 9: A single closed-eye frame NEVER declares a drowsiness event."""
    closed_crop = find_crop_for_state(test_pipeline, "CLOSED")

    # Set mock detector to return closed eye crops for both eyes
    mock_det: MockEyeDetector = test_pipeline.detector  # type: ignore
    mock_det.set_result(
        EyeDetectionResult(
            status=DetectionStatus.SUCCESS,
            left_eye_crop=closed_crop,
            right_eye_crop=closed_crop,
            face_confidence=0.98,
            left_eye_confidence=0.99,
            right_eye_confidence=0.99,
            message="Both eyes detected.",
        )
    )

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = test_pipeline.process_frame(dummy_frame)

    assert result.overall_state == "CLOSED"
    assert result.overall_confidence >= 0.55
    assert result.consecutive_closed_frames == 1
    # Critical Safety Rule 9: Drowsiness event MUST NOT be declared on a single closed frame!
    assert not result.drowsiness_declared


def test_missing_face_handling_never_silent_open(test_pipeline: EyeRegionPipeline) -> None:
    """Verify Task 7 & 8: Missing faces remain UNCERTAIN and are never silently treated as OPEN."""
    mock_det: MockEyeDetector = test_pipeline.detector  # type: ignore
    mock_det.set_result(
        EyeDetectionResult(
            status=DetectionStatus.NO_FACE,
            message="No face detected in frame.",
        )
    )

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = test_pipeline.process_frame(dummy_frame)

    assert result.overall_state == "UNCERTAIN"
    assert result.overall_confidence == 0.0
    assert result.detection_status == DetectionStatus.NO_FACE
    assert not result.drowsiness_declared
    assert "no face" in result.status_message.lower()


def test_multiple_faces_handling_never_silent_open(test_pipeline: EyeRegionPipeline) -> None:
    """Verify Task 7 & 8: Multiple faces result in UNCERTAIN state and never default to OPEN."""
    mock_det: MockEyeDetector = test_pipeline.detector  # type: ignore
    mock_det.set_result(
        EyeDetectionResult(
            status=DetectionStatus.MULTIPLE_FACES,
            message="Multiple faces detected.",
        )
    )

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = test_pipeline.process_frame(dummy_frame)

    assert result.overall_state == "UNCERTAIN"
    assert result.overall_confidence == 0.0
    assert result.detection_status == DetectionStatus.MULTIPLE_FACES


def test_occlusion_handling_never_silent_open(test_pipeline: EyeRegionPipeline) -> None:
    """Verify Task 7 & 8: Occlusion results in UNCERTAIN state and never defaults to OPEN."""
    mock_det: MockEyeDetector = test_pipeline.detector  # type: ignore
    mock_det.set_result(
        EyeDetectionResult(
            status=DetectionStatus.OCCLUSION,
            message="Eyes occluded.",
        )
    )

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = test_pipeline.process_frame(dummy_frame)

    assert result.overall_state == "UNCERTAIN"
    assert result.overall_confidence == 0.0
    assert result.detection_status == DetectionStatus.OCCLUSION


def test_latency_breakdown_metrics(test_pipeline: EyeRegionPipeline) -> None:
    """Verify Task 10: Complete latency breakdown and FPS are tracked per frame."""
    open_crop = find_crop_for_state(test_pipeline, "OPEN")

    mock_det: MockEyeDetector = test_pipeline.detector  # type: ignore
    mock_det.set_result(
        EyeDetectionResult(
            status=DetectionStatus.SUCCESS,
            left_eye_crop=open_crop,
            right_eye_crop=open_crop,
            face_confidence=0.99,
            left_eye_confidence=0.95,
            right_eye_confidence=0.95,
        )
    )

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = test_pipeline.process_frame(dummy_frame)

    assert result.overall_state == "OPEN"
    assert "detection_ms" in result.latency_breakdown
    assert "preprocessing_ms" in result.latency_breakdown
    assert "inference_ms" in result.latency_breakdown
    assert "total_ms" in result.latency_breakdown
    assert result.latency_breakdown["total_ms"] > 0.0


def test_full_pipeline_with_real_driver_frame() -> None:
    """Verify live integration of real pose detector, preprocessor, and classifier on dataset frame."""
    driver_img_path = Path("dataset/Dataset 2/test/images/221023-091302_jpg.rf.17ebfd180b322cd4cc0bb71466308510.jpg")
    img = cv2.imread(str(driver_img_path))
    assert img is not None

    config = PipelineConfig()
    config.detection.allow_multiple_faces = True
    pipeline = EyeRegionPipeline(config)

    result = pipeline.process_frame(img)
    assert isinstance(result, PipelineFrameResult)
    assert result.detection_status == DetectionStatus.SUCCESS
    assert result.overall_state in ("OPEN", "CLOSED", "UNCERTAIN")
    assert result.overall_confidence > 0.0
    assert result.latency_breakdown["total_ms"] > 0.0
    assert result.temporal_status is not None


def test_pipeline_temporal_sustained_drowsiness_trigger(test_pipeline: EyeRegionPipeline) -> None:
    """Verify sustained closed frames through pipeline escalate to confirmed drowsiness."""
    closed_crop = find_crop_for_state(test_pipeline, "CLOSED")

    mock_det: MockEyeDetector = test_pipeline.detector  # type: ignore
    mock_det.set_result(
        EyeDetectionResult(
            status=DetectionStatus.SUCCESS,
            left_eye_crop=closed_crop,
            right_eye_crop=closed_crop,
            face_confidence=0.98,
            left_eye_confidence=0.99,
            right_eye_confidence=0.99,
        )
    )

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    t = 100.0

    # 1. First frame: consecutive=1, drowsiness_declared is False
    res1 = test_pipeline.process_frame(dummy_frame, timestamp=t)
    assert res1.overall_state == "CLOSED"
    assert not res1.drowsiness_declared

    # 2. Feed sustained closure past drowsy threshold (1.8s)
    for _ in range(65):
        t += 0.033
        res = test_pipeline.process_frame(dummy_frame, timestamp=t)

    assert res.overall_state == "CLOSED"
    assert res.temporal_status is not None
    assert res.temporal_status.alert_stage.value >= 2
    assert res.drowsiness_declared is True
