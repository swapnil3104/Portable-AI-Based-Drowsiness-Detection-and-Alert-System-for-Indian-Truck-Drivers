"""Integrated Camera and Eye-Region Pipeline for DrishtiDrive AI.

Coordinates camera capture, face and eye detection, preprocessing, and CNN inference.
Measures latency and FPS, explicitly handles degraded observations, prevents
silent OPEN fallbacks, and enforces the single-frame drowsiness isolation safety rule.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
import time
from typing import Any, Dict, List, Optional
import numpy as np

from backend.alert_manager import AlertManager
from backend.alert_visual import VisualAlertNotification
from backend.camera import CameraService
from backend.config import PipelineConfig
from backend.eye_detection import BaseEyeDetector, DetectionStatus, EyeDetectionResult, PoseEyeDetector
from backend.model_inference import EyeClassifier, EyeStatePrediction
from backend.preprocessing import EyePreprocessor
from backend.temporal_machine import TemporalDrowsinessDetector
from backend.temporal_state import TemporalStatus

logger = logging.getLogger("drishtidrive.pipeline")


@dataclass
class PipelineFrameResult:
    """Consolidated outputs and diagnostic telemetry for a single processed frame."""

    timestamp: float
    frame_index: int
    overall_state: str  # "OPEN", "CLOSED", or "UNCERTAIN"
    overall_confidence: float
    detection_status: DetectionStatus
    detection_result: EyeDetectionResult
    left_eye_prediction: Optional[EyeStatePrediction] = None
    right_eye_prediction: Optional[EyeStatePrediction] = None
    latency_breakdown: Dict[str, float] = field(default_factory=dict)
    camera_fps: float = 0.0
    consecutive_closed_frames: int = 0
    drowsiness_declared: bool = False
    status_message: str = ""
    temporal_status: Optional[TemporalStatus] = None
    visual_alert: Optional[VisualAlertNotification] = None
    alert_status: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert frame result to serializable dictionary."""
        return {
            "timestamp": round(self.timestamp, 4),
            "frame_index": self.frame_index,
            "overall_state": self.overall_state,
            "overall_confidence": round(self.overall_confidence, 4),
            "detection_status": self.detection_status.value,
            "latency_breakdown": self.latency_breakdown,
            "camera_fps": self.camera_fps,
            "consecutive_closed_frames": self.consecutive_closed_frames,
            "drowsiness_declared": self.drowsiness_declared,
            "status_message": self.status_message,
            "temporal_status": self.temporal_status.to_dict() if self.temporal_status else None,
            "visual_alert": self.visual_alert.to_dict() if self.visual_alert else None,
            "alert_status": self.alert_status,
        }


class EyeRegionPipeline:
    """End-to-end coordinator for the OpenCV Camera and Eye-Region Pipeline."""

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        camera: Optional[CameraService] = None,
        detector: Optional[BaseEyeDetector] = None,
        preprocessor: Optional[EyePreprocessor] = None,
        classifier: Optional[EyeClassifier] = None,
        temporal_detector: Optional[TemporalDrowsinessDetector] = None,
        alert_manager: Optional[AlertManager] = None,
    ) -> None:
        """Initialize pipeline with modular components and shared configuration."""
        self.config = config or PipelineConfig()
        self.camera = camera or CameraService(self.config.camera)
        self.preprocessor = preprocessor or EyePreprocessor(self.config.preprocessing)
        self.classifier = classifier or EyeClassifier(self.config.inference)
        self.detector = detector or PoseEyeDetector(self.config.detection)
        self.temporal_detector = temporal_detector or TemporalDrowsinessDetector(self.config.temporal)
        self.alert_manager = alert_manager or AlertManager(self.config.alert)

        self.frame_index: int = 0
        self.consecutive_closed_frames: int = 0
        self._is_active: bool = False

    def start(self) -> None:
        """Start the camera capture service and pipeline session."""
        self.camera.start()
        self.frame_index = 0
        self.consecutive_closed_frames = 0
        self.temporal_detector.reset()
        self._is_active = True
        logger.info("EyeRegionPipeline started.")

    def stop(self) -> None:
        """Stop camera capture and release resources gracefully."""
        self._is_active = False
        self.camera.stop()
        self.alert_manager.stop()
        logger.info("EyeRegionPipeline stopped.")

    def process_frame(self, frame: np.ndarray, timestamp: Optional[float] = None) -> PipelineFrameResult:
        """Execute full detection, preprocessing, and classification on a single frame."""
        t_start = time.perf_counter()
        now = timestamp if timestamp is not None else time.time()
        self.frame_index += 1

        latency: Dict[str, float] = {}

        # 1. Face and Eye Region Detection
        t0 = time.perf_counter()
        det_res = self.detector.detect(frame)
        latency["detection_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

        # 2. Handle Degraded Observations Safely (Rules 5, 8)
        if not det_res.is_valid:
            self.consecutive_closed_frames = 0
            temp_status = self.temporal_detector.step("UNCERTAIN", False, now)
            visual_notif = self.alert_manager.handle_temporal_status(temp_status)
            latency["total_ms"] = round((time.perf_counter() - t_start) * 1000.0, 2)
            return PipelineFrameResult(
                timestamp=now,
                frame_index=self.frame_index,
                overall_state="UNCERTAIN",
                overall_confidence=0.0,
                detection_status=det_res.status,
                detection_result=det_res,
                latency_breakdown=latency,
                camera_fps=self.camera.current_fps,
                consecutive_closed_frames=0,
                drowsiness_declared=False,
                status_message=det_res.message,
                temporal_status=temp_status,
                visual_alert=visual_notif,
                alert_status=self.alert_manager.get_status(),
            )

        # 3. Preprocess and Classify Eye Regions
        crops: List[np.ndarray] = []
        crop_roles: List[str] = []

        if det_res.left_eye_crop is not None:
            crops.append(det_res.left_eye_crop)
            crop_roles.append("left")
        if det_res.right_eye_crop is not None:
            crops.append(det_res.right_eye_crop)
            crop_roles.append("right")

        t0 = time.perf_counter()
        batch_tensor = self.preprocessor.preprocess_crops_batch(crops)
        latency["preprocessing_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

        t0 = time.perf_counter()
        preds = self.classifier.predict_batch(batch_tensor)
        latency["inference_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

        left_pred: Optional[EyeStatePrediction] = None
        right_pred: Optional[EyeStatePrediction] = None
        for role, pred in zip(crop_roles, preds):
            if role == "left":
                left_pred = pred
            elif role == "right":
                right_pred = pred

        # 4. Synthesize Combined Eye State (Preventing Silent OPEN Fallback)
        state, conf, msg = self._combine_eye_states(left_pred, right_pred)

        # 5. Track consecutive closed frames for immediate telemetry
        if state == "CLOSED":
            self.consecutive_closed_frames += 1
        else:
            self.consecutive_closed_frames = 0

        # 6. Apply Temporal State Machine (Phase 5)
        # Prevents single closed-eye frames from declaring confirmed drowsiness
        is_obs_valid = (state != "UNCERTAIN")
        mar_val = getattr(det_res, "mar", 0.14)
        ear_val = getattr(det_res, "ear", 0.10 if state == "CLOSED" else 0.28)
        temp_status = self.temporal_detector.step(state, is_obs_valid, now, mar=mar_val, ear=ear_val)
        visual_notif = self.alert_manager.handle_temporal_status(temp_status)
        drowsiness_event = (temp_status.alert_stage.value >= 2)

        latency["total_ms"] = round((time.perf_counter() - t_start) * 1000.0, 2)

        return PipelineFrameResult(
            timestamp=now,
            frame_index=self.frame_index,
            overall_state=state,
            overall_confidence=round(conf, 4),
            detection_status=det_res.status,
            detection_result=det_res,
            left_eye_prediction=left_pred,
            right_eye_prediction=right_pred,
            latency_breakdown=latency,
            camera_fps=self.camera.current_fps,
            consecutive_closed_frames=self.consecutive_closed_frames,
            drowsiness_declared=drowsiness_event,
            status_message=msg,
            temporal_status=temp_status,
            visual_alert=visual_notif,
            alert_status=self.alert_manager.get_status(),
        )

    def _combine_eye_states(
        self,
        left: Optional[EyeStatePrediction],
        right: Optional[EyeStatePrediction],
    ) -> Tuple[str, float, str]:
        """Fuse dual-eye predictions into a robust driver eye state."""
        if left is None and right is None:
            return "UNCERTAIN", 0.0, "No eye predictions available."

        if left is not None and right is not None:
            if left.state == "CLOSED" and right.state == "CLOSED":
                return "CLOSED", (left.confidence + right.confidence) / 2.0, "Both eyes closed."
            if left.state == "OPEN" and right.state == "OPEN":
                return "OPEN", (left.confidence + right.confidence) / 2.0, "Both eyes open."
            # Robust handling of glasses reflection / single eye closure
            if left.state == "CLOSED" and left.confidence >= 0.72:
                return "CLOSED", left.confidence * 0.9, "Left eye confirmed closed."
            if right.state == "CLOSED" and right.confidence >= 0.72:
                return "CLOSED", right.confidence * 0.9, "Right eye confirmed closed."
            if left.state == "UNCERTAIN" and right.state == "UNCERTAIN":
                return "UNCERTAIN", 0.0, "Both eye predictions below confidence threshold."
            confident = left if left.state != "UNCERTAIN" else right
            return confident.state, confident.confidence * 0.85, f"Single eye confirmed {confident.state}."

        single = left if left is not None else right
        return single.state, single.confidence, f"Monocular observation: {single.state}."

    def step(self) -> PipelineFrameResult:
        """Fetch next camera frame and process through pipeline."""
        frame, ts = self.camera.read_frame()
        return self.process_frame(frame, timestamp=ts)

    def __enter__(self) -> "EyeRegionPipeline":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.stop()
