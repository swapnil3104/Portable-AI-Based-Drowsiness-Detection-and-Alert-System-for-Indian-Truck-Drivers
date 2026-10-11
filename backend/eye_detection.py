"""Face and eye region detection service for DrishtiDrive AI.

Locates face and eye regions using facial pose keypoints or fallback detectors.
Explicitly detects and flags edge cases: missing faces, multiple occupants,
occlusions, low confidence, and poor frame quality (blur/darkness).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import List, Optional, Tuple
import cv2
import numpy as np

from backend.config import DetectionConfig


class DetectionStatus(str, Enum):
    """Enumeration of detection outcomes and degradation states."""

    SUCCESS = "SUCCESS"
    NO_FACE = "NO_FACE"
    MULTIPLE_FACES = "MULTIPLE_FACES"
    OCCLUSION = "OCCLUSION"
    POOR_QUALITY = "POOR_QUALITY"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"


@dataclass
class EyeDetectionResult:
    """Structured result containing localized eye and mouth bounding boxes and status."""

    status: DetectionStatus
    face_box: Optional[Tuple[int, int, int, int]] = None
    left_eye_box: Optional[Tuple[int, int, int, int]] = None
    right_eye_box: Optional[Tuple[int, int, int, int]] = None
    mouth_box: Optional[Tuple[int, int, int, int]] = None
    left_eye_crop: Optional[np.ndarray] = None
    right_eye_crop: Optional[np.ndarray] = None
    face_confidence: float = 0.0
    left_eye_confidence: float = 0.0
    right_eye_confidence: float = 0.0
    mar: float = 0.14
    is_yawning: bool = False
    message: str = ""

    @property
    def is_valid(self) -> bool:
        """Return True if observation is valid for confident downstream inference."""
        return self.status == DetectionStatus.SUCCESS and (
            self.left_eye_crop is not None or self.right_eye_crop is not None
        )


class BaseEyeDetector:
    """Abstract base detector interface."""

    def detect(self, frame: np.ndarray) -> EyeDetectionResult:
        """Detect face and eye regions from input frame."""
        raise NotImplementedError


class PoseEyeDetector(BaseEyeDetector):
    """Detects face and eye regions using Ultralytics YOLO Pose keypoints."""

    def __init__(self, config: Optional[DetectionConfig] = None) -> None:
        """Initialize detector with configuration and load pose model."""
        self.config = config or DetectionConfig()
        # pyrefly: ignore [missing-import]
        from ultralytics import YOLO

        self.model = YOLO(self.config.pose_weights)

    def detect(self, frame: np.ndarray) -> EyeDetectionResult:
        """Execute pose detection and extract verified eye regions."""
        # Step 1: Quality check on raw frame
        quality_status, quality_msg = self._check_frame_quality(frame)
        if quality_status != DetectionStatus.SUCCESS:
            return EyeDetectionResult(status=quality_status, message=quality_msg)

        # Step 2: YOLO Pose Inference
        results = self.model.predict(frame, verbose=False)
        if not results or len(results) == 0:
            return EyeDetectionResult(
                status=DetectionStatus.NO_FACE, message="No person or face detected in frame."
            )

        boxes = results[0].boxes
        keypoints = results[0].keypoints

        num_people = len(boxes) if boxes is not None else 0
        if num_people == 0 or keypoints is None or len(keypoints) == 0:
            return EyeDetectionResult(
                status=DetectionStatus.NO_FACE, message="No face keypoints found in frame."
            )

        # Step 3: Handle multiple occupants
        if num_people > 1 and not self.config.allow_multiple_faces:
            return EyeDetectionResult(
                status=DetectionStatus.MULTIPLE_FACES,
                message=f"Multiple occupants detected ({num_people}). Ambiguous driver context.",
            )

        # Step 4: Extract primary occupant box & keypoints (largest bounding box)
        all_boxes = boxes.xyxy.cpu().numpy().astype(int)
        all_confs = boxes.conf.cpu().numpy()
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in all_boxes]
        primary_idx = int(np.argmax(areas))

        primary_box = all_boxes[primary_idx]
        face_conf = float(all_confs[primary_idx])
        if face_conf < self.config.min_face_confidence:
            return EyeDetectionResult(
                status=DetectionStatus.LOW_CONFIDENCE,
                face_confidence=round(face_conf, 3),
                message="Face confidence below required threshold.",
            )

        # Keypoints: 1=left_eye, 2=right_eye (COCO indexing)
        kpts_xy = keypoints.xy[primary_idx].cpu().numpy()
        kpts_conf = (
            keypoints.conf[primary_idx].cpu().numpy()
            if keypoints.conf is not None

            else np.ones(len(kpts_xy))
        )

        left_pt, left_conf = kpts_xy[1], float(kpts_conf[1])
        right_pt, right_conf = kpts_xy[2], float(kpts_conf[2])

        min_kconf = self.config.min_keypoint_confidence
        left_valid = left_conf >= min_kconf and left_pt[0] > 0 and left_pt[1] > 0
        right_valid = right_conf >= min_kconf and right_pt[0] > 0 and right_pt[1] > 0

        if not left_valid and not right_valid:
            return EyeDetectionResult(
                status=DetectionStatus.OCCLUSION,
                face_confidence=round(face_conf, 3),
                left_eye_confidence=round(left_conf, 3),
                right_eye_confidence=round(right_conf, 3),
                message="Both eye regions occluded or keypoints below confidence threshold.",
            )

        # Step 5: Compute crop regions based on inter-ocular distance or face scale
        iod = (
            math.hypot(right_pt[0] - left_pt[0], right_pt[1] - left_pt[1])
            if (left_valid and right_valid)
            else (primary_box[2] - primary_box[0]) * 0.25
        )
        box_size = max(24, int(max(iod * 0.85, 32)))

        h, w = frame.shape[:2]
        left_crop, left_box = self._extract_crop(frame, left_pt, box_size, w, h) if left_valid else (None, None)
        right_crop, right_box = self._extract_crop(frame, right_pt, box_size, w, h) if right_valid else (None, None)

        # Step 6: Localize mouth region and calculate Mouth Aspect Ratio (MAR)
        mouth_box = None
        mar_val = 0.14
        nose_pt = kpts_xy[0]
        if left_valid and right_valid and nose_pt[0] > 0 and nose_pt[1] > 0:
            eye_y = (left_pt[1] + right_pt[1]) / 2.0
            nose_dist = max(12.0, float(nose_pt[1] - eye_y))
            mouth_cx = int(nose_pt[0])
            mouth_cy = int(nose_pt[1] + nose_dist * 0.95)
            mw = max(14, int(iod * 0.40))
            mh = max(10, int(nose_dist * 0.55))
            m_crop, m_box = self._extract_crop_rect(frame, mouth_cx, mouth_cy, mw * 2, mh * 2, w, h)
            if m_crop is not None:
                mouth_box = m_box
                mar_val = self._compute_mar(m_crop)

        return EyeDetectionResult(
            status=DetectionStatus.SUCCESS,
            face_box=(primary_box[0], primary_box[1], primary_box[2], primary_box[3]),
            left_eye_box=left_box,
            right_eye_box=right_box,
            mouth_box=mouth_box,
            left_eye_crop=left_crop,
            right_eye_crop=right_crop,
            face_confidence=round(face_conf, 3),
            left_eye_confidence=round(left_conf, 3),
            right_eye_confidence=round(right_conf, 3),
            mar=mar_val,
            is_yawning=mar_val >= 0.65,
            message="Eyes successfully detected and cropped.",
        )

    def _extract_crop_rect(
        self, frame: np.ndarray, cx: int, cy: int, bw: int, bh: int, width: int, height: int
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        """Crop rectangle region with border clamping."""
        x1, y1 = max(0, cx - bw // 2), max(0, cy - bh // 2)
        x2, y2 = min(width, cx + bw // 2), min(height, cy + bh // 2)
        if x2 - x1 < 10 or y2 - y1 < 10:
            return None, None
        return frame[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)

    def _compute_mar(self, crop: np.ndarray) -> float:
        """Estimate mouth openness (MAR) based on inner oral cavity depth."""
        if crop is None or crop.size == 0:
            return 0.14
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        h_c, w_c = gray.shape
        inner = gray[int(h_c * 0.25) : int(h_c * 0.75), int(w_c * 0.2) : int(w_c * 0.8)]
        if inner.size == 0:
            return 0.14
        mean_val = float(np.mean(gray))
        if mean_val < 15.0:
            return 0.14
        # A true yawning open oral cavity is significantly darker than surrounding lip tissue
        cavity_threshold = max(20.0, mean_val * 0.40)
        dark_pixels = np.sum(inner < cavity_threshold)
        open_ratio = float(dark_pixels) / float(inner.size)
        mar = 0.14 + float(open_ratio * 0.60)
        return round(float(np.clip(mar, 0.14, 0.85)), 2)

    def _extract_crop(
        self, frame: np.ndarray, pt: np.ndarray, size: int, width: int, height: int
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        """Crop square bounding box around eye center with border clamping."""
        half = size // 2
        cx, cy = int(pt[0]), int(pt[1])
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(width, cx + half)
        y2 = min(height, cy + half)
        if x2 - x1 < 10 or y2 - y1 < 10:
            return None, None
        return frame[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)

    def _check_frame_quality(self, frame: np.ndarray) -> Tuple[DetectionStatus, str]:
        """Verify brightness and focus sharpness of input frame."""
        if frame is None or frame.size == 0:
            return DetectionStatus.POOR_QUALITY, "Empty or invalid frame."

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        brightness = float(np.mean(gray))
        if brightness < self.config.min_brightness:
            return DetectionStatus.POOR_QUALITY, f"Frame too dark ({brightness:.1f} < {self.config.min_brightness})."

        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if laplacian_var < self.config.blur_threshold:
            return DetectionStatus.POOR_QUALITY, f"Frame excessively blurry ({laplacian_var:.1f} < {self.config.blur_threshold})."

        return DetectionStatus.SUCCESS, "Quality OK."


class MockEyeDetector(BaseEyeDetector):
    """Mock eye detector for deterministic unit tests and simulated fault injection."""

    def __init__(self, canned_result: Optional[EyeDetectionResult] = None) -> None:
        """Initialize mock detector with optional preset response."""
        self.canned_result = canned_result

    def set_result(self, result: EyeDetectionResult) -> None:
        """Update canned result returned by detect()."""
        self.canned_result = result

    def detect(self, frame: np.ndarray) -> EyeDetectionResult:
        """Return preset canned result or synthesize a valid dummy result."""
        if self.canned_result is not None:
            return self.canned_result

        h, w = frame.shape[:2] if frame is not None and frame.size > 0 else (100, 100)
        dummy_crop = np.zeros((48, 48, 3), dtype=np.uint8)
        return EyeDetectionResult(
            status=DetectionStatus.SUCCESS,
            face_box=(10, 10, w - 10, h - 10),
            left_eye_box=(20, 30, 40, 50),
            right_eye_box=(60, 30, 80, 50),
            left_eye_crop=dummy_crop.copy(),
            right_eye_crop=dummy_crop.copy(),
            face_confidence=0.99,
            left_eye_confidence=0.98,
            right_eye_confidence=0.97,
            message="Mock detection success.",
        )

