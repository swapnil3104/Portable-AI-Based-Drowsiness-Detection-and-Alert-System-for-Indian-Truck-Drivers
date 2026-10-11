"""Configuration definitions for DrishtiDrive AI camera and eye pipeline.

Centralizes all parameters for camera capture, face and eye detection,
preprocessing, model inference, and safety thresholds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class CameraConfig:
    """Configuration for video/camera capture source and resolution."""

    camera_index: int = 0
    video_source: Optional[str] = None
    frame_width: int = 640
    frame_height: int = 480
    target_fps: int = 30
    buffer_size: int = 2
    read_timeout_sec: float = 2.0


@dataclass
class DetectionConfig:
    """Configuration for face and eye region localization."""

    detector_type: str = "pose"  # "pose" or "haar"
    pose_weights: str = "yolo11n-pose.pt"
    face_cascade_path: Optional[str] = None
    eye_cascade_path: Optional[str] = None
    min_face_confidence: float = 0.5
    min_keypoint_confidence: float = 0.25
    eye_margin_ratio: float = 0.35
    min_face_size_pixels: int = 50
    blur_threshold: float = 20.0
    min_brightness: float = 25.0
    allow_multiple_faces: bool = True


@dataclass
class PreprocessingConfig:
    """Configuration for eye crop normalization and preprocessing."""

    image_size: int = 96
    channels: int = 3
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass
class InferenceConfig:
    """Configuration for CNN model inference and eye-state classification."""

    model_name: str = "mobilenetv2"
    model_checkpoint: str = "backend/models/mobilenetv2_best.pt"
    class_names: List[str] = field(default_factory=lambda: ["open", "closed"])
    uncertain_label: str = "UNCERTAIN"
    min_confidence: float = 0.55
    device: str = "cpu"


@dataclass
class TemporalConfig:
    """Configurable thresholds for time-based drowsiness state machine.

    SAFETY NOTICE: Eye closure is an indicator of possible drowsiness, not definitive proof.
    These thresholds are experimental engineering heuristics and are not medically validated.
    """

    blink_max_sec: float = 0.45
    sustained_early_sec: float = 0.8
    sustained_drowsy_sec: float = 1.8
    sustained_critical_sec: float = 3.0
    debounce_window_sec: float = 0.12
    recovery_time_sec: float = 1.5
    repeated_window_sec: float = 25.0
    repeated_event_threshold: int = 3
    uncertain_pause_sec: float = 0.5
    uncertain_reset_sec: float = 2.0
    alert_cooldown_sec: float = 2.0
    score_stage1_threshold: float = 75.0
    score_stage1_duration_sec: float = 60.0
    score_stage2_threshold: float = 60.0
    score_stage2_duration_sec: float = 60.0
    score_stage3_threshold: float = 30.0
    score_stage3_duration_sec: float = 30.0
    score_window_sec: float = 60.0


@dataclass
class AlertConfig:
    """Configurable settings for Phase 6 multi-stage audio and visual alerts."""

    enabled: bool = True
    volume: float = 0.8
    language: str = "en"
    use_tts: bool = False
    cooldown_sec: float = 3.0
    stage1_beep_path: str = "backend/assets/stage1_beep.wav"
    stage2_beep_path: str = "backend/assets/stage2_beep.wav"
    stage3_alarm_path: str = "backend/assets/stage3_alarm.wav"
    stage1_voice_path: str = "backend/assets/Stage1_Voice.mp3"
    stage2_voice_path: str = "backend/assets/Stage2_voice.mp3"
    stage3_voice_path: str = "backend/assets/Stage3_voice.mp3"


@dataclass
class PipelineConfig:
    """Master pipeline configuration combining all sub-component configs."""

    camera: CameraConfig = field(default_factory=CameraConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)
    save_frames_by_default: bool = False
    single_frame_guard: bool = True

