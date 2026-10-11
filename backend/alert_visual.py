"""Visual notification models and OpenCV overlay rendering for DrishtiDrive AI.

Displays active alert stages, safety directives, and audio playback status.
Uses restrained, professional color schemes aligned with vehicle HUD standards.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np

from backend.alert_catalog import AlertMessageCatalog
from backend.temporal_state import AlertStage


@dataclass
class VisualAlertNotification:
    """Visual alert presentation metadata for screen rendering."""

    stage: AlertStage
    title: str
    message: str
    color_bgr: Tuple[int, int, int]
    badge_text: str
    is_active: bool
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize visual notification state to dictionary."""
        return {
            "stage": self.stage.value,
            "title": self.title,
            "message": self.message,
            "badge_text": self.badge_text,
            "is_active": self.is_active,
            "timestamp": round(self.timestamp, 3),
        }


def get_visual_notification(
    stage: AlertStage,
    language: str = "en",
    timestamp: Optional[float] = None,
) -> VisualAlertNotification:
    """Construct visual notification styling and messages for an alert stage."""
    now = timestamp if timestamp is not None else time.time()
    title = AlertMessageCatalog.get_title(stage, language)
    message = AlertMessageCatalog.get_message(stage, language)

    # Restrained color tokens: BGR format for OpenCV
    color_map = {
        AlertStage.STAGE_0_NORMAL: (113, 179, 60),      # Muted forest green
        AlertStage.STAGE_1_EARLY: (30, 180, 255),       # Amber / gold
        AlertStage.STAGE_2_DROWSY: (0, 140, 255),       # Warm orange
        AlertStage.STAGE_3_CRITICAL: (34, 34, 220),     # Restrained crimson red
    }
    badge_map = {
        AlertStage.STAGE_0_NORMAL: "MONITORING - ATTENTIVE",
        AlertStage.STAGE_1_EARLY: "STAGE 1 - ATTENTION",
        AlertStage.STAGE_2_DROWSY: "STAGE 2 - WARNING",
        AlertStage.STAGE_3_CRITICAL: "STAGE 3 - DANGER",
    }

    return VisualAlertNotification(
        stage=stage,
        title=title,
        message=message,
        color_bgr=color_map.get(stage, (113, 179, 60)),
        badge_text=badge_map.get(stage, "NORMAL"),
        is_active=(stage > AlertStage.STAGE_0_NORMAL),
        timestamp=now,
    )


def render_alert_overlay(
    frame: np.ndarray,
    visual: VisualAlertNotification,
    audio_status_text: str = "",
) -> np.ndarray:
    """Render high-contrast, clean alert banner and HUD status onto an image frame.

    Args:
        frame: BGR numpy image frame.
        visual: Visual notification data.
        audio_status_text: Optional telemetry line for audio device/playback status.

    Returns:
        Frame with alert banner overlaid.
    """
    out_frame = frame.copy()
    h, w = out_frame.shape[:2]

    # Render top banner if alert is active (Stages 1, 2, 3)
    if visual.is_active:
        banner_h = 60
        # Draw translucent background banner
        overlay = out_frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), visual.color_bgr, -1)
        cv2.addWeighted(overlay, 0.85, out_frame, 0.15, 0, out_frame)

        # Draw outer perimeter warning border
        border_thickness = 4 if visual.stage == AlertStage.STAGE_3_CRITICAL else 2
        cv2.rectangle(out_frame, (0, 0), (w - 1, h - 1), visual.color_bgr, border_thickness)

        # Render Title and Directive text
        cv2.putText(
            out_frame,
            visual.badge_text,
            (16, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        # Truncate message if necessary for display width
        display_msg = visual.message[:70] + "..." if len(visual.message) > 70 else visual.message
        cv2.putText(
            out_frame,
            display_msg,
            (16, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # Render audio status tag in bottom corner
    if audio_status_text:
        cv2.putText(
            out_frame,
            f"AUDIO: {audio_status_text}",
            (16, h - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

    return out_frame
