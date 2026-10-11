"""Verification of project runtime prerequisites and configuration."""
import os
from pathlib import Path


def test_env_local_exists():
    """Verify that .env.local exists and contains required environment variables."""
    env_file = Path(".env.local")
    assert env_file.exists(), ".env.local configuration file must exist"
    content = env_file.read_text(encoding="utf-8")
    assert "APP_ENV=" in content
    assert "HOST=" in content
    assert "PORT=" in content
    assert "DEFAULT_CAMERA_INDEX=" in content


def test_core_python_dependencies():
    """Verify that essential Python packages are importable."""
    import cv2
    import numpy as np

    assert hasattr(cv2, "VideoCapture"), "OpenCV VideoCapture should be available"
    assert hasattr(cv2, "FaceDetectorYN") or hasattr(cv2, "CascadeClassifier"), "OpenCV face detector should be available"
    assert hasattr(cv2, "dnn"), "OpenCV dnn module should be available"

    sample_array = np.zeros((96, 96, 3), dtype=np.uint8)
    assert sample_array.shape == (96, 96, 3)


def test_prd_and_task_artifacts_exist():
    """Verify that PRD, summary, and tasks.json exist."""
    assert Path(".agent/prd/PRD.md").exists(), "PRD.md must exist in .agent/prd/"
    assert Path(".agent/prd/SUMMARY.md").exists(), "SUMMARY.md must exist in .agent/prd/"
    assert Path(".agent/tasks.json").exists(), "tasks.json must exist in .agent/"
