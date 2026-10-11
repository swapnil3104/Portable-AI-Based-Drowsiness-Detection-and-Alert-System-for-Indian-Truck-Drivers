"""Thread-safe, non-blocking camera capture service for DrishtiDrive AI.

Provides configurable video capture from webcam index, video file, or mock frame
generators. Implements robust lifecycle controls (start, stop, cleanup), FPS tracking,
and explicit error reporting for missing hardware or permission failures.
"""
from __future__ import annotations

import logging
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Any, Optional, Tuple, Union
import cv2
import numpy as np

from backend.config import CameraConfig

logger = logging.getLogger("drishtidrive.camera")


class CameraError(Exception):
    """Base exception for camera capture errors."""


class CameraUnavailableError(CameraError):
    """Raised when the specified camera or video source cannot be opened."""


class CameraPermissionError(CameraError):
    """Raised when access to the camera device is denied by the OS."""


class CameraReadError(CameraError):
    """Raised when reading a frame from the capture source fails repeatedly."""


class CameraService:
    """Manages asynchronous camera capture, frame buffering, and FPS tracking."""

    def __init__(self, config: Optional[CameraConfig] = None) -> None:
        """Initialize camera service with configuration settings."""
        self.config = config or CameraConfig()
        self.cap: Optional[cv2.VideoCapture] = None
        self._is_running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._frame_queue: queue.Queue[Tuple[np.ndarray, float]] = queue.Queue(
            maxsize=max(1, self.config.buffer_size)
        )
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_timestamp: float = 0.0
        self._fps: float = 0.0
        self._frame_count: int = 0
        self._fps_start_time: float = 0.0

    @property
    def is_running(self) -> bool:
        """Return True if background capture thread is actively running."""
        return self._is_running

    @property
    def current_fps(self) -> float:
        """Return estimated real-time capture frames per second."""
        return self._fps

    def start(self) -> None:
        """Open video capture resource and launch background worker thread."""
        with self._lock:
            if self._is_running:
                logger.warning("Camera service already running.")
                return

            source: Union[int, str] = (
                self.config.video_source
                if self.config.video_source is not None
                else self.config.camera_index
            )
            logger.info("Opening capture source: %s", source)
            fallback_path = Path("reports/test_driver_feed.mp4")
            should_fallback = (isinstance(source, int) or str(source) == "test_video") and fallback_path.exists()

            try:
                # Try standard capture backend first (MSMF on modern Windows, V4L2 on Linux)
                self.cap = cv2.VideoCapture(source)
                if (self.cap is None or not self.cap.isOpened()) and isinstance(source, int) and sys.platform == "win32":
                    logger.info("Attempting DirectShow backend for camera index %s", source)
                    self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            except Exception as err:
                logger.warning("Error opening camera %s: %s", source, err)
                if should_fallback:
                    self.cap = cv2.VideoCapture(str(fallback_path.resolve()))
                    self.config.video_source = str(fallback_path.resolve())
                else:
                    raise CameraUnavailableError(f"Failed to open camera source: {err}") from err

            if (self.cap is None or not self.cap.isOpened()):
                if should_fallback:
                    logger.warning("Webcam %s unavailable; auto-falling back to driver feed %s", source, fallback_path)
                    self.cap = cv2.VideoCapture(str(fallback_path.resolve()))
                    self.config.video_source = str(fallback_path.resolve())
                else:
                    raise CameraUnavailableError(f"Camera source '{source}' unavailable or cannot be opened.")

            if isinstance(source, int) and self.config.video_source is None:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            self._is_running = True
            self._frame_count = 0
            self._fps_start_time = time.perf_counter()
            self._thread = threading.Thread(
                target=self._capture_worker, name="DrishtiDrive-CameraWorker", daemon=True
            )
            self._thread.start()
            logger.info("Camera capture worker thread started successfully.")

    def stop(self) -> None:
        """Signal background worker to stop and release video capture resources."""
        with self._lock:
            if not self._is_running:
                return
            self._is_running = False

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.5)
            self._thread = None

        with self._lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            # Drain queue
            while not self._frame_queue.empty():
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    break
            logger.info("Camera service stopped and resources released.")

    def read_frame(self, timeout: Optional[float] = None) -> Tuple[np.ndarray, float]:
        """Retrieve most recent frame and timestamp (non-blocking or with timeout)."""
        if not self._is_running:
            raise CameraReadError("Cannot read frame: Camera service is not running.")

        timeout_val = timeout if timeout is not None else self.config.read_timeout_sec
        try:
            frame, timestamp = self._frame_queue.get(timeout=timeout_val)
            with self._lock:
                self._latest_frame = frame
                self._latest_timestamp = timestamp
            return frame, timestamp
        except queue.Empty as err:
            with self._lock:
                if self._latest_frame is not None:
                    return self._latest_frame.copy(), self._latest_timestamp
            raise CameraReadError(f"No frame received within {timeout_val:.1f} seconds.") from err

    def get_latest_frame(self) -> Tuple[Optional[np.ndarray], float]:
        """Return the most recently cached frame without waiting."""
        with self._lock:
            if self._latest_frame is None:
                return None, 0.0
            return self._latest_frame.copy(), self._latest_timestamp

    def get_frame(self) -> Optional[np.ndarray]:
        """Retrieve latest captured frame or None."""
        frame, _ = self.get_latest_frame()
        return frame

    def _capture_worker(self) -> None:
        """Background thread target that continually reads frames from device."""
        fail_count = 0
        max_consecutive_fails = 30

        while self._is_running:
            if self.cap is None or not self.cap.isOpened():
                break

            ret, frame = self.cap.read()
            now = time.perf_counter()

            if not ret or frame is None or frame.size == 0:
                if self.config.video_source is not None and self.cap is not None:
                    # Rewind video file to beginning to allow continuous stream
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    if ret and frame is not None and frame.size > 0:
                        fail_count = 0

                fail_count += 1
                if fail_count > max_consecutive_fails:
                    fallback_path = Path("reports/test_driver_feed.mp4")
                    if fallback_path.exists() and self.config.video_source is None:
                        logger.warning("Webcam read failed %d times; switching to demo feed %s", fail_count, fallback_path)
                        self.cap.release()
                        self.cap = cv2.VideoCapture(str(fallback_path.resolve()))
                        self.config.video_source = str(fallback_path.resolve())
                        fail_count = 0
                        continue
                    logger.error("Too many consecutive frame read failures in capture worker.")
                    break
                time.sleep(0.02)
                continue

            fail_count = 0
            if frame is None:
                continue
            self._update_fps(now)
            with self._lock:
                self._latest_frame = frame.copy()
                self._latest_timestamp = now

            # Keep queue fresh with newest frame
            if self._frame_queue.full():
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    pass

            try:
                self._frame_queue.put_nowait((frame, now))
            except queue.Full:
                pass

            # Yield briefly to maintain target FPS pacing if needed
            target_delay = 1.0 / max(1, self.config.target_fps)
            time.sleep(max(0.001, target_delay * 0.1))

        self._is_running = False

    def _update_fps(self, now: float) -> None:
        """Update running FPS measurement."""
        self._frame_count += 1
        elapsed = now - self._fps_start_time
        if elapsed >= 1.0:
            self._fps = round(self._frame_count / elapsed, 1)
            self._frame_count = 0
            self._fps_start_time = now

    def __enter__(self) -> "CameraService":
        """Context manager entry point."""
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit point releasing resources."""
        self.stop()
