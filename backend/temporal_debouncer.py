"""Temporal debouncing and smoothing for eye prediction signals.

Filters high-frequency frame flicker, distinguishes normal physiological blinks
from sustained closure, and prevents single-frame noise from corrupting state timers.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

logger = logging.getLogger("drishtidrive.debouncer")


class EyeStateDebouncer:
    """Debounces raw frame-level eye states and detects blink completions."""

    def __init__(
        self,
        debounce_window_sec: float = 0.12,
        blink_max_sec: float = 0.45,
    ) -> None:
        """Initialize debouncer with persistence and blink duration thresholds."""
        self.debounce_window_sec = debounce_window_sec
        self.blink_max_sec = blink_max_sec

        self.smoothed_state: str = "OPEN"
        self._pending_state: Optional[str] = None
        self._pending_start_time: float = 0.0

        self._active_closure_start: Optional[float] = None
        self._last_timestamp: Optional[float] = None

    def reset(self) -> None:
        """Reset internal debouncing buffers and tracking timers."""
        self.smoothed_state = "OPEN"
        self._pending_state = None
        self._pending_start_time = 0.0
        self._active_closure_start = None
        self._last_timestamp = None

    def update(self, raw_state: str, timestamp: float) -> Tuple[str, Optional[float]]:
        """Update filter with raw frame observation and timestamp.

        Returns:
            Tuple of (smoothed_state, completed_blink_duration_sec).
            completed_blink_duration_sec is non-None only on the frame where a
            closure lasting <= blink_max_sec successfully transitions back to OPEN.
        """
        completed_blink: Optional[float] = None

        if self._last_timestamp is None:
            self._last_timestamp = timestamp
            self.smoothed_state = raw_state
            if raw_state == "CLOSED":
                self._active_closure_start = timestamp
            return self.smoothed_state, None

        # Guard against backwards or identical timestamps
        if timestamp < self._last_timestamp:
            timestamp = self._last_timestamp
        self._last_timestamp = timestamp

        if raw_state == self.smoothed_state:
            # Raw observation matches current smoothed state; cancel any pending flip
            self._pending_state = None
            self._pending_start_time = 0.0
            return self.smoothed_state, None

        # Raw state differs from current smoothed state
        if self._pending_state != raw_state:
            self._pending_state = raw_state
            self._pending_start_time = timestamp
            return self.smoothed_state, None

        # Pending state persists; check if debounce window has elapsed
        elapsed = timestamp - self._pending_start_time
        if elapsed >= self.debounce_window_sec:
            previous_smoothed = self.smoothed_state
            self.smoothed_state = raw_state
            self._pending_state = None
            self._pending_start_time = 0.0

            # Track closure start
            if self.smoothed_state == "CLOSED":
                if self._active_closure_start is None:
                    self._active_closure_start = timestamp - elapsed

            # Track closure end and evaluate if it qualifies as a physiological blink
            elif previous_smoothed == "CLOSED" and self.smoothed_state == "OPEN":
                if self._active_closure_start is not None:
                    duration = timestamp - self._active_closure_start
                    self._active_closure_start = None
                    if duration <= self.blink_max_sec:
                        completed_blink = max(0.05, duration)

        return self.smoothed_state, completed_blink
