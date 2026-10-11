"""Evaluation script for Phase 5 Temporal Drowsiness Logic.

Runs synthetic and held-out driving event sequences through the state machine
and saves performance metrics to reports/phase5_temporal_benchmark.json.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import TemporalConfig
from backend.temporal_validator import (
    SequenceStep,
    TemporalSequenceValidator,
    ValidationReport,
    generate_blinking_sequence,
    generate_sustained_closure_sequence,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("drishtidrive.benchmark_temporal")


def build_repeated_closures_sequence(fps: float = 30.0) -> List[SequenceStep]:
    """Generate 3 micro-sleep closure events in a 18s window to test repeated escalation."""
    steps: List[SequenceStep] = []
    dt = 1.0 / fps
    curr = 0.0

    def add_segment(state: str, duration: float, is_valid: bool = True) -> None:
        nonlocal curr
        frames = int(duration * fps)
        for _ in range(frames):
            steps.append(SequenceStep(timestamp=curr, raw_state=state, is_valid=is_valid))
            curr += dt

    # 3 episodes of 0.9s closed followed by 2.0s open
    for _ in range(3):
        add_segment("CLOSED", 0.9)
        add_segment("OPEN", 2.0)

    return steps


def build_sensor_dropout_sequence(fps: float = 30.0) -> List[SequenceStep]:
    """Generate camera loss / face occlusion outage to test safe degradation holding."""
    steps: List[SequenceStep] = []
    dt = 1.0 / fps
    curr = 0.0

    def add_segment(state: str, duration: float, is_valid: bool = True) -> None:
        nonlocal curr
        frames = int(duration * fps)
        for _ in range(frames):
            steps.append(SequenceStep(timestamp=curr, raw_state=state, is_valid=is_valid))
            curr += dt

    add_segment("OPEN", 1.0, is_valid=True)
    add_segment("UNCERTAIN", 3.0, is_valid=False)  # Face lost
    add_segment("OPEN", 1.5, is_valid=True)  # Recovered

    return steps


def build_debouncing_flicker_sequence(fps: float = 30.0) -> List[SequenceStep]:
    """Generate sustained closure with intermittent single-frame glitches."""
    steps: List[SequenceStep] = []
    dt = 1.0 / fps
    curr = 0.0

    # 2.5s closure with 2 single-frame OPEN flickers
    for i in range(int(2.5 * fps)):
        # Frame 25 and 50 are glitch OPEN frames (33ms each)
        state = "OPEN" if i in (25, 50) else "CLOSED"
        steps.append(SequenceStep(timestamp=curr, raw_state=state, is_valid=True))
        curr += dt

    return steps


def run_all_benchmarks() -> Dict[str, Any]:
    """Execute all validation scenarios and write results to reports."""
    validator = TemporalSequenceValidator(TemporalConfig())
    scenarios = {
        "attentive_driver_normal": generate_blinking_sequence(duration_sec=12.0, fps=30.0),
        "sustained_closure_microsleep": generate_sustained_closure_sequence(
            pre_open_sec=1.0, closure_sec=3.3, post_open_sec=2.0
        ),
        "repeated_micro_closures": build_repeated_closures_sequence(),
        "sensor_dropout_safety": build_sensor_dropout_sequence(),
        "flicker_debouncing_immunity": build_debouncing_flicker_sequence(),
    }

    results: Dict[str, Any] = {}
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    for name, seq in scenarios.items():
        report, _ = validator.run_sequence(name, seq)
        results[name] = report.to_dict()
        logger.info(
            "Scenario %s: Steps=%d, MaxStage=%d, Alerts=%d, Passed=%s",
            name,
            report.total_steps,
            report.max_alert_stage,
            report.alert_events_count,
            report.passed,
        )

    out_file = reports_dir / "phase5_temporal_benchmark.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Saved Phase 5 temporal benchmark results to %s", out_file)
    return results


if __name__ == "__main__":
    run_all_benchmarks()
