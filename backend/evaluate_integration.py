"""Phase 8 Integration and Evaluation Runner for DrishtiDrive AI.

Executes rigorous independent evaluation across:
1. Recorded video sequence (reports/test_driver_feed.mp4) through full pipeline.
2. Temporal driving event sequences measuring False Alarm Rate (FAR) and Missed Event Rate (MER).
3. Held-out test image cohorts evaluating optical impediments (glasses, bad lighting, reflections).
4. Generates structured benchmark results at reports/phase8_integration_benchmark.json.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from backend.alert_manager import AlertManager
from backend.audio_backend import MockAudioBackend
from backend.camera import CameraConfig, CameraService
from backend.config import PipelineConfig, TemporalConfig
from backend.eye_detection import DetectionStatus, EyeDetectionResult, MockEyeDetector, PoseEyeDetector
from backend.model_inference import EyeClassifier
from backend.pipeline import EyeRegionPipeline
from backend.preprocessing import EyePreprocessor
from backend.temporal_machine import TemporalDrowsinessDetector
from backend.temporal_state import AlertStage

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("drishtidrive.evaluator")


def evaluate_video_clip(video_path: Path) -> Dict[str, Any]:
    """Evaluate end-to-end pipeline on recorded driver video clip."""
    logger.info("Evaluating recorded video clip: %s", video_path)
    cam_cfg = CameraConfig(video_source=str(video_path), target_fps=30)
    camera = CameraService(cam_cfg)
    mock_audio = MockAudioBackend()
    alert_mgr = AlertManager(audio_backend=mock_audio)
    pipeline = EyeRegionPipeline(camera=camera, alert_manager=alert_mgr)

    pipeline.start()
    frame_results: List[Dict[str, Any]] = []
    latencies: List[float] = []
    state_counts: Dict[str, int] = {"OPEN": 0, "CLOSED": 0, "UNCERTAIN": 0}
    t_start = time.perf_counter()

    try:
        for _ in range(60):  # Full 60 frames of test_driver_feed.mp4
            res = pipeline.step()
            tot_ms = res.latency_breakdown.get("total_ms", 0.0)
            stg = res.temporal_status.alert_stage.value if res.temporal_status else 0
            frame_results.append({
                "frame": res.frame_index, "state": res.overall_state,
                "confidence": round(res.overall_confidence, 3),
                "stage": stg, "latency_ms": tot_ms,
            })
            latencies.append(tot_ms)
            state_counts[res.overall_state] = state_counts.get(res.overall_state, 0) + 1
    finally:
        pipeline.stop()

    total_time = time.perf_counter() - t_start
    avg_fps = len(frame_results) / max(0.001, total_time)

    return {
        "clip_path": str(video_path),
        "total_frames_processed": len(frame_results),
        "duration_seconds": round(total_time, 2),
        "measured_fps": round(avg_fps, 1),
        "avg_latency_ms": round(sum(latencies) / max(1, len(latencies)), 2),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)) if latencies else 0.0, 2),
        "state_distribution": state_counts,
        "alerts_dispatched": alert_mgr.total_alerts_emitted,
    }


def evaluate_temporal_scenarios() -> Dict[str, Any]:
    """Measure False Alarm Rate (FAR) and Missed Event Rate (MER) on controlled sequences."""
    logger.info("Evaluating temporal driving scenarios for FAR and MER...")
    cfg = TemporalConfig()

    scenarios = [
        # Scenario 1: Attentive Driver (12s, 360 frames @ 30 FPS, blinks every 3s)
        {
            "name": "attentive_driver_normal",
            "duration_sec": 12.0,
            "fps": 30.0,
            "generator": "normal_blinks",
            "expected_alerts": 0,
        },
        # Scenario 2: Prolonged Eye Closure / Microsleep (8s, 240 frames @ 30 FPS, 3.2s closure)
        {
            "name": "prolonged_closure_microsleep",
            "duration_sec": 8.0,
            "fps": 30.0,
            "generator": "microsleep",
            "expected_alerts": 1,  # Expect Stage 2 and Stage 3
        },
        # Scenario 3: Repeated Micro-Closures (20s, 600 frames @ 30 FPS, 3x 0.9s closures)
        {
            "name": "repeated_micro_closures",
            "duration_sec": 20.0,
            "fps": 30.0,
            "generator": "repeated",
            "expected_alerts": 1,  # Expect Stage 2 escalation
        },
        # Scenario 4: Sensor Loss / Occlusion (8s, 240 frames @ 30 FPS, 3.0s UNCERTAIN)
        {
            "name": "sensor_loss_dropout",
            "duration_sec": 8.0,
            "fps": 30.0,
            "generator": "dropout",
            "expected_alerts": 0,  # Must hold without false alarms
        },
    ]

    results: Dict[str, Any] = {}
    for sc in scenarios:
        detector = TemporalDrowsinessDetector(cfg)
        fps = sc["fps"]
        dt = 1.0 / fps
        total_steps = int(sc["duration_sec"] * fps)
        t = 0.0

        stage_reached = 0
        false_alarms = 0
        missed_events = 0
        alert_timestamps: List[Tuple[float, int]] = []

        for i in range(total_steps):
            if sc["generator"] == "normal_blinks":
                # Blink of 150ms every 3.0 seconds
                is_closed = (1.2 <= (t % 3.0) <= 1.35)
                raw_state = "CLOSED" if is_closed else "OPEN"
                is_valid = True
            elif sc["generator"] == "microsleep":
                # Pre-open 1.0s, closed for 3.2s (1.0 to 4.2), then recover
                is_closed = (1.0 <= t <= 4.2)
                raw_state = "CLOSED" if is_closed else "OPEN"
                is_valid = True
            elif sc["generator"] == "repeated":
                # 3 episodes of 0.9s closed followed by 2.0s open
                is_closed = (1.0 <= t <= 1.9) or (3.9 <= t <= 4.8) or (6.8 <= t <= 7.7)
                raw_state = "CLOSED" if is_closed else "OPEN"
                is_valid = True
            elif sc["generator"] == "dropout":
                # Open 1.0s, face lost 3.0s (1.0 to 4.0), then recover
                is_lost = (1.0 <= t <= 4.0)
                raw_state = "UNCERTAIN" if is_lost else "OPEN"
                is_valid = not is_lost

            status = detector.step(raw_state, is_valid, t)
            stg = status.alert_stage.value
            if stg > stage_reached:
                stage_reached = stg
                alert_timestamps.append((round(t, 2), stg))

            t += dt

        # Score outcomes
        if sc["expected_alerts"] == 0:
            false_alarms = 1 if stage_reached >= 2 else 0
            missed_events = 0
            far = 0.0 if false_alarms == 0 else 1.0
            mer = 0.0
        else:
            false_alarms = 0
            missed_events = 0 if stage_reached >= 2 else 1
            far = 0.0
            mer = 0.0 if missed_events == 0 else 1.0

        results[sc["name"]] = {
            "duration_sec": sc["duration_sec"],
            "total_frames": total_steps,
            "max_alert_stage_reached": stage_reached,
            "alert_transitions": alert_timestamps,
            "false_alarms": false_alarms,
            "missed_events": missed_events,
            "false_alarm_rate": far,
            "missed_event_rate": mer,
            "passed": (false_alarms == 0 and missed_events == 0),
        }

    return results


def evaluate_optical_impediments() -> Dict[str, Any]:
    """Evaluate CNN eye classifier across glasses, lighting, and reflections on test split."""
    logger.info("Evaluating held-out test split under optical impediments...")
    clf = EyeClassifier()
    preprocessor = EyePreprocessor()
    test_files = list(Path("data/eye_state/test").glob("*/*.png"))

    categories = {
        "all_held_out_test": [f for f in test_files if len(f.stem.split("_")) >= 8],
        "eyeglasses_present": [f for f in test_files if len(f.stem.split("_")) >= 8 and f.stem.split("_")[3] == "1"],
        "no_eyeglasses": [f for f in test_files if len(f.stem.split("_")) >= 8 and f.stem.split("_")[3] == "0"],
        "poor_lighting": [f for f in test_files if len(f.stem.split("_")) >= 8 and f.stem.split("_")[6] == "0"],
        "good_lighting": [f for f in test_files if len(f.stem.split("_")) >= 8 and f.stem.split("_")[6] == "1"],
        "severe_reflections": [f for f in test_files if len(f.stem.split("_")) >= 8 and f.stem.split("_")[5] == "2"],
    }

    metrics: Dict[str, Any] = {}
    for cat_name, files in categories.items():
        tp = 0  # True closed correctly predicted as closed
        fp = 0  # True open falsely predicted as closed (false alarm)
        tn = 0  # True open correctly predicted as open
        fn = 0  # True closed missed as open (missed closure)
        latencies = []

        for f in files:
            is_true_closed = "closed" in f.parent.name.lower()
            img = cv2.imread(str(f))
            tensor = preprocessor.preprocess_crop(img)
            pred = clf.predict_tensor(tensor)
            latencies.append(pred.latency_ms)

            pred_closed = (pred.state == "CLOSED")
            if is_true_closed and pred_closed:
                tp += 1
            elif not is_true_closed and pred_closed:
                fp += 1
            elif not is_true_closed and not pred_closed:
                tn += 1
            elif is_true_closed and not pred_closed:
                fn += 1

        total = len(files)
        acc = (tp + tn) / max(1, total)
        far = fp / max(1, fp + tn)  # False Alarm Rate (False Positive Rate)
        mer = fn / max(1, fn + tp)  # Missed Event Rate (False Negative Rate)
        closed_recall = tp / max(1, tp + fn)
        avg_lat = sum(latencies) / max(1, len(latencies))

        metrics[cat_name] = {
            "sample_count": total,
            "accuracy": round(acc * 100.0, 2),
            "false_alarm_rate_pct": round(far * 100.0, 2),
            "missed_event_rate_pct": round(mer * 100.0, 2),
            "closed_eye_recall_pct": round(closed_recall * 100.0, 2),
            "avg_latency_ms": round(avg_lat, 2),
            "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        }

    return metrics


def run_full_integration_evaluation() -> Dict[str, Any]:
    """Execute complete Phase 8 evaluation and save report."""
    video_path = Path("reports/test_driver_feed.mp4")
    video_results = evaluate_video_clip(video_path)
    temporal_results = evaluate_temporal_scenarios()
    optical_results = evaluate_optical_impediments()

    overall_report = {
        "benchmark_timestamp": time.time(),
        "phase": "Phase 8 — Integration and Testing",
        "evaluation_scope": {
            "camera_service": "Integrated (Video playback & USB Camera)",
            "detector": "YOLO11n-Pose (Dual Keypoint Localization)",
            "classifier": "Depthwise-Separable CNN (1.93ms, 0.38MB)",
            "temporal_engine": "Time-based state machine with debouncing",
            "alert_manager": "Three-stage audio and visual alert system",
            "frontend_api": "FastAPI + WebSocket telemetry stream",
        },
        "video_clip_evaluation": video_results,
        "temporal_scenarios_evaluation": temporal_results,
        "optical_impediments_evaluation": optical_results,
        "methodology_and_limitations": {
            "methodology": (
                "Evaluated using independent held-out MRL test partitions (stratified by eyeglasses, "
                "lighting, and lens reflections) and controlled continuous temporal event sequences "
                "measuring False Alarm Rate (FAR) and Missed Event Rate (MER). Video pipeline was benchmarked "
                "on a recorded test driver feed."
            ),
            "dataset_limitations": (
                "Evaluations are laboratory and simulation benchmarks on laptop/desktop CPU. Datasets comprise "
                "controlled subject recordings and webcam feeds. They do NOT represent real-world automotive "
                "crash or highway conditions and cannot validate safety compliance."
            ),
            "safety_notice": (
                "DrishtiDrive AI is an experimental driver-assistance prototype. It must not be connected to "
                "vehicle actuators (brakes, steering) and cannot guarantee drowsiness detection."
            ),
        },
    }
    out_file = Path("reports/phase8_integration_benchmark.json")
    out_file.write_text(json.dumps(overall_report, indent=2))
    logger.info("Evaluation report successfully written to %s", out_file)
    return overall_report


if __name__ == "__main__":
    run_full_integration_evaluation()
