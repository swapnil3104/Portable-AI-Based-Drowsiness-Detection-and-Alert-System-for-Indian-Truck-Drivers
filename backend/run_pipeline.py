"""CLI and benchmark runner for DrishtiDrive AI OpenCV Camera & Eye Pipeline.

Processes live webcam feed or test video/images, computes actual FPS and latency
measurements, and displays visual overlay or console telemetry.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Any, Dict, List
import cv2
import numpy as np

from backend.config import PipelineConfig
from backend.pipeline import EyeRegionPipeline


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("drishtidrive.runner")


def draw_overlay(frame: np.ndarray, result: Any) -> np.ndarray:
    """Render bounding boxes, eye labels, and telemetry stats onto frame."""
    canvas = frame.copy()
    det = result.detection_result

    # Draw face box
    if det.face_box is not None:
        fx1, fy1, fx2, fy2 = det.face_box
        cv2.rectangle(canvas, (fx1, fy1), (fx2, fy2), (255, 180, 0), 2)

    # Draw prominent eye boxes (primary focus)
    l_state = getattr(result.left_eye_prediction, "state", "OPEN") if getattr(result, "left_eye_prediction", None) else result.overall_state
    r_state = getattr(result.right_eye_prediction, "state", "OPEN") if getattr(result, "right_eye_prediction", None) else result.overall_state
    l_col = (0, 0, 235) if l_state == "CLOSED" else (0, 230, 0)
    r_col = (0, 0, 235) if r_state == "CLOSED" else (0, 230, 0)
    box_thick = 3 if (l_state == "CLOSED" or r_state == "CLOSED") else 2

    if det.left_eye_box is not None:
        lx1, ly1, lx2, ly2 = det.left_eye_box
        cv2.rectangle(canvas, (lx1, ly1), (lx2, ly2), l_col, box_thick)
        cv2.putText(canvas, f"L: {l_state}", (lx1, max(12, ly1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, l_col, 2)

    if det.right_eye_box is not None:
        rx1, ry1, rx2, ry2 = det.right_eye_box
        cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), r_col, box_thick)
        cv2.putText(canvas, f"R: {r_state}", (rx1, max(12, ry1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, r_col, 2)

    # Subtle mouth indicator
    if getattr(det, "mouth_box", None) is not None:
        mx1, my1, mx2, my2 = det.mouth_box
        cv2.rectangle(canvas, (mx1, my1), (mx2, my2), (160, 160, 160), 1)

    # Check temporal closure status for bold eye closing indicator
    closure_sec = 0.0
    if result.temporal_status is not None:
        closure_sec = result.temporal_status.closure_duration_sec

    # Eye Closing Alert Banner (if eyes are closed or sustained closure)
    is_eyes_closed = (result.overall_state == "CLOSED" or closure_sec > 0.15)
    if is_eyes_closed:
        # Prominent red indicator bar across top center
        h_frame, w_frame = canvas.shape[:2]
        bw, bh = min(420, w_frame - 32), 36
        bx1, by1 = (w_frame - bw) // 2, 12
        cv2.rectangle(canvas, (bx1, by1), (bx1 + bw, by1 + bh), (0, 0, 220), -1)
        cv2.rectangle(canvas, (bx1, by1), (bx1 + bw, by1 + bh), (255, 255, 255), 2)
        close_txt = f"EYES CLOSED: {closure_sec:.2f}s" if closure_sec > 0 else "EYES CLOSED"
        cv2.putText(canvas, close_txt, (bx1 + 18, by1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
    else:
        # Attentive green indicator badge
        status_color = (0, 200, 0)
        cv2.putText(canvas, "EYES: OPEN (ATTENTIVE)", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)

    # Diagnostic text
    det_ms = result.latency_breakdown.get("detection_ms", 0.0)
    inf_ms = result.latency_breakdown.get("inference_ms", 0.0)
    tot_ms = result.latency_breakdown.get("total_ms", 0.0)
    fps_text = f"FPS: {result.camera_fps:.1f} | Det: {det_ms:.1f}ms | Inf: {inf_ms:.1f}ms | Tot: {tot_ms:.1f}ms"
    cv2.putText(canvas, fps_text, (16, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)

    if result.temporal_status is not None:
        t_stat = result.temporal_status
        score_val = getattr(t_stat, "attention_score", 95.0)
        temp_text = f"Score: {score_val:.0f}% | Closure: {t_stat.closure_duration_sec:.2f}s | Open: {t_stat.continuous_open_sec:.2f}s | Blinks: {t_stat.blink_count}"
        cv2.putText(canvas, temp_text, (16, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 220, 255), 1)

    if result.status_message:
        cv2.putText(
            canvas,
            f"Status: {result.status_message}",
            (16, 96),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (180, 180, 180),
            1,
        )

    return canvas


def run_benchmark(
    source: str = "0",
    max_frames: int = 100,
    display: bool = False,
    allow_multiple_faces: bool = True,
) -> Dict[str, Any]:
    """Execute live or video-based pipeline benchmark and return telemetry."""
    config = PipelineConfig()
    config.detection.allow_multiple_faces = allow_multiple_faces
    try:
        config.camera.camera_index = int(source)
        config.camera.video_source = None
    except ValueError:
        config.camera.video_source = source

    pipeline = EyeRegionPipeline(config)

    latencies: List[float] = []
    det_latencies: List[float] = []
    inf_latencies: List[float] = []
    state_counts: Dict[str, int] = {"OPEN": 0, "CLOSED": 0, "UNCERTAIN": 0}

    logger.info("Starting pipeline benchmark (source=%s, max_frames=%d)...", source, max_frames)
    pipeline.start()

    t_start = time.perf_counter()
    processed_count = 0

    try:
        while max_frames <= 0 or processed_count < max_frames:
            try:
                result = pipeline.step()
            except Exception as err:
                logger.warning("Frame read/step error: %s", err)
                break

            processed_count += 1
            latencies.append(result.latency_breakdown.get("total_ms", 0.0))
            det_latencies.append(result.latency_breakdown.get("detection_ms", 0.0))
            inf_latencies.append(result.latency_breakdown.get("inference_ms", 0.0))
            state_counts[result.overall_state] = state_counts.get(result.overall_state, 0) + 1

            if display:
                latest_frame, _ = pipeline.camera.get_latest_frame()
                if latest_frame is not None:
                    vis = draw_overlay(latest_frame, result)
                    cv2.imshow("DrishtiDrive AI — Pipeline Preview", vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

    finally:
        pipeline.stop()
        if display:
            cv2.destroyAllWindows()

    total_time = time.perf_counter() - t_start
    actual_fps = round(processed_count / total_time, 2) if total_time > 0 else 0.0

    summary = {
        "frames_processed": processed_count,
        "total_time_seconds": round(total_time, 2),
        "actual_fps": actual_fps,
        "avg_total_latency_ms": round(float(np.mean(latencies)), 2) if latencies else 0.0,
        "p95_total_latency_ms": round(float(np.percentile(latencies, 95)), 2) if latencies else 0.0,
        "avg_detection_latency_ms": round(float(np.mean(det_latencies)), 2) if det_latencies else 0.0,
        "avg_inference_latency_ms": round(float(np.mean(inf_latencies)), 2) if inf_latencies else 0.0,
        "state_distribution": state_counts,
    }
    logger.info("Benchmark complete: %s", summary)
    return summary


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="DrishtiDrive AI Pipeline Runner")
    parser.add_argument("--source", type=str, default="0", help="Camera index or video file path")
    parser.add_argument("--frames", type=int, default=50, help="Number of frames to process")
    parser.add_argument("--display", action="store_true", help="Display video preview window")
    args = parser.parse_args()

    summary = run_benchmark(source=args.source, max_frames=args.frames, display=args.display)
    print("\n--- Benchmark Summary ---")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
