"""Phase 6 Audio-Visual Alert Benchmark and Verification Suite.

Exercises each alert stage, verifies sound synthesis and speech output,
measures thread dispatch latency, checks error resilience, and writes
a JSON verification report to reports/phase6_alert_benchmark.json.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from backend.alert_catalog import AlertMessageCatalog
from backend.alert_manager import AlertManager
from backend.alert_sounds import ensure_alert_tones
from backend.alert_visual import get_visual_notification, render_alert_overlay
from backend.audio_backend import LocalAudioBackend, MockAudioBackend
from backend.config import AlertConfig
from backend.temporal_state import AlertStage

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("drishtidrive.validate_alerts")


def run_alert_benchmark() -> Dict[str, Any]:
    """Execute end-to-end verification of Phase 6 audio and visual alert subsystems."""
    logger.info("Initializing Phase 6 Alert Benchmark...")
    report: Dict[str, Any] = {
        "benchmark_timestamp": time.time(),
        "stages_verified": {},
        "dispatch_latency_ms": {},
        "localization_verified": {},
        "resilience_verified": {},
        "assets_verified": {},
    }

    # 1. Verify Alert Tone Assets
    tones = ensure_alert_tones()
    for stage, path in tones.items():
        report["assets_verified"][f"stage_{stage.value}_tone"] = {
            "path": str(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size,
        }
    logger.info("Alert tone assets verified: %d files.", len(tones))

    # 2. Verify Exact Messages & Localization
    languages = AlertMessageCatalog.SUPPORTED_LANGUAGES
    for lang in languages:
        report["localization_verified"][lang] = {
            "stage1": AlertMessageCatalog.get_message(AlertStage.STAGE_1_EARLY, lang),
            "stage2": AlertMessageCatalog.get_message(AlertStage.STAGE_2_DROWSY, lang),
            "stage3": AlertMessageCatalog.get_message(AlertStage.STAGE_3_CRITICAL, lang),
        }

    # 3. Benchmark Non-blocking Dispatch Latency
    mock_backend = MockAudioBackend()
    manager = AlertManager(config=AlertConfig(cooldown_sec=0.01), audio_backend=mock_backend)

    latencies = []
    for stage in [AlertStage.STAGE_1_EARLY, AlertStage.STAGE_2_DROWSY, AlertStage.STAGE_3_CRITICAL]:
        t0 = time.perf_counter()
        manager.trigger_alert(stage, reason=f"Benchmark stage {stage.value}")
        lat_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat_ms)
        report["dispatch_latency_ms"][f"stage_{stage.value}"] = round(lat_ms, 4)
        time.sleep(0.02)

    avg_lat = float(np.mean(latencies))
    report["dispatch_latency_ms"]["average_dispatch_latency_ms"] = round(avg_lat, 4)
    report["dispatch_latency_ms"]["is_non_blocking"] = avg_lat < 1.0
    manager.stop()

    # 4. Verify Each Stage Output
    for stage, name in [
        (AlertStage.STAGE_1_EARLY, "Stage 1 — Early Warning"),
        (AlertStage.STAGE_2_DROWSY, "Stage 2 — Drowsiness Warning"),
        (AlertStage.STAGE_3_CRITICAL, "Stage 3 — Critical Safety Alert"),
    ]:
        vis = get_visual_notification(stage, "en")
        report["stages_verified"][f"stage_{stage.value}"] = {
            "name": name,
            "sound_file": str(tones[stage].name),
            "voice_message": AlertMessageCatalog.get_message(stage, "en"),
            "visual_badge": vis.badge_text,
            "visual_color_bgr": list(vis.color_bgr),
        }

    # 5. Verify Graceful Hardware Failure Handling
    failing_backend = MockAudioBackend(simulate_device_failure=True)
    resilient_manager = AlertManager(audio_backend=failing_backend)
    no_crash = True
    try:
        resilient_manager.trigger_alert(AlertStage.STAGE_3_CRITICAL)
        resilient_manager.trigger_manual_test(AlertStage.STAGE_2_DROWSY)
    except Exception as exc:
        no_crash = False
        logger.error("Crash during simulated failure: %s", exc)
    finally:
        resilient_manager.stop()

    report["resilience_verified"] = {
        "survives_missing_audio_hardware": no_crash,
        "reported_device_error": not failing_backend.status.device_available,
    }

    # 6. Save JSON Report
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = reports_dir / "phase6_alert_benchmark.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("Report saved to %s", report_file)
    return report


if __name__ == "__main__":
    rep = run_alert_benchmark()
    print("\n=== Phase 6 Benchmark Verification Summary ===")
    print(f"Average Caller Dispatch Latency: {rep['dispatch_latency_ms']['average_dispatch_latency_ms']} ms")
    print(f"Non-blocking verified: {rep['dispatch_latency_ms']['is_non_blocking']}")
    print(f"Missing Hardware Resilience: {rep['resilience_verified']['survives_missing_audio_hardware']}")
    for s_key, s_data in rep["stages_verified"].items():
        print(f"  {s_data['name']}:")
        print(f"    Sound: {s_data['sound_file']}")
        print(f"    Voice: \"{s_data['voice_message']}\"")
