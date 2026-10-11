"""DrishtiDrive AI — Unified Application Entry Point.

Tagline: A Watchful Eye on Every Journey
Launches the integrated software edition (FastAPI backend + React dashboard).
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
import webbrowser
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("drishtidrive.main")


def print_banner(host: str, port: int) -> None:
    """Display startup branding, safety notice, and access URL."""
    banner = f"""
========================================================================
             DrishtiDrive AI — Software Edition
          "A Watchful Eye on Every Journey"
========================================================================
 [i] Hardware Target: Standard Webcam & Desktop CPU
 [i] Primary Model:   MobileNetV2 (9.36ms CPU latency, 73.0% safety recall)
 [i] Face Detector:   YOLO11n-Pose Landmark Keypoints
 [i] Alert Engine:    3-Stage Audio-Visual Warning Manager
 [i] Privacy:         100% Offline, Zero Frames Persisted
 [!] SAFETY NOTICE:   Experimental Prototype only. Not an automotive
                      certified safety system.
------------------------------------------------------------------------
 Dashboard Web URL:   http://{host}:{port}/
 Interactive API:     http://{host}:{port}/docs
========================================================================
"""
    print(banner)


def main() -> None:
    """Parse CLI options and launch DrishtiDrive AI application."""
    parser = argparse.ArgumentParser(
        description="DrishtiDrive AI — Integrated Driver Monitoring & Alert System",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host IP (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open browser")
    parser.add_argument("--cli", action="store_true", help="Run standalone OpenCV window instead of web server")
    parser.add_argument("--video", type=str, default=None, help="Path to video file for simulated feed")

    args = parser.parse_args()

    if args.cli:
        from backend.run_pipeline import main as run_cli_pipeline
        sys.argv = [sys.argv[0]]
        if args.video:
            sys.argv.extend(["--video", args.video])
        run_cli_pipeline()
        return

    print_banner(args.host, args.port)

    if not args.no_browser:
        target_url = f"http://{args.host}:{args.port}/"
        try:
            webbrowser.open(target_url)
        except Exception as exc:
            logger.debug("Automatic browser launch skipped: %s", exc)

    uvicorn.run(
        "backend.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
