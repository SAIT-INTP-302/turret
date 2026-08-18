from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

from turret.app import TurretApp
from turret.config import load_config
from turret.selftest import format_report, run_selftest

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "default.yaml"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="turret", description="Red-target tracking turret")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML config path")
    parser.add_argument("--video", help="Use a video file instead of a camera")
    parser.add_argument("--camera", type=int, help="Camera index (forces OpenCV backend)")
    parser.add_argument(
        "--detector",
        choices=["hsv", "tflite", "opencv_dnn"],
        help="Override detector_backend from config",
    )
    parser.add_argument("--model", help="Override ml_detection.model_path")
    parser.add_argument("--mock", action="store_true", help="Force mock actuators")
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="Check all subsystems (axes, camera, detector, fire control) and exit",
    )
    parser.add_argument(
        "--actuate-trigger",
        action="store_true",
        help="With --selftest, also motion-test the trigger axis (default: skipped for safety)",
    )
    parser.add_argument(
        "--no-motion",
        action="store_true",
        help="With --selftest, skip axis motion tests entirely (construction/wiring check only)",
    )
    parser.add_argument(
        "--camera-frames",
        type=int,
        default=5,
        help="With --selftest, frames to read during the camera check",
    )
    parser.add_argument("--headless", action="store_true", help="No display windows")
    parser.add_argument("--show-mask", action="store_true", help="Show the detection mask")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--dashboard", action="store_true", help="Serve the event dashboard")
    parser.add_argument("--dashboard-port", type=int, default=8080)
    parser.add_argument("--db", default="turret_events.db", help="SQLite path for event log")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    if args.video is not None:
        cfg = replace(cfg, camera=replace(cfg.camera, backend="opencv", device=args.video))
    elif args.camera is not None:
        cfg = replace(cfg, camera=replace(cfg.camera, backend="opencv", device=args.camera))
    if args.detector is not None:
        cfg = replace(cfg, detector_backend=args.detector)
    if args.model is not None:
        cfg = replace(cfg, ml_detection=replace(cfg.ml_detection, model_path=args.model))

    if args.selftest:
        report = run_selftest(
            cfg,
            force_mock=args.mock,
            actuate_axes=not args.no_motion,
            actuate_trigger=args.actuate_trigger,
            camera_frames=args.camera_frames,
        )
        print(format_report(report))
        sys.exit(report.exit_code)

    TurretApp(
        cfg,
        force_mock=args.mock,
        headless=args.headless,
        show_mask=args.show_mask,
        dashboard=args.dashboard,
        dashboard_port=args.dashboard_port,
        db_path=args.db,
    ).run()


if __name__ == "__main__":
    main()
