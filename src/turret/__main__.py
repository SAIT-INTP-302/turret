from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path

from turret.app import TurretApp
from turret.config import load_config

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
    parser.add_argument("--headless", action="store_true", help="No display windows")
    parser.add_argument("--show-mask", action="store_true", help="Show the detection mask")
    parser.add_argument("--log-level", default="INFO")
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

    TurretApp(
        cfg,
        force_mock=args.mock,
        headless=args.headless,
        show_mask=args.show_mask,
    ).run()


if __name__ == "__main__":
    main()
