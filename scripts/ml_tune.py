#!/usr/bin/env python3
"""Live tuner and benchmark for the ML person-detector backends.

Live mode shows the candidate canvas (every pre-selection box, chosen box
highlighted) with a confidence trackbar and reports the selected target's
`area` -- which is what fire.min_area_px should be set from when switching
detector_backend away from "hsv" (ML bounding-box area is not directly
comparable to HSV contour area). Press 'p' to print paste-ready YAML,
'q' to quit.

    python scripts/ml_tune.py --detector tflite              # default camera
    python scripts/ml_tune.py --detector tflite clip.mp4     # video file
    python scripts/ml_tune.py --detector opencv_dnn
    python scripts/ml_tune.py --detector tflite --benchmark 100   # headless timing run
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import replace
from pathlib import Path

import cv2

from turret.camera.factory import open_camera
from turret.config import CameraConfig, TurretConfig, load_config
from turret.vision.factory import build_detector

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "default.yaml"
WINDOW = "ml_tune"


def _resolve_cfg(args: argparse.Namespace) -> TurretConfig:
    cfg = load_config(args.config)
    backend = args.detector or cfg.detector_backend
    if backend not in ("tflite", "opencv_dnn"):
        raise SystemExit(
            f"ml_tune.py is for the ML backends, not {backend!r}; pass --detector tflite "
            "or --detector opencv_dnn"
        )
    ml_cfg = cfg.ml_detection
    if args.model:
        ml_cfg = replace(ml_cfg, model_path=args.model)
    return replace(cfg, detector_backend=backend, ml_detection=ml_cfg)


def live(args: argparse.Namespace) -> None:
    cfg = _resolve_cfg(args)
    detector = build_detector(cfg, debug=True)

    device = args.input if args.input is not None else 0
    cam = open_camera(CameraConfig(backend="opencv", device=device))

    cv2.namedWindow(WINDOW)
    init_conf = round(cfg.ml_detection.conf_threshold * 100)
    cv2.createTrackbar("conf x100", WINDOW, init_conf, 100, lambda _v: None)

    prev = time.monotonic()
    fps = 0.0
    try:
        while True:
            frame = cam.read()
            if frame is None:
                break
            now = time.monotonic()
            dt = now - prev
            prev = now
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if dt > 0 else fps

            conf = cv2.getTrackbarPos("conf x100", WINDOW) / 100.0
            detector._cfg = replace(detector._cfg, conf_threshold=conf)

            t0 = time.monotonic()
            det, canvas = detector.detect(frame)
            infer_ms = (time.monotonic() - t0) * 1000

            cv2.putText(
                canvas,
                f"{fps:.1f} fps  infer {infer_ms:.0f} ms  conf {conf:.2f}",
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
            )
            if det is not None:
                cv2.putText(
                    canvas,
                    f"area {det.area:.0f}  bbox {det.bbox}",
                    (8, 44),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255),
                    1,
                )
            cv2.imshow(WINDOW, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("p"):
                print(f"detector_backend: {cfg.detector_backend}")
                print("ml_detection:")
                print(f"  conf_threshold: {conf:.2f}")
                if det is not None:
                    print("fire:")
                    print(f"  min_area_px: {int(det.area)}   # observed at your chosen engagement distance")
                else:
                    print("  # no target detected right now -- move into frame before pressing 'p'")
    finally:
        cam.close()
        cv2.destroyAllWindows()


def benchmark(args: argparse.Namespace) -> None:
    cfg = _resolve_cfg(args)
    detector = build_detector(cfg, debug=False)

    device = args.input if args.input is not None else 0
    cam = open_camera(CameraConfig(backend="opencv", device=device))
    times: list[float] = []
    try:
        frame = cam.read()
        if frame is None:
            raise SystemExit("no frames available from camera/video")
        for _ in range(3):  # warm up (first inference pays one-time setup cost)
            detector.detect(frame)

        for _ in range(args.benchmark):
            frame = cam.read()
            if frame is None:
                break
            t0 = time.monotonic()
            detector.detect(frame)
            times.append(time.monotonic() - t0)
    finally:
        cam.close()

    if not times:
        raise SystemExit("no frames were timed")

    times.sort()
    mean_ms = sum(times) / len(times) * 1000
    p50_ms = times[len(times) // 2] * 1000
    p95_ms = times[int(len(times) * 0.95)] * 1000
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0

    print(f"backend: {cfg.detector_backend}")
    print(f"frames:  {len(times)}")
    print(f"mean: {mean_ms:.1f} ms   p50: {p50_ms:.1f} ms   p95: {p95_ms:.1f} ms   ~{fps:.1f} FPS")
    if fps < 3:
        print(
            "verdict: <3 FPS -- try raising ml_detection.num_threads, or fall back to "
            "detector_backend: opencv_dnn / hsv"
        )
    elif fps < 8:
        print("verdict: usable but marginal -- consider lowering camera.fps to match, or raising num_threads")
    else:
        print("verdict: comfortably real-time for this control loop")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", nargs="?", help="video file path (default: camera device 0)")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--detector", choices=["tflite", "opencv_dnn"], help="override detector_backend")
    parser.add_argument("--model", help="override ml_detection.model_path")
    parser.add_argument("--benchmark", type=int, metavar="N", help="headless: run N frames and report timing")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.benchmark:
        benchmark(args)
    else:
        live(args)


if __name__ == "__main__":
    main()
