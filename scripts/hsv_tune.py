#!/usr/bin/env python3
"""Interactive HSV threshold tuner.

Shows the camera/video feed and the combined red mask with trackbars for
both hue-wrap ranges. Press 'p' to print the current values as YAML ready
to paste into config/default.yaml, 'q' to quit.

    python scripts/hsv_tune.py            # default camera
    python scripts/hsv_tune.py clip.mp4   # video file
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from turret.camera.factory import open_camera  # noqa: E402
from turret.config import CameraConfig, DetectionConfig  # noqa: E402

WINDOW = "hsv_tune"

BARS = [
    ("H1 lo", 0, 180), ("S1 lo", 120, 255), ("V1 lo", 70, 255),
    ("H1 hi", 10, 180), ("S1 hi", 255, 255), ("V1 hi", 255, 255),
    ("H2 lo", 170, 180), ("S2 lo", 120, 255), ("V2 lo", 70, 255),
    ("H2 hi", 180, 180), ("S2 hi", 255, 255), ("V2 hi", 255, 255),
]


def read_ranges() -> tuple[np.ndarray, ...]:
    v = [cv2.getTrackbarPos(name, WINDOW) for name, _, _ in BARS]
    return (
        np.array(v[0:3]), np.array(v[3:6]),
        np.array(v[6:9]), np.array(v[9:12]),
    )


def main() -> None:
    device: int | str = sys.argv[1] if len(sys.argv) > 1 else 0
    cam = open_camera(CameraConfig(backend="opencv", device=device))
    cv2.namedWindow(WINDOW)
    for name, default, maximum in BARS:
        cv2.createTrackbar(name, WINDOW, default, maximum, lambda _v: None)

    defaults = DetectionConfig()
    try:
        while True:
            frame = cam.read()
            if frame is None:
                break
            lo1, hi1, lo2, hi2 = read_ranges()
            hsv = cv2.cvtColor(
                cv2.GaussianBlur(frame, (defaults.blur_ksize,) * 2, 0),
                cv2.COLOR_BGR2HSV,
            )
            mask = cv2.inRange(hsv, lo1, hi1) | cv2.inRange(hsv, lo2, hi2)
            cv2.imshow(WINDOW, frame)
            cv2.imshow("mask", mask)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
            if key == ord("p"):
                print("detection:")
                print(f"  red_low_1: {list(map(int, lo1))}")
                print(f"  red_high_1: {list(map(int, hi1))}")
                print(f"  red_low_2: {list(map(int, lo2))}")
                print(f"  red_high_2: {list(map(int, hi2))}")
    finally:
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
