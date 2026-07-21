"""Raspberry Pi camera backend via picamera2 (apt: python3-picamera2)."""

from __future__ import annotations

import logging

import numpy as np

from turret.camera.base import Camera
from turret.config import CameraConfig

log = logging.getLogger(__name__)


class PiCamera(Camera):
    def __init__(self, cfg: CameraConfig) -> None:
        from picamera2 import Picamera2  # apt-only; guarded import

        self._cam = Picamera2()
        config = self._cam.create_video_configuration(
            main={"size": (cfg.width, cfg.height), "format": "RGB888"}
        )
        self._cam.configure(config)
        self._cam.start()
        self._resolution = (cfg.width, cfg.height)

    def read(self) -> np.ndarray | None:
        # Despite the name, picamera2's "RGB888" arrays are BGR channel order,
        # matching OpenCV.
        return self._cam.capture_array("main")

    @property
    def resolution(self) -> tuple[int, int]:
        return self._resolution

    def close(self) -> None:
        self._cam.stop()
        self._cam.close()
