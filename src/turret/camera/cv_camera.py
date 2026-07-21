"""OpenCV VideoCapture backend: USB webcams and video files.

A string device is treated as a video file and looped on EOF so dev runs
against a test clip are repeatable.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from turret.camera.base import Camera
from turret.config import CameraConfig

log = logging.getLogger(__name__)


class CvCamera(Camera):
    def __init__(self, cfg: CameraConfig) -> None:
        self._is_file = isinstance(cfg.device, str)
        self._cap = cv2.VideoCapture(cfg.device)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera/video {cfg.device!r}")
        if not self._is_file:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
            self._cap.set(cv2.CAP_PROP_FPS, cfg.fps)
        self._resolution = (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or cfg.width,
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or cfg.height,
        )

    def read(self) -> np.ndarray | None:
        ok, frame = self._cap.read()
        if not ok and self._is_file:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop the clip
            ok, frame = self._cap.read()
        return frame if ok else None

    @property
    def resolution(self) -> tuple[int, int]:
        return self._resolution

    def close(self) -> None:
        self._cap.release()
