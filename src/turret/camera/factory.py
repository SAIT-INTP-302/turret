from __future__ import annotations

import logging

from turret.camera.base import Camera
from turret.config import CameraConfig

log = logging.getLogger(__name__)


def open_camera(cfg: CameraConfig) -> Camera:
    """Open the configured camera backend; 'auto' prefers picamera2."""
    if cfg.backend in ("auto", "picamera2"):
        try:
            from turret.camera.picam import PiCamera

            cam = PiCamera(cfg)
            log.info("Using picamera2 backend")
            return cam
        except Exception as exc:
            if cfg.backend == "picamera2":
                raise
            log.debug("picamera2 unavailable (%s); falling back to OpenCV", exc)

    from turret.camera.cv_camera import CvCamera

    cam = CvCamera(cfg)
    log.info("Using OpenCV backend for device %r", cfg.device)
    return cam
