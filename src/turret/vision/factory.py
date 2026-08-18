from __future__ import annotations

from turret.config import TurretConfig
from turret.vision.base import Detector
from turret.vision.detector import RedBlobDetector

_BACKENDS = ("hsv", "tflite", "opencv_dnn")


def build_detector(cfg: TurretConfig, *, debug: bool = False) -> Detector:
    """Construct the detector named by cfg.detector_backend.

    Never silently falls back to a different backend: a requested ML
    backend that can't load raises with the exact command to fix it, rather
    than quietly reverting to HSV.
    """
    backend = cfg.detector_backend
    if backend == "hsv":
        return RedBlobDetector(cfg.detection)
    if backend == "tflite":
        from turret.vision.tflite_detector import TFLiteDetector

        return TFLiteDetector(cfg.ml_detection, debug=debug)
    if backend == "opencv_dnn":
        from turret.vision.dnn_detector import DnnDetector

        return DnnDetector(cfg.ml_detection, debug=debug)
    raise ValueError(
        f"Unknown detector_backend {backend!r}; expected one of: {', '.join(_BACKENDS)}"
    )
