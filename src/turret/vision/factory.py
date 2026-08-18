from __future__ import annotations

from turret.config import MLDetectionConfig, TurretConfig
from turret.vision.base import Detector
from turret.vision.detector import RedBlobDetector

_BACKENDS = ("hsv", "tflite", "opencv_dnn")


def build_detector(
    cfg: TurretConfig, *, debug: bool = False, ml_detection: MLDetectionConfig | None = None
) -> Detector:
    """Construct the detector named by cfg.detector_backend.

    Never silently falls back to a different backend: a requested ML
    backend that can't load raises with the exact command to fix it, rather
    than quietly reverting to HSV.

    `ml_detection` overrides `cfg.ml_detection` when given — used to hand
    an ML backend a live-tunable mirror (see turret.live_tuning) instead of
    the static config.
    """
    ml_cfg = ml_detection if ml_detection is not None else cfg.ml_detection
    backend = cfg.detector_backend
    if backend == "hsv":
        return RedBlobDetector(cfg.detection)
    if backend == "tflite":
        from turret.vision.tflite_detector import TFLiteDetector

        return TFLiteDetector(ml_cfg, debug=debug)
    if backend == "opencv_dnn":
        from turret.vision.dnn_detector import DnnDetector

        return DnnDetector(ml_cfg, debug=debug)
    raise ValueError(
        f"Unknown detector_backend {backend!r}; expected one of: {', '.join(_BACKENDS)}"
    )
