"""Configuration dataclasses and YAML loading.

Every tunable lives here; config/default.yaml mirrors these structures and
overrides fields per-section.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)

# src/turret/config.py -> repo root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ServoAxisConfig:
    pin: int = 0
    # SG90-family calibration (see ESP32Servo DEFAULT_uS_LOW/HIGH)
    min_us: int = 500
    max_us: int = 2400
    # Software travel limits — keep the turret from over-driving its mount
    min_angle: float = 20.0
    max_angle: float = 160.0
    start_angle: float = 90.0
    max_deg_per_s: float = 120.0
    invert: bool = False


@dataclass(frozen=True)
class StepperAxisConfig:
    pins: tuple[int, int, int, int] = (0, 0, 0, 0)  # ULN2003 IN1..IN4
    steps_per_rev: int = 4096  # 28BYJ-48 half-stepping
    max_steps_per_s: float = 500.0
    spin_steps_per_s: float = 400.0


@dataclass(frozen=True)
class AxisConfig:
    backend: str = "mock"  # "servo" | "stepper" | "mock"
    servo: ServoAxisConfig = field(default_factory=ServoAxisConfig)
    stepper: StepperAxisConfig = field(default_factory=StepperAxisConfig)


@dataclass(frozen=True)
class CameraConfig:
    backend: str = "auto"  # "auto" | "picamera2" | "opencv"
    device: int | str = 0  # index, or a video file path
    width: int = 640
    height: int = 480
    fps: int = 30


@dataclass(frozen=True)
class DetectionConfig:
    # Red wraps around hue 0 in HSV, so two ranges are OR'd together
    red_low_1: tuple[int, int, int] = (0, 120, 70)
    red_high_1: tuple[int, int, int] = (10, 255, 255)
    red_low_2: tuple[int, int, int] = (170, 120, 70)
    red_high_2: tuple[int, int, int] = (180, 255, 255)
    blur_ksize: int = 5
    morph_ksize: int = 5
    min_area_px: int = 400
    max_aspect: float = 4.0


@dataclass(frozen=True)
class ControlConfig:
    kp_yaw: float = 0.05  # degrees per pixel of error
    kp_pitch: float = 0.05
    deadband_px: int = 15
    max_step_deg: float = 4.0
    lost_target_timeout_s: float = 2.0


@dataclass(frozen=True)
class FireConfig:
    mode: str = "log"  # "log" | "roll_spin" | "servo_pull"
    center_tol_px: int = 25
    min_area_px: int = 3000  # blob area proxy for "in range"
    dwell_s: float = 0.5
    cooldown_s: float = 2.0
    spin_duration_s: float = 1.0  # roll_spin only
    trigger_pull_angle: float = 90.0  # servo_pull only: angle when pulled
    trigger_hold_s: float = 0.15  # servo_pull only: how long to hold before releasing


@dataclass(frozen=True)
class MLDetectionConfig:
    # Paths are resolved against PROJECT_ROOT when relative. Fetch with
    # scripts/download_models.py.
    model_path: str = "models/ssd_mobilenet_v2_coco_quant_postprocess.tflite"
    labels_path: str = "models/coco_labels.txt"  # tflite backend only
    target_classes: tuple[str, ...] = ("person",)
    conf_threshold: float = 0.5
    num_threads: int = 2  # tflite backend only; 4-core Pi, leave headroom for the control loop
    # opencv_dnn backend only (NanoDet architecture constant -- don't change
    # unless swapping in a different model; tflite reads its size from the
    # model instead of this field).
    input_size: int = 416
    # opencv_dnn backend only: NanoDet has no NMS baked into the graph, so
    # it runs cv2.dnn.NMSBoxes itself using this IoU threshold.
    nms_iou_threshold: float = 0.6


def _default_axes() -> dict[str, AxisConfig]:
    return {
        "yaw": AxisConfig(backend="servo", servo=ServoAxisConfig(pin=17)),
        "pitch": AxisConfig(backend="servo", servo=ServoAxisConfig(pin=27)),
        "roll": AxisConfig(
            backend="servo",
            servo=ServoAxisConfig(
                pin=5,
                min_angle=0.0,
                max_angle=90.0,
                start_angle=0.0,
                max_deg_per_s=400.0,
            ),
        ),
    }


@dataclass(frozen=True)
class TurretConfig:
    axes: dict[str, AxisConfig] = field(default_factory=_default_axes)
    camera: CameraConfig = field(default_factory=CameraConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    detector_backend: str = "hsv"  # "hsv" | "tflite" | "opencv_dnn"
    ml_detection: MLDetectionConfig = field(default_factory=MLDetectionConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    fire: FireConfig = field(default_factory=FireConfig)
    allow_mock_fallback: bool = True


def _merge_dataclass(base: Any, data: dict[str, Any]) -> Any:
    """Return a copy of dataclass `base` with fields overridden from `data`."""
    updates: dict[str, Any] = {}
    valid = {f.name: f for f in fields(base)}
    for key, value in data.items():
        if key not in valid:
            log.warning("Unknown config key %r for %s", key, type(base).__name__)
            continue
        current = getattr(base, key)
        if is_dataclass(current) and isinstance(value, dict):
            updates[key] = _merge_dataclass(current, value)
        elif isinstance(current, tuple) and isinstance(value, list):
            updates[key] = tuple(value)
        else:
            updates[key] = value
    return replace(base, **updates)


def _merge_axes(data: dict[str, Any]) -> dict[str, AxisConfig]:
    axes = dict(_default_axes())
    for name, axis_data in data.items():
        base = axes.get(name, AxisConfig())
        axes[name] = _merge_dataclass(base, axis_data)
    return axes


def load_config(path: str | Path | None = None) -> TurretConfig:
    """Load config from YAML, falling back to defaults for anything unset."""
    cfg = TurretConfig()
    if path is None:
        return cfg
    path = Path(path)
    if not path.exists():
        log.warning("Config file %s not found; using defaults", path)
        return cfg
    data = yaml.safe_load(path.read_text()) or {}
    if "axes" in data:
        cfg = replace(cfg, axes=_merge_axes(data.pop("axes")))
    return _merge_dataclass(cfg, data)
