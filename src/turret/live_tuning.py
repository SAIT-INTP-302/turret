"""Live-tunable subset of TurretConfig, adjustable from the dashboard while
the turret is running.

Deliberately narrow and session-only: never touches config/default.yaml,
and never includes fire.mode or axis/servo calibration — those change what
the turret is capable of doing, not how sensitive it is, and stay a
deliberate config-file + restart action.
"""

from __future__ import annotations

import dataclasses
import threading
from typing import Any

from turret.config import TurretConfig

CONTROL_BOUNDS: dict[str, tuple[float, float]] = {
    "kp_yaw": (0.0, 1.0),
    "kp_pitch": (0.0, 1.0),
    "deadband_px": (0.0, 200.0),
}
FIRE_BOUNDS: dict[str, tuple[float, float]] = {
    "center_tol_px": (0.0, 300.0),
    "min_area_px": (0.0, 500_000.0),
    "dwell_s": (0.0, 10.0),
    "cooldown_s": (0.0, 30.0),
}
ML_BOUNDS: dict[str, tuple[float, float]] = {
    "conf_threshold": (0.0, 1.0),
}

ALL_BOUNDS: dict[str, tuple[float, float]] = {**CONTROL_BOUNDS, **FIRE_BOUNDS, **ML_BOUNDS}


class LiveTunable:
    """Thread-safe mutable mirror of a frozen config dataclass.

    Every field of `cfg` becomes a plain attribute, so this is a drop-in
    substitute anywhere the frozen dataclass was used (Tracker, FireDecider,
    the ML detectors don't need to know this isn't the real thing). Only
    the fields named in `bounds` can change after construction, and every
    write is clamped to its bound.
    """

    def __init__(self, cfg: Any, bounds: dict[str, tuple[float, float]]) -> None:
        self._lock = threading.Lock()
        self._bounds = bounds
        for field in dataclasses.fields(cfg):
            setattr(self, field.name, getattr(cfg, field.name))

    def update(self, **values: float) -> dict[str, float]:
        with self._lock:
            for name, value in values.items():
                if name not in self._bounds:
                    raise ValueError(f"{name!r} is not a live-tunable setting")
                lo, hi = self._bounds[name]
                setattr(self, name, min(max(float(value), lo), hi))
            return self._snapshot_locked()

    def snapshot(self) -> dict[str, float]:
        with self._lock:
            return self._snapshot_locked()

    def _snapshot_locked(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in self._bounds}


class LiveTuning:
    """Bundles the three tunable mirrors the control loop reads from, plus
    one flat get/update surface for the dashboard API."""

    def __init__(self, cfg: TurretConfig) -> None:
        self.control = LiveTunable(cfg.control, CONTROL_BOUNDS)
        self.fire = LiveTunable(cfg.fire, FIRE_BOUNDS)
        self.ml_detection = LiveTunable(cfg.ml_detection, ML_BOUNDS)
        self._by_name: dict[str, LiveTunable] = {
            **dict.fromkeys(CONTROL_BOUNDS, self.control),
            **dict.fromkeys(FIRE_BOUNDS, self.fire),
            **dict.fromkeys(ML_BOUNDS, self.ml_detection),
        }

    def snapshot(self) -> dict[str, float]:
        out: dict[str, float] = {}
        out.update(self.control.snapshot())
        out.update(self.fire.snapshot())
        out.update(self.ml_detection.snapshot())
        return out

    def update(self, **values: float) -> dict[str, float]:
        by_target: dict[int, tuple[LiveTunable, dict[str, float]]] = {}
        for name, value in values.items():
            target = self._by_name.get(name)
            if target is None:
                raise ValueError(f"{name!r} is not a live-tunable setting")
            _, kv = by_target.setdefault(id(target), (target, {}))
            kv[name] = value
        for target, kv in by_target.values():
            target.update(**kv)
        return self.snapshot()
