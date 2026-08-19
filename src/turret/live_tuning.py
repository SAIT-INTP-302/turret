"""Live-tunable subset of TurretConfig, adjustable from the dashboard while
the turret is running.

Deliberately narrow: never includes fire.mode or axis/servo calibration —
those change what the turret is capable of doing, not how sensitive it is,
and stay a deliberate config-file + restart action.

Persistence is opt-in and explicit: nothing here ever touches
config/default.yaml. A Save writes the current values to a separate,
gitignored override file (DEFAULT_TUNING_PATH), which is loaded back on top
of the real config the next time a LiveTuning is constructed. "Reset"
always means "back to config/default.yaml's value" (captured before the
override file is applied), never "back to what I last saved."
"""

from __future__ import annotations

import dataclasses
import threading
from pathlib import Path
from typing import Any

import yaml

from turret.config import PROJECT_ROOT, TurretConfig

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

DEFAULT_TUNING_PATH = PROJECT_ROOT / "config" / "tuning.local.yaml"


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
        self._defaults = self._snapshot_locked()

    def update(self, **values: float) -> dict[str, float]:
        with self._lock:
            for name, value in values.items():
                if name not in self._bounds:
                    raise ValueError(f"{name!r} is not a live-tunable setting")
                lo, hi = self._bounds[name]
                setattr(self, name, min(max(float(value), lo), hi))
            return self._snapshot_locked()

    def reset(self, name: str | None = None) -> dict[str, float]:
        with self._lock:
            names = [name] if name is not None else list(self._bounds)
            for n in names:
                if n not in self._bounds:
                    raise ValueError(f"{n!r} is not a live-tunable setting")
                setattr(self, n, self._defaults[n])
            return self._snapshot_locked()

    def snapshot(self) -> dict[str, float]:
        with self._lock:
            return self._snapshot_locked()

    def defaults(self) -> dict[str, float]:
        return dict(self._defaults)

    def _snapshot_locked(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in self._bounds}


class LiveTuning:
    """Bundles the three tunable mirrors the control loop reads from, plus
    one flat get/update/reset/save surface for the dashboard API."""

    def __init__(
        self, cfg: TurretConfig, *, override_path: Path | None = DEFAULT_TUNING_PATH
    ) -> None:
        self.control = LiveTunable(cfg.control, CONTROL_BOUNDS)
        self.fire = LiveTunable(cfg.fire, FIRE_BOUNDS)
        self.ml_detection = LiveTunable(cfg.ml_detection, ML_BOUNDS)
        self._by_name: dict[str, LiveTunable] = {
            **dict.fromkeys(CONTROL_BOUNDS, self.control),
            **dict.fromkeys(FIRE_BOUNDS, self.fire),
            **dict.fromkeys(ML_BOUNDS, self.ml_detection),
        }
        self._override_path = override_path
        if override_path is not None and override_path.exists():
            data = yaml.safe_load(override_path.read_text()) or {}
            saved = {k: v for k, v in data.items() if k in self._by_name}
            if saved:
                self.update(**saved)

    def snapshot(self) -> dict[str, float]:
        out: dict[str, float] = {}
        out.update(self.control.snapshot())
        out.update(self.fire.snapshot())
        out.update(self.ml_detection.snapshot())
        return out

    def defaults(self) -> dict[str, float]:
        out: dict[str, float] = {}
        out.update(self.control.defaults())
        out.update(self.fire.defaults())
        out.update(self.ml_detection.defaults())
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

    def reset(self, name: str | None = None) -> dict[str, float]:
        if name is not None:
            target = self._by_name.get(name)
            if target is None:
                raise ValueError(f"{name!r} is not a live-tunable setting")
            target.reset(name)
        else:
            self.control.reset()
            self.fire.reset()
            self.ml_detection.reset()
        return self.snapshot()

    def save(self) -> dict[str, float]:
        if self._override_path is None:
            raise RuntimeError("no tuning file configured; cannot save")
        snapshot = self.snapshot()
        self._override_path.parent.mkdir(parents=True, exist_ok=True)
        self._override_path.write_text(yaml.safe_dump(snapshot, sort_keys=True))
        return snapshot
