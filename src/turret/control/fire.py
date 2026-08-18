"""Fire decision and fire actuation.

FireDecider says *when* to fire (target centered, big enough, held long
enough, cooldown elapsed). FireControl says *how*: log-only stub, spinning
the roll axis (a stepper-driven barrel that fires by rotating), or pulling a
trigger servo to a fire angle and releasing it back to rest.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from turret.actuators.base import Axis
from turret.config import FireConfig
from turret.vision.types import Detection

log = logging.getLogger(__name__)


class FireControl(ABC):
    @abstractmethod
    def fire(self) -> None: ...

    def update(self, dt: float) -> None:
        """Advance time-based behavior (e.g. spin-down)."""


class LogFireControl(FireControl):
    def fire(self) -> None:
        log.warning("FIRE!")


class RollSpinFireControl(FireControl):
    def __init__(self, roll: Axis, cfg: FireConfig) -> None:
        self._roll = roll
        self._cfg = cfg
        self._spin_left_s = 0.0

    def fire(self) -> None:
        log.warning("FIRE! (spinning roll axis for %.1fs)", self._cfg.spin_duration_s)
        self._roll.spin(forward=True)
        self._spin_left_s = self._cfg.spin_duration_s

    def update(self, dt: float) -> None:
        if self._spin_left_s > 0.0:
            self._spin_left_s -= dt
            if self._spin_left_s <= 0.0:
                self._roll.stop_spin()


class ServoPullFireControl(FireControl):
    def __init__(self, roll: Axis, cfg: FireConfig) -> None:
        self._roll = roll
        self._cfg = cfg
        self._rest_angle = roll.target
        self._hold_left_s = 0.0

    def fire(self) -> None:
        log.warning("FIRE! (pulling trigger servo to %.1f deg)", self._cfg.trigger_pull_angle)
        self._roll.set_target(self._cfg.trigger_pull_angle)
        self._hold_left_s = self._cfg.trigger_hold_s

    def update(self, dt: float) -> None:
        if self._hold_left_s > 0.0:
            self._hold_left_s -= dt
            if self._hold_left_s <= 0.0:
                self._roll.set_target(self._rest_angle)


def make_fire_control(cfg: FireConfig, roll: Axis | None) -> FireControl:
    if cfg.mode == "roll_spin":
        if roll is None:
            raise ValueError("fire mode 'roll_spin' requires a roll axis")
        if not roll.supports_spin:
            raise ValueError(
                f"fire mode 'roll_spin' requires a continuous-rotation axis, but "
                f"roll axis ({type(roll).__name__}) doesn't support spin()"
            )
        return RollSpinFireControl(roll, cfg)
    if cfg.mode == "servo_pull":
        if roll is None:
            raise ValueError("fire mode 'servo_pull' requires a roll axis")
        return ServoPullFireControl(roll, cfg)
    if cfg.mode == "log":
        return LogFireControl()
    raise ValueError(f"Unknown fire mode {cfg.mode!r}")


class FireDecider:
    def __init__(self, cfg: FireConfig, frame_size: tuple[int, int]) -> None:
        self._cfg = cfg
        self._cx = frame_size[0] / 2
        self._cy = frame_size[1] / 2
        self._lock_s = 0.0
        self._cooldown_s = 0.0

    def locked(self, det: Detection | None) -> bool:
        if det is None:
            return False
        return (
            abs(det.cx - self._cx) <= self._cfg.center_tol_px
            and abs(det.cy - self._cy) <= self._cfg.center_tol_px
            and det.area >= self._cfg.min_area_px
        )

    def update(self, det: Detection | None, dt: float) -> bool:
        """Advance timers; True means fire now."""
        self._cooldown_s = max(0.0, self._cooldown_s - dt)
        if not self.locked(det):
            self._lock_s = 0.0
            return False
        self._lock_s += dt
        if self._lock_s >= self._cfg.dwell_s and self._cooldown_s == 0.0:
            self._lock_s = 0.0
            self._cooldown_s = self._cfg.cooldown_s
            return True
        return False
