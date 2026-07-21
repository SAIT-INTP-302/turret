"""Axis interface shared by all actuator backends.

Motion model (ported from the ESP32Servo Sweep example): callers set a target
angle, and update(dt) slews the current angle toward it at a bounded rate so
no backend ever slams the mechanism.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

log = logging.getLogger(__name__)


class ContinuousNotSupported(RuntimeError):
    """Raised when spin() is called on an axis that can't rotate continuously."""


class Axis(ABC):
    """A single rotational axis (yaw, pitch, or roll)."""

    def __init__(
        self,
        name: str,
        *,
        min_angle: float,
        max_angle: float,
        start_angle: float,
        max_deg_per_s: float,
    ) -> None:
        self.name = name
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.max_deg_per_s = max_deg_per_s
        self._current = self._clamp(start_angle)
        self._target = self._current

    def _clamp(self, deg: float) -> float:
        return min(max(deg, self.min_angle), self.max_angle)

    def set_target(self, deg: float) -> None:
        self._target = self._clamp(deg)

    @property
    def target(self) -> float:
        return self._target

    def angle(self) -> float:
        """Last commanded angle in degrees."""
        return self._current

    def update(self, dt: float) -> None:
        """Slew toward the target by at most max_deg_per_s * dt."""
        error = self._target - self._current
        if error == 0.0:
            return
        step = self.max_deg_per_s * dt
        if abs(error) <= step:
            self._current = self._target
        else:
            self._current += step if error > 0 else -step
        self._apply(self._current)

    @abstractmethod
    def _apply(self, deg: float) -> None:
        """Drive the hardware to `deg`."""

    def spin(self, forward: bool = True) -> None:
        """Rotate continuously (used to fire via barrel rotation)."""
        raise ContinuousNotSupported(f"axis {self.name!r} cannot spin continuously")

    def stop_spin(self) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        """Detach / de-energize the actuator."""
