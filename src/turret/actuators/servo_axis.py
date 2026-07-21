"""Hobby-servo axis driven through pigpio.

Port of the ESP32Servo Arduino library's Servo class. The LEDC timer/tick
math is dropped entirely — pigpio.set_servo_pulsewidth() takes microseconds
directly and produces hardware-timed 50 Hz pulses.
"""

from __future__ import annotations

import logging

from turret.actuators.base import Axis
from turret.config import ServoAxisConfig

log = logging.getLogger(__name__)

# Hard pulse-width limits enforced by ESP32Servo (MIN/MAX_PULSE_WIDTH)
MIN_PULSE_US = 500
MAX_PULSE_US = 2500


def angle_to_us(angle: float, min_us: int, max_us: int) -> int:
    """Map an angle in [0, 180] to a pulse width, clamping like ESP32Servo."""
    angle = min(max(angle, 0.0), 180.0)
    us = min_us + (angle / 180.0) * (max_us - min_us)
    return round(min(max(us, MIN_PULSE_US), MAX_PULSE_US))


def us_to_angle(us: int, min_us: int, max_us: int) -> float:
    """Inverse of angle_to_us for the calibrated range."""
    us = min(max(us, min_us), max_us)
    return (us - min_us) / (max_us - min_us) * 180.0


class ServoAxis(Axis):
    """SG92R-style servo on a GPIO pin, pulsed by the pigpio daemon."""

    def __init__(self, pi, name: str, cfg: ServoAxisConfig) -> None:
        super().__init__(
            name,
            min_angle=cfg.min_angle,
            max_angle=cfg.max_angle,
            start_angle=cfg.start_angle,
            max_deg_per_s=cfg.max_deg_per_s,
        )
        self._pi = pi
        self._cfg = cfg
        self._last_us = 0
        self.attach()

    def attach(self) -> None:
        self._apply(self._current)

    def _apply(self, deg: float) -> None:
        if self._cfg.invert:
            deg = 180.0 - deg
        self.write_microseconds(angle_to_us(deg, self._cfg.min_us, self._cfg.max_us))

    def write_microseconds(self, us: int) -> None:
        us = min(max(us, MIN_PULSE_US), MAX_PULSE_US)
        self._pi.set_servo_pulsewidth(self._cfg.pin, us)
        self._last_us = us

    def read_microseconds(self) -> int:
        return self._last_us

    def detach(self) -> None:
        # Pulse width 0 stops the pulse train (ESP32Servo release())
        self._pi.set_servo_pulsewidth(self._cfg.pin, 0)
        self._last_us = 0

    def close(self) -> None:
        self.detach()
