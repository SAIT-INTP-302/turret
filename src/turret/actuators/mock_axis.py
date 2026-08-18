"""Simulated axis for development machines without GPIO hardware."""

from __future__ import annotations

import logging

from turret.actuators.base import Axis

log = logging.getLogger(__name__)


class MockAxis(Axis):
    supports_spin = True

    def __init__(
        self,
        name: str,
        *,
        min_angle: float = 0.0,
        max_angle: float = 180.0,
        start_angle: float = 90.0,
        max_deg_per_s: float = 120.0,
    ) -> None:
        super().__init__(
            name,
            min_angle=min_angle,
            max_angle=max_angle,
            start_angle=start_angle,
            max_deg_per_s=max_deg_per_s,
        )
        self.spinning = False
        self.closed = False

    def _apply(self, deg: float) -> None:
        log.debug("mock axis %s -> %.1f deg (target %.1f)", self.name, deg, self._target)

    def spin(self, forward: bool = True) -> None:
        self.spinning = True
        log.info("mock axis %s spinning %s", self.name, "forward" if forward else "reverse")

    def stop_spin(self) -> None:
        self.spinning = False
        log.info("mock axis %s spin stopped", self.name)

    def close(self) -> None:
        self.closed = True
        log.debug("mock axis %s closed", self.name)
