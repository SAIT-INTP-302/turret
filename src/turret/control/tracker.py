"""Proportional tracking: pixel error from frame center -> yaw/pitch targets."""

from __future__ import annotations

import logging

from turret.actuators.base import Axis
from turret.config import ControlConfig
from turret.vision.types import Detection

log = logging.getLogger(__name__)


def _clamp(v: float, limit: float) -> float:
    return min(max(v, -limit), limit)


class Tracker:
    def __init__(
        self,
        cfg: ControlConfig,
        frame_size: tuple[int, int],
        yaw: Axis,
        pitch: Axis,
    ) -> None:
        self._cfg = cfg
        self._cx = frame_size[0] / 2
        self._cy = frame_size[1] / 2
        self._yaw = yaw
        self._pitch = pitch
        self._lost_s = 0.0

    def update(self, det: Detection | None, dt: float) -> None:
        cfg = self._cfg
        if det is None:
            self._lost_s += dt
            if self._lost_s >= cfg.lost_target_timeout_s:
                # Give up and return to center to widen the search view
                self._yaw.set_target(90.0)
                self._pitch.set_target(90.0)
            return
        self._lost_s = 0.0

        # Target right of center -> positive error -> yaw right (increase
        # angle); image y grows downward -> pitch down. Physical direction
        # is corrected per-axis with the `invert` config flag.
        err_x = det.cx - self._cx
        err_y = det.cy - self._cy
        if abs(err_x) > cfg.deadband_px:
            delta = _clamp(cfg.kp_yaw * err_x, cfg.max_step_deg)
            self._yaw.set_target(self._yaw.angle() + delta)
        if abs(err_y) > cfg.deadband_px:
            delta = _clamp(cfg.kp_pitch * err_y, cfg.max_step_deg)
            self._pitch.set_target(self._pitch.angle() + delta)
