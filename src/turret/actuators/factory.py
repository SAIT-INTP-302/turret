"""Build Axis instances from config.

This is the only module that decides whether real GPIO hardware is available.
Everything else works with the Axis interface.
"""

from __future__ import annotations

import logging

from turret.actuators.base import Axis
from turret.actuators.mock_axis import MockAxis
from turret.config import AxisConfig, TurretConfig

log = logging.getLogger(__name__)


def _connect_pigpio():
    try:
        import pigpio
    except ImportError:
        log.warning("pigpio not installed; hardware axes unavailable")
        return None
    pi = pigpio.pi()
    if not pi.connected:
        log.warning("pigpio daemon not running (start with: sudo systemctl start pigpiod)")
        return None
    return pi


def _mock_for(name: str, cfg: AxisConfig) -> MockAxis:
    if cfg.backend == "servo":
        s = cfg.servo
        return MockAxis(
            name,
            min_angle=s.min_angle,
            max_angle=s.max_angle,
            start_angle=s.start_angle,
            max_deg_per_s=s.max_deg_per_s,
        )
    return MockAxis(name)


def build_axes(cfg: TurretConfig, *, force_mock: bool = False) -> dict[str, Axis]:
    needs_hw = not force_mock and any(
        ax.backend in ("servo", "stepper") for ax in cfg.axes.values()
    )
    pi = _connect_pigpio() if needs_hw else None
    if needs_hw and pi is None:
        if not cfg.allow_mock_fallback:
            raise RuntimeError(
                "GPIO hardware unavailable and allow_mock_fallback is false. "
                "Install pigpio and start pigpiod, or run with --mock."
            )
        log.warning("Falling back to mock axes for all hardware backends")
        force_mock = True

    axes: dict[str, Axis] = {}
    for name, axis_cfg in cfg.axes.items():
        if force_mock or axis_cfg.backend == "mock":
            axes[name] = _mock_for(name, axis_cfg)
        elif axis_cfg.backend == "servo":
            from turret.actuators.servo_axis import ServoAxis

            axes[name] = ServoAxis(pi, name, axis_cfg.servo)
        elif axis_cfg.backend == "stepper":
            from turret.actuators.stepper_axis import StepperAxis

            axes[name] = StepperAxis(pi, name, axis_cfg.stepper)
        else:
            raise ValueError(f"Unknown axis backend {axis_cfg.backend!r} for {name!r}")
    return axes
