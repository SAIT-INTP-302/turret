"""Pre-run system self-test: exercise every subsystem and report pass/fail.

Checks axes, camera, detector, and fire-control construction against the
exact TurretConfig a real run would use, without ever calling
FireControl.fire() and without ever actuating the trigger axis (the "roll"
axis, whichever fire mechanism it drives) unless explicitly asked to.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from turret.actuators.base import Axis
from turret.actuators.factory import build_axes
from turret.actuators.mock_axis import MockAxis
from turret.camera.factory import open_camera
from turret.config import TurretConfig
from turret.control.fire import make_fire_control
from turret.vision.factory import build_detector

log = logging.getLogger(__name__)

TRIGGER_AXIS_NAME = "roll"


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str  # "pass" | "warn" | "fail"
    detail: str


@dataclass(frozen=True)
class SelftestReport:
    results: list[CheckResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(r.status == "fail" for r in self.results)

    @property
    def exit_code(self) -> int:
        return 0 if self.ok else 1


def _nudge_test(axis: Axis, *, timeout_s: float = 3.0) -> CheckResult:
    """Move `axis` a small bounded amount and back; confirm it actually moved."""
    start = axis.angle()
    span = axis.max_angle - axis.min_angle
    offset = min(5.0, span / 4) if span > 0 else 0.0
    target = start + offset if start + offset <= axis.max_angle else start - offset

    def _settle(goal: float) -> float:
        axis.set_target(goal)
        deadline = time.monotonic() + timeout_s
        dt = 0.02
        while time.monotonic() < deadline and axis.angle() != axis.target:
            axis.update(dt)
            time.sleep(dt)
        return axis.angle()

    try:
        reached = _settle(target)
        moved = abs(reached - start) > 0.01
        _settle(start)  # always try to return to the original position
        if not moved:
            return CheckResult(
                axis.name, "warn", "didn't reach target — check wiring/power"
            )
        return CheckResult(axis.name, "pass", f"moved {start:.1f} -> {reached:.1f} -> rest")
    except Exception as exc:  # noqa: BLE001 - report, don't crash the run
        return CheckResult(axis.name, "fail", f"{type(exc).__name__}: {exc}")


def _check_axes(
    cfg: TurretConfig,
    *,
    force_mock: bool,
    actuate_axes: bool,
    actuate_trigger: bool,
) -> tuple[list[CheckResult], dict[str, Axis] | None]:
    try:
        axes = build_axes(cfg, force_mock=force_mock)
    except Exception as exc:  # noqa: BLE001
        return [CheckResult("axes", "fail", f"{type(exc).__name__}: {exc}")], None

    results: list[CheckResult] = []
    try:
        for name, axis in axes.items():
            if isinstance(axis, MockAxis):
                results.append(CheckResult(name, "warn", "mock axis (no real hardware)"))
            elif name == TRIGGER_AXIS_NAME and not actuate_trigger:
                results.append(
                    CheckResult(
                        name,
                        "pass",
                        "hardware axis constructed (motion test skipped — trigger "
                        "axis; pass --actuate-trigger to test motion)",
                    )
                )
            elif not actuate_axes:
                results.append(
                    CheckResult(name, "pass", "hardware axis constructed (motion test skipped)")
                )
            else:
                results.append(_nudge_test(axis))
    finally:
        for axis in axes.values():
            axis.close()
    return results, axes


def _check_camera(cfg: TurretConfig, *, frames: int) -> tuple[CheckResult, np.ndarray | None]:
    try:
        camera = open_camera(cfg.camera)
    except Exception as exc:  # noqa: BLE001
        return CheckResult("camera", "fail", f"{type(exc).__name__}: {exc}"), None

    last_good: np.ndarray | None = None
    good = 0
    try:
        for _ in range(frames):
            frame = camera.read()
            if frame is not None:
                good += 1
                last_good = frame
        w, h = camera.resolution
    finally:
        camera.close()

    if good == 0:
        return CheckResult("camera", "fail", f"0/{frames} frames read"), None
    if good < frames:
        return (
            CheckResult("camera", "warn", f"{frames - good}/{frames} frames dropped"),
            last_good,
        )
    return CheckResult("camera", "pass", f"{good}/{frames} frames ok ({w}x{h})"), last_good


def _check_detector(cfg: TurretConfig, frame: np.ndarray | None) -> CheckResult:
    try:
        detector = build_detector(cfg, debug=False)
        if frame is None:
            frame = np.zeros((cfg.camera.height, cfg.camera.width, 3), dtype=np.uint8)
        det, _ = detector.detect(frame)
        note = "target seen" if det is not None else "no target in frame (ok)"
        return CheckResult("detector", "pass", f"{cfg.detector_backend}: {note}")
    except Exception as exc:  # noqa: BLE001
        return CheckResult("detector", "fail", f"{type(exc).__name__}: {exc}")


def _check_fire_control(cfg: TurretConfig, axes: dict[str, Axis] | None) -> CheckResult:
    roll = axes.get(TRIGGER_AXIS_NAME) if axes is not None else None
    try:
        fc = make_fire_control(cfg.fire, roll)
        return CheckResult("fire_control", "pass", f"mode={cfg.fire.mode} ({type(fc).__name__})")
    except Exception as exc:  # noqa: BLE001
        return CheckResult("fire_control", "fail", f"{type(exc).__name__}: {exc}")


def run_selftest(
    cfg: TurretConfig,
    *,
    force_mock: bool = False,
    actuate_axes: bool = True,
    actuate_trigger: bool = False,
    camera_frames: int = 5,
) -> SelftestReport:
    results: list[CheckResult] = []

    axis_results, axes = _check_axes(
        cfg, force_mock=force_mock, actuate_axes=actuate_axes, actuate_trigger=actuate_trigger
    )
    results.extend(axis_results)

    camera_result, frame = _check_camera(cfg, frames=camera_frames)
    results.append(camera_result)

    results.append(_check_detector(cfg, frame))

    if axes is not None:
        results.append(_check_fire_control(cfg, axes))

    return SelftestReport(results)


def format_report(report: SelftestReport) -> str:
    lines = []
    for r in report.results:
        lines.append(f"[{r.status.upper():4}] {r.name}: {r.detail}")
    passed = sum(1 for r in report.results if r.status == "pass")
    warned = sum(1 for r in report.results if r.status == "warn")
    failed = sum(1 for r in report.results if r.status == "fail")
    lines.append(f"{passed} passed, {warned} warned, {failed} failed")
    return "\n".join(lines)
