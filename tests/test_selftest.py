from dataclasses import replace

import cv2
import numpy as np
import pytest

from turret.actuators.base import Axis
from turret.actuators.mock_axis import MockAxis
from turret.config import CameraConfig, FireConfig, TurretConfig
from turret.selftest import format_report, run_selftest


class FakeHardwareAxis(Axis):
    """Stands in for a real (non-mock) axis without needing GPIO hardware."""

    def __init__(self, name: str, **kwargs) -> None:
        super().__init__(
            name,
            min_angle=kwargs.pop("min_angle", 0.0),
            max_angle=kwargs.pop("max_angle", 180.0),
            start_angle=kwargs.pop("start_angle", 90.0),
            max_deg_per_s=kwargs.pop("max_deg_per_s", 1000.0),
        )

    def _apply(self, deg: float) -> None:
        pass

    def close(self) -> None:
        pass


@pytest.fixture
def clip(tmp_path):
    path = str(tmp_path / "clip.avi")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    assert writer.isOpened()
    for _ in range(10):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()
    return path


def working_cfg(clip) -> TurretConfig:
    cfg = TurretConfig()
    return replace(cfg, camera=CameraConfig(backend="opencv", device=clip))


def test_all_mock_axes_warn_but_ok(clip):
    report = run_selftest(working_cfg(clip), force_mock=True)
    axis_results = {r.name: r for r in report.results if r.name in ("yaw", "pitch", "roll")}
    assert set(axis_results) == {"yaw", "pitch", "roll"}
    assert all(r.status == "warn" for r in axis_results.values())
    assert report.ok
    assert report.exit_code == 0


def test_camera_failure_is_fail_but_other_checks_still_run(clip):
    cfg = replace(
        working_cfg(clip), camera=CameraConfig(backend="opencv", device="/nonexistent/x.avi")
    )
    report = run_selftest(cfg, force_mock=True)
    by_name = {r.name: r for r in report.results}
    assert by_name["camera"].status == "fail"
    assert by_name["detector"].status == "pass"  # falls back to a synthetic blank frame
    assert not report.ok
    assert report.exit_code == 1


def test_roll_spin_with_mock_axis_passes(clip):
    cfg = replace(working_cfg(clip), fire=FireConfig(mode="roll_spin"))
    report = run_selftest(cfg, force_mock=True)
    fc = next(r for r in report.results if r.name == "fire_control")
    assert fc.status == "pass"


def test_roll_spin_with_non_spinning_axis_fails(clip, monkeypatch):
    def fake_build_axes(cfg, *, force_mock=False):
        yaw = MockAxis("yaw")
        pitch = MockAxis("pitch")
        roll = MockAxis("roll")
        roll.supports_spin = False  # simulate a real ServoAxis
        return {"yaw": yaw, "pitch": pitch, "roll": roll}

    monkeypatch.setattr("turret.selftest.build_axes", fake_build_axes)
    cfg = replace(working_cfg(clip), fire=FireConfig(mode="roll_spin"))
    report = run_selftest(cfg)
    fc = next(r for r in report.results if r.name == "fire_control")
    assert fc.status == "fail"
    assert "continuous-rotation" in fc.detail


def test_trigger_axis_never_actuated_by_default(clip, monkeypatch):
    calls: list[str] = []

    class SpyAxis(FakeHardwareAxis):
        def set_target(self, deg: float) -> None:
            calls.append("set_target")
            super().set_target(deg)

        def spin(self, forward: bool = True) -> None:
            calls.append("spin")
            super().spin(forward)

    def fake_build_axes(cfg, *, force_mock=False):
        return {"yaw": MockAxis("yaw"), "pitch": MockAxis("pitch"), "roll": SpyAxis("roll")}

    monkeypatch.setattr("turret.selftest.build_axes", fake_build_axes)
    run_selftest(working_cfg(clip), actuate_axes=True, actuate_trigger=False)
    assert calls == []


def test_actuate_trigger_opt_in_moves_the_axis(clip, monkeypatch):
    calls: list[str] = []

    class SpyAxis(FakeHardwareAxis):
        def set_target(self, deg: float) -> None:
            calls.append("set_target")
            super().set_target(deg)

    def fake_build_axes(cfg, *, force_mock=False):
        return {"yaw": MockAxis("yaw"), "pitch": MockAxis("pitch"), "roll": SpyAxis("roll")}

    monkeypatch.setattr("turret.selftest.build_axes", fake_build_axes)
    run_selftest(working_cfg(clip), actuate_axes=True, actuate_trigger=True)
    assert "set_target" in calls


def test_axes_build_failure_still_runs_camera_and_detector(clip, monkeypatch):
    def raising_build_axes(cfg, *, force_mock=False):
        raise RuntimeError("pigpio daemon not running")

    monkeypatch.setattr("turret.selftest.build_axes", raising_build_axes)
    report = run_selftest(working_cfg(clip))
    by_name = {r.name: r for r in report.results}
    assert by_name["axes"].status == "fail"
    assert "pigpio" in by_name["axes"].detail
    assert by_name["camera"].status == "pass"
    assert by_name["detector"].status == "pass"
    assert "fire_control" not in by_name  # nothing to check without axes
    assert not report.ok


def test_format_report_includes_summary(clip):
    report = run_selftest(working_cfg(clip), force_mock=True)
    text = format_report(report)
    assert "passed" in text and "warned" in text and "failed" in text
