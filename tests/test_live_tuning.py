import pytest

from turret.actuators.mock_axis import MockAxis
from turret.config import ControlConfig, TurretConfig
from turret.control.tracker import Tracker
from turret.live_tuning import (
    CONTROL_BOUNDS,
    FIRE_BOUNDS,
    ML_BOUNDS,
    LiveTunable,
    LiveTuning,
)
from turret.vision.types import Detection


def test_live_tunable_clamps_to_bounds():
    lt = LiveTunable(ControlConfig(), CONTROL_BOUNDS)
    lt.update(deadband_px=10_000.0)
    assert lt.deadband_px == CONTROL_BOUNDS["deadband_px"][1]
    lt.update(deadband_px=-50.0)
    assert lt.deadband_px == CONTROL_BOUNDS["deadband_px"][0]


def test_live_tunable_rejects_unknown_field():
    lt = LiveTunable(ControlConfig(), CONTROL_BOUNDS)
    with pytest.raises(ValueError):
        lt.update(max_deg_per_s=999.0)  # a real ControlConfig field, but not in bounds


def test_live_tunable_copies_non_tunable_fields_and_leaves_them_readable():
    cfg = ControlConfig(max_step_deg=4.0, lost_target_timeout_s=2.0)
    lt = LiveTunable(cfg, CONTROL_BOUNDS)
    assert lt.max_step_deg == 4.0
    assert lt.lost_target_timeout_s == 2.0


def test_live_tunable_snapshot_only_includes_bounded_fields():
    lt = LiveTunable(ControlConfig(), CONTROL_BOUNDS)
    snap = lt.snapshot()
    assert set(snap) == set(CONTROL_BOUNDS)


def test_live_tuning_snapshot_covers_all_eight():
    lt = LiveTuning(TurretConfig())
    snap = lt.snapshot()
    assert set(snap) == set(CONTROL_BOUNDS) | set(FIRE_BOUNDS) | set(ML_BOUNDS)


def test_live_tuning_update_routes_mixed_fields_to_the_right_mirrors():
    lt = LiveTuning(TurretConfig())
    result = lt.update(kp_yaw=0.2, min_area_px=1234.0)
    assert result["kp_yaw"] == 0.2
    assert result["min_area_px"] == 1234.0
    assert lt.control.kp_yaw == 0.2
    assert lt.fire.min_area_px == 1234.0


def test_live_tuning_update_rejects_unknown_name_without_partial_apply():
    lt = LiveTuning(TurretConfig())
    before = lt.snapshot()
    with pytest.raises(ValueError):
        lt.update(kp_yaw=0.9, bogus_field=1.0)
    assert lt.snapshot() == before  # nothing applied


def test_live_tunable_is_a_drop_in_for_tracker():
    cfg = ControlConfig(deadband_px=15.0, kp_yaw=0.05, kp_pitch=0.05, max_step_deg=4.0)
    live = LiveTunable(cfg, CONTROL_BOUNDS)
    yaw = MockAxis("yaw")
    pitch = MockAxis("pitch")
    tracker = Tracker(live, (640, 480), yaw, pitch)

    live.update(deadband_px=0.0)  # tighten the deadband after construction
    det = Detection(cx=325, cy=240, bbox=(300, 220, 40, 40), area=1600)  # 5px off-center
    tracker.update(det, dt=1.0)
    assert yaw.target != 90.0  # moved, because the *live* (tightened) deadband is honored
