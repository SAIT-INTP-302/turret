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
    lt = LiveTuning(TurretConfig(), override_path=None)
    snap = lt.snapshot()
    assert set(snap) == set(CONTROL_BOUNDS) | set(FIRE_BOUNDS) | set(ML_BOUNDS)


def test_live_tuning_update_routes_mixed_fields_to_the_right_mirrors():
    lt = LiveTuning(TurretConfig(), override_path=None)
    result = lt.update(kp_yaw=0.2, min_area_px=1234.0)
    assert result["kp_yaw"] == 0.2
    assert result["min_area_px"] == 1234.0
    assert lt.control.kp_yaw == 0.2
    assert lt.fire.min_area_px == 1234.0


def test_live_tuning_update_rejects_unknown_name_without_partial_apply():
    lt = LiveTuning(TurretConfig(), override_path=None)
    before = lt.snapshot()
    with pytest.raises(ValueError):
        lt.update(kp_yaw=0.9, bogus_field=1.0)
    assert lt.snapshot() == before  # nothing applied


def test_live_tunable_reset_one_field_leaves_others_untouched():
    lt = LiveTunable(ControlConfig(kp_yaw=0.05, kp_pitch=0.05, deadband_px=15.0), CONTROL_BOUNDS)
    lt.update(kp_yaw=0.9, kp_pitch=0.9)
    lt.reset("kp_yaw")
    assert lt.kp_yaw == 0.05
    assert lt.kp_pitch == 0.9  # untouched


def test_live_tunable_reset_all_restores_every_bounded_field():
    lt = LiveTunable(ControlConfig(kp_yaw=0.05, kp_pitch=0.05, deadband_px=15.0), CONTROL_BOUNDS)
    lt.update(kp_yaw=0.9, kp_pitch=0.9, deadband_px=100.0)
    lt.reset()
    assert lt.snapshot() == {"kp_yaw": 0.05, "kp_pitch": 0.05, "deadband_px": 15.0}


def test_live_tunable_reset_rejects_unknown_field():
    lt = LiveTunable(ControlConfig(), CONTROL_BOUNDS)
    with pytest.raises(ValueError):
        lt.reset("bogus_field")


def test_live_tunable_defaults_reflect_construction_not_current():
    lt = LiveTunable(ControlConfig(kp_yaw=0.05), CONTROL_BOUNDS)
    lt.update(kp_yaw=0.9)
    assert lt.defaults()["kp_yaw"] == 0.05
    assert lt.kp_yaw == 0.9


def test_live_tuning_reset_routes_to_the_right_mirror():
    lt = LiveTuning(TurretConfig(), override_path=None)
    lt.update(kp_yaw=0.9, min_area_px=99999.0)
    lt.reset("kp_yaw")
    assert lt.control.kp_yaw == TurretConfig().control.kp_yaw
    assert lt.fire.min_area_px == 99999.0  # untouched


def test_live_tuning_reset_all_with_no_name():
    lt = LiveTuning(TurretConfig(), override_path=None)
    lt.update(kp_yaw=0.9, min_area_px=99999.0)
    lt.reset(None)
    assert lt.snapshot() == lt.defaults()


def test_live_tuning_reset_rejects_unknown_name():
    lt = LiveTuning(TurretConfig(), override_path=None)
    with pytest.raises(ValueError):
        lt.reset("bogus_field")


def test_live_tuning_save_requires_an_override_path():
    lt = LiveTuning(TurretConfig(), override_path=None)
    with pytest.raises(RuntimeError):
        lt.save()


def test_live_tuning_save_and_reload_round_trips(tmp_path):
    path = tmp_path / "tuning.local.yaml"
    lt = LiveTuning(TurretConfig(), override_path=path)
    lt.update(kp_yaw=0.42, min_area_px=12345.0)
    lt.save()
    assert path.exists()

    reloaded = LiveTuning(TurretConfig(), override_path=path)
    assert reloaded.control.kp_yaw == 0.42
    assert reloaded.fire.min_area_px == 12345.0
    # fields never saved stay at their config default
    assert reloaded.control.kp_pitch == TurretConfig().control.kp_pitch


def test_live_tuning_reset_after_reload_goes_to_config_default_not_last_saved(tmp_path):
    """Reset means config/default.yaml's value, never "back to what I saved"."""
    path = tmp_path / "tuning.local.yaml"
    lt = LiveTuning(TurretConfig(), override_path=path)
    lt.update(kp_yaw=0.42)
    lt.save()

    reloaded = LiveTuning(TurretConfig(), override_path=path)
    assert reloaded.control.kp_yaw == 0.42  # loaded the saved override
    reloaded.reset("kp_yaw")
    assert reloaded.control.kp_yaw == TurretConfig().control.kp_yaw  # not 0.42


def test_live_tuning_ignores_unknown_keys_in_override_file(tmp_path):
    path = tmp_path / "tuning.local.yaml"
    path.write_text("kp_yaw: 0.42\nsome_removed_field: 1.0\n")
    lt = LiveTuning(TurretConfig(), override_path=path)  # must not raise
    assert lt.control.kp_yaw == 0.42


def test_live_tuning_missing_override_file_is_a_no_op(tmp_path):
    path = tmp_path / "does_not_exist.yaml"
    lt = LiveTuning(TurretConfig(), override_path=path)
    assert lt.snapshot() == lt.defaults()


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
