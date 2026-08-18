import pytest

from turret.actuators.mock_axis import MockAxis
from turret.config import FireConfig
from turret.control.fire import (
    FireDecider,
    LogFireControl,
    RollSpinFireControl,
    ServoPullFireControl,
    make_fire_control,
)
from turret.vision.types import Detection

FRAME = (640, 480)


def centered_det(area=5000.0):
    return Detection(cx=320, cy=240, bbox=(300, 220, 40, 40), area=area)


def test_fires_after_dwell():
    fd = FireDecider(FireConfig(dwell_s=0.5, cooldown_s=2.0), FRAME)
    assert fd.update(centered_det(), 0.3) is False
    assert fd.update(centered_det(), 0.3) is True  # 0.6s of lock


def test_lock_resets_when_target_lost():
    fd = FireDecider(FireConfig(dwell_s=0.5), FRAME)
    fd.update(centered_det(), 0.4)
    fd.update(None, 0.05)
    assert fd.update(centered_det(), 0.4) is False  # dwell restarted


def test_cooldown_blocks_refire():
    fd = FireDecider(FireConfig(dwell_s=0.2, cooldown_s=2.0), FRAME)
    assert fd.update(centered_det(), 0.3) is True
    assert fd.update(centered_det(), 0.3) is False  # cooling down
    results = [fd.update(centered_det(), 0.3) for _ in range(10)]
    assert results.count(True) == 1  # exactly one refire once cooldown elapses


def test_small_or_offcenter_target_not_locked():
    fd = FireDecider(FireConfig(min_area_px=3000, center_tol_px=25), FRAME)
    assert not fd.locked(centered_det(area=100))  # too small / out of range
    assert not fd.locked(Detection(cx=400, cy=240, bbox=(0, 0, 1, 1), area=5000))
    assert fd.locked(centered_det())


def test_roll_spin_fire_control_spins_then_stops():
    roll = MockAxis("roll")
    fc = RollSpinFireControl(roll, FireConfig(spin_duration_s=1.0))
    fc.fire()
    assert roll.spinning
    fc.update(0.5)
    assert roll.spinning
    fc.update(0.6)
    assert not roll.spinning


def test_servo_pull_fire_control_pulls_then_releases():
    roll = MockAxis("roll", min_angle=0.0, max_angle=90.0, start_angle=0.0, max_deg_per_s=1000.0)
    fc = ServoPullFireControl(roll, FireConfig(trigger_pull_angle=90.0, trigger_hold_s=0.15))
    fc.fire()
    assert roll.target == 90.0
    fc.update(0.1)
    assert roll.target == 90.0
    fc.update(0.1)
    assert roll.target == 0.0  # released back to rest


def test_make_fire_control():
    assert isinstance(make_fire_control(FireConfig(mode="log"), None), LogFireControl)
    rc = make_fire_control(FireConfig(mode="roll_spin"), MockAxis("roll"))
    assert isinstance(rc, RollSpinFireControl)
    sc = make_fire_control(FireConfig(mode="servo_pull"), MockAxis("roll"))
    assert isinstance(sc, ServoPullFireControl)
    with pytest.raises(ValueError):
        make_fire_control(FireConfig(mode="roll_spin"), None)
    with pytest.raises(ValueError):
        make_fire_control(FireConfig(mode="servo_pull"), None)
    with pytest.raises(ValueError):
        make_fire_control(FireConfig(mode="nope"), None)


def test_roll_spin_rejects_non_spinning_axis():
    servo_like = MockAxis("roll")
    servo_like.supports_spin = False  # e.g. a real ServoAxis
    with pytest.raises(ValueError, match="continuous-rotation"):
        make_fire_control(FireConfig(mode="roll_spin"), servo_like)
