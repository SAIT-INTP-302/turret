import pytest

from turret.actuators.mock_axis import MockAxis
from turret.config import ControlConfig
from turret.control.tracker import Tracker
from turret.vision.types import Detection

FRAME = (640, 480)  # center (320, 240)


def det_at(cx, cy, area=5000.0):
    return Detection(cx=cx, cy=cy, bbox=(cx - 20, cy - 20, 40, 40), area=area)


@pytest.fixture
def axes():
    return MockAxis("yaw"), MockAxis("pitch")


def make_tracker(yaw, pitch, **overrides):
    return Tracker(ControlConfig(**overrides), FRAME, yaw, pitch)


def test_deadband_holds_position(axes):
    yaw, pitch = axes
    tr = make_tracker(yaw, pitch, deadband_px=15)
    tr.update(det_at(330, 245), 0.03)  # 10px, 5px — inside deadband
    assert yaw.target == 90.0
    assert pitch.target == 90.0


def test_quadrant_signs(axes):
    yaw, pitch = axes
    tr = make_tracker(yaw, pitch)
    tr.update(det_at(420, 140), 0.03)  # right of and above center
    assert yaw.target > 90.0  # yaw right
    assert pitch.target < 90.0  # pitch up (image y up = smaller angle)

    yaw2, pitch2 = MockAxis("yaw"), MockAxis("pitch")
    tr2 = make_tracker(yaw2, pitch2)
    tr2.update(det_at(220, 340), 0.03)  # left of and below center
    assert yaw2.target < 90.0
    assert pitch2.target > 90.0


def test_step_clamped(axes):
    yaw, pitch = axes
    tr = make_tracker(yaw, pitch, kp_yaw=1.0, max_step_deg=4.0)
    tr.update(det_at(640, 240), 0.03)  # 320px error * 1.0 would be huge
    assert yaw.target == 94.0


def test_lost_target_recenters_after_timeout(axes):
    yaw, pitch = axes
    tr = make_tracker(yaw, pitch, lost_target_timeout_s=1.0)
    tr.update(det_at(500, 400), 0.03)
    moved = (yaw.target, pitch.target)
    assert moved != (90.0, 90.0)
    tr.update(None, 0.5)  # lost, but under timeout: hold
    assert (yaw.target, pitch.target) == moved
    tr.update(None, 0.6)  # timeout exceeded: recenter
    assert (yaw.target, pitch.target) == (90.0, 90.0)


def test_detection_resets_lost_timer(axes):
    yaw, pitch = axes
    tr = make_tracker(yaw, pitch, lost_target_timeout_s=1.0)
    tr.update(None, 0.9)
    tr.update(det_at(500, 240), 0.03)  # re-acquired
    tr.update(None, 0.9)  # timer restarted, still under timeout
    assert yaw.target != 90.0
