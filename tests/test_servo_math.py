from turret.actuators.mock_axis import MockAxis
from turret.actuators.servo_axis import (
    MAX_PULSE_US,
    MIN_PULSE_US,
    angle_to_us,
    us_to_angle,
)


def test_angle_endpoints_sg90_calibration():
    assert angle_to_us(0, 500, 2400) == 500
    assert angle_to_us(180, 500, 2400) == 2400
    assert angle_to_us(90, 500, 2400) == 1450


def test_angle_clamped_to_0_180():
    assert angle_to_us(-45, 500, 2400) == 500
    assert angle_to_us(400, 500, 2400) == 2400


def test_pulse_clamped_to_hard_limits():
    # even a wild calibration can't push past ESP32Servo's hard limits
    assert angle_to_us(180, 500, 9000) == MAX_PULSE_US
    assert angle_to_us(0, 100, 2400) == MIN_PULSE_US


def test_roundtrip():
    for angle in (0, 45, 90, 135, 180):
        us = angle_to_us(angle, 1000, 2000)
        assert abs(us_to_angle(us, 1000, 2000) - angle) < 0.2


def test_slew_limits_motion():
    ax = MockAxis("yaw", start_angle=90, max_deg_per_s=100)
    ax.set_target(180)
    ax.update(0.1)  # can move at most 10 deg per update
    assert ax.angle() == 100
    ax.update(0.1)
    assert ax.angle() == 110
    for _ in range(20):
        ax.update(0.1)
    assert ax.angle() == 180


def test_target_clamped_to_travel_limits():
    ax = MockAxis("pitch", min_angle=45, max_angle=135, start_angle=90)
    ax.set_target(500)
    assert ax.target == 135
    ax.set_target(-500)
    assert ax.target == 45


def test_update_converges_exactly():
    ax = MockAxis("yaw", start_angle=90, max_deg_per_s=1000)
    ax.set_target(95.5)
    ax.update(1.0)
    assert ax.angle() == 95.5
