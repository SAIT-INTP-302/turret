from pathlib import Path

from turret.config import AxisConfig, TurretConfig, load_config

DEFAULT_YAML = Path(__file__).parent.parent / "config" / "default.yaml"


def test_defaults_without_file():
    cfg = load_config(None)
    assert isinstance(cfg, TurretConfig)
    assert set(cfg.axes) == {"yaw", "pitch", "roll"}
    assert cfg.axes["yaw"].backend == "servo"
    assert cfg.axes["roll"].backend == "stepper"


def test_missing_file_falls_back(tmp_path):
    cfg = load_config(tmp_path / "nope.yaml")
    assert cfg == TurretConfig()


def test_default_yaml_loads():
    cfg = load_config(DEFAULT_YAML)
    assert cfg.axes["yaw"].servo.pin == 17
    assert cfg.axes["roll"].stepper.pins == (5, 6, 13, 19)
    assert cfg.detection.red_low_2 == (170, 120, 70)
    assert cfg.fire.mode == "log"


def test_partial_override(tmp_path):
    p = tmp_path / "partial.yaml"
    p.write_text(
        """
axes:
  yaw:
    servo:
      pin: 22
control:
  deadband_px: 30
"""
    )
    cfg = load_config(p)
    assert cfg.axes["yaw"].servo.pin == 22
    # untouched fields keep defaults
    assert cfg.axes["yaw"].servo.max_us == 2400
    assert cfg.axes["pitch"].servo.pin == 27
    assert cfg.control.deadband_px == 30
    assert cfg.control.kp_yaw == 0.05


def test_unknown_key_ignored(tmp_path):
    p = tmp_path / "weird.yaml"
    p.write_text("control:\n  bogus_key: 1\n")
    cfg = load_config(p)
    assert cfg.control == TurretConfig().control


def test_axis_config_defaults():
    ax = AxisConfig()
    assert ax.backend == "mock"
