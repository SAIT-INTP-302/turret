import time

from turret.actuators.stepper_axis import (
    HALF_STEP_SEQ,
    StepperAxis,
    deg_to_steps,
    step_phase,
    steps_to_deg,
)
from turret.config import StepperAxisConfig


class FakePi:
    def __init__(self):
        self.levels = {}
        self.modes = {}

    def set_mode(self, pin, mode):
        self.modes[pin] = mode

    def write(self, pin, level):
        self.levels[pin] = level


def test_sequence_is_valid_half_step():
    assert len(HALF_STEP_SEQ) == 8
    # adjacent phases differ by exactly one coil (gray-code-like)
    for i in range(8):
        a, b = HALF_STEP_SEQ[i], HALF_STEP_SEQ[(i + 1) % 8]
        assert sum(x != y for x, y in zip(a, b)) == 1


def test_step_phase_wraps_and_handles_negatives():
    assert step_phase(0) == HALF_STEP_SEQ[0]
    assert step_phase(8) == HALF_STEP_SEQ[0]
    assert step_phase(-1) == HALF_STEP_SEQ[7]
    assert step_phase(-9) == HALF_STEP_SEQ[7]


def test_deg_step_conversion():
    assert deg_to_steps(360, 4096) == 4096
    assert deg_to_steps(90, 4096) == 1024
    assert steps_to_deg(2048, 4096) == 180.0


def test_stepper_axis_steps_toward_target():
    pi = FakePi()
    cfg = StepperAxisConfig(pins=(1, 2, 3, 4), max_steps_per_s=2000.0)
    ax = StepperAxis(pi, "roll", cfg)
    try:
        ax.set_target(10.0)
        deadline = time.monotonic() + 2.0
        while ax._step_pos < deg_to_steps(10.0, cfg.steps_per_rev):
            ax.update(0.05)
            assert time.monotonic() < deadline, "stepper never reached target"
            time.sleep(0.01)
    finally:
        ax.close()
    # coils de-energized on close
    assert all(pi.levels[p] == 0 for p in cfg.pins)


def test_stepper_spin_and_stop():
    pi = FakePi()
    cfg = StepperAxisConfig(pins=(1, 2, 3, 4), spin_steps_per_s=2000.0)
    ax = StepperAxis(pi, "roll", cfg)
    try:
        ax.spin(forward=True)
        time.sleep(0.1)
        ax.stop_spin()
        pos = ax._step_pos
        assert pos > 0
        time.sleep(0.05)
        assert ax._step_pos in (pos, pos + 1)  # at most one in-flight step
    finally:
        ax.close()
