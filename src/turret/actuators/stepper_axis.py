"""28BYJ-48 stepper axis via a ULN2003 driver board.

A background thread paces half-steps toward the target step count, or
free-runs while spinning (the roll axis fires by rotating the barrel).
"""

from __future__ import annotations

import logging
import threading
import time

from turret.actuators.base import Axis
from turret.config import StepperAxisConfig

log = logging.getLogger(__name__)

# Half-step energization sequence for IN1..IN4
HALF_STEP_SEQ = (
    (1, 0, 0, 0),
    (1, 1, 0, 0),
    (0, 1, 0, 0),
    (0, 1, 1, 0),
    (0, 0, 1, 0),
    (0, 0, 1, 1),
    (0, 0, 0, 1),
    (1, 0, 0, 1),
)


def deg_to_steps(deg: float, steps_per_rev: int) -> int:
    return round(deg / 360.0 * steps_per_rev)


def steps_to_deg(steps: int, steps_per_rev: int) -> float:
    return steps / steps_per_rev * 360.0


def step_phase(n: int) -> tuple[int, int, int, int]:
    """Coil states for absolute step index n (negative-safe)."""
    return HALF_STEP_SEQ[n % len(HALF_STEP_SEQ)]


class StepperAxis(Axis):
    def __init__(self, pi, name: str, cfg: StepperAxisConfig) -> None:
        super().__init__(
            name,
            min_angle=0.0,
            max_angle=360.0,
            start_angle=0.0,
            max_deg_per_s=steps_to_deg(round(cfg.max_steps_per_s), cfg.steps_per_rev),
        )
        self._pi = pi
        self._cfg = cfg
        self._step_pos = 0
        self._step_target = 0
        self._spinning = 0  # 0 stopped, +1 forward, -1 reverse
        self._lock = threading.Lock()
        self._stop = threading.Event()
        for pin in cfg.pins:
            pi.set_mode(pin, 1)  # pigpio.OUTPUT == 1
            pi.write(pin, 0)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # The base-class slew already rate-limits the angle; the thread just
    # chases whatever step target _apply sets.
    def _apply(self, deg: float) -> None:
        with self._lock:
            self._step_target = deg_to_steps(deg, self._cfg.steps_per_rev)

    def spin(self, forward: bool = True) -> None:
        with self._lock:
            self._spinning = 1 if forward else -1

    def stop_spin(self) -> None:
        with self._lock:
            self._spinning = 0
            # Adopt wherever the spin ended as the new setpoint so the
            # thread doesn't rewind to the pre-spin target. Normalizing to
            # one rev keeps future set_target() moves short; steps_per_rev
            # is a multiple of 8 so the coil phase index is preserved.
            self._step_pos %= self._cfg.steps_per_rev
            self._step_target = self._step_pos
        angle = steps_to_deg(self._step_pos, self._cfg.steps_per_rev) % 360.0
        self._current = angle
        self._target = angle

    def _write_phase(self) -> None:
        for pin, level in zip(self._cfg.pins, step_phase(self._step_pos)):
            self._pi.write(pin, level)

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                spinning = self._spinning
                delta = self._step_target - self._step_pos
            if spinning:
                self._step_pos += spinning
                self._write_phase()
                time.sleep(1.0 / self._cfg.spin_steps_per_s)
            elif delta:
                self._step_pos += 1 if delta > 0 else -1
                self._write_phase()
                time.sleep(1.0 / self._cfg.max_steps_per_s)
            else:
                time.sleep(0.01)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        for pin in self._cfg.pins:
            self._pi.write(pin, 0)  # de-energize coils
