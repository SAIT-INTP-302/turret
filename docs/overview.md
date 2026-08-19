# Overview

## What it is

The turret is a camera-guided pan/tilt/fire platform built for SAIT
INTP-302: a Raspberry Pi watches a video feed, tracks a target, steers two
servo axes (yaw and pitch) to keep it centered, and — once the target has
been centered and in range for long enough — actuates a third axis (roll)
that pulls a trigger. Everything is configurable, everything is
observable from a web dashboard, and firing is off by default at every
level of the stack until someone deliberately turns it on.

It's a small, self-contained example of a pattern that shows up constantly
in embedded/robotics work: **sense → decide → actuate**, with a real-time
loop driving hardware, wrapped in enough tooling (config, tests, self-test,
dashboard) that it can be developed on a laptop with no hardware attached
and then deployed to the actual device with a high degree of confidence
it'll behave the same way.

## Why it looks the way it does

Three constraints shaped almost every decision documented in
[Stack & Rationale](stack.md):

1. **It has to run on a Raspberry Pi 4 (2 GB, CPU-only).** No GPU
   acceleration, limited RAM headroom, and (for the recommended ML
   detector) a 64-bit OS requirement that rules out some otherwise-obvious
   library choices.
2. **It has to be safe by construction.** A physical mechanism that fires
   something needs multiple, independent points where a mistake or a bad
   config value fails closed rather than open — this shows up as the
   `fire.mode` gate, the dwell/cooldown timers, the self-test's refusal to
   ever actuate the trigger axis on its own, and the live-tuning
   dashboard's deliberately narrow scope.
3. **It has to be developable without the hardware in front of you.** Every
   piece of hardware (servos, stepper, camera, GPIO) sits behind an
   interface with a mock implementation, so the full test suite — and most
   day-to-day development — runs on a laptop with zero physical hardware
   attached, then gets verified for real via a dedicated pre-run
   self-test once it's on the Pi.

## What it can do

- **Track a target** two ways: an HSV red-blob detector (zero dependencies,
  the original baseline) or one of two ML person-detectors (TensorFlow
  Lite SSD-MobileNet, or an OpenCV DNN NanoDet fallback) — see
  [Detector backends](stack.md#detection-three-tiers-not-one).
- **Steer two servo axes** (yaw, pitch) with slew-rate-limited motion so
  nothing snaps to a target, using proportional control tuned by a
  deadband and a gain per axis.
- **Fire, in three selectable modes**: `log` (the safe default — logs what
  *would* happen, actuates nothing), `servo_pull` (a trigger servo pulls
  and releases), or `roll_spin` (a continuously-spinning stepper-driven
  barrel, for hardware built that way).
- **Show a live dashboard**: an MJPEG camera preview with the tracking
  overlay, a running event log (every sighting and every fire), and a
  tuning panel that adjusts eight sensitivity parameters on the running
  turret — with persistence, entirely separate from the checked-in config.
- **Check itself before running**: `--selftest` exercises every subsystem
  (axes, camera, detector, fire control) against the real hardware and
  reports pass/fail, without ever actuating the trigger axis unless
  explicitly told to.
- **Deploy as a real service**: systemd units for the control loop and a
  standalone WiFi access point, verified against real Raspberry Pi
  hardware (see [Deployment & Operations](deployment-and-ops.md)).

## Scope boundaries (on purpose)

These aren't missing features — they're deliberate lines drawn for safety
or simplicity, documented here so they don't look like oversights:

- **One target at a time.** The detector always resolves to a single best
  candidate (largest blob / largest confident box); there's no multi-target
  tracking or target selection UI.
- **`fire.mode` is a config-file + restart decision, never a dashboard
  toggle.** The live-tuning dashboard can change *how sensitive* the
  turret is, never *whether it's armed*. See
  [Architecture → Safety model](architecture.md#safety-model).
- **No axis/servo calibration from the dashboard.** Pulse widths, angle
  limits, and pin assignments are `scripts/axis_test.py`'s job — a
  deliberate, one-axis-at-a-time bring-up process, not a live control.
- **No auth on the dashboard.** It's designed for a local network / the
  turret's own access point, not the public internet.

## See also

- [Architecture](architecture.md) — how the pieces above actually fit together
- [Stack & Rationale](stack.md) — why each piece is built the way it is
- [Show Notes](show-notes.md) — the short version, for talking through this out loud
