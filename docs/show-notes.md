# Show Notes

Cue cards, not a script. Short phrases to glance at and expand on out
loud — full explanations live in [Overview](overview.md),
[Architecture](architecture.md), and [Stack & Rationale](stack.md); this
page is the index into your own head, not something to read verbatim.

## Elevator pitch

- Camera-guided pan/tilt/fire turret, Raspberry Pi 4, SAIT INTP-302
- Sense → decide → actuate loop, wrapped in enough tooling to develop with
  zero hardware attached and deploy with confidence
- Safe by construction, not by convention — fire is off by default at
  every layer

## Architecture at a glance

- Camera → **Detector** (pluggable: HSV / TFLite / OpenCV DNN) → **Tracker**
  → yaw/pitch servos
- Same detection → **FireDecider** (locked? dwelled? cooled down?) →
  **FireControl** → roll axis (trigger)
- Everything hardware-shaped sits behind a small interface with a mock
  implementation (`Axis`, `Camera`, `Detector`) — that's *why* 137 tests
  run with zero physical hardware
- Web dashboard: live MJPEG preview + event log + live tuning panel,
  reading off the same running loop

## Key technical decisions (one line each)

- **Three detector backends, one interface** — HSV needs nothing, TFLite
  is the best on a Pi 4 but aarch64-only, OpenCV DNN is the
  dependency-free fallback for 32-bit
- **Trigger moved from stepper to servo** — mechanism changed from
  spin-a-barrel to pull-and-release; `Axis.supports_spin` keeps both
  paths validated at startup, not first-shot
- **`LiveTunable`: a duck-typed mutable mirror of a frozen config
  dataclass** — lets the dashboard retune a running `Tracker`/`FireDecider`/
  detector with *zero changes* to those classes
- **Two config files, on purpose** — `default.yaml` (git-tracked,
  reviewed) vs `tuning.local.yaml` (gitignored, Save-button-only) — a live
  deployment's working tree should never silently drift from git
- **MJPEG over WebSockets/WebRTC** — `<img src=stream.mjpg>` just works,
  zero client JS, good enough for one operator on a local network
- **systemd `Restart=always`, not `on-failure`** — the loop exits 0 when
  the camera drops frames; that's a retry case, not a crash

## Live demo checklist

1. `python -m turret --mock --selftest` — dry run, no hardware, show the
   pass/warn/fail report
2. `python -m turret --selftest` — same, against real hardware (call out:
   trigger axis never actuated unless `--actuate-trigger`)
3. `python -m turret --dashboard --detector tflite` — open the dashboard,
   point at a person, show the live overlay + event log filling in
4. Drag a tuning slider (e.g. `conf_threshold`) live — point out the
   preview reacting in real time
5. Hit **Save**, restart the process, show the value survived
6. (If comfortable) `fire.mode: servo_pull` in config, restart, show a
   deliberate, config-file-gated live fire

## Safety story

- `fire.mode: log` is the default everywhere — fresh checkout, fresh
  config, fresh Pi image
- Dashboard can change *sensitivity*, never *armed/disarmed* — `fire.mode`
  isn't a dashboard control and never will be
- Config mismatches fail at startup, not mid-engagement — e.g.
  `roll_spin` mode against a servo that can't spin
- Dwell + cooldown are load-bearing state machines, not cosmetic delays
- Self-test's one honest limitation: a "PASS" on yaw/pitch means pigpio
  accepted the pulses, not that a physical servo moved — no position
  feedback in this design

## Stack cheat-sheet

- **Python 3.13**, NumPy, OpenCV, PyYAML, Flask
- **pigpio** (servo/stepper PWM via a daemon) + **picamera2** (Pi camera)
- **ai-edge-litert** (TFLite/LiteRT) for the recommended ML detector;
  `cv2.dnn` for the dependency-free fallback
- **devenv/Nix** for a reproducible dev environment; **pytest + ruff**
- **systemd** in production — `turret.service` + `turret-ap.service` +
  `pigpiod.service`
- **SQLite** for the event log, **YAML** for config (both human-inspectable)

## Anticipated questions

- *"Why not just always use the ML detector?"* → HSV needs zero setup and
  zero model download; it's the fallback that always works, and still the
  default.
- *"Why Flask and not [framework]?"* → local-network single-operator tool;
  a dev server and vanilla JS is the whole requirement.
- *"Does live tuning ever touch the committed config?"* → No — separate
  gitignored file, explicit Save button, "Reset" always means the
  committed file's value.
- *"What happens if the camera dies mid-run?"* → Loop logs an error and
  exits 0; systemd restarts it (`Restart=always`).
- *"How do you test hardware code without hardware?"* → Mock
  implementations behind the same interface real hardware uses — see
  `MockAxis`, in-memory video clips, fake ML interpreters.

## Numbers to have ready

- ~2,600 lines of source, 14 test files, **137 tests**, all hardware-free
- 4 architecture diagrams (PlantUML, rendered to SVG, committed)
- 3 contributors, project started 2026-07-20
- Deployed and validated on a real **Raspberry Pi 4B**
- Measured **~10 fps** with the recommended TFLite detector on that
  hardware

## See also

- [Overview](overview.md) · [Architecture](architecture.md) · [Stack & Rationale](stack.md) · [Deployment & Operations](deployment-and-ops.md)
