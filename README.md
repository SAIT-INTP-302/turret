# turret

SAIT INTP-302 — Roll/Pitch/Yaw turret driven by a Raspberry Pi with an OpenCV
pipeline that tracks people wearing red and fires when the target is centered
and in range.

## How it works

```
camera ──► RedBlobDetector ──► Tracker ──► yaw/pitch axes (slew-limited)
                 │
                 └─────────► FireDecider ──► FireControl (log | spin roll axis)
```

- **Pitch/Yaw**: SG92R micro servos via the pigpio daemon. The angle→pulse
  math (linear map over a calibrated 500–2400 µs range, hard-clamped to
  500–2500 µs, 50 Hz) is ported from the vendored
  [`ESP32Servo-master/`](ESP32Servo-master/) Arduino library, as is the
  slew-rate-limited motion of its Sweep example. The LEDC timer/tick code was
  dropped — `pigpio.set_servo_pulsewidth()` takes microseconds directly.
- **Roll**: 28BYJ-48 stepper on a ULN2003 driver (half-stepping, 4096
  steps/rev). Firing is done by rotating the barrel, which needs continuous
  rotation a 90°-travel servo can't provide — hence the stepper.
- **Detection**: HSV red threshold over both hue-wrap ranges (0–10 and
  170–180), morphological open/close, contour filtering by area and aspect,
  largest blob wins.
- **Fire decision**: target centered within tolerance AND blob area above the
  in-range threshold, held for a dwell time, with a cooldown between shots.

Everything is configured in [`config/default.yaml`](config/default.yaml)
(pins, calibration, HSV ranges, gains, fire behavior, axis→backend mapping).

## Dev setup (any machine with Nix)

```sh
devenv shell            # or `devenv allow` once; installs the venv + deps
pytest                  # unit tests, no hardware needed
python -m turret --mock --video clip.mp4   # full pipeline with mock actuators
python -m turret --mock --dashboard        # same, plus the web dashboard on :8080
```

`--mock` forces simulated axes; without it, missing pigpio just logs a
warning and falls back to mocks (`allow_mock_fallback: true`).

## Raspberry Pi setup

```sh
sudo apt install python3-picamera2 python3-opencv python3-pigpio pigpio
sudo systemctl enable --now pigpiod
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -e . --no-deps
```

`--system-site-packages` is required: picamera2 has no usable PyPI wheel and
must come from apt.

### Wiring (BCM pins, defaults from config/default.yaml)

| Axis  | Actuator          | Pins                     |
|-------|-------------------|--------------------------|
| Yaw   | SG92R servo       | 17 (signal)              |
| Pitch | SG92R servo       | 27 (signal)              |
| Roll  | 28BYJ-48 stepper  | 5, 6, 13, 19 (IN1–IN4)   |

Power servos and the stepper from a separate 5 V supply with common ground —
not from the Pi's 5 V rail.

### Bring-up order

1. `python scripts/axis_test.py yaw --sweep 30 150 --speed 20` — one axis at
   a time, slowly; find each servo's real `min_us`/`max_us` and mechanical
   travel limits, then record them in the config.
2. `python scripts/hsv_tune.py` — tune the red HSV ranges under your
   lighting; press `p` to print YAML-ready values.
3. `python -m turret --headless --log-level DEBUG` — full loop with
   `fire.mode: log` (no firing hardware engaged).
4. Switch `fire.mode: roll_spin` when the barrel mechanism is mounted.

## Web dashboard

A local Flask app logs every sighting (target locked on) and every fire to
SQLite, and serves a live-updating page showing the event feed and running
counts. No cloud involved, everything runs on the Pi.

```
turret loop ──► EventStore (SQLite) ◄── Flask API ──► dashboard (browser)
   (sighting / fired events)          /api/events        auto-refreshes 2s
```

Run it alongside the main loop:

```sh
python -m turret --dashboard --mock            # add --headless if no display
```

Then open `http://<pi-ip>:8080` in a browser on the same network. Flags:

| Flag               | Default            | What it does                          |
|--------------------|--------------------|----------------------------------------|
| `--dashboard`       | off                | Enables the dashboard server           |
| `--dashboard-port`  | `8080`             | Port the dashboard listens on          |
| `--db`              | `turret_events.db` | SQLite file where events are stored    |

Each event records the kind (`sighting` or `fired`), timestamp, and the
target's pixel position/blob area at that moment. `sighting` is logged once
when a lock-on starts (not every frame); `fired` is logged whenever
`FireControl.fire()` actually runs.

The server can also run standalone (e.g. for testing without a camera):

```sh
python -m turret.webapp.server --db turret_events.db
curl -X POST http://localhost:8080/api/events \
  -H "Content-Type: application/json" \
  -d '{"kind": "sighting", "cx": 320, "cy": 240, "area": 1500}'
```

Code lives in [`src/turret/webapp/`](src/turret/webapp/): `store.py` (SQLite
event log), `server.py` (Flask API + static page server), `static/index.html`
(the dashboard UI).

## Safety

- Software travel limits clamp every target before pulse conversion.
- All motion is slew-rate limited; nothing slams to a target.
- Servos are detached (pulse 0) and stepper coils de-energized on exit,
  including Ctrl-C.
- Keep `fire.mode: log` until the mechanism is mechanically safe to spin.

## Credits

Servo control concepts ported from
[ESP32Servo](https://github.com/madhephaestus/ESP32Servo) (John Kenneth
Bennett / RoboticsBrno), vendored under `ESP32Servo-master/`.
