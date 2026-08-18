# turret

SAIT INTP-302 — Roll/Pitch/Yaw turret driven by a Raspberry Pi with an OpenCV
pipeline that tracks people wearing red and fires when the target is centered
and in range.

## How it works

```
camera ──► Detector (hsv | tflite | opencv_dnn) ──► Tracker ──► yaw/pitch axes (slew-limited)
                 │
                 └─────────► FireDecider ──► FireControl (log | pull trigger servo | spin roll axis)
```

- **Pitch/Yaw**: SG92R micro servos via the pigpio daemon. The angle→pulse
  math (linear map over a calibrated 500–2400 µs range, hard-clamped to
  500–2500 µs, 50 Hz) is ported from the vendored
  [`ESP32Servo-master/`](ESP32Servo-master/) Arduino library, as is the
  slew-rate-limited motion of its Sweep example. The LEDC timer/tick code was
  dropped — `pigpio.set_servo_pulsewidth()` takes microseconds directly.
- **Roll**: SG92R micro servo (same driver as pitch/yaw). Firing pulls the
  trigger servo to `fire.trigger_pull_angle`, holds it for
  `fire.trigger_hold_s`, then releases it back to rest (`fire.mode:
  servo_pull`). A 28BYJ-48 stepper + ULN2003 driver is still supported for a
  continuous-rotation, spin-the-barrel mechanism (`fire.mode: roll_spin`,
  `axes.roll.backend: stepper`) if that's how your hardware is built.
- **Detection**: pluggable, selected by `detector_backend` in
  [`config/default.yaml`](config/default.yaml) or `--detector` on the CLI.
  Default is HSV red threshold over both hue-wrap ranges (0–10 and 170–180),
  morphological open/close, contour filtering by area and aspect, largest
  blob wins. Two ML person-detector backends are also available — see
  [ML person detection](#ml-person-detection) below.
- **Fire decision**: target centered within tolerance AND target area above
  the in-range threshold, held for a dwell time, with a cooldown between
  shots. `area` means different things for different detectors (see below),
  so `fire.min_area_px` needs retuning per detector.

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
| Roll  | SG92R servo (trigger pull) | 5 (signal)      |

Power servos from a separate 5 V supply with common ground — not from the
Pi's 5 V rail.

### Bring-up order

1. `python scripts/axis_test.py yaw --sweep 30 150 --speed 20` — one axis at
   a time, slowly; find each servo's real `min_us`/`max_us` and mechanical
   travel limits, then record them in the config.
2. `python scripts/hsv_tune.py` — tune the red HSV ranges under your
   lighting; press `p` to print YAML-ready values.
   - *(ML only)* `python scripts/download_models.py --set all &&
     python scripts/download_models.py --verify` — fetch model weights and
     confirm the label mapping prints `person`.
   - *(ML only)* `python scripts/ml_tune.py --detector tflite --benchmark 100`
     on the Pi, then live mode to set `conf_threshold` **and note the
     reported `area`** for `fire.min_area_px` (see below).
3. `python -m turret --selftest` — checks every subsystem (axes, camera,
   detector, fire control) against the real config and reports
   `[PASS]/[WARN]/[FAIL]` per check, non-zero exit on any failure. Run
   `python -m turret --mock --selftest` first for a dry run with no hardware
   at all. It never actuates the trigger axis (only constructs it) unless
   you pass `--actuate-trigger`; yaw/pitch get a small nudge-and-return
   motion test by default (skip with `--no-motion`).
4. `python -m turret --headless --log-level DEBUG` — full loop with
   `fire.mode: log` (no firing hardware engaged). Add `--detector tflite` (or
   `opencv_dnn`) to use an ML backend instead of the HSV default.
5. Switch `fire.mode: servo_pull` when the trigger mechanism is mounted
   (or `fire.mode: roll_spin` if using a stepper-driven spinning barrel
   instead).

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

## ML person detection

Two ML backends are available alongside the default HSV detector, selected
via `detector_backend` in `config/default.yaml` or `--detector` on the CLI.
Both implement the same `Detector.detect(frame) -> (Detection | None, debug_img)`
interface as the HSV detector, so `Tracker` and `FireDecider` don't change.

| Backend      | Model                                   | Size   | Extra install                | Notes |
|--------------|------------------------------------------|--------|-------------------------------|-------|
| `tflite`     | SSD-MobileNet-V2 (COCO, int8 quantized)  | 6.2 MB | `pip install -e '.[ml]'`      | Recommended; ~5–10 FPS expected on Pi 4 (unverified on real hardware, see below) |
| `opencv_dnn` | NanoDet-Plus (int8, block-quantized)     | 1.1 MB | none (`cv2.dnn` is built in)  | Fallback; smaller and less accurate, FPS on Pi not yet measured |

**Recommended: `tflite`.** It needs the `ai-edge-litert` runtime, which only
publishes wheels for **64-bit (aarch64)** Linux:

```sh
uname -m   # must print aarch64 -- if it prints armv7l, use opencv_dnn instead
```

**Fallback: `opencv_dnn`.** No extra dependency (uses `cv2.dnn`, already
part of `opencv-python`), so it works on 32-bit OSes and needs nothing
installed beyond what's already required. It's a smaller, less accurate
model, included as a backend that works everywhere rather than for
raw performance. Note: `cv2.dnn`'s Caffe importer was removed in OpenCV 5,
which is why this isn't the classic Caffe MobileNet-SSD — see
`src/turret/vision/dnn_detector.py` for the full story.

### Setup

```sh
pip install -e '.[ml]'                       # only needed for detector_backend: tflite
python scripts/download_models.py --set all  # fetches both backends' weights into models/
python scripts/download_models.py --verify   # sanity-checks what's on disk, no camera needed
```

`models/` is gitignored — weights are fetched, not committed. Model URLs
and SHA-256 checksums are pinned in `scripts/download_models.py`; re-running
the download is idempotent (skips files whose checksum already matches).

### Selecting a backend

```sh
python -m turret --detector tflite                              # config/default.yaml default model path
python -m turret --detector opencv_dnn --model models/object_detection_nanodet_2022nov_int8bq.onnx
```

The `opencv_dnn` backend needs `--model` (or `ml_detection.model_path` in
config) pointed at its `.onnx` file explicitly — the config default points
at the `tflite` model, since `tflite` is the recommended backend.

### Tuning

`scripts/ml_tune.py` is the ML equivalent of `hsv_tune.py`: live mode shows
every candidate detection with a confidence trackbar, `--benchmark N` runs
headless and reports FPS/latency so you can validate real-world performance
on the actual Pi before enabling `fire.mode: servo_pull` (or `roll_spin`).

```sh
python scripts/ml_tune.py --detector tflite --benchmark 100   # timing only, no display
python scripts/ml_tune.py --detector tflite clip.mp4          # live tuning; 'p' prints YAML
```

**Important:** `fire.min_area_px` is calibrated for HSV *contour* area by
default. ML backends report full-person *bounding-box* area, which is
typically 5–20x larger at the same distance — reusing the HSV-tuned value
means the turret will consider itself "in range" from much farther away
than intended. Use the `area` reported by `ml_tune.py` at your intended
engagement distance to set `fire.min_area_px` before enabling
`fire.mode: servo_pull` (or `roll_spin`) with an ML backend.

### Adding another backend

Implement `Detector.detect()` in a new `src/turret/vision/*.py` file, add a
branch to `build_detector()` in `src/turret/vision/factory.py`, and (if it
needs a new dependency) add an extras group in `pyproject.toml`. An `onnx`
backend (e.g. YOLOv8n via `onnxruntime`) was considered and deliberately
left out: YOLO-family models benchmark around 1 FPS on a Pi 4 CPU in public
benchmarks, which wasn't worth building for the 2 GB CPU-only hardware this
project targets.

## Safety

- Software travel limits clamp every target before pulse conversion.
- All motion is slew-rate limited; nothing slams to a target.
- Servos are detached (pulse 0) and stepper coils de-energized on exit,
  including Ctrl-C.
- Keep `fire.mode: log` until the trigger/barrel mechanism is mechanically
  safe to actuate.

## Credits

Servo control concepts ported from
[ESP32Servo](https://github.com/madhephaestus/ESP32Servo) (John Kenneth
Bennett / RoboticsBrno), vendored under `ESP32Servo-master/`.
