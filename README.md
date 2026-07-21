# turret

SAIT INTP-302 — Roll/Pitch/Yaw turret driven by a Raspberry Pi with an OpenCV
pipeline that tracks people wearing red and fires when the target is centered
and in range.

- **Pitch/Yaw**: SG92R micro servos via pigpio (angle→pulse math ported from
  the vendored [`ESP32Servo-master/`](ESP32Servo-master/) Arduino library).
- **Roll**: 28BYJ-48 stepper (ULN2003) — firing is done by rotating the barrel,
  which needs continuous rotation a 90° servo can't provide.
- **Detection**: HSV red-blob detection (both red hue-wrap ranges).

## Dev setup (any machine with Nix)

```sh
devenv shell     # or `direnv allow` once
pytest
python -m turret --mock --video sample.mp4
```

## Raspberry Pi setup

See the Pi section below (added as hardware bring-up progresses).
