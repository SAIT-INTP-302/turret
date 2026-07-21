#!/usr/bin/env python3
"""Per-axis bring-up and calibration tool for the Pi.

Examples:
    python scripts/axis_test.py yaw --angle 90
    python scripts/axis_test.py yaw --sweep 20 160 --speed 30
    python scripts/axis_test.py roll --spin 2.0
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import replace
from pathlib import Path

from turret.actuators.factory import build_axes
from turret.config import load_config

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "default.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("axis", help="axis name from config (yaw/pitch/roll)")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--angle", type=float, help="go to this angle and hold")
    parser.add_argument("--sweep", nargs=2, type=float, metavar=("LOW", "HIGH"))
    parser.add_argument("--spin", type=float, metavar="SECONDS", help="continuous spin")
    parser.add_argument("--speed", type=float, default=30.0, help="deg/s during moves")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    if args.axis not in cfg.axes:
        parser.error(f"unknown axis {args.axis!r}; have {sorted(cfg.axes)}")

    # Slow the axis down for safe bring-up
    axis_cfg = cfg.axes[args.axis]
    cfg = replace(
        cfg,
        axes={
            **cfg.axes,
            args.axis: replace(axis_cfg, servo=replace(axis_cfg.servo, max_deg_per_s=args.speed)),
        },
    )

    axes = build_axes(cfg, force_mock=args.mock)
    axis = axes[args.axis]
    try:
        if args.spin is not None:
            print(f"spinning {args.axis} for {args.spin}s")
            axis.spin(forward=True)
            time.sleep(args.spin)
            axis.stop_spin()
        elif args.sweep:
            low, high = args.sweep
            print(f"sweeping {args.axis} {low} <-> {high} at {args.speed} deg/s (Ctrl-C to stop)")
            target = low
            while True:
                axis.set_target(target)
                while abs(axis.angle() - axis.target) > 0.5:
                    axis.update(0.02)
                    time.sleep(0.02)
                print(f"  reached {axis.angle():.1f}")
                target = high if target == low else low
        else:
            angle = args.angle if args.angle is not None else 90.0
            print(f"moving {args.axis} to {angle} and holding (Ctrl-C to exit)")
            axis.set_target(angle)
            while True:
                axis.update(0.02)
                time.sleep(0.02)
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        for ax in axes.values():
            ax.close()


if __name__ == "__main__":
    main()
