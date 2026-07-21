"""Main application loop: camera -> detector -> tracker -> axes -> fire."""

from __future__ import annotations

import logging
import time

import cv2

from turret import viz
from turret.actuators.factory import build_axes
from turret.camera.factory import open_camera
from turret.config import TurretConfig
from turret.control.fire import FireDecider, make_fire_control
from turret.control.tracker import Tracker
from turret.vision.detector import RedBlobDetector

log = logging.getLogger(__name__)


class TurretApp:
    def __init__(
        self,
        cfg: TurretConfig,
        *,
        force_mock: bool = False,
        headless: bool = False,
        show_mask: bool = False,
    ) -> None:
        self._cfg = cfg
        self._force_mock = force_mock
        self._headless = headless
        self._show_mask = show_mask

    def run(self) -> None:
        cfg = self._cfg
        axes = build_axes(cfg, force_mock=self._force_mock)
        camera = open_camera(cfg.camera)
        try:
            frame_size = camera.resolution
            detector = RedBlobDetector(cfg.detection)
            tracker = Tracker(cfg.control, frame_size, axes["yaw"], axes["pitch"])
            decider = FireDecider(cfg.fire, frame_size)
            fire_control = make_fire_control(cfg.fire, axes.get("roll"))

            period = 1.0 / cfg.camera.fps
            prev = time.monotonic()
            fps = 0.0
            fired_flash = 0.0
            log.info("Turret running (%dx%d)  Ctrl-C or 'q' to quit", *frame_size)
            while True:
                frame = camera.read()
                if frame is None:
                    log.error("Camera returned no frame; stopping")
                    break

                now = time.monotonic()
                dt = now - prev
                prev = now
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if dt > 0 else fps

                det, mask = detector.detect(frame)
                tracker.update(det, dt)
                for axis in axes.values():
                    axis.update(dt)
                if decider.update(det, dt):
                    fire_control.fire()
                    fired_flash = 0.5
                fire_control.update(dt)
                fired_flash = max(0.0, fired_flash - dt)

                if not self._headless:
                    viz.draw(frame, det, cfg, fired=fired_flash > 0, fps=fps)
                    cv2.imshow("turret", frame)
                    if self._show_mask:
                        cv2.imshow("mask", mask)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                # Pace the loop to the configured fps
                sleep_for = period - (time.monotonic() - now)
                if sleep_for > 0:
                    time.sleep(sleep_for)
        except KeyboardInterrupt:
            log.info("Interrupted")
        finally:
            for axis in axes.values():
                axis.close()
            camera.close()
            if not self._headless:
                cv2.destroyAllWindows()
