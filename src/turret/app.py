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
from turret.vision.factory import build_detector
from turret.webapp.frames import FrameStore
from turret.webapp.store import EventStore

log = logging.getLogger(__name__)


class TurretApp:
    def __init__(
        self,
        cfg: TurretConfig,
        *,
        force_mock: bool = False,
        headless: bool = False,
        show_mask: bool = False,
        dashboard: bool = False,
        dashboard_port: int = 8080,
        db_path: str = "turret_events.db",
    ) -> None:
        self._cfg = cfg
        self._force_mock = force_mock
        self._headless = headless
        self._show_mask = show_mask
        self._dashboard = dashboard
        self._dashboard_port = dashboard_port
        self._store = EventStore(db_path) if dashboard else None
        self._frames = FrameStore() if dashboard else None

    def run(self) -> None:
        cfg = self._cfg
        axes = build_axes(cfg, force_mock=self._force_mock)
        camera = open_camera(cfg.camera)
        if self._dashboard and self._store is not None:
            from turret.webapp.server import run_in_thread

            run_in_thread(self._store, self._frames, port=self._dashboard_port)
        try:
            frame_size = camera.resolution
            detector = build_detector(cfg, debug=self._show_mask)
            tracker = Tracker(cfg.control, frame_size, axes["yaw"], axes["pitch"])
            decider = FireDecider(cfg.fire, frame_size)
            fire_control = make_fire_control(cfg.fire, axes.get("roll"))

            period = 1.0 / cfg.camera.fps
            prev = time.monotonic()
            fps = 0.0
            fired_flash = 0.0
            was_locked = False
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

                is_locked = decider.locked(det)
                if self._store is not None and is_locked and not was_locked and det is not None:
                    self._store.log("sighting", cx=det.cx, cy=det.cy, area=det.area)
                was_locked = is_locked

                if decider.update(det, dt):
                    fire_control.fire()
                    fired_flash = 0.5
                    if self._store is not None and det is not None:
                        self._store.log("fired", cx=det.cx, cy=det.cy, area=det.area)
                fire_control.update(dt)
                fired_flash = max(0.0, fired_flash - dt)

                if not self._headless or self._frames is not None:
                    viz.draw(frame, det, cfg, fired=fired_flash > 0, fps=fps)
                if self._frames is not None:
                    self._frames.set_frame(frame)
                if not self._headless:
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
