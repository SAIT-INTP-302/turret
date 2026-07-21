"""Debug overlay drawing for the live view."""

from __future__ import annotations

import cv2
import numpy as np

from turret.config import TurretConfig
from turret.vision.types import Detection

_GREEN = (0, 255, 0)
_YELLOW = (0, 255, 255)
_RED = (0, 0, 255)
_WHITE = (255, 255, 255)


def draw(
    frame: np.ndarray,
    det: Detection | None,
    cfg: TurretConfig,
    *,
    fired: bool,
    fps: float,
) -> np.ndarray:
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    # Crosshair
    cv2.line(frame, (cx - 12, cy), (cx + 12, cy), _WHITE, 1)
    cv2.line(frame, (cx, cy - 12), (cx, cy + 12), _WHITE, 1)

    # Deadband (tracking holds) and fire-tolerance boxes
    db = cfg.control.deadband_px
    cv2.rectangle(frame, (cx - db, cy - db), (cx + db, cy + db), _WHITE, 1)
    ft = cfg.fire.center_tol_px
    cv2.rectangle(frame, (cx - ft, cy - ft), (cx + ft, cy + ft), _YELLOW, 1)

    if det is not None:
        x, y, bw, bh = det.bbox
        in_range = det.area >= cfg.fire.min_area_px
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), _RED if in_range else _GREEN, 2)
        cv2.circle(frame, (det.cx, det.cy), 4, _RED, -1)
        cv2.putText(
            frame,
            f"area {det.area:.0f}",
            (x, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            _GREEN,
            1,
        )

    cv2.putText(frame, f"{fps:.1f} fps", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, _WHITE, 1)
    if fired:
        cv2.putText(frame, "FIRE", (cx - 40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, _RED, 3)
    return frame
