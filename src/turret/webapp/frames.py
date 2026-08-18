"""Thread-safe holder for the latest annotated frame, JPEG-encoded so the
dashboard can stream it without re-touching the camera or re-encoding per
viewer.
"""

from __future__ import annotations

import threading

import cv2
import numpy as np


class FrameStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None

    def set_frame(self, frame_bgr: np.ndarray, *, quality: int = 80) -> None:
        ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            with self._lock:
                self._jpeg = buf.tobytes()

    def get_jpeg(self) -> bytes | None:
        with self._lock:
            return self._jpeg
