"""Red-blob target detection.

Targets are people wearing red: threshold both red hue-wrap ranges in HSV,
clean the mask morphologically, and pick the largest plausible contour.
"""

from __future__ import annotations

import cv2
import numpy as np

from turret.config import DetectionConfig
from turret.vision.types import Detection


class RedBlobDetector:
    def __init__(self, cfg: DetectionConfig) -> None:
        self._cfg = cfg
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (cfg.morph_ksize, cfg.morph_ksize)
        )

    def detect(self, frame_bgr: np.ndarray) -> tuple[Detection | None, np.ndarray]:
        """Return (largest red target or None, binary mask for debugging)."""
        cfg = self._cfg
        blurred = cv2.GaussianBlur(frame_bgr, (cfg.blur_ksize, cfg.blur_ksize), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv, np.array(cfg.red_low_1), np.array(cfg.red_high_1)
        ) | cv2.inRange(hsv, np.array(cfg.red_low_2), np.array(cfg.red_high_2))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best: Detection | None = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < cfg.min_area_px:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            aspect = max(w, h) / max(min(w, h), 1)
            if aspect > cfg.max_aspect:
                continue
            if best is None or area > best.area:
                m = cv2.moments(contour)
                best = Detection(
                    cx=int(m["m10"] / m["m00"]),
                    cy=int(m["m01"] / m["m00"]),
                    bbox=(x, y, w, h),
                    area=area,
                )
        return best, mask
