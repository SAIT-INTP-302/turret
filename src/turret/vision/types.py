from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    cx: int
    cy: int
    bbox: tuple[int, int, int, int]  # x, y, w, h
    area: float  # contour area in px^2
    confidence: float | None = None  # ML score; None for HSV detection