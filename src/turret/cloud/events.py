"""Cloud event schema for turret detections."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from uuid import uuid4

from turret.vision.types import Detection


@dataclass(frozen=True)
class DetectionEvent:
    event_id: str
    timestamp_utc: str
    detector_backend: str
    target_detected: bool
    confidence: float | None
    center_x: int | None
    center_y: int | None
    bbox: tuple[int, int, int, int] | None
    area: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_detection_event(
    detection: Detection | None,
    *,
    detector_backend: str,
) -> DetectionEvent:
    return DetectionEvent(
        event_id=str(uuid4()),
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        detector_backend=detector_backend,
        target_detected=detection is not None,
        confidence=detection.confidence if detection else None,
        center_x=detection.cx if detection else None,
        center_y=detection.cy if detection else None,
        bbox=detection.bbox if detection else None,
        area=detection.area if detection else None,
    )