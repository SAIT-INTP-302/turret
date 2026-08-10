from turret.cloud.events import build_detection_event
from turret.vision.types import Detection


def test_build_detection_event_with_ml_detection():
    detection = Detection(
        cx=320,
        cy=240,
        bbox=(280, 180, 80, 120),
        area=9600.0,
        confidence=0.87,
    )

    event = build_detection_event(
        detection,
        detector_backend="tflite",
    )

    assert event.target_detected is True
    assert event.detector_backend == "tflite"
    assert event.confidence == 0.87
    assert event.center_x == 320
    assert event.center_y == 240
    assert event.bbox == (280, 180, 80, 120)
    assert event.area == 9600.0


def test_build_detection_event_without_target():
    event = build_detection_event(
        None,
        detector_backend="hsv",
    )

    assert event.target_detected is False
    assert event.confidence is None
    assert event.bbox is None