import numpy as np
import pytest

from turret.config import DetectionConfig
from turret.vision.detector import RedBlobDetector

# BGR colors chosen to land in each red hue range after HSV conversion
PURE_RED = (0, 0, 255)  # hue 0 (range 1)
CRIMSON = (60, 0, 255)  # hue ~172 (range 2, wraps below 180)
BLUE = (255, 0, 0)


def frame_with_square(color, x, y, size, shape=(240, 320, 3)):
    frame = np.zeros(shape, dtype=np.uint8)
    frame[y : y + size, x : x + size] = color
    return frame


@pytest.fixture
def det():
    return RedBlobDetector(DetectionConfig())


def test_red_square_detected_at_centroid(det):
    frame = frame_with_square(PURE_RED, 100, 60, 50)
    d, mask = det.detect(frame)
    assert d is not None
    assert abs(d.cx - 125) <= 2 and abs(d.cy - 85) <= 2
    x, y, w, h = d.bbox
    assert abs(w - 50) <= 4 and abs(h - 50) <= 4
    assert mask.shape == frame.shape[:2]


def test_wraparound_red_detected(det):
    d, _ = det.detect(frame_with_square(CRIMSON, 50, 50, 40))
    assert d is not None


def test_blue_square_ignored(det):
    d, _ = det.detect(frame_with_square(BLUE, 100, 60, 50))
    assert d is None


def test_largest_of_two_wins(det):
    frame = frame_with_square(PURE_RED, 20, 20, 30)
    frame[150:210, 200:260] = PURE_RED  # bigger square
    d, _ = det.detect(frame)
    assert d is not None
    assert abs(d.cx - 230) <= 2 and abs(d.cy - 180) <= 2


def test_below_min_area_rejected():
    det = RedBlobDetector(DetectionConfig(min_area_px=400))
    d, _ = det.detect(frame_with_square(PURE_RED, 100, 100, 10))
    assert d is None


def test_extreme_aspect_rejected():
    det = RedBlobDetector(DetectionConfig(max_aspect=4.0))
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[100:104, 10:310] = PURE_RED  # 300x4 stripe
    d, _ = det.detect(frame)
    assert d is None
