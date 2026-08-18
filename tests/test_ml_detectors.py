"""ML detector backend tests via injected fake interpreter/net -- no real
model file and no ai-edge-litert/opencv-dnn model download required.
"""

from __future__ import annotations

import numpy as np
import pytest

from turret.config import MLDetectionConfig
from turret.vision.tflite_detector import TFLiteDetector

LABELS_TEXT = "person\nbicycle\ncar\n"


def _write_labels(tmp_path):
    p = tmp_path / "labels.txt"
    p.write_text(LABELS_TEXT)
    return p


class FakeInterpreter:
    """Stands in for ai_edge_litert.interpreter.Interpreter.

    Returns whatever raw boxes/classes/scores/count arrays are set on it,
    and records what tensor was fed via set_tensor for assertions.
    """

    def __init__(self, *, in_h=300, in_w=300, in_dtype=np.uint8):
        self._in_h = in_h
        self._in_w = in_w
        self._in_dtype = in_dtype
        self.last_input_tensor: np.ndarray | None = None
        self.boxes = np.zeros((1, 0, 4), dtype=np.float32)
        self.classes = np.zeros((1, 0), dtype=np.float32)
        self.scores = np.zeros((1, 0), dtype=np.float32)
        self.count = np.array([0.0], dtype=np.float32)

    def get_input_details(self):
        return [
            {
                "name": "normalized_input_image_tensor",
                "index": 0,
                "shape": np.array([1, self._in_h, self._in_w, 3]),
                "dtype": self._in_dtype,
            }
        ]

    def get_output_details(self):
        return [
            {"name": "TFLite_Detection_PostProcess", "index": 10},
            {"name": "TFLite_Detection_PostProcess:1", "index": 11},
            {"name": "TFLite_Detection_PostProcess:2", "index": 12},
            {"name": "TFLite_Detection_PostProcess:3", "index": 13},
        ]

    def set_tensor(self, index, tensor):
        assert index == 0
        self.last_input_tensor = tensor

    def invoke(self):
        pass

    def get_tensor(self, index):
        return {10: self.boxes, 11: self.classes, 12: self.scores, 13: self.count}[index]


def _make_detector(tmp_path, *, fake=None, debug=False, target_classes=("person",), conf_threshold=0.5):
    fake = fake or FakeInterpreter()
    labels_path = _write_labels(tmp_path)
    cfg = MLDetectionConfig(
        model_path="unused-because-interpreter-is-injected",
        labels_path=str(labels_path),
        target_classes=target_classes,
        conf_threshold=conf_threshold,
    )
    det = TFLiteDetector(cfg, debug=debug, interpreter=fake)
    return det, fake


def test_detect_returns_detection_at_expected_pixel_centroid(tmp_path):
    fake = FakeInterpreter()
    fake.boxes = np.array([[[0.25, 0.25, 0.75, 0.75]]], dtype=np.float32)
    fake.classes = np.array([[0.0]], dtype=np.float32)
    fake.scores = np.array([[0.9]], dtype=np.float32)
    fake.count = np.array([1.0], dtype=np.float32)

    det, _ = _make_detector(tmp_path, fake=fake)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result, _mask = det.detect(frame)

    assert result is not None
    assert result.bbox == (160, 120, 320, 240)
    assert result.cx == 320
    assert result.cy == 240


def test_uint8_model_feeds_unnormalized_tensor(tmp_path):
    fake = FakeInterpreter(in_dtype=np.uint8)
    det, fake = _make_detector(tmp_path, fake=fake)
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    det.detect(frame)
    assert fake.last_input_tensor.dtype == np.uint8
    assert fake.last_input_tensor.shape == (1, 300, 300, 3)
    # unnormalized: raw pixel values pass through resize untouched
    assert fake.last_input_tensor.max() <= 255


def test_float_model_feeds_normalized_tensor(tmp_path):
    fake = FakeInterpreter(in_dtype=np.float32)
    det, fake = _make_detector(tmp_path, fake=fake)
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    det.detect(frame)
    assert fake.last_input_tensor.dtype == np.float32
    # (255 - 127.5) / 127.5 == 1.0
    assert np.allclose(fake.last_input_tensor, 1.0)


def test_input_tensor_shape_comes_from_model_not_config(tmp_path):
    fake = FakeInterpreter(in_h=224, in_w=224)
    det, fake = _make_detector(tmp_path, fake=fake)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    det.detect(frame)
    assert fake.last_input_tensor.shape == (1, 224, 224, 3)


def test_input_tensor_is_rgb_ordered(tmp_path):
    fake = FakeInterpreter()
    det, fake = _make_detector(tmp_path, fake=fake)
    # Pure-blue BGR frame -> should arrive as pure-blue-in-last-channel RGB.
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, :, 0] = 255  # BGR channel 0 = blue
    det.detect(frame)
    tensor = fake.last_input_tensor[0]
    assert tensor[..., 2].mean() > 200  # blue now in RGB channel 2
    assert tensor[..., 0].mean() < 50  # red channel stayed low


def test_two_people_largest_wins(tmp_path):
    fake = FakeInterpreter()
    fake.boxes = np.array(
        [[[0.4, 0.4, 0.6, 0.6], [0.0, 0.0, 0.5, 0.5]]], dtype=np.float32
    )  # small box, then a larger box
    fake.classes = np.array([[0.0, 0.0]], dtype=np.float32)
    fake.scores = np.array([[0.9, 0.8]], dtype=np.float32)
    fake.count = np.array([2.0], dtype=np.float32)

    det, _ = _make_detector(tmp_path, fake=fake)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result, _ = det.detect(frame)
    assert result is not None
    assert result.bbox == (0, 0, 320, 240)  # the larger of the two


def test_nothing_above_threshold_returns_none(tmp_path):
    fake = FakeInterpreter()
    fake.boxes = np.array([[[0.25, 0.25, 0.75, 0.75]]], dtype=np.float32)
    fake.classes = np.array([[0.0]], dtype=np.float32)
    fake.scores = np.array([[0.1]], dtype=np.float32)
    fake.count = np.array([1.0], dtype=np.float32)

    det, _ = _make_detector(tmp_path, fake=fake)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result, mask = det.detect(frame)
    assert result is None
    assert mask.shape == (480, 640)


def test_debug_false_returns_2d_array(tmp_path):
    det, _ = _make_detector(tmp_path, debug=False)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    _, mask = det.detect(frame)
    assert mask.ndim == 2
    assert mask.shape == (480, 640)


def test_debug_true_returns_3d_candidate_canvas(tmp_path):
    fake = FakeInterpreter()
    fake.boxes = np.array([[[0.25, 0.25, 0.75, 0.75]]], dtype=np.float32)
    fake.classes = np.array([[0.0]], dtype=np.float32)
    fake.scores = np.array([[0.9]], dtype=np.float32)
    fake.count = np.array([1.0], dtype=np.float32)

    det, _ = _make_detector(tmp_path, fake=fake, debug=True)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    _, canvas = det.detect(frame)
    assert canvas.ndim == 3
    assert canvas.shape == (480, 640, 3)


def test_missing_model_file_raises_with_download_hint(tmp_path):
    labels_path = _write_labels(tmp_path)
    cfg = MLDetectionConfig(model_path=str(tmp_path / "nope.tflite"), labels_path=str(labels_path))
    with pytest.raises(FileNotFoundError, match="download_models.py"):
        TFLiteDetector(cfg)


def test_unknown_target_class_raises(tmp_path):
    fake = FakeInterpreter()
    with pytest.raises(ValueError, match="not found in"):
        _make_detector(tmp_path, fake=fake, target_classes=("dinosaur",))


def test_resolve_path_relative_resolves_against_project_root(monkeypatch, tmp_path):
    import turret.vision.tflite_detector as m
    from turret.config import PROJECT_ROOT

    monkeypatch.chdir(tmp_path)
    resolved = m._resolve_path("models/foo.tflite")
    assert resolved == PROJECT_ROOT / "models" / "foo.tflite"


def test_resolve_path_absolute_passthrough(tmp_path):
    import turret.vision.tflite_detector as m

    abs_path = tmp_path / "model.tflite"
    resolved = m._resolve_path(str(abs_path))
    assert resolved == abs_path
