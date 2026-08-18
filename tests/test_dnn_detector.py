"""DnnDetector tests via an injected fake cv2.dnn.Net -- no real model file
or ONNX download required. Output shapes below mirror what the pinned
NanoDet-Plus int8bq model actually returns (verified against the real
model file): three (cls, box) pairs at strides (8, 16, 32) for a 416x416
input, class heads grouped before box heads in
`getUnconnectedOutLayersNames()` order (NOT interleaved).
"""

from __future__ import annotations

import numpy as np
import pytest

from turret.config import MLDetectionConfig
from turret.vision.dnn_detector import COCO80_LABELS, DnnDetector

_STRIDES = (8, 16, 32)
_REG_MAX = 7
_INPUT_SIZE = 416


def _side(stride: int) -> int:
    return _INPUT_SIZE // stride


def _zero_outputs() -> list[np.ndarray]:
    """Six all-zero output tensors shaped like the real model's, grouped
    cls-first then box-first (matching the real getUnconnectedOutLayersNames
    order this codebase discovered empirically).
    """
    cls_outs = [
        np.zeros((1, _side(s) * _side(s), len(COCO80_LABELS)), dtype=np.float32) for s in _STRIDES
    ]
    box_outs = [
        np.zeros((1, _side(s) * _side(s), 4 * (_REG_MAX + 1)), dtype=np.float32) for s in _STRIDES
    ]
    return cls_outs + box_outs


def _inject_person_box(outs: list[np.ndarray], stride_idx: int, cell_idx: int, *, score: float) -> None:
    """Set a high person-class score at one grid cell so decode finds a box.

    The DFL regression stays all-zero (softmax of zeros -> uniform ->
    projected distance = mean of 0..reg_max), which still produces a valid,
    decodable box -- exact geometry isn't the point of these tests.
    """
    outs[stride_idx][0, cell_idx, 0] = score  # class 0 == person


class FakeNet:
    def __init__(self, outs: list[np.ndarray]):
        self._outs = outs
        self.last_blob = None
        self.last_input_name = None

    def getUnconnectedOutLayersNames(self):
        return ["cls0", "cls1", "cls2", "box0", "box1", "box2"]

    def setInput(self, blob, name):
        self.last_blob = blob
        self.last_input_name = name

    def forward(self, names):
        return self._outs


def _make_detector(*, outs=None, debug=False, target_classes=("person",), conf_threshold=0.5):
    outs = outs if outs is not None else _zero_outputs()
    fake = FakeNet(outs)
    cfg = MLDetectionConfig(
        model_path="unused-because-net-is-injected.onnx",
        target_classes=target_classes,
        conf_threshold=conf_threshold,
    )
    det = DnnDetector(cfg, debug=debug, net=fake)
    return det, fake


def test_detect_finds_person_at_high_score_cell():
    outs = _zero_outputs()
    _inject_person_box(outs, stride_idx=1, cell_idx=100, score=10.0)  # stride 16
    det, _ = _make_detector(outs=outs)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result, _mask = det.detect(frame)

    assert result is not None
    assert result.bbox[2] >= 1 and result.bbox[3] >= 1  # valid, non-degenerate box
    assert 0 <= result.cx < 640
    assert 0 <= result.cy < 480


def test_no_detections_when_nothing_above_threshold():
    det, _ = _make_detector()  # all-zero scores everywhere
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result, mask = det.detect(frame)
    assert result is None
    assert mask.shape == (480, 640)


def test_wrong_class_filtered_out():
    outs = _zero_outputs()
    outs[0][0, 0, 5] = 10.0  # some non-person class at stride 8, cell 0
    det, _ = _make_detector(outs=outs, target_classes=("person",))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result, _ = det.detect(frame)
    assert result is None


def test_blob_fed_at_configured_input_size():
    det, fake = _make_detector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    det.detect(frame)
    # blobFromImage on a (416,416,3) image -> NCHW (1,3,416,416)
    assert fake.last_blob.shape == (1, 3, 416, 416)
    assert fake.last_input_name == "input.1"


def test_debug_false_returns_2d_array():
    det, _ = _make_detector(debug=False)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    _, mask = det.detect(frame)
    assert mask.ndim == 2


def test_debug_true_returns_3d_candidate_canvas():
    outs = _zero_outputs()
    _inject_person_box(outs, stride_idx=0, cell_idx=0, score=10.0)
    det, _ = _make_detector(outs=outs, debug=True)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    _, canvas = det.detect(frame)
    assert canvas.ndim == 3
    assert canvas.shape == (480, 640, 3)


def test_unexpected_output_shapes_raise():
    bad_outs = [np.zeros((1, 100, 999), dtype=np.float32) for _ in range(6)]
    det, _ = _make_detector(outs=bad_outs)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="Unexpected NanoDet output shapes"):
        det.detect(frame)


def test_non_onnx_model_path_rejected(tmp_path):
    cfg = MLDetectionConfig(model_path=str(tmp_path / "model.tflite"))
    with pytest.raises(ValueError, match="ONNX"):
        DnnDetector(cfg)


def test_missing_model_file_raises_with_download_hint(tmp_path):
    cfg = MLDetectionConfig(model_path=str(tmp_path / "nope.onnx"))
    with pytest.raises(FileNotFoundError, match="download_models.py"):
        DnnDetector(cfg)


def test_unknown_target_class_raises():
    with pytest.raises(ValueError, match="not found in"):
        _make_detector(target_classes=("dinosaur",))


def test_describe_reports_person_target():
    det, _ = _make_detector()
    info = det.describe()
    assert info["targets"] == {0: "person"}
    assert info["input_size"] == (416, 416)
