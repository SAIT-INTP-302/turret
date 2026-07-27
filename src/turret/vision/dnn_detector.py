"""Person detection via NanoDet-Plus (int8, block-quantized) through cv2.dnn.

Zero extra pip dependency -- opencv-python already ships cv2.dnn. This is
the fallback path for when the tflite backend's LiteRT runtime isn't
available (e.g. a 32-bit/armv7l OS has no ai-edge-litert wheel), at the
cost of a smaller, less accurate model that needs its own NMS
(cv2.dnn.NMSBoxes) rather than having it baked into the graph.

Model note: cv2.dnn's Caffe importer was removed in OpenCV 5 ("Caffe
importer has been removed. Please use ONNX-converted models or use an
older OpenCV version"), which rules out the classic MobileNet-SSD/Caffe
combo. The ONNX Model Zoo's tf2onnx-exported SSD models embed ONNX
control-flow ops (If/Loop) for their postprocessing subgraph, which
OpenCV's ONNX importer doesn't support either. NanoDet-Plus, distributed
by the OpenCV team itself in https://github.com/opencv/opencv_zoo
specifically for cv2.dnn, is the smallest verified-compatible option.

NanoDet is an FCOS-style anchor-free detector: for each of 3 feature-map
strides (8, 16, 32) it predicts a per-cell class score and a box regressed
via Distribution Focal Loss (a softmax over `reg_max+1` bins per edge,
projected to a distance). The decode below is ported from opencv_zoo's own
reference implementation (models/object_detection_nanodet/nanodet.py),
with one deliberate change: outputs are paired to strides by matching
each tensor's channel width and anchor count rather than by trusting
`getUnconnectedOutLayersNames()` order, because that order was found
(empirically, against this exact model) to group all class heads before
all box heads rather than interleave them as the reference script assumes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from turret.config import PROJECT_ROOT, MLDetectionConfig
from turret.vision.base import Detector
from turret.vision.postprocess import (
    Candidate,
    draw_candidates,
    resolve_class_ids,
    select_largest,
)
from turret.vision.types import Detection

log = logging.getLogger(__name__)

# Standard 80-class COCO detection ordering (person first), matching
# opencv_zoo's object_detection_nanodet/demo.py `classes` tuple.
COCO80_LABELS: tuple[str, ...] = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
)  # fmt: skip

# Architecture constants for the pinned NanoDet-Plus export -- not user
# configurable, since changing them requires a different/retrained model.
_STRIDES = (8, 16, 32)
_REG_MAX = 7
_INPUT_NAME = "input.1"  # ONNX graph input name baked into the pinned export
_INPUT_MEAN = np.array([103.53, 116.28, 123.675], dtype=np.float32).reshape(1, 1, 3)
_INPUT_STD = np.array([57.375, 57.12, 58.395], dtype=np.float32).reshape(1, 1, 3)


def _resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _make_anchors(input_size: int) -> list[np.ndarray]:
    anchors = []
    for s in _STRIDES:
        side = input_size // s
        shift = np.arange(side) * s
        xv, yv = np.meshgrid(shift, shift)
        cx = xv.flatten() + 0.5 * (s - 1)
        cy = yv.flatten() + 0.5 * (s - 1)
        anchors.append(np.column_stack((cx, cy)))
    return anchors


def _decode_dfl(bbox_pred: np.ndarray) -> np.ndarray:
    """(N, 4*(reg_max+1)) DFL logits -> (N, 4) distances via softmax expectation."""
    project = np.arange(_REG_MAX + 1)
    x = bbox_pred.reshape(-1, _REG_MAX + 1)
    x = np.exp(x - x.max(axis=1, keepdims=True))
    x /= x.sum(axis=1, keepdims=True)
    return (x @ project).reshape(-1, 4)


class DnnDetector(Detector):
    def __init__(
        self,
        cfg: MLDetectionConfig,
        *,
        debug: bool = False,
        net: Any | None = None,
    ) -> None:
        self._cfg = cfg
        self._debug = debug
        self._blank_key: tuple[int, int] | None = None
        self._blank_img: np.ndarray | None = None
        self._input_size = cfg.input_size

        model_path = _resolve_path(cfg.model_path)
        if model_path.suffix != ".onnx":
            raise ValueError(
                f"opencv_dnn backend expects an ONNX (.onnx) model, got {model_path.name!r}. "
                "Override ml_detection.model_path, e.g. "
                "--model models/object_detection_nanodet_2022nov_int8bq.onnx"
            )
        if net is None:
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model not found: {model_path}\n"
                    "Download it with: python scripts/download_models.py --set opencv_dnn"
                )
            net = cv2.dnn.readNetFromONNX(str(model_path))
        self._net = net
        self._out_names = list(self._net.getUnconnectedOutLayersNames())

        self._labels = dict(enumerate(COCO80_LABELS))
        self._wanted = resolve_class_ids(self._labels, cfg.target_classes, source="COCO80_LABELS")
        self._anchors = _make_anchors(self._input_size)

        log.info(
            "opencv_dnn detector: nanodet %dx%d, targets %s",
            self._input_size,
            self._input_size,
            {i: self._labels[i] for i in sorted(self._wanted)},
        )

    def detect(self, frame_bgr: np.ndarray) -> tuple[Detection | None, np.ndarray]:
        frame_h, frame_w = frame_bgr.shape[:2]
        s = self._input_size

        resized = cv2.resize(frame_bgr, (s, s), interpolation=cv2.INTER_LINEAR)
        img = (resized.astype(np.float32) - _INPUT_MEAN) / _INPUT_STD
        blob = cv2.dnn.blobFromImage(img)
        self._net.setInput(blob, _INPUT_NAME)
        outs = self._net.forward(self._out_names)

        cands = self._decode(outs, frame_w=frame_w, frame_h=frame_h)
        chosen = select_largest(cands)

        debug_img = (
            draw_candidates(frame_bgr, cands, self._labels, chosen)
            if self._debug
            else self._blank(frame_h, frame_w)
        )
        return chosen, debug_img

    def _decode(self, outs: list[np.ndarray], *, frame_w: int, frame_h: int) -> list[Candidate]:
        num_classes = len(COCO80_LABELS)
        box_width = 4 * (_REG_MAX + 1)

        cls_outs = sorted(
            (o.reshape(-1, num_classes) for o in outs if o.shape[-1] == num_classes),
            key=lambda a: -a.shape[0],
        )
        box_outs = sorted(
            (o.reshape(-1, box_width) for o in outs if o.shape[-1] == box_width),
            key=lambda a: -a.shape[0],
        )
        if len(cls_outs) != len(_STRIDES) or len(box_outs) != len(_STRIDES):
            raise RuntimeError(
                f"Unexpected NanoDet output shapes {[o.shape for o in outs]}; expected "
                f"{len(_STRIDES)} class heads (width {num_classes}) and "
                f"{len(_STRIDES)} box heads (width {box_width})"
            )

        s = self._input_size
        boxes_all: list[np.ndarray] = []
        scores_all: list[np.ndarray] = []
        for stride, cls_score, bbox_pred, anchors in zip(_STRIDES, cls_outs, box_outs, self._anchors):
            dist = _decode_dfl(bbox_pred) * stride
            x1 = np.clip(anchors[:, 0] - dist[:, 0], 0, s)
            y1 = np.clip(anchors[:, 1] - dist[:, 1], 0, s)
            x2 = np.clip(anchors[:, 0] + dist[:, 2], 0, s)
            y2 = np.clip(anchors[:, 1] + dist[:, 3], 0, s)
            boxes_all.append(np.column_stack([x1, y1, x2, y2]))
            scores_all.append(cls_score)

        boxes = np.concatenate(boxes_all, axis=0)
        scores = np.concatenate(scores_all, axis=0)
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        boxes_wh = boxes.copy()
        boxes_wh[:, 2:4] -= boxes_wh[:, 0:2]

        indices = cv2.dnn.NMSBoxes(
            boxes_wh.tolist(),
            confidences.tolist(),
            self._cfg.conf_threshold,
            self._cfg.nms_iou_threshold,
        )
        indices = np.array(indices).flatten()

        sx, sy = frame_w / s, frame_h / s
        cands: list[Candidate] = []
        for i in indices:
            cid = int(class_ids[i])
            if cid not in self._wanted:
                continue
            x, y, w, h = boxes_wh[i]
            cands.append(
                Candidate(
                    x=round(x * sx),
                    y=round(y * sy),
                    w=max(1, round(w * sx)),
                    h=max(1, round(h * sy)),
                    class_id=cid,
                    score=float(confidences[i]),
                )
            )
        return cands

    def _blank(self, h: int, w: int) -> np.ndarray:
        if self._blank_key != (h, w):
            self._blank_key = (h, w)
            self._blank_img = np.zeros((h, w), dtype=np.uint8)
        return self._blank_img

    def describe(self) -> dict[str, object]:
        """Diagnostic summary for scripts/download_models.py --verify."""
        return {
            "input_size": (self._input_size, self._input_size),
            "output_names": self._out_names,
            "num_labels": len(self._labels),
            "targets": {i: self._labels[i] for i in sorted(self._wanted)},
        }
