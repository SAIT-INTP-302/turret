"""Person detection via a TFLite/LiteRT SSD-MobileNet-V2 (COCO) model.

Primary recommended backend for a CPU-only Raspberry Pi: quantized
SSD-MobileNet is markedly faster than YOLO-family models on this class of
hardware, and the pinned model's TFLite_Detection_PostProcess op bakes NMS
into the graph, so there's no manual NMS here.
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
    decode_ssd_boxes,
    draw_candidates,
    load_labels,
    resolve_class_ids,
    resolve_ssd_output_indices,
    select_largest,
)
from turret.vision.types import Detection

log = logging.getLogger(__name__)


def _load_interpreter_class() -> type:
    """ai_edge_litert (current) -> tflite_runtime (legacy) -> tensorflow.lite.

    Tries the modern LiteRT package first, then packages some existing Pi
    images may already carry. Raises RuntimeError naming all three and the
    fix if none import.
    """
    errors: list[str] = []
    try:
        from ai_edge_litert.interpreter import Interpreter

        return Interpreter
    except ImportError as exc:
        errors.append(f"ai_edge_litert: {exc}")
    try:
        from tflite_runtime.interpreter import (
            Interpreter,  # type: ignore[import-not-found]
        )

        return Interpreter
    except ImportError as exc:
        errors.append(f"tflite_runtime: {exc}")
    try:
        from tensorflow.lite import Interpreter  # type: ignore[import-not-found]

        return Interpreter
    except ImportError as exc:
        errors.append(f"tensorflow: {exc}")

    raise RuntimeError(
        "detector_backend='tflite' needs a LiteRT runtime; none of "
        "ai_edge_litert, tflite_runtime, tensorflow could be imported "
        f"({'; '.join(errors)}). Install with: pip install -e '.[ml]'\n"
        "Note: ai-edge-litert publishes aarch64 wheels only -- check "
        "`uname -m` reports aarch64, not armv7l. On 32-bit, use "
        "detector_backend: opencv_dnn instead."
    )


def _resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


class TFLiteDetector(Detector):
    def __init__(
        self,
        cfg: MLDetectionConfig,
        *,
        debug: bool = False,
        interpreter: Any | None = None,
    ) -> None:
        self._cfg = cfg
        self._debug = debug
        self._blank_key: tuple[int, int] | None = None
        self._blank_img: np.ndarray | None = None

        model_path = _resolve_path(cfg.model_path)
        labels_path = _resolve_path(cfg.labels_path)

        if interpreter is None:
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model not found: {model_path}\n"
                    "Download it with: python scripts/download_models.py"
                )
            interpreter_cls = _load_interpreter_class()
            interpreter = interpreter_cls(model_path=str(model_path), num_threads=cfg.num_threads)
            interpreter.allocate_tensors()
        self._interp = interpreter

        in_detail = self._interp.get_input_details()[0]
        self._in_index = in_detail["index"]
        _, self._in_h, self._in_w, _ = (int(v) for v in in_detail["shape"])
        self._in_dtype = in_detail["dtype"]

        out_details = self._interp.get_output_details()
        boxes_i, classes_i, scores_i, count_i = resolve_ssd_output_indices(out_details)
        self._boxes_index = out_details[boxes_i]["index"]
        self._classes_index = out_details[classes_i]["index"]
        self._scores_index = out_details[scores_i]["index"]
        self._count_index = out_details[count_i]["index"]
        self._out_names = {
            "boxes": out_details[boxes_i]["name"],
            "classes": out_details[classes_i]["name"],
            "scores": out_details[scores_i]["name"],
            "count": out_details[count_i]["name"],
        }

        if not labels_path.exists():
            raise FileNotFoundError(
                f"Labels file not found: {labels_path}\n"
                "Download it with: python scripts/download_models.py"
            )
        self._labels = load_labels(labels_path)
        self._wanted = resolve_class_ids(self._labels, cfg.target_classes, source=str(labels_path))

        log.info(
            "tflite detector: %dx%d %s, %d threads, targets %s",
            self._in_w,
            self._in_h,
            getattr(self._in_dtype, "__name__", self._in_dtype),
            cfg.num_threads,
            {i: self._labels[i] for i in sorted(self._wanted)},
        )

    def detect(self, frame_bgr: np.ndarray) -> tuple[Detection | None, np.ndarray]:
        frame_h, frame_w = frame_bgr.shape[:2]

        # SSD-MobileNet is RGB-trained; feeding BGR degrades accuracy silently.
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self._in_w, self._in_h), interpolation=cv2.INTER_LINEAR)
        if self._in_dtype == np.uint8:
            tensor = np.expand_dims(resized, 0).astype(np.uint8)
        else:
            tensor = np.expand_dims((resized.astype(np.float32) - 127.5) / 127.5, 0)

        self._interp.set_tensor(self._in_index, tensor)
        self._interp.invoke()

        boxes = self._interp.get_tensor(self._boxes_index)[0]
        classes = self._interp.get_tensor(self._classes_index)[0]
        scores = self._interp.get_tensor(self._scores_index)[0]
        count = self._interp.get_tensor(self._count_index)[0]

        cands = decode_ssd_boxes(
            boxes,
            classes,
            scores,
            count,
            frame_w=frame_w,
            frame_h=frame_h,
            wanted=self._wanted,
            conf_threshold=self._cfg.conf_threshold,
        )
        chosen = select_largest(cands)

        debug_img = (
            draw_candidates(frame_bgr, cands, self._labels, chosen)
            if self._debug
            else self._blank(frame_h, frame_w)
        )
        return chosen, debug_img

    def _blank(self, h: int, w: int) -> np.ndarray:
        if self._blank_key != (h, w):
            self._blank_key = (h, w)
            self._blank_img = np.zeros((h, w), dtype=np.uint8)
        return self._blank_img

    def describe(self) -> dict[str, object]:
        """Diagnostic summary for scripts/download_models.py --verify."""
        return {
            "input_shape": (1, self._in_h, self._in_w, 3),
            "input_dtype": getattr(self._in_dtype, "__name__", str(self._in_dtype)),
            "output_names": self._out_names,
            "num_labels": len(self._labels),
            "targets": {i: self._labels[i] for i in sorted(self._wanted)},
        }
