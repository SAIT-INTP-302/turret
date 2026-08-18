"""Pure postprocessing helpers shared by the ML detector backends.

Nothing here touches a model, an interpreter, or a camera — every function
takes plain numpy arrays / dicts in and returns plain values out, so the
box-decode and coordinate-rescaling logic (the place a subtly wrong turret
comes from) can be unit tested without a model file or an ML runtime
installed.
"""

from __future__ import annotations

import difflib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from turret.vision.types import Detection


@dataclass(frozen=True)
class Candidate:
    """One surviving box, already rescaled to original-frame pixel space."""

    x: int
    y: int
    w: int
    h: int
    class_id: int
    score: float


def load_labels(path: Path) -> dict[int, str]:
    """Parse a labels file into {class_id: name}.

    Handles the formats seen in the wild:
      - bare list, one name per line, id == line index (Coral coco_labels.txt)
      - bare list whose first line is '???' (a placeholder background class)
        -- dropped, then id == index of what remains
      - 'id<space/tab>name' per line -- explicit ids, used as-is

    Blank lines are skipped and do not consume an index.
    """
    lines = [ln.strip() for ln in Path(path).read_text().splitlines()]
    lines = [ln for ln in lines if ln]
    if lines and lines[0] == "???":
        lines = lines[1:]

    labels: dict[int, str] = {}
    next_id = 0
    for line in lines:
        parts = line.split(None, 1)
        if len(parts) == 2 and parts[0].isdigit():
            labels[int(parts[0])] = parts[1].strip()
        else:
            labels[next_id] = line
            next_id += 1
    return labels


def resolve_class_ids(
    labels: Mapping[int, str], names: Sequence[str], *, source: str = "labels"
) -> frozenset[int]:
    """Map target_classes names -> ids, raising a helpful error on typos."""
    name_to_id = {name: cid for cid, name in labels.items()}
    ids: set[int] = set()
    for name in names:
        if name not in name_to_id:
            suggestion = difflib.get_close_matches(name, name_to_id.keys(), n=1)
            hint = f" Did you mean: {suggestion[0]}?" if suggestion else ""
            available = ", ".join(sorted(name_to_id))
            raise ValueError(
                f"target_classes entry {name!r} not found in {source} "
                f"({len(labels)} labels loaded).{hint} Available: {available}"
            )
        ids.add(name_to_id[name])
    return frozenset(ids)


def resolve_ssd_output_indices(output_details: Sequence[Mapping[str, Any]]) -> tuple[int, int, int, int]:
    """Return (boxes_idx, classes_idx, scores_idx, count_idx) into output_details.

    TFLite's SSD postprocess op emits four outputs from one graph node,
    named '<op_name>', '<op_name>:1', '<op_name>:2', '<op_name>:3' for
    boxes, classes, scores, and detection count respectively. Resolved by
    that naming suffix rather than assumed list order, since different
    converters are free to reorder output_details.
    """
    if len(output_details) != 4:
        raise RuntimeError(
            f"Expected 4 SSD postprocess outputs, got {len(output_details)}: "
            f"{[d.get('name') for d in output_details]}"
        )
    by_suffix: dict[str, int] = {}
    for i, d in enumerate(output_details):
        name = d.get("name", "")
        if ":" in name:
            _, _, suffix = name.rpartition(":")
        else:
            suffix = "0"
        by_suffix[suffix] = i
    if {"0", "1", "2", "3"} - set(by_suffix):
        raise RuntimeError(
            "Could not resolve SSD postprocess outputs by name; expected "
            "'<name>', '<name>:1', '<name>:2', '<name>:3'. Got: "
            f"{[d.get('name') for d in output_details]}"
        )
    return by_suffix["0"], by_suffix["1"], by_suffix["2"], by_suffix["3"]


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(v, hi))


def _rescale(
    xmin: float, ymin: float, xmax: float, ymax: float, frame_w: int, frame_h: int
) -> tuple[int, int, int, int]:
    """Normalized [0,1] box -> clamped (x, y, w, h) pixel box in frame_w x frame_h."""
    x = _clamp(round(xmin * frame_w), 0, frame_w - 1)
    y = _clamp(round(ymin * frame_h), 0, frame_h - 1)
    w = _clamp(round((xmax - xmin) * frame_w), 1, frame_w - x)
    h = _clamp(round((ymax - ymin) * frame_h), 1, frame_h - y)
    return x, y, w, h


def decode_ssd_boxes(
    boxes: np.ndarray,
    class_ids: np.ndarray,
    scores: np.ndarray,
    count: float,
    *,
    frame_w: int,
    frame_h: int,
    wanted: frozenset[int],
    conf_threshold: float,
    class_id_offset: int = 0,
) -> list[Candidate]:
    """Decode a TFLite SSD postprocess output into frame-space Candidates.

    boxes: (N,4) normalized [ymin, xmin, ymax, xmax]. class_ids, scores:
    (N,). count: number of valid entries -- SSD postprocess tensors are
    fixed-length and carry garbage past this point, so entries beyond it are
    ignored even if they carry a high score.
    """
    boxes = np.asarray(boxes)
    class_ids = np.asarray(class_ids)
    scores = np.asarray(scores)
    n = min(round(float(count)), boxes.shape[0])

    out: list[Candidate] = []
    for i in range(n):
        score = float(scores[i])
        if score < conf_threshold:
            continue
        cid = round(float(class_ids[i])) + class_id_offset
        if cid not in wanted:
            continue
        ymin, xmin, ymax, xmax = (float(v) for v in boxes[i])
        x, y, w, h = _rescale(xmin, ymin, xmax, ymax, frame_w, frame_h)
        out.append(Candidate(x=x, y=y, w=w, h=h, class_id=cid, score=score))
    return out


def select_largest(cands: Sequence[Candidate]) -> Detection | None:
    """Largest w*h wins -- mirrors RedBlobDetector's largest-contour policy."""
    if not cands:
        return None
    best = max(cands, key=lambda c: c.w * c.h)
    return Detection(
        cx=best.x + best.w // 2,
        cy=best.y + best.h // 2,
        bbox=(best.x, best.y, best.w, best.h),
        area=float(best.w * best.h),
    )


def draw_candidates(
    frame_bgr: np.ndarray,
    cands: Sequence[Candidate],
    labels: Mapping[int, str],
    chosen: Detection | None,
) -> np.ndarray:
    """Debug canvas: every candidate box + 'name score', chosen one highlighted."""
    canvas = frame_bgr.copy()
    for c in cands:
        is_chosen = chosen is not None and (c.x, c.y, c.w, c.h) == chosen.bbox
        color = (0, 0, 255) if is_chosen else (0, 255, 0)
        thickness = 2 if is_chosen else 1
        cv2.rectangle(canvas, (c.x, c.y), (c.x + c.w, c.y + c.h), color, thickness)
        name = labels.get(c.class_id, str(c.class_id))
        cv2.putText(
            canvas,
            f"{name} {c.score:.2f}",
            (c.x, max(0, c.y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
        )
    return canvas
