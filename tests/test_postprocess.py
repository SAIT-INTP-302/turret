import numpy as np
import pytest

from turret.vision.postprocess import (
    Candidate,
    decode_ssd_boxes,
    draw_candidates,
    load_labels,
    resolve_class_ids,
    resolve_ssd_output_indices,
    select_largest,
)

# ---- load_labels ----------------------------------------------------------


def test_load_labels_bare_list(tmp_path):
    p = tmp_path / "labels.txt"
    p.write_text("person\nbicycle\ncar\n")
    labels = load_labels(p)
    assert labels == {0: "person", 1: "bicycle", 2: "car"}


def test_load_labels_placeholder_prefix(tmp_path):
    p = tmp_path / "labelmap.txt"
    p.write_text("???\nperson\nbicycle\n")
    labels = load_labels(p)
    assert labels == {0: "person", 1: "bicycle"}


def test_load_labels_explicit_ids(tmp_path):
    p = tmp_path / "labels.txt"
    p.write_text("0 person\n1 bicycle\n2 car\n")
    labels = load_labels(p)
    assert labels == {0: "person", 1: "bicycle", 2: "car"}


def test_load_labels_skips_blank_lines(tmp_path):
    p = tmp_path / "labels.txt"
    p.write_text("person\n\nbicycle\n\n\ncar\n")
    labels = load_labels(p)
    assert labels == {0: "person", 1: "bicycle", 2: "car"}


# ---- resolve_class_ids ------------------------------------------------------


def test_resolve_class_ids_happy_path():
    labels = {0: "person", 1: "bicycle", 15: "car"}
    assert resolve_class_ids(labels, ["person"]) == frozenset({0})
    assert resolve_class_ids(labels, ["person", "car"]) == frozenset({0, 15})


def test_resolve_class_ids_unknown_name_raises():
    labels = {0: "person", 1: "bicycle"}
    with pytest.raises(ValueError, match="persno"):
        resolve_class_ids(labels, ["persno"])


def test_resolve_class_ids_suggests_close_match():
    labels = {0: "person", 1: "bicycle"}
    with pytest.raises(ValueError, match="Did you mean: person"):
        resolve_class_ids(labels, ["persno"])


# ---- resolve_ssd_output_indices --------------------------------------------


def test_resolve_ssd_output_indices_tf1_naming():
    output_details = [
        {"name": "TFLite_Detection_PostProcess"},
        {"name": "TFLite_Detection_PostProcess:1"},
        {"name": "TFLite_Detection_PostProcess:2"},
        {"name": "TFLite_Detection_PostProcess:3"},
    ]
    assert resolve_ssd_output_indices(output_details) == (0, 1, 2, 3)


def test_resolve_ssd_output_indices_reordered_list():
    # Same tensor names, different position in output_details -- resolver
    # must go by name suffix, not list order.
    output_details = [
        {"name": "TFLite_Detection_PostProcess:2"},
        {"name": "TFLite_Detection_PostProcess:3"},
        {"name": "TFLite_Detection_PostProcess"},
        {"name": "TFLite_Detection_PostProcess:1"},
    ]
    boxes_idx, classes_idx, scores_idx, count_idx = resolve_ssd_output_indices(output_details)
    assert boxes_idx == 2
    assert classes_idx == 3
    assert scores_idx == 0
    assert count_idx == 1


def test_resolve_ssd_output_indices_unrecognized_raises():
    output_details = [{"name": "a"}, {"name": "b"}, {"name": "c"}, {"name": "d"}]
    with pytest.raises(RuntimeError, match="Could not resolve"):
        resolve_ssd_output_indices(output_details)


def test_resolve_ssd_output_indices_wrong_count_raises():
    output_details = [{"name": "TFLite_Detection_PostProcess"}]
    with pytest.raises(RuntimeError, match="Expected 4"):
        resolve_ssd_output_indices(output_details)


# ---- decode_ssd_boxes -------------------------------------------------------


def test_decode_ssd_boxes_exact_pixel_rescale():
    boxes = np.array([[0.25, 0.25, 0.75, 0.75]])  # ymin, xmin, ymax, xmax
    class_ids = np.array([0.0])
    scores = np.array([0.9])
    cands = decode_ssd_boxes(
        boxes,
        class_ids,
        scores,
        count=1,
        frame_w=640,
        frame_h=480,
        wanted=frozenset({0}),
        conf_threshold=0.5,
    )
    assert len(cands) == 1
    c = cands[0]
    assert (c.x, c.y, c.w, c.h) == (160, 120, 320, 240)


def test_decode_ssd_boxes_non_square_frame_no_axis_swap():
    # A box that's wide-short in normalized space must stay wide-short in
    # pixel space -- catches a latent frame_w/frame_h swap.
    boxes = np.array([[0.4, 0.1, 0.6, 0.9]])  # ymin,xmin,ymax,xmax -> wide box
    cands = decode_ssd_boxes(
        boxes,
        np.array([0.0]),
        np.array([0.9]),
        count=1,
        frame_w=640,
        frame_h=480,
        wanted=frozenset({0}),
        conf_threshold=0.5,
    )
    c = cands[0]
    assert c.w > c.h  # wide box stays wide, not rotated into tall


def test_decode_ssd_boxes_out_of_range_clamped():
    boxes = np.array([[-0.1, -0.1, 1.2, 1.2]])
    cands = decode_ssd_boxes(
        boxes,
        np.array([0.0]),
        np.array([0.9]),
        count=1,
        frame_w=640,
        frame_h=480,
        wanted=frozenset({0}),
        conf_threshold=0.5,
    )
    c = cands[0]
    assert c.x >= 0 and c.y >= 0
    assert c.x + c.w <= 640
    assert c.y + c.h <= 480
    assert c.w >= 1 and c.h >= 1


def test_decode_ssd_boxes_ignores_entries_beyond_count():
    boxes = np.array([[0.25, 0.25, 0.75, 0.75], [0.0, 0.0, 0.5, 0.5]])
    class_ids = np.array([0.0, 0.0])
    scores = np.array([0.9, 0.99])  # second entry has a higher score
    cands = decode_ssd_boxes(
        boxes,
        class_ids,
        scores,
        count=1,  # only the first entry is valid
        frame_w=640,
        frame_h=480,
        wanted=frozenset({0}),
        conf_threshold=0.5,
    )
    assert len(cands) == 1
    assert cands[0].score == pytest.approx(0.9)


def test_decode_ssd_boxes_drops_below_threshold():
    boxes = np.array([[0.25, 0.25, 0.75, 0.75]])
    cands = decode_ssd_boxes(
        boxes,
        np.array([0.0]),
        np.array([0.4]),
        count=1,
        frame_w=640,
        frame_h=480,
        wanted=frozenset({0}),
        conf_threshold=0.5,
    )
    assert cands == []


def test_decode_ssd_boxes_drops_wrong_class():
    boxes = np.array([[0.25, 0.25, 0.75, 0.75]])
    cands = decode_ssd_boxes(
        boxes,
        np.array([2.0]),  # e.g. "car"
        np.array([0.9]),
        count=1,
        frame_w=640,
        frame_h=480,
        wanted=frozenset({0}),  # only "person"
        conf_threshold=0.5,
    )
    assert cands == []


def test_decode_ssd_boxes_class_id_offset():
    boxes = np.array([[0.25, 0.25, 0.75, 0.75]])
    cands = decode_ssd_boxes(
        boxes,
        np.array([0.0]),
        np.array([0.9]),
        count=1,
        frame_w=640,
        frame_h=480,
        wanted=frozenset({1}),  # expecting the offset id
        conf_threshold=0.5,
        class_id_offset=1,
    )
    assert len(cands) == 1
    assert cands[0].class_id == 1


# ---- select_largest ----------------------------------------------------------


def test_select_largest_picks_biggest_box():
    small = Candidate(x=0, y=0, w=10, h=10, class_id=0, score=0.9)
    big = Candidate(x=100, y=100, w=50, h=40, class_id=0, score=0.6)
    det = select_largest([small, big])
    assert det is not None
    assert det.bbox == (100, 100, 50, 40)
    assert det.area == 2000.0
    assert det.cx == 100 + 25
    assert det.cy == 100 + 20


def test_select_largest_empty_returns_none():
    assert select_largest([]) is None


# ---- draw_candidates ----------------------------------------------------------


def test_draw_candidates_returns_same_shape_copy():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cand = Candidate(x=10, y=10, w=50, h=50, class_id=0, score=0.8)
    chosen = select_largest([cand])
    canvas = draw_candidates(frame, [cand], {0: "person"}, chosen)
    assert canvas.shape == frame.shape
    assert canvas is not frame
    assert canvas.any()  # something was drawn onto it


def test_draw_candidates_handles_no_chosen():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cand = Candidate(x=10, y=10, w=50, h=50, class_id=0, score=0.8)
    canvas = draw_candidates(frame, [cand], {0: "person"}, None)
    assert canvas.shape == frame.shape
