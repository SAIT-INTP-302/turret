import pytest

from turret.config import MLDetectionConfig, TurretConfig
from turret.vision.detector import RedBlobDetector
from turret.vision.factory import build_detector


def test_hsv_backend_returns_red_blob_detector():
    cfg = TurretConfig(detector_backend="hsv")
    det = build_detector(cfg)
    assert isinstance(det, RedBlobDetector)


def test_unknown_backend_raises_with_valid_options():
    cfg = TurretConfig(detector_backend="bogus")
    with pytest.raises(ValueError, match="hsv, tflite, opencv_dnn"):
        build_detector(cfg)


def test_tflite_missing_model_file_raises(tmp_path):
    cfg = TurretConfig(
        detector_backend="tflite",
        ml_detection=MLDetectionConfig(
            model_path=str(tmp_path / "nope.tflite"), labels_path=str(tmp_path / "labels.txt")
        ),
    )
    with pytest.raises(FileNotFoundError, match="download_models.py"):
        build_detector(cfg)


def test_tflite_missing_runtime_raises_actionable_error(monkeypatch, tmp_path):
    import turret.vision.tflite_detector as tflite_mod

    dummy_model = tmp_path / "model.tflite"
    dummy_model.write_bytes(b"not a real model")
    labels = tmp_path / "labels.txt"
    labels.write_text("person\n")

    def _boom():
        raise RuntimeError("detector_backend='tflite' needs a LiteRT runtime ... pip install -e '.[ml]' ...")

    monkeypatch.setattr(tflite_mod, "_load_interpreter_class", _boom)

    cfg = TurretConfig(
        detector_backend="tflite",
        ml_detection=MLDetectionConfig(model_path=str(dummy_model), labels_path=str(labels)),
    )
    with pytest.raises(RuntimeError, match="pip install"):
        build_detector(cfg)


def test_opencv_dnn_missing_model_file_raises(tmp_path):
    cfg = TurretConfig(
        detector_backend="opencv_dnn",
        ml_detection=MLDetectionConfig(model_path=str(tmp_path / "nope.onnx")),
    )
    with pytest.raises(FileNotFoundError, match="download_models.py"):
        build_detector(cfg)


def test_opencv_dnn_non_onnx_model_rejected(tmp_path):
    dummy = tmp_path / "model.caffemodel"
    dummy.write_bytes(b"not really a model")
    cfg = TurretConfig(
        detector_backend="opencv_dnn",
        ml_detection=MLDetectionConfig(model_path=str(dummy)),
    )
    with pytest.raises(ValueError, match="ONNX"):
        build_detector(cfg)
