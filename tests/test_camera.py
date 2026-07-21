import cv2
import numpy as np
import pytest

from turret.camera.cv_camera import CvCamera
from turret.camera.factory import open_camera
from turret.config import CameraConfig


@pytest.fixture
def clip(tmp_path):
    path = str(tmp_path / "clip.avi")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    assert writer.isOpened()
    for _ in range(10):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()
    return path


def test_video_file_loops(clip):
    with CvCamera(CameraConfig(device=clip)) as cam:
        assert cam.resolution == (64, 48)
        frames = [cam.read() for _ in range(15)]  # more than the clip holds
    assert all(f is not None for f in frames)
    assert frames[0].shape == (48, 64, 3)


def test_factory_falls_back_to_opencv(clip):
    cam = open_camera(CameraConfig(backend="auto", device=clip))
    try:
        assert cam.read() is not None
    finally:
        cam.close()


def test_missing_device_raises():
    with pytest.raises(RuntimeError):
        CvCamera(CameraConfig(device="/nonexistent/video.avi"))
