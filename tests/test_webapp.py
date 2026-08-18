import cv2
import numpy as np

from turret.webapp.frames import FrameStore
from turret.webapp.server import _mjpeg_frames, create_app
from turret.webapp.store import EventStore


def test_frame_store_round_trip():
    store = FrameStore()
    assert store.get_jpeg() is None

    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    frame[:, :, 2] = 255  # solid red, BGR
    store.set_frame(frame)

    jpeg = store.get_jpeg()
    assert jpeg is not None
    decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == frame.shape


def test_stream_without_frames_is_503(tmp_path):
    store = EventStore(tmp_path / "events.db")
    app = create_app(store, frames=None)
    client = app.test_client()
    resp = client.get("/api/stream.mjpg")
    assert resp.status_code == 503
    assert resp.get_json()["error"]


def test_stream_with_frames_is_multipart(tmp_path):
    store = EventStore(tmp_path / "events.db")
    frames = FrameStore()
    frames.set_frame(np.zeros((48, 64, 3), dtype=np.uint8))
    app = create_app(store, frames=frames)
    client = app.test_client()
    resp = client.get("/api/stream.mjpg")
    try:
        assert resp.status_code == 200
        assert resp.mimetype == "multipart/x-mixed-replace"
    finally:
        resp.close()  # stop the generator without draining the infinite stream


def test_mjpeg_frames_formats_multipart_chunks():
    frames = FrameStore()
    frames.set_frame(np.zeros((48, 64, 3), dtype=np.uint8))
    chunks = list(_mjpeg_frames(frames, interval_s=0, max_frames=2))
    assert len(chunks) == 2
    for chunk in chunks:
        assert chunk.startswith(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
        assert chunk.endswith(b"\r\n")


def test_mjpeg_frames_skips_when_no_frame_yet():
    frames = FrameStore()  # never had set_frame() called
    chunks = list(_mjpeg_frames(frames, interval_s=0, max_frames=3))
    assert chunks == []
