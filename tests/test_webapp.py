import cv2
import numpy as np

from turret.config import TurretConfig
from turret.live_tuning import LiveTuning
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


def test_tuning_routes_without_tuning_are_503(tmp_path):
    store = EventStore(tmp_path / "events.db")
    app = create_app(store, tuning=None)
    client = app.test_client()
    assert client.get("/api/tuning").status_code == 503
    assert client.post("/api/tuning", json={"kp_yaw": 0.1}).status_code == 503


def test_get_tuning_returns_values_and_bounds(tmp_path):
    store = EventStore(tmp_path / "events.db")
    tuning = LiveTuning(TurretConfig(), override_path=None)
    app = create_app(store, tuning=tuning)
    client = app.test_client()
    resp = client.get("/api/tuning")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["values"]["deadband_px"] == TurretConfig().control.deadband_px
    assert body["bounds"]["deadband_px"] == [0.0, 200.0]
    assert body["defaults"]["deadband_px"] == TurretConfig().control.deadband_px


def test_post_tuning_updates_a_value(tmp_path):
    store = EventStore(tmp_path / "events.db")
    tuning = LiveTuning(TurretConfig(), override_path=None)
    app = create_app(store, tuning=tuning)
    client = app.test_client()
    resp = client.post("/api/tuning", json={"min_area_px": 2500})
    assert resp.status_code == 200
    assert resp.get_json()["values"]["min_area_px"] == 2500.0
    assert tuning.fire.min_area_px == 2500.0


def test_post_tuning_unknown_field_is_400(tmp_path):
    store = EventStore(tmp_path / "events.db")
    tuning = LiveTuning(TurretConfig(), override_path=None)
    app = create_app(store, tuning=tuning)
    client = app.test_client()
    resp = client.post("/api/tuning", json={"fire_mode": "roll_spin"})
    assert resp.status_code == 400


def test_reset_routes_without_tuning_are_503(tmp_path):
    store = EventStore(tmp_path / "events.db")
    app = create_app(store, tuning=None)
    client = app.test_client()
    assert client.post("/api/tuning/reset", json={}).status_code == 503
    assert client.post("/api/tuning/save").status_code == 503


def test_post_tuning_reset_one_field(tmp_path):
    store = EventStore(tmp_path / "events.db")
    tuning = LiveTuning(TurretConfig(), override_path=None)
    tuning.update(kp_yaw=0.9, min_area_px=99999.0)
    app = create_app(store, tuning=tuning)
    client = app.test_client()
    resp = client.post("/api/tuning/reset", json={"key": "kp_yaw"})
    assert resp.status_code == 200
    assert resp.get_json()["values"]["kp_yaw"] == TurretConfig().control.kp_yaw
    assert tuning.fire.min_area_px == 99999.0  # untouched


def test_post_tuning_reset_all(tmp_path):
    store = EventStore(tmp_path / "events.db")
    tuning = LiveTuning(TurretConfig(), override_path=None)
    tuning.update(kp_yaw=0.9, min_area_px=99999.0)
    app = create_app(store, tuning=tuning)
    client = app.test_client()
    resp = client.post("/api/tuning/reset", json={})
    assert resp.status_code == 200
    assert resp.get_json()["values"] == tuning.defaults()


def test_post_tuning_reset_unknown_key_is_400(tmp_path):
    store = EventStore(tmp_path / "events.db")
    tuning = LiveTuning(TurretConfig(), override_path=None)
    app = create_app(store, tuning=tuning)
    client = app.test_client()
    resp = client.post("/api/tuning/reset", json={"key": "bogus_field"})
    assert resp.status_code == 400


def test_post_tuning_save_writes_the_override_file(tmp_path):
    store = EventStore(tmp_path / "events.db")
    override_path = tmp_path / "tuning.local.yaml"
    tuning = LiveTuning(TurretConfig(), override_path=override_path)
    tuning.update(kp_yaw=0.42)
    app = create_app(store, tuning=tuning)
    client = app.test_client()
    resp = client.post("/api/tuning/save")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["saved"] is True
    assert body["values"]["kp_yaw"] == 0.42
    assert override_path.exists()

    reloaded = LiveTuning(TurretConfig(), override_path=override_path)
    assert reloaded.control.kp_yaw == 0.42


def test_post_tuning_save_without_override_path_is_500(tmp_path):
    store = EventStore(tmp_path / "events.db")
    tuning = LiveTuning(TurretConfig(), override_path=None)
    app = create_app(store, tuning=tuning)
    client = app.test_client()
    resp = client.post("/api/tuning/save")
    assert resp.status_code == 500
