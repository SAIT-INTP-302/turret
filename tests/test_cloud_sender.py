import json
from urllib.error import URLError

from turret.cloud.events import DetectionEvent
from turret.cloud.sender import HttpEventSender


def _event() -> DetectionEvent:
    return DetectionEvent(
        event_id="evt-1",
        timestamp_utc="2026-08-09T21:00:00+00:00",
        detector_backend="tflite",
        target_detected=True,
        confidence=0.87,
        center_x=320,
        center_y=240,
        bbox=(280, 180, 80, 120),
        area=9600.0,
    )


class _FakeResponse:
    status = 202

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_sender_posts_json(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(
            request.data.decode("utf-8")
        )
        return _FakeResponse()

    monkeypatch.setattr(
        "turret.cloud.sender.urlopen",
        fake_urlopen,
    )

    sender = HttpEventSender(
        "https://example.test/api/events",
        timeout_s=1.5,
    )

    assert sender.send(_event()) is True
    assert captured["url"] == "https://example.test/api/events"
    assert captured["timeout"] == 1.5
    assert captured["body"]["event_id"] == "evt-1"
    assert captured["body"]["confidence"] == 0.87


def test_sender_network_failure_returns_false(monkeypatch):
    def fail(*args, **kwargs):
        raise URLError("offline")

    monkeypatch.setattr(
        "turret.cloud.sender.urlopen",
        fail,
    )

    sender = HttpEventSender(
        "https://example.test/api/events"
    )

    assert sender.send(_event()) is False