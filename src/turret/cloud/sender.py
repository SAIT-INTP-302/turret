"""HTTP sender for cloud detection events."""

from __future__ import annotations

import json
import logging
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from turret.cloud.events import DetectionEvent

log = logging.getLogger(__name__)


class HttpEventSender:
    def __init__(
        self,
        endpoint: str,
        *,
        timeout_s: float = 2.0,
        api_key: str | None = None,
    ) -> None:
        if not endpoint:
            raise ValueError("endpoint must not be empty")

        self._endpoint = endpoint
        self._timeout_s = timeout_s
        self._api_key = api_key

    def send(self, event: DetectionEvent) -> bool:
        payload = json.dumps(event.to_dict()).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
        }

        if self._api_key:
            headers["x-functions-key"] = self._api_key

        request = Request(
            self._endpoint,
            data=payload,
            headers=headers,
            method="POST",
        )

        try:
            with urlopen(request, timeout=self._timeout_s) as response:
                status = getattr(response, "status", 200)
                return 200 <= status < 300

        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            log.warning("Cloud event send failed: %s", exc)
            return False