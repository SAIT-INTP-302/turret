"""Local web dashboard: shows every sighting and fire the turret logs.

Run standalone with:  python -m turret.webapp.server
Or embedded: call `run_in_thread(store)` from app.py to serve alongside
the main control loop.
"""

from __future__ import annotations

import argparse
import logging
import threading
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from turret.webapp.store import EventStore

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


def create_app(store: EventStore) -> Flask:
    app = Flask(__name__, static_folder=None)

    @app.get("/")
    def index():
        return send_from_directory(STATIC_DIR, "index.html")

    @app.get("/api/events")
    def get_events():
        limit = request.args.get("limit", default=100, type=int)
        since = request.args.get("since", default=None, type=float)
        return jsonify(
            {
                "events": store.recent(limit=limit, since=since),
                "counts": store.counts(),
            }
        )

    @app.post("/api/events")
    def post_event():
        """Optional: let a remote process (e.g. the Pi) push events over HTTP
        instead of writing to the same SQLite file directly."""
        data = request.get_json(force=True, silent=True) or {}
        kind = data.get("kind")
        if kind not in ("sighting", "fired"):
            return jsonify({"error": "kind must be 'sighting' or 'fired'"}), 400
        store.log(
            kind,
            cx=data.get("cx"),
            cy=data.get("cy"),
            area=data.get("area"),
            note=data.get("note"),
        )
        return jsonify({"ok": True}), 201

    return app


def run_in_thread(store: EventStore, host: str = "0.0.0.0", port: int = 8080) -> threading.Thread:
    """Serve the dashboard in a background thread alongside the main app loop."""
    app = create_app(store)
    thread = threading.Thread(
        target=lambda: app.run(host=host, port=port, debug=False, use_reloader=False),
        daemon=True,
    )
    thread.start()
    log.info("Dashboard running at http://%s:%d", host, port)
    return thread


def main() -> None:
    parser = argparse.ArgumentParser(description="Turret event dashboard")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--db", default="turret_events.db")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    store = EventStore(args.db)
    app = create_app(store)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
