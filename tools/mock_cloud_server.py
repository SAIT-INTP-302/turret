"""Simple local HTTP server for testing turret cloud events."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


OUTPUT_FILE = Path("mock_events.jsonl")


class EventHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        if self.path != "/api/events":
            self.send_response(404)
            self.end_headers()
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            event = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            self.send_response(400)
            self.end_headers()
            return

        with OUTPUT_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

        print("\nReceived event:")
        print(json.dumps(event, indent=2))

        self.send_response(202)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status":"accepted"}')

    def log_message(self, format, *args):
        return


def main() -> None:
    server = HTTPServer(("localhost", 8000), EventHandler)

    print("Mock cloud server running")
    print("Endpoint: http://localhost:8000/api/events")
    print("Press Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()