#!/usr/bin/env python3
"""
sse_server.py -- single-command demo server for Kindred's village viz.

Standard library only. Serves the `viz/` directory as static files (so
village.html loads normally) and exposes GET /agent-stream as a
Server-Sent-Events endpoint that replays agent-conversation events forever.

LIVE-MODE WIRING (design choice, kept deliberately simple):
  village.html only goes into "live" mode if `window.KINDRED_STREAM_URL` is
  set before its own <script> block runs (see the comment in village.html
  around line 147). This server does NOT rewrite village.html or inject
  anything -- it serves the file byte-for-byte from disk, untouched.
  To drive the live village, add one line to viz/village.html yourself,
  just before the big <script> tag that defines AGENTS/DEMO_CARD:

      <script>window.KINDRED_STREAM_URL = "/agent-stream";</script>

  That's it -- one script tag, no build step, no server-side templating.
  (Alternative approaches considered: serving a tiny wrapper HTML that sets
  the var and then iframes/redirects to village.html, or rewriting the
  response bytes on the fly to inject the script tag. Both work but add
  moving parts for zero real benefit in a demo script -- a single manual
  line in the HTML is simplest and most transparent.)

Usage:
    python viz/sse_server.py --port 8090 --interval 1.8 --events run.json
"""

from __future__ import annotations

import argparse
import json
import os
import socketserver
import time
from http.server import SimpleHTTPRequestHandler
from typing import Any

VIZ_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(VIZ_DIR)


def load_events(events_path: str | None) -> list[dict[str, Any]]:
    """Load the flat list of events to stream.

    Resolution order:
      1. --events path, if given explicitly.
      2. run.json in the repo root, if present.
      3. viz/phase_cards.json, flattened (each card's `script` array is
         concatenated in order; card-level fields like round/objective are
         ignored -- only the individual event dicts are streamed).
    """
    if events_path:
        path = events_path
    else:
        default_run = os.path.join(REPO_ROOT, "run.json")
        if os.path.exists(default_run):
            path = default_run
        else:
            path = os.path.join(VIZ_DIR, "phase_cards.json")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # run.json is assumed to already be a flat list of events (or a dict
    # with an "events" key holding one). phase_cards.json has a "cards"
    # list, each with a "script" list of events -- flatten those.
    if isinstance(data, list):
        events = data
    elif isinstance(data, dict) and "cards" in data:
        events = []
        for card in data["cards"]:
            events.extend(card.get("script", []))
    elif isinstance(data, dict) and "events" in data:
        events = data["events"]
    else:
        raise ValueError(f"Don't know how to extract events from {path}")

    print(f"Loaded {len(events)} events from {path}")
    return events


class Handler(SimpleHTTPRequestHandler):
    events: list[dict[str, Any]] = []
    interval: float = 1.8

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=VIZ_DIR, **kwargs)

    def log_message(self, fmt: str, *args: Any) -> None:
        # Keep the console readable; SimpleHTTPRequestHandler is chatty.
        print("[%s] %s" % (self.address_string(), fmt % args))

    def do_GET(self) -> None:
        if self.path.split("?", 1)[0] == "/agent-stream":
            self.handle_stream()
            return
        super().do_GET()

    def handle_stream(self) -> None:
        if not self.events:
            self.send_error(503, "No events loaded")
            return
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache, no-store")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            # Each client independently loops through the full event list
            # from the start -- simplest correct approach for multiple
            # concurrent viewers, no shared cursor to coordinate.
            while True:
                for event in self.events:
                    payload = json.dumps(event)
                    chunk = f"data: {payload}\n\n".encode("utf-8")
                    self.wfile.write(chunk)
                    self.wfile.flush()
                    time.sleep(self.interval)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            # Client disconnected -- exit quietly, nothing to clean up.
            return


class ThreadingHTTPServer(socketserver.ThreadingMixIn, __import__("http.server", fromlist=["HTTPServer"]).HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> None:
    parser = argparse.ArgumentParser(description="Kindred village viz demo server (SSE + static files).")
    parser.add_argument("--port", type=int, default=8090, help="Port to serve on (default: 8090).")
    parser.add_argument(
        "--events",
        type=str,
        default=None,
        help="Path to a JSON events file. Defaults to run.json in the repo root if present, "
        "else viz/phase_cards.json (flattened).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.8,
        help="Seconds to sleep between streamed events (default: 1.8).",
    )
    args = parser.parse_args()

    events = load_events(args.events)

    Handler.events = events
    Handler.interval = args.interval

    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)

    print(f"Serving http://localhost:{args.port}/village.html")
    print(f"Streaming http://localhost:{args.port}/agent-stream")
    print(
        "To drive the live village, add this line to viz/village.html just before its main "
        '<script> tag: <script>window.KINDRED_STREAM_URL = "/agent-stream";</script>'
    )

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
