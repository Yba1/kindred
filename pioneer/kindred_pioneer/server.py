"""A tiny POST /score shim, for callers that would rather have HTTP than an import.

    python -m kindred_pioneer.server --port 8099

    POST /score        {"a": {...person...}, "b": {...person...}}     -> {"score": 0.81, ...}
    POST /score/batch  {"pairs": [{"a": {...}, "b": {...}}, ...]}     -> {"scores": [...]}
    POST /explain      {"a": {...}, "b": {...}}                       -> score + top drivers
    GET  /health                                                      -> backend + readiness

Stdlib http.server on purpose: the FastAPI service is workstream A's, and the
evolution loop shouldn't need a second web framework installed to ask one
question. In-process callers should just `from kindred_pioneer import score_pair`
and skip the network entirely.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from . import scorer

MAX_BODY_BYTES = 4 * 1024 * 1024
MAX_BATCH = 5000


class ScoreHandler(BaseHTTPRequestHandler):
    server_version = "kindred-pioneer/0.1"

    # ---- plumbing -------------------------------------------------------

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            raise ValueError("empty request body")
        if length > MAX_BODY_BYTES:
            raise ValueError(f"request body over {MAX_BODY_BYTES} bytes")
        blob = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(blob, dict):
            raise ValueError("request body must be a JSON object")
        return blob

    def log_message(self, fmt: str, *args: Any) -> None:
        pass  # the loop scores thousands of pairs; don't drown its stdout

    # ---- routes ---------------------------------------------------------

    def do_OPTIONS(self) -> None:  # noqa: N802 - http.server naming
        self._send(204, {})

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") in ("/health", ""):
            self._send(200, {"ok": True, **scorer.info()})
        else:
            self._send(404, {"error": f"no route {self.path}"})

    def do_POST(self) -> None:  # noqa: N802
        route = self.path.rstrip("/") or "/"
        try:
            body = self._read_json()
            if route == "/score":
                a, b = _require_pair(body)
                self._send(200, {"score": scorer.score_pair(a, b), "threshold": scorer.threshold()})
            elif route == "/score/batch":
                pairs = body.get("pairs")
                if not isinstance(pairs, list):
                    raise ValueError("'pairs' must be a list of {a, b} objects")
                if len(pairs) > MAX_BATCH:
                    raise ValueError(f"batch over {MAX_BATCH} pairs")
                parsed = [_require_pair(item) for item in pairs]
                self._send(200, {
                    "scores": scorer.score_pairs(parsed),
                    "threshold": scorer.threshold(),
                })
            elif route == "/explain":
                a, b = _require_pair(body)
                self._send(200, scorer.explain(a, b))
            else:
                self._send(404, {"error": f"no route {self.path}"})
        except scorer.ScorerNotTrained as exc:
            self._send(503, {"error": str(exc)})
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            self._send(400, {"error": str(exc)})
        except Exception as exc:  # keep the judge up even on a bad row
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})


def _require_pair(body: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    a, b = body.get("a"), body.get("b")
    if not isinstance(a, dict) or not isinstance(b, dict):
        raise ValueError("expected objects at 'a' and 'b'")
    return a, b


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve the Kindred pair scorer over HTTP.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8099)
    args = parser.parse_args(argv)

    state = scorer.info()
    if not state.get("ready"):
        print(f"warning: scorer not ready -> {state.get('error', state)}")

    httpd = ThreadingHTTPServer((args.host, args.port), ScoreHandler)
    print(f"kindred scorer ({state['backend']}) on http://{args.host}:{args.port}  "
          f"[POST /score, /score/batch, /explain · GET /health]")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
