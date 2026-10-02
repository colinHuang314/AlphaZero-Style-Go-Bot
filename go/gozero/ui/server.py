"""Local web UI for go-zero.

    python -m gozero.ui.server [--model runs/eval_night1/candidate_c79.pt] [--port 8765]

Then open http://localhost:8765 . Uses only the standard library HTTP server;
the page polls /api/state a few times per second.
"""
import argparse
import json
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .engine import UIEngine

STATIC = os.path.join(os.path.dirname(__file__), "static")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def make_handler(engine):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the console quiet
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                with open(os.path.join(STATIC, "index.html"), "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            elif self.path == "/api/state":
                self._send(200, json.dumps(engine.snapshot()))
            elif self.path == "/api/sgf":
                self._send(200, engine.sgf(), "application/x-go-sgf")
            else:
                self._send(404, json.dumps({"error": "not found"}))

        def do_POST(self):
            if self.path != "/api/action":
                self._send(404, json.dumps({"error": "not found"}))
                return
            length = int(self.headers.get("Content-Length", 0))
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
                res = engine.action(payload)
            except Exception as e:
                res = {"ok": False, "message": repr(e)}
            self._send(200, json.dumps(res))

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="checkpoint path relative to the project root")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    engine = UIEngine(ROOT, args.model)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(engine))
    url = f"http://localhost:{args.port}"
    print(f"go-zero UI at {url}  (model: {engine.settings['model']})  Ctrl+C to quit", flush=True)
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
