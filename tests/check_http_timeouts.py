"""Run a local silent-serving check without credentials or annotation writes."""
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import requests
import supervisely as sly

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import importlib.util
spec = importlib.util.spec_from_file_location(
    "request_control_check", Path(__file__).resolve().parents[1] / "src/tracking/request_control.py"
)
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)


class SilentServing(BaseHTTPRequestHandler):
    def do_POST(self):
        time.sleep(0.5)

with ThreadingHTTPServer(("127.0.0.1", 0), SilentServing) as server:
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    url = f"http://127.0.0.1:{server.server_port}"
    api = control.TrackingApi(sly.Api(url, "local-test"), threading.Event())
    started = time.monotonic()
    try:
        with patch.object(control, "READ_TIMEOUT_SECONDS", 0.1):
            try:
                api.task.send_request(1, "track-api", {})
                raise AssertionError("A silent server must time out")
            except requests.ReadTimeout:
                elapsed = time.monotonic() - started
                assert elapsed < 1, elapsed
                print(f"Silent model HTTP response timed out after {elapsed:.3f}s (limit 0.1s).")
    finally:
        server.shutdown()
        worker.join()
