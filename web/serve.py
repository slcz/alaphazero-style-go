#!/usr/bin/env python3
"""serve.py — dev server for the web GUI: repo root, no caching (every
asset is Cache-Control: no-store, so worker/JS edits reload instantly).
For development only; a real deploy is any static host.

    python3 web/serve.py [port]
"""
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class NoCache(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


port = int(sys.argv[1]) if len(sys.argv) > 1 else 8642
srv = ThreadingHTTPServer(("127.0.0.1", port), NoCache)
root = __file__.rsplit("/", 2)[0]
handler = partial(NoCache, directory=root)
srv.RequestHandlerClass = handler
print(f"serving {root} at http://127.0.0.1:{port}/web/ (no-store)")
srv.serve_forever()
