#!/usr/bin/env python3
"""
============================================================================
Teamprojekt AIN7 - Portabler Python Webserver
Startet einen lokalen oder Server-Webserver ohne externe Abhängigkeiten.
Speichert alle Änderungen in 'data/mindmap_data.json'.

Starten mit:
    python server.py
    oder
    python server.py 8080
============================================================================
"""

import os
import sys
import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
from datetime import datetime

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("PORT", 8080))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_FILE = os.path.join(DATA_DIR, "mindmap_data.json")

os.makedirs(DATA_DIR, exist_ok=True)

class MindmapHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=BASE_DIR, **kwargs)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        path_only = self.path.split("?")[0]
        if path_only in ["/api/data", "/api.php", "/api"]:
            query = self.path.split("?")[1] if "?" in self.path else ""
            if "action=ping" in query:
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                res = json.dumps({
                    "status": "ok",
                    "backend": "python",
                    "dataFileExists": os.path.exists(DATA_FILE),
                    "serverTime": datetime.now().isoformat()
                })
                self.wfile.write(res.encode("utf-8"))
                return

            if os.path.exists(DATA_FILE):
                try:
                    with open(DATA_FILE, "r", encoding="utf-8") as f:
                        content = f.read()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(content.encode("utf-8"))
                    return
                except Exception as e:
                    pass

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            res = json.dumps({"status": "ok", "exists": False, "message": "No data yet"})
            self.wfile.write(res.encode("utf-8"))
            return

        super().do_GET()

    def do_POST(self):
        path_only = self.path.split("?")[0]
        if path_only in ["/api/data", "/api.php", "/api"]:
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length == 0:
                self.send_response(400)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(b'{"success":false,"error":"Empty body"}')
                return

            body = self.rfile.read(content_length).decode("utf-8")
            try:
                data = json.loads(body)
                data["updatedAt"] = datetime.now().isoformat()
                with open(DATA_FILE, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                res = json.dumps({"success": True, "updatedAt": data["updatedAt"]})
                self.wfile.write(res.encode("utf-8"))
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Saved updated mindmap data ({len(body)} bytes)")
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                res = json.dumps({"success": False, "error": str(e)})
                self.wfile.write(res.encode("utf-8"))
            return

        super().do_POST()

if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", PORT), MindmapHandler)
    print("=" * 65)
    print(f"  Teamprojekt Webserver läuft!")
    print(f"  URL: http://localhost:{PORT}")
    print(f"  Speicherort für Änderungen: {DATA_FILE}")
    print(f"  Beenden mit Strg+C")
    print("=" * 65)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer beendet.")
