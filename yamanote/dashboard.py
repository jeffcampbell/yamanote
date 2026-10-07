"""Dashboard: JSON API, live Server-Sent Events, Prometheus /metrics, and the
single-page UI in web/. Stdlib only."""
from __future__ import annotations

import hmac
import json
import logging
import mimetypes
import os
import queue
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from . import decisions, settings, stats
from .metrics import METRICS, render_prometheus
from .store import ACTIVE_STATUSES, STATIONS, public_item

log = logging.getLogger("yamanote.dashboard")


def state(factory) -> dict:
    store = factory.store
    now = time.time()
    midnight = time.mktime(time.localtime()[:3] + (0, 0, 0, 0, 0, -1))
    active = [public_item(i) for i in store.items(ACTIVE_STATUSES + ("held",))]
    running = {r["item_id"]: r for r in store.active_runs() if r["item_id"]}
    for item in active:
        run = running.get(item["id"])
        item["run"] = {"id": run["id"], "role": run["role"], "model": run["model"], "steps": run["steps"],
                       "cost_usd": run["cost_usd"], "started_at": run["started_at"]} if run else None
    line_jobs = {k: True for k in list(factory.jobs) if not k.startswith("item:")}
    return {
        "now": now,
        "uptime": now - factory.started_at,
        "paused": factory.paused,
        "suspended": factory.suspension(),
        "stations": [{"key": k, "code": c, "name": n, "desc": d} for k, c, n, d in STATIONS],
        "trains": [{"id": t, "item_id": factory.trains_in_use().get(t)} for t in factory.trains],
        "items": active,
        "arrivals": [public_item(i) for i in store.recent_terminal(40)],
        "line_jobs": {
            "dispatcher": "dispatcher" in line_jobs,
            "signal": any(k.startswith("signal:") for k in line_jobs),
            "ops": "ops" in line_jobs,
        },
        "line_runs": [r for r in store.active_runs() if not r["item_id"]],
        "budget": {"spent_today": store.spend_since(midnight), "daily_limit": settings.DAILY_BUDGET_USD,
                   "item_limit": settings.ITEM_BUDGET_USD},
        "stats": store.stats(),
        "service_classes": settings.SERVICE_CLASSES,
        "station_models": settings.STATION_MODELS,
        "jev": decisions.status(),
        "projects": factory.projects(),
        "dispatcher_next": max(0, (store.kv_get("dispatcher_last", 0) or 0) + settings.DISPATCHER_INTERVAL - now),
        "ops_report": next((e for e in reversed(store.events(limit=300)) if e["kind"] == "ops_report"), None),
        "auth_required": bool(settings.DASHBOARD_TOKEN),
        "routing": routing(factory),
        "playbook": {p["path"]: factory.playbook(p["path"]) for p in factory.projects()},
        "retro_enabled": settings.RETRO_ENABLED,
        "telemetry": factory.telemetry.status() if getattr(factory, "telemetry", None) else {"enabled": False},
        "autopilot": {**factory.autopilot_state(), "config": factory.autopilot_config(),
                      "preview": factory.schedule_preview(),
                      "untested": [p["name"] for p in factory.projects()
                                   if not settings.project_commands(p["path"])["test"]]},
        "autopilot_report": next((e for e in reversed(store.events(limit=500)) if e["kind"] == "autopilot_report"), None),
        "retros": [{"item_id": r["item_id"], "title": r["title"], "ts": r["ts"], "outcome": r["outcome"],
                    "class_fit": r["class_fit"], "summary": r["data"].get("summary", ""),
                    "added": len(r["data"].get("notes_added") or []), "retired": len(r["data"].get("notes_retired") or [])}
                   for r in store.recent_retros(8)],
        "checks": {p["path"]: settings.project_commands(p["path"]) for p in factory.projects()},
        "notify": {"enabled": bool(settings.NOTIFY_CMD or settings.NOTIFY_WEBHOOK),
                   "events": sorted(settings.NOTIFY_EVENTS)},
        "watches": {p["path"]: w for p in factory.projects()
                    if (w := store.kv_get(f"watch:{p['path']}")) and w.get("until", 0) > now},
    }


def routing(factory) -> dict:
    """Difficulty → class routing with the history behind it (first-pass rate, fare)."""
    stats = factory.store.routing_stats()
    rows = []
    for level in settings.DIFFICULTY_LEVELS:
        cls, why = factory.class_for_difficulty(level)
        history = [{"cls": r["cls"], "n": r["n"], "first_pass": r["first_pass"], "arrived": r["arrived"],
                    "avg_cost": r["avg_cost"]} for r in stats if r["difficulty"] == level]
        rows.append({"difficulty": level, "default": settings.DIFFICULTY_TO_CLASS[level], "class": cls,
                     "why": why, "history": history})
    for row in rows:
        for h in row["history"]:
            src = next((r for r in stats if r["difficulty"] == row["difficulty"] and r["cls"] == h["cls"]), {})
            h.update(underpowered=src.get("underpowered") or 0, overpowered=src.get("overpowered") or 0,
                     retros=src.get("retros") or 0)
    return {"adaptive": settings.ADAPTIVE_ROUTING, "min_samples": settings.ADAPTIVE_MIN_SAMPLES, "levels": rows}


def item_detail(factory, item_id: int) -> dict | None:
    item = factory.store.get_item(item_id)
    if not item:
        return None
    return {"item": item, "events": factory.store.events(item_id, limit=500),
            "runs": factory.store.runs(item_id), "retro": factory.store.get_retro(item_id)}


def update_gauges(factory) -> None:
    store = factory.store
    items = store.items(ACTIVE_STATUSES)
    METRICS.backlog_size.set(sum(1 for i in items if not i.get("train")))
    METRICS.trains_active.set(sum(1 for i in items if i.get("train")))
    METRICS.launches_last_hour.set(store.runs_since(time.time() - 3600))
    METRICS.sleep_mode_active.set(1 if factory.suspension() else 0)
    METRICS.spend_24h_usd.set(store.spend_since(time.time() - 86400))
    METRICS.uptime_seconds.set(time.time() - factory.started_at)


class Broadcaster:
    """Fan store notifications out to every connected SSE client."""

    def __init__(self, store):
        self.clients: set[queue.Queue] = set()
        self._lock = threading.Lock()
        store.subscribe(self.publish)

    def publish(self, kind: str, payload: dict):
        if kind == "item":
            payload = public_item(payload)
        msg = f"event: {kind}\ndata: {json.dumps(payload, default=str)}\n\n"
        with self._lock:
            for q in list(self.clients):
                try:
                    q.put_nowait(msg)
                except queue.Full:
                    self.clients.discard(q)  # slow client: drop it; it will reconnect

    def add(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=2000)
        with self._lock:
            self.clients.add(q)
        return q

    def remove(self, q):
        with self._lock:
            self.clients.discard(q)


def make_handler(factory, broadcaster: Broadcaster):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            pass

        # ── helpers ──
        def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data, code: int = 200):
            self._send(code, json.dumps(data, default=str).encode(), "application/json")

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if not n:
                return {}
            try:
                data = json.loads(self.rfile.read(min(n, 1_000_000)))
                return data if isinstance(data, dict) else {}
            except json.JSONDecodeError:
                return {}

        def _authorized(self, qs: dict | None = None) -> bool:
            """With a dashboard token configured, every API read and action needs
            it: Authorization header, ?token= (EventSource can't send headers),
            or the yamanote_token cookie."""
            if not settings.DASHBOARD_TOKEN:
                return True
            want = settings.DASHBOARD_TOKEN
            given = [self.headers.get("Authorization", "").removeprefix("Bearer "),
                     ((qs or {}).get("token") or [""])[0]]
            for part in (self.headers.get("Cookie") or "").split(";"):
                k, _, v = part.strip().partition("=")
                if k == "yamanote_token":
                    given.append(v)
            return any(g and hmac.compare_digest(g, want) for g in given)

        # ── GET ──
        def do_GET(self):
            url = urlparse(self.path)
            path, qs = url.path, parse_qs(url.query)
            if (path.startswith("/api/") or path == "/metrics") and not self._authorized(qs):
                return self._json({"error": "unauthorized"}, 401)
            try:
                if path in ("/", "/index.html"):
                    return self._static("index.html")
                if path.startswith("/static/"):
                    return self._static(path[len("/static/"):])
                if path == "/api/state":
                    return self._json(state(factory))
                if m := re.fullmatch(r"/api/items/(\d+)", path):
                    detail = item_detail(factory, int(m[1]))
                    return self._json(detail) if detail else self._json({"error": "not found"}, 404)
                if m := re.fullmatch(r"/api/runs/(\d+)/steps", path):
                    after = int((qs.get("after") or ["0"])[0])
                    return self._json({"run": factory.store.get_run(int(m[1])),
                                       "steps": factory.store.steps(int(m[1]), after_id=after)})
                if path == "/api/events":
                    after = int((qs.get("after") or ["0"])[0])
                    return self._json({"events": factory.store.events(limit=200, after_id=after)})
                if path == "/api/stats":
                    try:
                        days = float((qs.get("days") or ["7"])[0])
                    except ValueError:
                        days = 7.0
                    days = max(1 / 24, min(365.0, days))
                    project = (qs.get("project") or [""])[0] or None
                    return self._json(stats.compute(factory.store, days, project))
                if path == "/api/diagram":
                    hours = max(1, min(168, int((qs.get("hours") or ["12"])[0])))
                    since = time.time() - hours * 3600
                    return self._json({"since": since, "now": time.time(), "trains": factory.store.diagram(since)})
                if path == "/api/stream":
                    return self._stream()
                if path == "/metrics":
                    update_gauges(factory)
                    return self._send(200, render_prometheus().encode(), "text/plain; version=0.0.4; charset=utf-8")
                if path == "/healthz":
                    return self._json({"ok": True})
                self._json({"error": "not found"}, 404)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _static(self, name: str):
            root = os.path.realpath(settings.WEB_DIR)
            full = os.path.realpath(os.path.join(root, name))
            if not full.startswith(root + os.sep) or not os.path.isfile(full):
                return self._json({"error": "not found"}, 404)
            ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
            with open(full, "rb") as f:
                self._send(200, f.read(), ctype + ("; charset=utf-8" if ctype.startswith("text/") or
                                                   ctype.endswith("javascript") else ""))

        def _stream(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            q = broadcaster.add()
            try:
                self.wfile.write(b"retry: 3000\n\n")
                self.wfile.flush()
                while True:
                    try:
                        msg = q.get(timeout=15)
                    except queue.Empty:
                        msg = ": keepalive\n\n"
                    self.wfile.write(msg.encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                broadcaster.remove(q)
                self.close_connection = True

        # ── POST ──
        def do_POST(self):
            # A custom header can't be sent by a cross-site form post, so this blocks CSRF.
            if self.headers.get("X-Yamanote") != "1":
                return self._json({"error": "missing X-Yamanote header"}, 403)
            if not self._authorized():
                return self._json({"error": "unauthorized"}, 401)
            path = urlparse(self.path).path
            body = self._body()
            try:
                if path == "/api/items":
                    item = factory.create_item(str(body.get("title", "")), str(body.get("description", "")),
                                               body.get("project") or None, kind=str(body.get("kind", "feature")),
                                               priority=str(body.get("priority", "medium")))
                    return self._json({"ok": True, "item": public_item(item)})
                if m := re.fullmatch(r"/api/items/(\d+)/(approve|reject|cancel|retry)", path):
                    item_id, action = int(m[1]), m[2]
                    if action == "reject":
                        item = factory.reject(item_id, str(body.get("reason", ""))[:500])
                    else:
                        item = getattr(factory, action)(item_id)
                    return self._json({"ok": True, "item": public_item(item)})
                if path == "/api/pause":
                    factory.set_paused(True)
                    return self._json({"ok": True})
                if path == "/api/resume":
                    factory.set_paused(False)
                    return self._json({"ok": True})
                if path == "/api/playbook/delete":
                    notes = factory.delete_note(str(body.get("project", "")), int(body.get("id", -1)))
                    return self._json({"ok": True, "playbook": notes})
                if path == "/api/autopilot":
                    return self._json({"ok": True, "autopilot": factory.set_autopilot(bool(body.get("on")))})
                if path == "/api/autopilot/settings":
                    allowed = {k: body[k] for k in ("schedule_enabled", "on_cron", "off_cron", "merge_without_tests",
                                                    "supervised_gates") if k in body}
                    cfg = factory.save_autopilot_config(**allowed)
                    return self._json({"ok": True, "config": cfg, "preview": factory.schedule_preview()})
                if path == "/api/dispatch":
                    factory.dispatch_now()
                    return self._json({"ok": True})
                self._json({"error": "unknown action"}, 404)
            except (ValueError, TypeError) as e:
                self._json({"ok": False, "error": str(e)}, 400)

    return Handler


def start_dashboard(factory, port: int, host: str | None = None) -> ThreadingHTTPServer | None:
    host = host or settings.DASHBOARD_HOST
    broadcaster = Broadcaster(factory.store)
    try:
        server = ThreadingHTTPServer((host, port), make_handler(factory, broadcaster))
    except OSError as e:
        log.error("Dashboard failed to start on %s:%d: %s", host, port, e)
        return None
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True, name="dashboard").start()
    log.info("Dashboard at http://%s:%d/", host, port)
    if host not in ("127.0.0.1", "localhost", "::1") and not settings.DASHBOARD_TOKEN:
        log.warning("Dashboard is exposed on %s without AGENT_TEAM_DASHBOARD_TOKEN; anyone who can reach it "
                    "can create work items", host)
    return server
