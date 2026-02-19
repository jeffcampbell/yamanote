"""Optional web dashboard for the Yamanote orchestrator.

Serves a dark-themed status page and a JSON API endpoint.
Started as a daemon thread — does not block orchestrator shutdown.
"""

import glob
import json
import os
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

import config

# Cache the HTML file at import time (zero disk I/O per request)
_HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")
try:
    with open(_HTML_PATH, "rb") as _f:
        _HTML_BYTES = _f.read()
except FileNotFoundError:
    _HTML_BYTES = b"<h1>dashboard.html not found</h1>"


def _build_status_payload(station_manager) -> dict:
    """Snapshot mutable StationManager state into a JSON-safe dict.

    Thread safety: we copy all mutable containers at the top so the rest
    of the function operates on local, immutable snapshots.
    """
    now = time.time()

    # ── Snapshot mutable containers (GIL-safe dict()/list() copies) ──
    active_agents = dict(station_manager.active_agents)
    launch_times = list(station_manager.launch_times)
    agent_cooldowns = dict(station_manager.agent_cooldowns)
    consecutive_failures = dict(station_manager.consecutive_failures)
    last_launch_times = dict(station_manager.last_launch_times)
    rework_counts = dict(station_manager.rework_counts)

    # ── Scalars (GIL-safe direct reads) ──
    start_time = getattr(station_manager, "start_time", now)
    current_spec = station_manager.current_conductor_spec
    current_branch = station_manager.current_conductor_branch
    working_dir = station_manager.current_working_dir
    sleep_until = station_manager.sleep_until

    # ── Agents ──
    agents_out = {}
    for name in ("dispatcher", "conductor", "inspector", "signal", "station_manager", "ops"):
        agent = active_agents.get(name)
        cooldown_until = agent_cooldowns.get(name, 0)
        in_cooldown = now < cooldown_until

        if agent is not None and agent.proc is not None and agent.proc.poll() is None:
            status = "running"
            pid = agent.proc.pid
            running_for = now - (agent.start_time or now)
            cooldown_rem = None
        elif in_cooldown:
            status = "cooldown"
            pid = None
            running_for = None
            cooldown_rem = cooldown_until - now
        else:
            status = "idle"
            pid = None
            running_for = None
            cooldown_rem = None

        last_launch = last_launch_times.get(name)
        min_interval = config.AGENT_MIN_INTERVALS.get(name, 0)
        next_run = None
        if min_interval > 0 and last_launch is not None and status == "idle":
            remaining = (last_launch + min_interval) - now
            if remaining > 0:
                next_run = remaining

        agents_out[name] = {
            "status": status,
            "pid": pid,
            "running_for_seconds": round(running_for, 1) if running_for is not None else None,
            "cooldown_remaining_seconds": round(cooldown_rem, 1) if cooldown_rem is not None else None,
            "next_run_seconds": round(next_run, 1) if next_run is not None else None,
            "last_launch": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last_launch)) if last_launch else None,
            "consecutive_failures": consecutive_failures.get(name, 0),
            "model": config.AGENT_MODELS.get(name, "unknown"),
        }

    # ── Pipeline ──
    if active_agents.get("conductor") is not None:
        stage = "transit"
    elif active_agents.get("inspector") is not None:
        stage = "checkpoint"
    elif current_branch and any(
        rework_counts.get(k, 0) > 0 for k in rework_counts
    ):
        stage = "reroute"
    elif current_branch:
        stage = "checkpoint"
    else:
        stage = "idle"

    spec_title = None
    if current_spec:
        try:
            spec_read = current_spec + ".in_progress"
            if not os.path.exists(spec_read):
                spec_read = current_spec
            with open(spec_read) as f:
                spec_title = json.load(f).get("title")
        except (json.JSONDecodeError, OSError):
            spec_title = os.path.basename(current_spec)

    pipeline_out = {
        "current_spec": spec_title,
        "current_branch": current_branch,
        "working_dir": working_dir,
        "rework_count": max(rework_counts.values()) if rework_counts else 0,
        "max_rework": config.MAX_REWORK_ATTEMPTS,
        "stage": stage,
    }

    # ── Backlog (filesystem read — read-only, safe) ──
    specs_out = []
    in_progress_count = 0
    for pattern in ("*.json", "*.json.in_progress"):
        for path in sorted(glob.glob(os.path.join(config.BACKLOG_DIR, pattern))):
            try:
                with open(path) as f:
                    data = json.load(f)
                is_ip = path.endswith(".in_progress")
                if is_ip:
                    in_progress_count += 1
                specs_out.append({
                    "filename": os.path.basename(path),
                    "title": data.get("title", "(untitled)"),
                    "description": data.get("description", ""),
                    "priority": data.get("priority", "medium"),
                    "created_by": data.get("created_by", "?"),
                })
            except (json.JSONDecodeError, OSError):
                continue
    json_only = len(sorted(glob.glob(os.path.join(config.BACKLOG_DIR, "*.json"))))

    backlog_out = {
        "count": json_only,
        "in_progress_count": in_progress_count,
        "specs": specs_out,
    }

    # ── Stats ──
    recent = [t for t in launch_times if t > now - 3600]
    sleep_active = now < sleep_until
    stats_out = {
        "launches_last_hour": len(recent),
        "max_launches_per_hour": config.MAX_AGENT_LAUNCHES_PER_HOUR,
        "sleep_mode_active": sleep_active,
        "sleep_remaining_seconds": round(sleep_until - now, 1) if sleep_active else 0,
    }

    # ── Activity log tail (filesystem read — read-only) ──
    activity_lines = []
    all_lines = []
    try:
        with open(config.ACTIVITY_LOG, "r") as f:
            all_lines = f.readlines()
            activity_lines = [l.rstrip("\n") for l in all_lines[-80:]]
    except (OSError, FileNotFoundError):
        pass

    # ── Recently completed (derived from TERMINUS/MERGED entries in activity log) ──
    completed_out = []
    for raw_line in reversed(all_lines):
        if len(completed_out) >= 20:
            break
        line = raw_line.strip()
        # Support both new "TERMINUS" and old "MERGED" keywords
        if ("TERMINUS" not in line and "MERGED" not in line) or "branch feature/" not in line:
            continue
        # Format: [YYYY-MM-DD HH:MM:SS]  TERMINUS — branch feature/title merged to trunk.
        try:
            ts = line[1:20]  # "YYYY-MM-DD HH:MM:SS"
            branch_start = line.index("branch feature/") + len("branch ")
            branch_end = line.index(" merged to trunk")
            branch = line[branch_start:branch_end]
            title = branch.replace("feature/", "")
            completed_out.append({"title": title, "merged_at": ts})
        except (ValueError, IndexError):
            continue

    # ── Config summary ──
    config_out = {
        "tick_interval": config.TICK_INTERVAL,
        "agent_timeout": config.AGENT_TIMEOUT_SECONDS,
        "sleep_mode_duration": config.SLEEP_MODE_DURATION,
        "max_launches_per_hour": config.MAX_AGENT_LAUNCHES_PER_HOUR,
        "entropy_threshold": config.ENTROPY_FIX_COMMIT_THRESHOLD,
        "max_rework": config.MAX_REWORK_ATTEMPTS,
        "models": dict(config.AGENT_MODELS),
        "intervals": dict(config.AGENT_MIN_INTERVALS),
    }

    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "uptime_seconds": round(now - start_time, 1),
        "agents": agents_out,
        "pipeline": pipeline_out,
        "backlog": backlog_out,
        "completed": completed_out,
        "stats": stats_out,
        "activity": activity_lines,
        "config": config_out,
    }


def _make_handler(station_manager):
    """Factory returning a request handler class bound to the given StationManager."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/api/status":
                payload = json.dumps(_build_status_payload(station_manager)).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            elif self.path == "/" or self.path == "/index.html":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(_HTML_BYTES)))
                self.end_headers()
                self.wfile.write(_HTML_BYTES)
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            # Suppress per-request stderr logging
            pass

    return Handler


def start_dashboard(station_manager, port: int):
    """Start the dashboard HTTP server on a daemon thread.

    Logs an error and returns (without crashing) if the port is in use.
    """
    import logging
    log = logging.getLogger("orchestrator")

    try:
        handler = _make_handler(station_manager)
        server = HTTPServer(("0.0.0.0", port), handler)
    except OSError as exc:
        log.error("Dashboard failed to start on port %d: %s", port, exc)
        return

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    log.info("Dashboard running at http://0.0.0.0:%d/", port)
