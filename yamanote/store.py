"""SQLite store: work items (trains), their event timeline, agent runs, and
each run's steps. Replaces the old folder message bus and activity.log.

Thread-safe: one connection guarded by a lock (agent runs write from worker
threads; the dashboard reads from its HTTP threads).
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

# The loop line. Each work item travels these stations in order; rework sends
# it back to BUILD. Codes follow the JR Yamanote numbering style.
STATIONS = [
    ("intake",  "JY01", "Intake",   "Work enters the line from Dispatcher, Signal, or a human"),
    ("triage",  "JY02", "Triage",   "Worth building now? Jev pre-screen, then an LLM gate"),
    ("spec",    "JY03", "Spec",     "Acceptance criteria, plan, and holdout scenarios"),
    ("build",   "JY04", "Build",    "Builder agent implements on a branch in a worktree"),
    ("inspect", "JY05", "Inspect",  "Code review of the diff against the spec"),
    ("verify",  "JY06", "Verify",   "Holdout scenarios run against the build"),
    ("merge",   "JY07", "Merge",    "Merge to trunk (optional human gate)"),
    ("deploy",  "JY08", "Deploy",   "Restart or deploy the service"),
    ("retro",   "JY09", "Retro",    "Retrospective: what this journey teaches the factory"),
]
STATION_KEYS = [s[0] for s in STATIONS]

# queued: waiting at a station for capacity; running: an agent/job is working;
# waiting: held at a human gate; held: triage HOLD, recycled later;
# done/rejected/failed/cancelled: terminal.
ACTIVE_STATUSES = ("queued", "running", "waiting")
TERMINAL_STATUSES = ("done", "rejected", "failed", "cancelled")

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'feature',
    source TEXT NOT NULL DEFAULT 'human',
    project TEXT NOT NULL,
    priority TEXT NOT NULL DEFAULT 'medium',
    description TEXT NOT NULL DEFAULT '',
    spec TEXT,              -- JSON: acceptance criteria, plan, relevant files
    scenarios TEXT,         -- JSON list: holdout scenarios (never shown to the builder)
    difficulty TEXT,
    difficulty_score REAL,
    service_class TEXT,
    station TEXT NOT NULL DEFAULT 'intake',
    status TEXT NOT NULL DEFAULT 'queued',
    gate TEXT,
    branch TEXT,
    worktree TEXT,
    train TEXT,
    attempt INTEGER NOT NULL DEFAULT 0,
    conflicts INTEGER NOT NULL DEFAULT 0,
    feedback TEXT,
    satisfaction REAL,
    cost_usd REAL NOT NULL DEFAULT 0,
    tokens_in INTEGER NOT NULL DEFAULT 0,
    tokens_out INTEGER NOT NULL DEFAULT 0,
    outcome TEXT,
    hold_until REAL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL
);
CREATE INDEX IF NOT EXISTS items_status ON items(status);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id INTEGER,
    ts REAL NOT NULL,
    station TEXT,
    kind TEXT NOT NULL,
    message TEXT NOT NULL,
    data TEXT
);
CREATE INDEX IF NOT EXISTS events_item ON events(item_id, id);

CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id INTEGER,
    role TEXT NOT NULL,
    station TEXT,
    model TEXT,
    service_class TEXT,
    status TEXT NOT NULL DEFAULT 'running',
    started_at REAL NOT NULL,
    ended_at REAL,
    steps INTEGER NOT NULL DEFAULT 0,
    tokens_in INTEGER NOT NULL DEFAULT 0,
    tokens_out INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0,
    summary TEXT,
    error TEXT
);
CREATE INDEX IF NOT EXISTS runs_item ON runs(item_id, id);
CREATE INDEX IF NOT EXISTS runs_started ON runs(started_at);

CREATE TABLE IF NOT EXISTS steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    ts REAL NOT NULL,
    kind TEXT NOT NULL,
    name TEXT,
    detail TEXT,
    tokens_in INTEGER NOT NULL DEFAULT 0,
    tokens_out INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS steps_run ON steps(run_id, id);

CREATE TABLE IF NOT EXISTS retros (
    item_id INTEGER PRIMARY KEY,
    ts REAL NOT NULL,
    outcome TEXT,           -- done | failed
    class_fit TEXT,         -- right | underpowered | overpowered
    data TEXT NOT NULL      -- JSON: summary, went_well, went_wrong, root_cause, notes added/retired
);

CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""

ITEM_FIELDS = {
    "title", "kind", "source", "project", "priority", "description", "spec", "scenarios",
    "difficulty", "difficulty_score", "service_class", "station", "status", "gate", "branch",
    "worktree", "train", "attempt", "conflicts", "feedback", "satisfaction", "outcome",
    "parent_id", "holds", "first_class", "pending_status",
    "hold_until", "started_at", "finished_at",
}
JSON_FIELDS = ("spec", "scenarios")


class Store:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=NORMAL")
            self._db.executescript(SCHEMA)
            self._migrate()
        self._listeners: list = []

    # ─── low level ──────────────────────────────────────────────────────

    def _exec(self, sql: str, params=()) -> sqlite3.Cursor:
        with self._lock:
            return self._db.execute(sql, params)

    def _all(self, sql: str, params=()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._db.execute(sql, params).fetchall()]

    def _one(self, sql: str, params=()) -> dict | None:
        with self._lock:
            row = self._db.execute(sql, params).fetchone()
        return dict(row) if row else None

    # Columns added after the first release; existing databases get them on open.
    MIGRATIONS = {
        "items": [("parent_id", "INTEGER"), ("holds", "INTEGER NOT NULL DEFAULT 0"), ("first_class", "TEXT"),
                  ("pending_status", "TEXT")],
        "runs": [("cached_tokens", "INTEGER NOT NULL DEFAULT 0")],
    }

    def _migrate(self) -> None:
        for table, cols in self.MIGRATIONS.items():
            have = {r[1] for r in self._db.execute(f"PRAGMA table_info({table})")}
            for name, decl in cols:
                if name not in have:
                    self._db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def subscribe(self, fn) -> None:
        """fn(kind, payload) is called after every event/step write (for SSE)."""
        self._listeners.append(fn)

    def unsubscribe(self, fn) -> None:
        if fn in self._listeners:
            self._listeners.remove(fn)

    def _notify(self, kind: str, payload: dict) -> None:
        for fn in list(self._listeners):
            try:
                fn(kind, payload)
            except Exception:
                pass

    # ─── items ──────────────────────────────────────────────────────────

    def create_item(self, *, title: str, project: str, description: str = "", kind: str = "feature",
                    source: str = "human", priority: str = "medium", station: str = "triage") -> dict:
        now = time.time()
        cur = self._exec(
            "INSERT INTO items (title, kind, source, project, priority, description, station, status,"
            " created_at, updated_at) VALUES (?,?,?,?,?,?,?,'queued',?,?)",
            (title, kind, source, project, priority, description, station, now, now))
        item = self.get_item(cur.lastrowid)
        self.event(item["id"], "intake", "created",
                   f"{source} added {kind} '{title}'", {"priority": priority})
        return item

    def get_item(self, item_id: int) -> dict | None:
        return _decode(self._one("SELECT * FROM items WHERE id=?", (item_id,)))

    def update_item(self, item_id: int, **fields) -> dict:
        bad = set(fields) - ITEM_FIELDS
        if bad:
            raise ValueError(f"unknown item fields: {bad}")
        for f in JSON_FIELDS:
            if f in fields and fields[f] is not None and not isinstance(fields[f], str):
                fields[f] = json.dumps(fields[f])
        fields["updated_at"] = time.time()
        cols = ", ".join(f"{k}=?" for k in fields)
        self._exec(f"UPDATE items SET {cols} WHERE id=?", (*fields.values(), item_id))
        item = self.get_item(item_id)
        self._notify("item", _public_item(item))
        return item

    def transition(self, item_id: int, expect: tuple[str, ...], **fields) -> dict | None:
        """Compare-and-set: update only if the item's status is still one of
        `expect`. Returns the updated item, or None if someone else (a human
        cancel, the SLA reaper) changed it first."""
        bad = set(fields) - ITEM_FIELDS
        if bad:
            raise ValueError(f"unknown item fields: {bad}")
        for f in JSON_FIELDS:
            if f in fields and fields[f] is not None and not isinstance(fields[f], str):
                fields[f] = json.dumps(fields[f])
        fields["updated_at"] = time.time()
        cols = ", ".join(f"{k}=?" for k in fields)
        marks = ",".join("?" * len(expect))
        cur = self._exec(f"UPDATE items SET {cols} WHERE id=? AND status IN ({marks})",
                         (*fields.values(), item_id, *expect))
        if cur.rowcount == 0:
            return None
        item = self.get_item(item_id)
        self._notify("item", _public_item(item))
        return item

    def add_item_cost(self, item_id: int, cost: float, tokens_in: int, tokens_out: int) -> None:
        self._exec("UPDATE items SET cost_usd=cost_usd+?, tokens_in=tokens_in+?, tokens_out=tokens_out+?,"
                   " updated_at=? WHERE id=?", (cost, tokens_in, tokens_out, time.time(), item_id))

    def items(self, statuses: tuple[str, ...] | None = None, limit: int = 500) -> list[dict]:
        if statuses:
            marks = ",".join("?" * len(statuses))
            rows = self._all(f"SELECT * FROM items WHERE status IN ({marks}) ORDER BY id LIMIT ?",
                             (*statuses, limit))
        else:
            rows = self._all("SELECT * FROM items ORDER BY id DESC LIMIT ?", (limit,))
        return [_decode(r) for r in rows]

    def recent_terminal(self, limit: int = 50) -> list[dict]:
        marks = ",".join("?" * len(TERMINAL_STATUSES))
        return [_decode(r) for r in self._all(
            f"SELECT * FROM items WHERE status IN ({marks}) ORDER BY finished_at DESC, id DESC LIMIT ?",
            (*TERMINAL_STATUSES, limit))]

    def history(self, project: str, statuses: tuple[str, ...], days: int = 30, limit: int = 20) -> list[dict]:
        marks = ",".join("?" * len(statuses))
        return [_decode(r) for r in self._all(
            f"SELECT * FROM items WHERE project=? AND status IN ({marks}) AND updated_at>? "
            "ORDER BY updated_at DESC LIMIT ?",
            (project, *statuses, time.time() - days * 86400, limit))]

    # ─── events ─────────────────────────────────────────────────────────

    def event(self, item_id: int | None, station: str | None, kind: str, message: str,
              data: dict | None = None) -> None:
        ts = time.time()
        cur = self._exec("INSERT INTO events (item_id, ts, station, kind, message, data) VALUES (?,?,?,?,?,?)",
                         (item_id, ts, station, kind, message, json.dumps(data) if data else None))
        self._notify("event", {"id": cur.lastrowid, "item_id": item_id, "ts": ts, "station": station,
                               "kind": kind, "message": message, "data": data})

    def events(self, item_id: int | None = None, limit: int = 200, after_id: int = 0) -> list[dict]:
        if item_id is None:
            rows = self._all("SELECT * FROM events WHERE id>? ORDER BY id DESC LIMIT ?", (after_id, limit))
        else:
            rows = self._all("SELECT * FROM events WHERE item_id=? AND id>? ORDER BY id DESC LIMIT ?",
                             (item_id, after_id, limit))
        for r in rows:
            r["data"] = json.loads(r["data"]) if r["data"] else None
        return list(reversed(rows))

    # ─── runs & steps ───────────────────────────────────────────────────

    def start_run(self, item_id: int | None, role: str, station: str | None, model: str,
                  service_class: str | None = None) -> int:
        cur = self._exec("INSERT INTO runs (item_id, role, station, model, service_class, started_at)"
                         " VALUES (?,?,?,?,?,?)", (item_id, role, station, model, service_class, time.time()))
        run = self.get_run(cur.lastrowid)
        self._notify("run", run)
        return cur.lastrowid

    def add_step(self, run_id: int, kind: str, name: str, detail: str, tokens_in: int = 0,
                 tokens_out: int = 0, cost_usd: float = 0.0, model: str = "", cached_tokens: int = 0) -> None:
        ts = time.time()
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO steps (run_id, ts, kind, name, detail, tokens_in, tokens_out, cost_usd)"
                " VALUES (?,?,?,?,?,?,?,?)", (run_id, ts, kind, name, detail, tokens_in, tokens_out, cost_usd))
            self._db.execute(
                "UPDATE runs SET steps=steps+?, tokens_in=tokens_in+?, tokens_out=tokens_out+?,"
                " cached_tokens=cached_tokens+?, cost_usd=cost_usd+?, model=COALESCE(NULLIF(?, ''), model)"
                " WHERE id=?",
                (1 if kind == "model" else 0, tokens_in, tokens_out, cached_tokens, cost_usd, model, run_id))
            row = self._db.execute("SELECT item_id FROM runs WHERE id=?", (run_id,)).fetchone()
        if row and row["item_id"] and (cost_usd or tokens_in or tokens_out):
            self.add_item_cost(row["item_id"], cost_usd, tokens_in, tokens_out)
        self._notify("step", {"id": cur.lastrowid, "run_id": run_id, "item_id": row["item_id"] if row else None,
                              "ts": ts, "kind": kind, "name": name, "detail": detail,
                              "cost_usd": cost_usd})

    def finish_run(self, run_id: int, status: str, summary: str = "", error: str = "") -> None:
        self._exec("UPDATE runs SET status=?, ended_at=?, summary=?, error=? WHERE id=?",
                   (status, time.time(), summary[:4000], error[:2000], run_id))
        self._notify("run", self.get_run(run_id))

    def get_run(self, run_id: int) -> dict | None:
        return self._one("SELECT * FROM runs WHERE id=?", (run_id,))

    def runs(self, item_id: int | None = None, limit: int = 100) -> list[dict]:
        if item_id is None:
            return self._all("SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,))
        return self._all("SELECT * FROM runs WHERE item_id=? ORDER BY id", (item_id,))

    def active_runs(self) -> list[dict]:
        return self._all("SELECT * FROM runs WHERE status='running' ORDER BY id")

    def steps(self, run_id: int, after_id: int = 0, limit: int = 500) -> list[dict]:
        return self._all("SELECT * FROM steps WHERE run_id=? AND id>? ORDER BY id LIMIT ?",
                         (run_id, after_id, limit))

    def interrupt_running(self) -> int:
        """On startup: runs left 'running' by a previous process are dead."""
        cur = self._exec("UPDATE runs SET status='interrupted', ended_at=? WHERE status='running'", (time.time(),))
        return cur.rowcount

    # ─── accounting ─────────────────────────────────────────────────────

    def spend_since(self, since: float) -> float:
        row = self._one("SELECT COALESCE(SUM(cost_usd),0) AS c FROM runs WHERE started_at>=?", (since,))
        return float(row["c"]) if row else 0.0

    def runs_since(self, since: float) -> int:
        row = self._one("SELECT COUNT(*) AS n FROM runs WHERE started_at>=? AND role NOT IN ('jev')", (since,))
        return int(row["n"]) if row else 0

    def stats(self) -> dict:
        day = time.time() - 86400
        out = {r["status"]: r["n"] for r in self._all("SELECT status, COUNT(*) AS n FROM items GROUP BY status")}
        done = self._all("SELECT started_at, finished_at, cost_usd FROM items WHERE status='done'"
                         " AND finished_at>? AND started_at IS NOT NULL", (day,))
        out["done_24h"] = len(done)
        out["lead_time_avg_24h"] = (sum(r["finished_at"] - r["started_at"] for r in done) / len(done)) if done else None
        out["cost_per_merge_24h"] = (sum(r["cost_usd"] for r in done) / len(done)) if done else None
        out["spend_24h"] = self.spend_since(day)
        by_model = self._all("SELECT model, COUNT(*) AS runs, SUM(cost_usd) AS cost, SUM(tokens_in) AS tin,"
                             " SUM(cached_tokens) AS cached,"
                             " SUM(tokens_out) AS tout FROM runs WHERE started_at>? GROUP BY model"
                             " ORDER BY cost DESC", (day,))
        out["by_model_24h"] = by_model
        sat = self._one("SELECT AVG(satisfaction) AS s FROM items WHERE satisfaction IS NOT NULL AND updated_at>?",
                        (time.time() - 7 * 86400,))
        out["satisfaction_7d"] = sat["s"] if sat else None
        return out

    def prune(self, older_than_days: int) -> dict:
        """Drop step-level detail and events of items that finished long ago
        (items and run summaries are kept for history and routing stats)."""
        cutoff = time.time() - older_than_days * 86400
        marks = ",".join("?" * len(TERMINAL_STATUSES))
        old = f"SELECT id FROM items WHERE status IN ({marks}) AND finished_at < ?"
        with self._lock:
            steps = self._db.execute(
                f"DELETE FROM steps WHERE run_id IN (SELECT id FROM runs WHERE item_id IN ({old})"
                " OR (item_id IS NULL AND started_at < ?))", (*TERMINAL_STATUSES, cutoff, cutoff)).rowcount
            events = self._db.execute(
                f"DELETE FROM events WHERE (item_id IN ({old})"
                " AND kind NOT IN ('created','done','failed','rejected','cancelled'))"
                " OR (item_id IS NULL AND ts < ? AND kind != 'ops_report')", (*TERMINAL_STATUSES, cutoff, cutoff)).rowcount
        return {"steps": steps, "events": events}

    def routing_stats(self, days: int = 60) -> list[dict]:
        """Per (difficulty, first service class): how often trains arrived on
        their first build attempt, what they cost, and what their retros said
        about the class. Feeds adaptive routing."""
        return self._all(
            "SELECT i.difficulty, i.first_class AS cls, COUNT(*) AS n,"
            " SUM(CASE WHEN i.status='done' AND i.attempt<=1 THEN 1 ELSE 0 END) AS first_pass,"
            " SUM(CASE WHEN i.status='done' THEN 1 ELSE 0 END) AS arrived,"
            " AVG(i.cost_usd) AS avg_cost,"
            " SUM(CASE WHEN r.class_fit='underpowered' THEN 1 ELSE 0 END) AS underpowered,"
            " SUM(CASE WHEN r.class_fit='overpowered' THEN 1 ELSE 0 END) AS overpowered,"
            " COUNT(r.item_id) AS retros"
            " FROM items i LEFT JOIN retros r ON r.item_id = i.id"
            " WHERE i.status IN ('done','failed') AND i.difficulty IS NOT NULL AND i.first_class IS NOT NULL"
            " AND i.attempt>0 AND i.finished_at>? GROUP BY i.difficulty, i.first_class",
            (time.time() - days * 86400,))

    # ─── retrospectives ─────────────────────────────────────────────────

    def save_retro(self, item_id: int, outcome: str, class_fit: str | None, data: dict) -> None:
        self._exec("INSERT INTO retros (item_id, ts, outcome, class_fit, data) VALUES (?,?,?,?,?)"
                   " ON CONFLICT(item_id) DO UPDATE SET ts=excluded.ts, outcome=excluded.outcome,"
                   " class_fit=excluded.class_fit, data=excluded.data",
                   (item_id, time.time(), outcome, class_fit, json.dumps(data)))

    def get_retro(self, item_id: int) -> dict | None:
        row = self._one("SELECT * FROM retros WHERE item_id=?", (item_id,))
        if row:
            row["data"] = json.loads(row["data"])
        return row

    def recent_retros(self, limit: int = 20) -> list[dict]:
        rows = self._all("SELECT r.*, i.title FROM retros r LEFT JOIN items i ON i.id = r.item_id"
                         " ORDER BY r.ts DESC LIMIT ?", (limit,))
        for r in rows:
            r["data"] = json.loads(r["data"])
        return rows

    DIAGRAM_KINDS = ("created", "arrived", "departed", "gate", "approved", "rework", "conflict", "held",
                     "recycled", "done", "failed", "rejected", "cancelled", "retried", "recovered", "retro")

    def diagram(self, since: float) -> list[dict]:
        """Station-vs-time points for every item active since `since`."""
        marks = ",".join("?" * len(self.DIAGRAM_KINDS))
        rows = self._all(
            f"SELECT e.item_id, e.ts, e.station, e.kind FROM events e WHERE e.item_id IN"
            f" (SELECT DISTINCT item_id FROM events WHERE ts>=? AND item_id IS NOT NULL)"
            f" AND e.kind IN ({marks}) AND e.station IS NOT NULL ORDER BY e.item_id, e.ts",
            (since, *self.DIAGRAM_KINDS))
        by_item: dict[int, dict] = {}
        for r in rows:
            entry = by_item.setdefault(r["item_id"], {"id": r["item_id"], "points": []})
            entry["points"].append([round(r["ts"], 1), r["station"], r["kind"]])
        for item_id, entry in by_item.items():
            it = self.get_item(item_id) or {}
            entry.update(title=it.get("title"), status=it.get("status"),
                         cls=it.get("first_class") or it.get("service_class"), station=it.get("station"))
        return list(by_item.values())

    def events_since(self, since: float, project: str | None = None, kinds: tuple[str, ...] | None = None) -> list[dict]:
        """Events at or after `since`, optionally for one project's items and of given kinds."""
        sql, params = "SELECT e.* FROM events e", []
        if project:
            sql += " JOIN items i ON i.id = e.item_id"
        sql += " WHERE e.ts >= ?"
        params.append(since)
        if project:
            sql += " AND i.project = ?"
            params.append(project)
        if kinds:
            sql += f" AND e.kind IN ({','.join('?' * len(kinds))})"
            params.extend(kinds)
        return self._all(sql + " ORDER BY e.id", tuple(params))

    def kv_delete_prefix(self, prefix: str) -> None:
        self._exec("DELETE FROM kv WHERE key LIKE ? ESCAPE '\\'",
                   (prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%",))

    # ─── kv (small persistent state: timers, stall flags) ───────────────

    def kv_get(self, key: str, default=None):
        row = self._one("SELECT value FROM kv WHERE key=?", (key,))
        return json.loads(row["value"]) if row else default

    def kv_set(self, key: str, value) -> None:
        self._exec("INSERT INTO kv (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                   (key, json.dumps(value)))


def _decode(row: dict | None) -> dict | None:
    if row is None:
        return None
    for f in JSON_FIELDS:
        if row.get(f):
            try:
                row[f] = json.loads(row[f])
            except (TypeError, json.JSONDecodeError):
                pass
    return row


def _public_item(item: dict | None) -> dict | None:
    """Item as shown to the UI/SSE: holdout scenarios reduced to a count."""
    if item is None:
        return None
    out = dict(item)
    scenarios = out.pop("scenarios", None) or []
    out["scenario_count"] = len(scenarios) if isinstance(scenarios, list) else 0
    return out


public_item = _public_item
