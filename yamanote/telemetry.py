"""OpenTelemetry export (OTLP over HTTP with JSON encoding, stdlib only).

Configured with the standard OpenTelemetry environment variables, so it plugs
into any collector or backend that speaks OTLP/HTTP (OpenTelemetry Collector,
Grafana Alloy/Tempo/Mimir, Jaeger, SigNoz, Honeycomb, Langfuse, ...):

  OTEL_EXPORTER_OTLP_ENDPOINT           base URL, e.g. http://localhost:4318
  OTEL_EXPORTER_OTLP_TRACES_ENDPOINT    full URL override for traces
  OTEL_EXPORTER_OTLP_METRICS_ENDPOINT   full URL override for metrics
  OTEL_EXPORTER_OTLP_HEADERS            "key=value,key2=value2" (URL-encoded values)
  OTEL_EXPORTER_OTLP_PROTOCOL           must be http/json (the only one stdlib can speak)
  OTEL_SERVICE_NAME                     default "yamanote"
  OTEL_RESOURCE_ATTRIBUTES              "key=value,..." added to the resource
  OTEL_METRIC_EXPORT_INTERVAL           milliseconds, default 60000
  OTEL_SDK_DISABLED                     "true" turns export off

Traces: one trace per train, written when its journey closes —
  train span → one span per station visit → agent runs ("invoke_agent {role}"
  or "chat {model}") → model turns ("chat {model}") and tool calls
  ("execute_tool {name}"), using the GenAI semantic conventions
  (gen_ai.operation.name, gen_ai.provider.name, gen_ai.request.model,
  gen_ai.usage.input_tokens / output_tokens / cache_read.input_tokens, ...).
  Line-wide runs (Dispatcher, Signal, Ops) are traces of their own.
Metrics: cumulative counters, a journey-time histogram and gauges, recomputed
  from the database each interval (so they survive restarts).

Export reads the database on a cursor that only advances after the backend
accepts a batch, so nothing is lost while a collector is down.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from . import settings
from .store import STATIONS, Store

log = logging.getLogger("yamanote.telemetry")

SCOPE = {"name": "yamanote", "version": "1.0"}
CHAT_ROLES = {"redactor", "retrospective", "signal", "ops"}  # single model call, no tools
JOURNEY_BUCKETS = [60, 300, 600, 1800, 3600, 7200, 14400, 43200]  # seconds
SPAN_LIMIT = 2000
KIND_INTERNAL, KIND_CLIENT = 1, 3
STATUS_OK, STATUS_ERROR = 1, 2


# ─── configuration ───────────────────────────────────────────────────────────

def _parse_pairs(raw: str) -> dict:
    out = {}
    for part in (raw or "").split(","):
        if "=" in part:
            k, _, v = part.partition("=")
            if k.strip():
                out[k.strip()] = urllib.parse.unquote(v.strip())
    return out


def config() -> dict:
    env = os.environ
    base = env.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").rstrip("/")
    traces = env.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") or (base + "/v1/traces" if base else "")
    metrics = env.get("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT") or (base + "/v1/metrics" if base else "")
    protocol = env.get("OTEL_EXPORTER_OTLP_PROTOCOL", "http/json")
    disabled = env.get("OTEL_SDK_DISABLED", "").lower() == "true"
    problem = None
    if protocol != "http/json" and (traces or metrics):
        problem = f"OTEL_EXPORTER_OTLP_PROTOCOL={protocol} isn't supported; Yamanote speaks http/json"
    try:
        interval = max(5.0, int(env.get("OTEL_METRIC_EXPORT_INTERVAL", "60000")) / 1000)
    except ValueError:
        interval = 60.0
    return {"traces": traces, "metrics": metrics, "headers": _parse_pairs(env.get("OTEL_EXPORTER_OTLP_HEADERS", "")),
            "service": env.get("OTEL_SERVICE_NAME", "yamanote"),
            "resource": _parse_pairs(env.get("OTEL_RESOURCE_ATTRIBUTES", "")),
            "interval": interval, "enabled": bool((traces or metrics) and not disabled and not problem),
            "problem": problem}


# ─── OTLP JSON helpers ───────────────────────────────────────────────────────

def _value(v):
    if isinstance(v, bool):
        return {"boolValue": v}
    if isinstance(v, int):
        return {"intValue": str(v)}
    if isinstance(v, float):
        return {"doubleValue": v}
    if isinstance(v, (list, tuple)):
        return {"arrayValue": {"values": [_value(x) for x in v]}}
    return {"stringValue": str(v)}


def attrs(d: dict) -> list:
    return [{"key": k, "value": _value(v)} for k, v in d.items() if v is not None and v != ""]


def _nanos(ts: float) -> str:
    return str(int(ts * 1e9))


def _hex(seed: str, n: int) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()[:n]


def resource(cfg: dict) -> dict:
    return {"attributes": attrs({"service.name": cfg["service"], "service.namespace": "yamanote",
                                 "telemetry.sdk.name": "yamanote", "telemetry.sdk.language": "python",
                                 **cfg["resource"]})}


def _span(trace_id: str, span_id: str, parent: str | None, name: str, start: float, end: float,
          attributes: dict, kind: int = KIND_INTERNAL, error: str | None = None) -> dict:
    s = {"traceId": trace_id, "spanId": span_id, "name": name[:200], "kind": kind,
         "startTimeUnixNano": _nanos(start), "endTimeUnixNano": _nanos(max(end, start)),
         "attributes": attrs(attributes),
         "status": {"code": STATUS_ERROR, "message": error[:500]} if error else {"code": STATUS_OK}}
    if parent:
        s["parentSpanId"] = parent
    return s


# ─── traces ──────────────────────────────────────────────────────────────────

def _run_spans(store: Store, trace_id: str, parent: str, run: dict, now: float) -> list[dict]:
    end = run["ended_at"] or now
    role, model = run["role"], run["model"] or ""
    rid = _hex(f"{trace_id}-run-{run['id']}", 16)
    failed = run["status"] not in ("ok", "running")
    if role == "ci":
        steps = store.steps(run["id"])
        name = f"check {steps[0]['name'] if steps else 'command'}"
        return [_span(trace_id, rid, parent, name, run["started_at"], end,
                      {"yamanote.run.role": "checks", "yamanote.run.status": run["status"],
                       "yamanote.check.summary": (run["summary"] or "")[:300]},
                      error=(run["summary"] or "check failed") if failed else None)]
    common = {"gen_ai.request.model": model, "gen_ai.response.model": model,
              "gen_ai.usage.input_tokens": run["tokens_in"] or 0, "gen_ai.usage.output_tokens": run["tokens_out"] or 0,
              "gen_ai.usage.cache_read.input_tokens": run.get("cached_tokens") or 0,
              "yamanote.cost_usd": round(run["cost_usd"] or 0, 6), "yamanote.run.id": run["id"],
              "yamanote.run.role": role, "yamanote.run.status": run["status"],
              "yamanote.service_class": run.get("service_class")}
    if role == "jev":
        return [_span(trace_id, rid, parent, "decide jev", run["started_at"], end,
                      {**common, "gen_ai.operation.name": "decide", "gen_ai.provider.name": "typesafe",
                       "yamanote.decision": (run["summary"] or "")[:300]}, kind=KIND_CLIENT)]
    op = "chat" if role in CHAT_ROLES else "invoke_agent"
    name = f"chat {model}" if op == "chat" else f"invoke_agent {role}"
    out = [_span(trace_id, rid, parent, name, run["started_at"], end,
                 {**common, "gen_ai.operation.name": op, "gen_ai.provider.name": "openrouter",
                  "gen_ai.agent.name": role if op == "invoke_agent" else None},
                 kind=KIND_CLIENT if op == "chat" else KIND_INTERNAL,
                 error=(run["error"] or run["status"]) if failed else None)]
    if op == "chat":
        return out
    prev = run["started_at"]
    for st in store.steps(run["id"]):
        sid = _hex(f"{trace_id}-step-{st['id']}", 16)
        if st["kind"] == "model":
            out.append(_span(trace_id, sid, rid, f"chat {st['name'] or model}", prev, st["ts"],
                             {"gen_ai.operation.name": "chat", "gen_ai.provider.name": "openrouter",
                              "gen_ai.request.model": model, "gen_ai.response.model": st["name"] or model,
                              "gen_ai.usage.input_tokens": st["tokens_in"] or 0,
                              "gen_ai.usage.output_tokens": st["tokens_out"] or 0,
                              "yamanote.cost_usd": round(st["cost_usd"] or 0, 6)}, kind=KIND_CLIENT))
        elif st["kind"] == "tool":
            first = (st["detail"] or "").splitlines()[0][:200] if st["detail"] else ""
            out.append(_span(trace_id, sid, rid, f"execute_tool {st['name']}", prev, st["ts"],
                             {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": st["name"],
                              "gen_ai.tool.type": "function", "yamanote.tool.call": first}))
        elif st["kind"] == "error":
            out.append(_span(trace_id, sid, rid, "error", prev, st["ts"], {"yamanote.error": (st["detail"] or "")[:500]},
                             error=(st["detail"] or "error")[:500]))
        prev = st["ts"]
    return out


# Runs on internal stations belong under the line station they serve.
STATION_ALIASES = {"redact": "verify", "retro_retry": "retro", "lessons": "retro"}


def journey_start(store: Store, item: dict) -> float:
    """A retried train starts a new journey; its trace covers only that journey."""
    retried = [e["ts"] for e in store.events(item["id"], limit=5000) if e["kind"] == "retried"]
    return max(retried) if retried else item["created_at"]


def train_trace(store: Store, item: dict) -> list[dict]:
    """All spans for one train's (latest) journey."""
    now = time.time()
    begin = journey_start(store, item)
    trace_id = _hex(f"yamanote-item-{item['id']}-{begin}", 32)  # a retry gets a new trace
    root = _hex(f"{trace_id}-root", 16)
    end = item.get("finished_at") or now
    failed = item["status"] == "failed"
    spans = [_span(trace_id, root, None, f"train #{item['id']} {item['title']}", begin, end, {
        "yamanote.item.id": item["id"], "yamanote.item.title": item["title"], "yamanote.item.kind": item["kind"],
        "yamanote.item.source": item["source"], "yamanote.item.priority": item["priority"],
        "yamanote.project": os.path.basename(item["project"]), "yamanote.project.path": item["project"],
        "yamanote.item.status": item["status"], "yamanote.item.outcome": (item.get("outcome") or "")[:500],
        "yamanote.service_class": item.get("service_class"), "yamanote.first_class": item.get("first_class"),
        "yamanote.difficulty": item.get("difficulty"), "yamanote.attempts": item.get("attempt") or 0,
        "yamanote.conflicts": item.get("conflicts") or 0, "yamanote.satisfaction": item.get("satisfaction"),
        "yamanote.cost_usd": round(item.get("cost_usd") or 0, 6),
        "gen_ai.usage.input_tokens": item.get("tokens_in") or 0, "gen_ai.usage.output_tokens": item.get("tokens_out") or 0,
    }, error=item.get("outcome") if failed else None)]

    names = {k: (code, name) for k, code, name, _ in STATIONS}
    events = [e for e in store.events(item["id"], limit=5000) if e["station"] in names and e["ts"] >= begin]
    segments = []  # (station, start, end)
    for cur, nxt in zip(events, events[1:] + [None]):
        stop = nxt["ts"] if nxt else end
        if segments and segments[-1][0] == cur["station"]:
            segments[-1] = (cur["station"], segments[-1][1], stop)
        else:
            segments.append((cur["station"], cur["ts"], stop))
    seg_ids = []
    for n, (station, seg_start, seg_stop) in enumerate(segments):
        sid = _hex(f"{trace_id}-station-{n}", 16)
        seg_ids.append((station, seg_start, seg_stop, sid))
        code, name = names[station]
        spans.append(_span(trace_id, sid, root, f"station {code} {name}", seg_start, seg_stop,
                           {"yamanote.station": station, "yamanote.station.code": code, "yamanote.visit": n}))

    for run in store.runs(item["id"]):
        if run["started_at"] < begin - 1:
            continue  # an earlier journey's run
        station = STATION_ALIASES.get(run["station"], run["station"])
        parent = next((sid for st, a, b, sid in seg_ids if st == station and a - 1 <= run["started_at"] <= b + 1),
                      root)
        spans.extend(_run_spans(store, trace_id, parent, run, now))
        if len(spans) >= SPAN_LIMIT:
            break
    return spans[:SPAN_LIMIT]


def line_run_trace(store: Store, run: dict) -> list[dict]:
    """A line-wide run (Dispatcher, Signal, Ops) as its own trace."""
    trace_id = _hex(f"yamanote-run-{run['id']}-{run['started_at']}", 32)
    return _run_spans(store, trace_id, None, run, time.time())


def traces_payload(cfg: dict, spans: list[dict]) -> dict:
    return {"resourceSpans": [{"resource": resource(cfg), "scopeSpans": [{"scope": SCOPE, "spans": spans}]}]}


# ─── metrics ─────────────────────────────────────────────────────────────────

def metrics_payload(cfg: dict, store: Store, factory=None, now: float | None = None) -> dict:
    now = now or time.time()
    first = store._one("SELECT MIN(created_at) AS t FROM items")
    start = (first or {}).get("t") or now
    t0, t1 = _nanos(start), _nanos(now)

    def point(value, attributes=None, double=False):
        p = {"attributes": attrs(attributes or {}), "startTimeUnixNano": t0, "timeUnixNano": t1}
        p["asDouble" if double else "asInt"] = float(value) if double else str(int(value))
        return p

    def counter(name, unit, desc, points):
        return {"name": name, "unit": unit, "description": desc,
                "sum": {"aggregationTemporality": 2, "isMonotonic": True, "dataPoints": points}}

    def gauge(name, unit, desc, points):
        return {"name": name, "unit": unit, "description": desc, "gauge": {"dataPoints": points}}

    metrics = []
    # Counted from journey-end events, which are append-only (and kept by retention), so
    # the counter never goes down — e.g. when a failed train is retried.
    rows = store._all("SELECT e.kind AS status, i.project AS project, COUNT(*) AS n FROM events e"
                      " JOIN items i ON i.id = e.item_id WHERE e.kind IN ('done','failed','rejected','cancelled')"
                      " GROUP BY e.kind, i.project")
    metrics.append(counter("yamanote.trains.finished", "{train}", "Journeys that ended, by outcome",
                           [point(r["n"], {"yamanote.item.status": r["status"],
                                           "yamanote.project": os.path.basename(r["project"])}) for r in rows]))
    by_model = store._all("SELECT model, role, status, COUNT(*) AS n, SUM(tokens_in) AS tin, SUM(tokens_out) AS tout,"
                          " SUM(cached_tokens) AS cached, SUM(cost_usd) AS cost FROM runs WHERE role != 'ci'"
                          " GROUP BY model, role, status")
    # Tokens and cost only ever grow, so in-progress runs count toward them; the run
    # counter only counts finished runs (a status label that later changes would make
    # a "running" series go down).
    tok, cost, runs = [], [], []
    agg: dict[str, dict] = {}
    for r in by_model:
        model = r["model"] or "unknown"
        if r["status"] != "running":
            runs.append(point(r["n"], {"gen_ai.request.model": model, "yamanote.run.role": r["role"],
                                       "yamanote.run.status": r["status"]}))
        a = agg.setdefault(model, {"in": 0, "out": 0, "cached": 0, "cost": 0.0})
        a["in"] += r["tin"] or 0
        a["out"] += r["tout"] or 0
        a["cached"] += r["cached"] or 0
        a["cost"] += r["cost"] or 0
    for model, a in agg.items():
        for kind in ("in", "out", "cached"):
            tok.append(point(a[kind], {"gen_ai.request.model": model,
                                       "gen_ai.token.type": {"in": "input", "out": "output", "cached": "cache_read"}[kind]}))
        cost.append(point(a["cost"], {"gen_ai.request.model": model}, double=True))
    metrics.append(counter("yamanote.llm.tokens", "{token}", "Tokens used by agent runs, by model and type", tok))
    metrics.append(counter("yamanote.llm.cost", "USD", "Spend reported by OpenRouter (and Jev), by model", cost))
    metrics.append(counter("yamanote.agent.runs", "{run}", "Agent runs by model, role and outcome", runs))

    journeys = [r["d"] for r in store._all("SELECT finished_at - started_at AS d FROM items WHERE status='done'"
                                           " AND started_at IS NOT NULL AND finished_at IS NOT NULL")]
    counts = [0] * (len(JOURNEY_BUCKETS) + 1)
    for d in journeys:
        counts[next((i for i, b in enumerate(JOURNEY_BUCKETS) if d <= b), len(JOURNEY_BUCKETS))] += 1
    hist = {"attributes": [], "startTimeUnixNano": t0, "timeUnixNano": t1, "count": str(len(journeys)),
            "sum": float(sum(journeys)), "bucketCounts": [str(c) for c in counts],
            "explicitBounds": [float(b) for b in JOURNEY_BUCKETS]}
    if journeys:
        hist.update(min=float(min(journeys)), max=float(max(journeys)))
    metrics.append({"name": "yamanote.train.journey.duration", "unit": "s",
                    "description": "Time from boarding a train set to arrival, for trains that arrived",
                    "histogram": {"aggregationTemporality": 2, "dataPoints": [hist]}})

    added = removed = merges = 0
    for r in store._all("SELECT key, value FROM kv WHERE key LIKE 'mergestat:%'"):
        stat = json.loads(r["value"]) or {}
        if stat:
            merges += 1
            added += stat.get("insertions", 0)
            removed += stat.get("deletions", 0)
    metrics.append(counter("yamanote.merges", "{merge}", "Trains merged to trunk", [point(merges)]))
    metrics.append(counter("yamanote.lines.changed", "{line}", "Lines merged to trunk",
                           [point(added, {"yamanote.change.type": "added"}),
                            point(removed, {"yamanote.change.type": "removed"})]))

    active = store._all("SELECT status, COUNT(*) AS n FROM items WHERE status IN ('queued','running','waiting','held')"
                        " GROUP BY status")
    metrics.append(gauge("yamanote.trains.active", "{train}", "Trains on the line, by status",
                         [point(r["n"], {"yamanote.item.status": r["status"]}) for r in active] or [point(0)]))
    midnight = time.mktime(time.localtime(now)[:3] + (0, 0, 0, 0, 0, -1))
    metrics.append(gauge("yamanote.spend.today", "USD", "Spend since local midnight",
                         [point(store.spend_since(midnight), double=True)]))
    if factory is not None:
        metrics.append(gauge("yamanote.autopilot", "1", "1 while the line runs in autopilot",
                             [point(1 if factory.autopilot_on else 0)]))
        metrics.append(gauge("yamanote.paused", "1", "1 while the line is paused", [point(1 if factory.paused else 0)]))
    return {"resourceMetrics": [{"resource": resource(cfg), "scopeMetrics": [{"scope": SCOPE, "metrics": metrics}]}]}


# ─── exporter ────────────────────────────────────────────────────────────────

class Exporter:
    """Background thread: ships finished trains as traces and periodic metrics."""

    def __init__(self, store: Store, factory=None, cfg: dict | None = None):
        self.store, self.factory = store, factory
        self.cfg = cfg or config()
        self._stop = threading.Event()
        self._last_metrics = 0.0
        self.last_error: str | None = None
        self.exported = {"traces": 0, "metrics": 0}

    def status(self) -> dict:
        return {"enabled": self.cfg["enabled"], "traces_url": self.cfg["traces"], "metrics_url": self.cfg["metrics"],
                "problem": self.cfg["problem"], "last_error": self.last_error,
                "traces_sent": self.exported["traces"], "metric_exports": self.exported["metrics"]}

    def start(self):
        if not self.cfg["enabled"]:
            if self.cfg["problem"]:
                log.warning(self.cfg["problem"])
            return None
        if self.store.kv_get("otel_cursor") is None:  # first run: start from now, don't replay history
            self.store.kv_set("otel_cursor", {"items": time.time(), "item_id": 0, "runs": time.time(), "run_id": 0})
        t = threading.Thread(target=self._loop, daemon=True, name="otel")
        t.start()
        log.info("OpenTelemetry export on: traces → %s, metrics → %s", self.cfg["traces"] or "-", self.cfg["metrics"] or "-")
        return t

    def stop(self):
        self._stop.set()

    def _loop(self):
        backoff = 5.0
        while not self._stop.is_set():
            try:
                self.flush()
                backoff = 5.0
            except Exception as e:  # never let telemetry hurt the line
                self.last_error = str(e)[:300]
                log.warning("OTLP export failed: %s", e)
                backoff = min(backoff * 2, 300)
            self._stop.wait(backoff if self.last_error else 10)

    def flush(self, force_metrics: bool = False):
        """Export pending traces and (if due) metrics. Raises on a failed POST."""
        self.last_error = None
        if self.cfg["traces"]:
            self._export_traces()
        if self.cfg["metrics"] and (force_metrics or time.time() - self._last_metrics >= self.cfg["interval"]):
            self._post(self.cfg["metrics"], metrics_payload(self.cfg, self.store, self.factory))
            self._last_metrics = time.time()
            self.exported["metrics"] += 1

    def _export_traces(self):
        # The cursor is (time, id) so trains finishing at the same instant across a
        # batch boundary are never skipped.
        cur = self.store.kv_get("otel_cursor") or {}
        cur.setdefault("items", 0); cur.setdefault("item_id", 0); cur.setdefault("runs", 0); cur.setdefault("run_id", 0)
        items = self.store._all("SELECT * FROM items WHERE status IN ('done','failed','rejected','cancelled')"
                                " AND (finished_at > ? OR (finished_at = ? AND id > ?)) ORDER BY finished_at, id LIMIT 20",
                                (cur["items"], cur["items"], cur["item_id"]))
        if items:
            spans = []
            for it in items:
                spans.extend(train_trace(self.store, self.store.get_item(it["id"])))
            self._post(self.cfg["traces"], traces_payload(self.cfg, spans))
            cur["items"], cur["item_id"] = items[-1]["finished_at"], items[-1]["id"]
            self.store.kv_set("otel_cursor", cur)
            self.exported["traces"] += len(items)
        runs = self.store._all("SELECT * FROM runs WHERE item_id IS NULL AND ended_at IS NOT NULL"
                               " AND (ended_at > ? OR (ended_at = ? AND id > ?)) ORDER BY ended_at, id LIMIT 50",
                               (cur["runs"], cur["runs"], cur["run_id"]))
        if runs:
            spans = []
            for r in runs:
                spans.extend(line_run_trace(self.store, r))
            self._post(self.cfg["traces"], traces_payload(self.cfg, spans))
            cur["runs"], cur["run_id"] = runs[-1]["ended_at"], runs[-1]["id"]
            self.store.kv_set("otel_cursor", cur)
            self.exported["traces"] += len(runs)

    def _post(self, url: str, body: dict):
        req = urllib.request.Request(url, json.dumps(body).encode(),
                                     {"Content-Type": "application/json", **self.cfg["headers"]}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                r.read()
        except urllib.error.HTTPError as e:
            detail = e.read()[:200].decode(errors="replace")
            e.close()
            raise RuntimeError(f"{url} → HTTP {e.code}: {detail}") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            raise RuntimeError(f"{url} unreachable: {e}") from None
