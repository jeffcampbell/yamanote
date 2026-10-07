"""Statistics for the dashboard's Stats view, computed from the SQLite store.

Everything is derived on request for a time range (and optional project):
KPIs with the previous period for comparison, per-bucket series (hourly for
ranges up to two days, daily beyond), where trains spend their time by
station, and a model leaderboard.
"""
from __future__ import annotations

import datetime as dt
import os
import statistics
import time

from . import gitops, settings
from .store import STATIONS, STATION_KEYS, Store

TERMINAL = ("done", "failed", "rejected", "cancelled")


def _buckets(since: float, now: float, hourly: bool) -> list[float]:
    start = dt.datetime.fromtimestamp(since)
    start = start.replace(minute=0, second=0, microsecond=0) if hourly else \
        start.replace(hour=0, minute=0, second=0, microsecond=0)
    step = dt.timedelta(hours=1) if hourly else dt.timedelta(days=1)
    out, t = [], start
    while t.timestamp() <= now:
        out.append(t.timestamp())
        t += step
    return out


def _bucket_of(ts: float, buckets: list[float]) -> int | None:
    if not buckets or ts < buckets[0]:
        return None
    lo, hi = 0, len(buckets) - 1
    while lo < hi:  # last bucket start <= ts
        mid = (lo + hi + 1) // 2
        if buckets[mid] <= ts:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _items(store: Store, since: float, until: float, project: str | None) -> list[dict]:
    marks = ",".join("?" * len(TERMINAL))
    rows = store._all(f"SELECT * FROM items WHERE status IN ({marks}) AND finished_at>=? AND finished_at<?"
                      + (" AND project=?" if project else ""),
                      (*TERMINAL, since, until, *([project] if project else [])))
    return rows


def _runs(store: Store, since: float, until: float, project: str | None) -> list[dict]:
    if project:
        return store._all("SELECT r.* FROM runs r JOIN items i ON i.id=r.item_id"
                          " WHERE r.started_at>=? AND r.started_at<? AND i.project=?", (since, until, project))
    return store._all("SELECT * FROM runs WHERE started_at>=? AND started_at<?", (since, until))


def merge_stat(store: Store, item: dict) -> dict | None:
    """Lines/commits a train merged, cached in kv (backfilled from git once)."""
    cached = store.kv_get(f"mergestat:{item['id']}")
    if cached is not None:
        return cached or None
    commit = store.kv_get(f"merged:{item['id']}")
    if not commit and os.path.isdir(item["project"]):
        # merged before the commit was recorded: find it by the line every merge message carries
        _, commit, _ = gitops.git("log", "--merges", "-n", "1", "--format=%H",
                                  f"--grep=^Yamanote item #{item['id']}$", cwd=item["project"])
    stat = gitops.merge_stats(item["project"], commit) if commit and os.path.isdir(item["project"]) else None
    store.kv_set(f"mergestat:{item['id']}", stat or {})
    return stat


def _period(store: Store, since: float, until: float, project: str | None) -> dict:
    items = _items(store, since, until, project)
    runs = _runs(store, since, until, project)
    done = [i for i in items if i["status"] == "done"]
    failed = [i for i in items if i["status"] == "failed"]
    journeys = [i["finished_at"] - i["started_at"] for i in done if i.get("started_at")]
    finished = done + failed
    model_runs = [r for r in runs if r["role"] not in ("ci", "jev")]
    tokens_in = sum(r["tokens_in"] or 0 for r in model_runs)
    cached = sum(r.get("cached_tokens") or 0 for r in model_runs)
    stats = [merge_stat(store, i) for i in done]
    return {
        "arrived": len(done),
        "failed": len(failed),
        "rejected": sum(1 for i in items if i["status"] == "rejected"),
        "failure_rate": (len(failed) / len(finished)) if finished else None,
        "journey_median": statistics.median(journeys) if journeys else None,
        "journey_p90": (sorted(journeys)[int(0.9 * (len(journeys) - 1))] if journeys else None),
        "spend": sum(r["cost_usd"] or 0 for r in runs),
        "cost_per_arrival": (sum(i["cost_usd"] or 0 for i in done) / len(done)) if done else None,
        "first_pass": (sum(1 for i in done if (i["attempt"] or 0) <= 1) / len(finished)) if finished else None,
        "satisfaction": (statistics.mean(v) if (v := [i["satisfaction"] for i in done if i["satisfaction"] is not None])
                         else None),
        "merges": len(done),
        "lines_added": sum(s["insertions"] for s in stats if s),
        "lines_removed": sum(s["deletions"] for s in stats if s),
        "commits": sum(s["commits"] for s in stats if s),
        "cache_rate": (cached / tokens_in) if tokens_in else None,
        "llm_runs": len(model_runs),
        "jev_decisions": sum(1 for r in runs if r["role"] == "jev"),
        "jev_cost": sum(r["cost_usd"] or 0 for r in runs if r["role"] == "jev"),
    }


def _station_times(store: Store, items: list[dict]) -> list[dict]:
    """Average minutes per train at each station, split into time an agent or
    check was working and time spent waiting (queues, gates, backoff)."""
    totals = {k: {"total": 0.0, "working": 0.0, "trains": 0} for k in STATION_KEYS}
    for item in items:
        if not item.get("finished_at"):
            continue
        events = [e for e in store.events(item["id"], limit=2000) if e["station"] in totals]
        segments: dict[str, float] = {}
        for cur, nxt in zip(events, events[1:] + [None]):
            end = nxt["ts"] if nxt else item["finished_at"]
            segments[cur["station"]] = segments.get(cur["station"], 0.0) + max(0.0, end - cur["ts"])
        working: dict[str, float] = {}
        for r in store.runs(item["id"]):
            if r["station"] in totals and r["ended_at"]:
                working[r["station"]] = working.get(r["station"], 0.0) + (r["ended_at"] - r["started_at"])
        for st, secs in segments.items():
            totals[st]["total"] += secs
            totals[st]["working"] += min(working.get(st, 0.0), secs)
            totals[st]["trains"] += 1
    n = max(1, sum(1 for i in items if i.get("finished_at")))
    return [{"station": key, "code": code, "name": name,
             "working_min": totals[key]["working"] / n / 60,
             "waiting_min": max(0.0, totals[key]["total"] - totals[key]["working"]) / n / 60}
            for key, code, name, _ in STATIONS if key != "intake" or totals[key]["total"]]


def _models(runs: list[dict]) -> list[dict]:
    by: dict[str, dict] = {}
    for r in runs:
        if r["role"] == "ci":
            continue
        m = by.setdefault(r["model"] or "?", {"model": r["model"] or "?", "runs": 0, "tokens_in": 0, "tokens_out": 0,
                                               "cached": 0, "cost": 0.0, "seconds": 0.0, "roles": set(),
                                               "errors": 0})
        m["runs"] += 1
        m["tokens_in"] += r["tokens_in"] or 0
        m["tokens_out"] += r["tokens_out"] or 0
        m["cached"] += r.get("cached_tokens") or 0
        m["cost"] += r["cost_usd"] or 0
        m["seconds"] += ((r["ended_at"] or r["started_at"]) - r["started_at"])
        m["roles"].add("jev" if r["role"] == "jev" else r["role"])
        m["errors"] += 1 if r["status"] not in ("ok", "running") else 0
    total = sum(m["cost"] for m in by.values()) or 1
    out = []
    for m in by.values():
        out.append({**m, "roles": sorted(m["roles"]), "share": m["cost"] / total,
                    "cache_rate": (m["cached"] / m["tokens_in"]) if m["tokens_in"] else None,
                    "avg_cost": m["cost"] / m["runs"] if m["runs"] else 0,
                    "avg_seconds": m["seconds"] / m["runs"] if m["runs"] else 0})
    return sorted(out, key=lambda m: m["cost"], reverse=True)


def compute(store: Store, days: float = 7, project: str | None = None, now: float | None = None) -> dict:
    now = now or time.time()
    since = now - days * 86400
    hourly = days <= 2
    buckets = _buckets(since, now, hourly)
    since = min(since, buckets[0]) if buckets else since
    items = _items(store, since, now + 1, project)
    runs = _runs(store, since, now + 1, project)

    series = [{"t": b, "spend": 0.0, "done": 0, "failed": 0, "rejected": 0, "finished": 0, "first_pass": 0,
               "sat_sum": 0.0, "sat_n": 0, "merges": 0, "lines_added": 0, "lines_removed": 0} for b in buckets]
    for r in runs:
        i = _bucket_of(r["started_at"], buckets)
        if i is not None:
            series[i]["spend"] += r["cost_usd"] or 0
    for it in items:
        i = _bucket_of(it["finished_at"], buckets)
        if i is None:
            continue
        b = series[i]
        if it["status"] in ("done", "failed", "rejected"):
            b[it["status"]] += 1
        if it["status"] in ("done", "failed"):
            b["finished"] += 1
            if it["status"] == "done" and (it["attempt"] or 0) <= 1:
                b["first_pass"] += 1
        if it["status"] == "done":
            b["merges"] += 1
            if it["satisfaction"] is not None:
                b["sat_sum"] += it["satisfaction"]
                b["sat_n"] += 1
            stat = merge_stat(store, it)
            if stat:
                b["lines_added"] += stat["insertions"]
                b["lines_removed"] += stat["deletions"]
    for b in series:
        b["first_pass_rate"] = (b["first_pass"] / b["finished"]) if b["finished"] else None
        b["satisfaction"] = (b["sat_sum"] / b["sat_n"]) if b["sat_n"] else None
        for k in ("first_pass", "sat_sum", "sat_n"):
            del b[k]

    current = _period(store, since, now + 1, project)
    previous = _period(store, since - (now - since), since, project)
    events = store.events_since(since, project, ("arrived", "rejected", "held", "playbook", "disputed"))
    efficiency = {
        "jev_fast_pass": sum(1 for e in events if e["kind"] == "arrived" and e["message"].startswith("Jev fast-pass")),
        "jev_rejects": sum(1 for e in events if e["kind"] == "rejected" and "Jev:" in e["message"]),
        "jev_holds": sum(1 for e in events if e["kind"] == "held" and "Jev:" in e["message"]),
        "playbook_notes": sum(1 for e in events if e["kind"] == "playbook" and e["message"].startswith("New ")),
        "disputes": sum(1 for e in events if e["kind"] == "disputed"),
    }
    return {
        "since": since, "now": now, "days": days, "bucket": "hour" if hourly else "day", "project": project,
        "kpis": current, "previous": previous, "efficiency": efficiency,
        "series": series,
        "daily_budget": settings.DAILY_BUDGET_USD,
        "stations": _station_times(store, [i for i in items if i["status"] == "done"]),
        "models": _models(runs),
    }
