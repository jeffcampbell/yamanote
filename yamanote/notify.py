"""Notification hooks.

Yamanote doesn't talk to any messaging service itself. It hands each
notification to whatever you configure:

  AGENT_TEAM_NOTIFY_CMD      a shell command; the event JSON arrives on stdin,
                             and YAMANOTE_EVENT / _TITLE / _MESSAGE / _URL /
                             _ITEM_ID are set in its environment
  AGENT_TEAM_NOTIFY_WEBHOOK  a URL that receives the same JSON as a POST
  AGENT_TEAM_NOTIFY_EVENTS   which events to send (default:
                             gate,failed,suspended,regression,reverted,stalled;
                             "done" is also available)

Delivery runs on a background thread so a slow hook never stalls the line.
Failures are logged and reported back through `on_error`.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from . import settings

log = logging.getLogger("yamanote.notify")

EVENTS = {
    "gate": "A train is waiting at a human gate (spec or merge approval)",
    "failed": "A train was terminated (gave up, SLA breach, persistent conflict, over budget)",
    "done": "A train arrived (merged and deployed)",
    "suspended": "The line stopped departures (daily budget, out of OpenRouter credit, rate limits)",
    "regression": "New errors appeared in the app logs shortly after a deploy",
    "reverted": "A merge was reverted automatically after a regression",
    "stalled": "A project's dispatcher paused after repeated rejections",
    "autopilot": "Autopilot switched on or off (switching off includes the 'while you were away' report)",
}

_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="notify")


def payload(event: str, title: str, message: str, item: dict | None = None, data: dict | None = None) -> dict:
    out = {"event": event, "title": title, "message": message, "ts": time.time(), "data": data or {}}
    if item:
        out["item"] = {k: item.get(k) for k in ("id", "title", "kind", "project", "station", "status",
                                               "service_class", "cost_usd", "attempt", "branch")}
        if settings.PUBLIC_URL:
            out["url"] = f"{settings.PUBLIC_URL}/#item-{item['id']}"
    elif settings.PUBLIC_URL:
        out["url"] = settings.PUBLIC_URL + "/"
    return out


def enabled() -> bool:
    return bool(settings.NOTIFY_CMD or settings.NOTIFY_WEBHOOK)


def send(event: str, title: str, message: str, item: dict | None = None, data: dict | None = None,
         on_error=None):
    """Queue a notification (no-op when unconfigured or the event is filtered out)."""
    if event not in settings.NOTIFY_EVENTS or not enabled():
        return None
    body = payload(event, title, message, item, data)
    return _pool.submit(_deliver, body, on_error)


def _deliver(body: dict, on_error=None) -> list[str]:
    errors = []
    raw = json.dumps(body, default=str)
    if settings.NOTIFY_CMD:
        env = {**os.environ, "YAMANOTE_EVENT": body["event"], "YAMANOTE_TITLE": body["title"],
               "YAMANOTE_MESSAGE": body["message"], "YAMANOTE_URL": body.get("url", ""),
               "YAMANOTE_ITEM_ID": str((body.get("item") or {}).get("id", ""))}
        try:
            r = subprocess.run(["bash", "-c", settings.NOTIFY_CMD], input=raw, text=True, env=env,
                               capture_output=True, timeout=settings.NOTIFY_TIMEOUT_SECONDS)
            if r.returncode != 0:
                errors.append(f"notify command exited {r.returncode}: {(r.stderr or r.stdout)[:200]}")
        except (OSError, subprocess.TimeoutExpired) as e:
            errors.append(f"notify command failed: {e}")
    if settings.NOTIFY_WEBHOOK:
        req = urllib.request.Request(settings.NOTIFY_WEBHOOK, raw.encode(),
                                     {"Content-Type": "application/json", "User-Agent": "yamanote"})
        try:
            with urllib.request.urlopen(req, timeout=settings.NOTIFY_TIMEOUT_SECONDS) as r:
                r.read()
        except Exception as e:  # any delivery failure is reported, never raised
            errors.append(f"notify webhook failed: {e}")
    for err in errors:
        log.warning(err)
        if on_error:
            try:
                on_error(err)
            except Exception:
                pass
    return errors
