"""The factory: a tick loop that moves work items (trains) around the line.

Each queued item at a station gets a job on a worker thread; the job runs that
station's agent(s) and moves the item to its next station. Capacity: at most
MAX_TRAINS items between BUILD and DEPLOY (each holds a named train set), and
MAX_PLANNING_JOBS concurrent triage/spec jobs. Dispatcher, Signal and Ops are
line-wide jobs that feed or observe the line.

Status changes are compare-and-set (Store.transition), so a human cancel or the
SLA reaper can't be overwritten by a job finishing at the same moment.
"""
from __future__ import annotations

import glob
import json
import logging
import os
import re
import shlex
import subprocess
import threading
import datetime as dt
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor

from . import cron, decisions, gitops, notify, prompts, settings
from .agent import READ_TOOLS, RUN_TOOLS, WRITE_TOOLS, Agent, AgentResult, single_shot
from .checks import CheckResult, run_check
from .llm import UNREACHABLE, Client
from .metrics import METRICS
from .store import ACTIVE_STATUSES, STATION_KEYS, TERMINAL_STATUSES, Store

log = logging.getLogger("yamanote")

LINE_STATIONS = ("build", "inspect", "verify", "merge", "deploy")  # stations that need a train
PLANNING_STATIONS = ("triage", "spec")
PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}
_WATCH_PATTERN = re.compile(r"\b(ERROR|CRITICAL|FATAL|Traceback|panic:)\b", re.I)


class StoreRecorder:
    def __init__(self, store: Store, run_id: int):
        self.store, self.run_id = store, run_id

    def step(self, kind, name, detail, *, tokens_in=0, tokens_out=0, cost_usd=0.0, model="", cached_tokens=0):
        self.store.add_step(self.run_id, kind, name, detail, tokens_in, tokens_out, cost_usd, model, cached_tokens)


class ItemGone(Exception):
    """The item was cancelled or otherwise finished while a job was running."""


class LineSuspended(Exception):
    """OpenRouter can't serve us (no credit, bad key, persistent rate limits,
    unreachable): pause departures instead of burning every train's attempts."""

    def __init__(self, reason: str, seconds: int):
        super().__init__(reason)
        self.reason, self.seconds = reason, seconds


class OverBudget(Exception):
    """The item has spent its per-item budget."""


class Factory:
    def __init__(self, store: Store, client: Client | None = None):
        self.store = store
        self.client = client or Client()
        self.pool = ThreadPoolExecutor(max_workers=settings.MAX_TRAINS + settings.MAX_PLANNING_JOBS + 6,
                                       thread_name_prefix="yamanote")
        self.jobs: dict[str, Future] = {}
        self.job_stations: dict[str, str] = {}  # job key -> station it is working
        self.agents: dict[str, Agent] = {}
        self._lock = threading.RLock()
        self._merge_locks: dict[str, threading.Lock] = {}
        self._cancel_events: dict[int, threading.Event] = {}  # interrupts deterministic checks on cancel
        self.started_at = time.time()
        self._last_tick = time.time()
        self._log_offsets: dict[str, int] = {}
        self._railway_polled: dict[str, float] = {}
        self._railway_last: dict[str, str] = {}
        self._signal_seen: dict[str, float] = {}
        self._sig_history: dict[str, dict[str, float]] = {}  # project -> error signature -> last seen
        self._notices: dict[str, float] = {}
        self._last_gc = 0.0
        self._stopping = False
        self.trains = [f"E235-{i + 1:02d}" for i in range(settings.MAX_TRAINS)]
        self._recover()

    # ─── lifecycle ──────────────────────────────────────────────────────

    def _recover(self):
        n = self.store.interrupt_running()
        for item in self.store.items(("running",)):
            # Anything a verifier or inspector left in the worktree is junk; only
            # a builder's uncommitted work is worth keeping.
            if item.get("worktree") and item["station"] != "build" and os.path.isdir(item["worktree"]):
                gitops.discard_changes(item["worktree"])
            fields = {"status": "queued"}
            if item["station"] == "build" and (item.get("attempt") or 0) > 0:
                fields["attempt"] = item["attempt"] - 1  # the interrupted attempt didn't finish; don't count it
            self.store.update_item(item["id"], **fields)
            self.store.event(item["id"], item["station"], "recovered",
                             "Factory restarted; train re-queued at this station")
        if n:
            log.info("Marked %d interrupted runs", n)

    def run_forever(self):
        self.store.event(None, None, "line_open", f"Yamanote line open — {len(self.trains)} train sets, "
                         f"budget ${settings.DAILY_BUDGET_USD:.2f}/day, Jev {'on' if decisions.available() else 'off'}")
        try:
            while True:
                try:
                    self.tick()
                except Exception:
                    log.exception("tick failed")
                time.sleep(settings.TICK_INTERVAL)
        except KeyboardInterrupt:
            self.shutdown()

    def shutdown(self):
        self._stopping = True
        self.store.event(None, None, "line_closed", "Last train — stopping agents")
        for agent in list(self.agents.values()):
            agent.stop()
        self.pool.shutdown(wait=False, cancel_futures=True)
        deadline = time.time() + 10  # jobs blocked in an HTTP call may not notice for minutes
        while time.time() < deadline and any(not f.done() for f in list(self.jobs.values())):
            time.sleep(0.2)

    # ─── state helpers ──────────────────────────────────────────────────

    @property
    def paused(self) -> bool:
        return settings.PAUSE_FILE.exists()

    def set_paused(self, paused: bool):
        if paused:
            settings.PAUSE_FILE.parent.mkdir(parents=True, exist_ok=True)
            settings.PAUSE_FILE.touch()
            self.store.event(None, None, "paused", "Line paused — running jobs will finish, no new departures")
        else:
            settings.PAUSE_FILE.unlink(missing_ok=True)
            self.store.kv_set("suspended_until", 0)  # resuming also clears an API suspension
            self.store.event(None, None, "resumed", "Line resumed")

    def suspension(self) -> str | None:
        """Why new launches are blocked right now, if they are."""
        until = self.store.kv_get("suspended_until", 0) or 0
        if until > time.time():
            return f"{self.store.kv_get('suspended_reason', 'API problem')} (resumes in {int(until - time.time()) // 60 + 1} min)"
        midnight = time.mktime(time.localtime()[:3] + (0, 0, 0, 0, 0, -1))
        spent = self.store.spend_since(midnight)
        if spent >= settings.DAILY_BUDGET_USD:
            return f"daily budget reached (${spent:.2f} of ${settings.DAILY_BUDGET_USD:.2f})"
        n = self.store.runs_since(time.time() - 3600)
        if n >= settings.MAX_RUNS_PER_HOUR:
            return f"fare limit: {n} agent runs in the last hour (max {settings.MAX_RUNS_PER_HOUR})"
        return None

    def _suspend(self, reason: str, seconds: int):
        until = time.time() + seconds
        if (self.store.kv_get("suspended_until", 0) or 0) >= until - 5:
            return
        self.store.kv_set("suspended_until", until)
        self.store.kv_set("suspended_reason", reason)
        self.store.event(None, None, "suspended", f"SERVICE SUSPENDED for {seconds // 60} min — {reason}")
        self._notify("suspended", "Yamanote: line suspended", reason)

    def _notice(self, key: str, message: str, every: float = 900, item_id=None, station=None, notify_event=None):
        """Emit an event at most once per `every` seconds."""
        now = time.time()
        if now - self._notices.get(key, 0) >= every:
            self._notices[key] = now
            self.store.event(item_id, station, "notice", message)
            if notify_event:
                self._notify(notify_event, "Yamanote", message)

    def _notify(self, event: str, title: str, message: str, item: dict | None = None, data: dict | None = None):
        notify.send(event, title, message, item, data,
                    on_error=lambda err: self.store.event(None, None, "notice", f"Notification failed: {err}"))

    def trains_in_use(self) -> dict[str, int]:
        return {i["train"]: i["id"] for i in self.store.items(ACTIVE_STATUSES) if i.get("train")}

    def model_for(self, station: str, item: dict | None) -> tuple[str, str | None]:
        choice = settings.STATION_MODELS.get(station, "rapid")
        cls = (item or {}).get("service_class") or "rapid"
        if choice == "builder":
            c = cls
        elif choice == "builder+1":
            idx = settings.CLASS_ORDER.index(cls) if cls in settings.CLASS_ORDER else 1
            c = settings.CLASS_ORDER[min(idx + 1, len(settings.CLASS_ORDER) - 1)]
        elif choice in settings.SERVICE_CLASSES:
            c = choice
        else:
            return choice, None
        return settings.SERVICE_CLASSES[c]["model"], c

    def class_for_difficulty(self, level: str) -> tuple[str, str]:
        """Service class for a difficulty level, and why. With adaptive routing
        on, history can move a level up a class (too many reworks) or try one
        class cheaper (consistently passing first time)."""
        base = settings.DIFFICULTY_TO_CLASS.get(level, "rapid")
        if not settings.ADAPTIVE_ROUTING:
            return base, "default routing"
        stats = {(r["difficulty"], r["cls"]): r for r in self.store.routing_stats()}
        order = settings.CLASS_ORDER
        cls, why = base, "default routing"
        row = stats.get((level, base))
        if row and row["n"] >= settings.ADAPTIVE_MIN_SAMPLES:
            rate = row["first_pass"] / row["n"]
            retros = row.get("retros") or 0
            weak = retros and (row.get("underpowered") or 0) / retros >= 0.5
            overkill = retros >= settings.ADAPTIVE_MIN_SAMPLES and (row.get("overpowered") or 0) / retros >= 0.6
            if (rate < 0.5 or weak) and order.index(base) + 1 < len(order) - 1:  # never auto-route to Shinkansen
                cls = order[order.index(base) + 1]
                why = (f"adaptive: {base} passed first time on only {rate:.0%} of {row['n']} {level} items" if rate < 0.5
                       else f"adaptive: retrospectives judged {base} underpowered for {level} work")
            elif overkill and rate >= 0.8 and order.index(base) > 0:
                cls = order[order.index(base) - 1]
                why = f"adaptive: retrospectives judged {base} overkill for {level} work; trying {cls}"
            elif rate >= 0.9 and row["n"] >= 2 * settings.ADAPTIVE_MIN_SAMPLES and order.index(base) > 0:
                cheaper = order[order.index(base) - 1]
                crow = stats.get((level, cheaper))
                if not crow or crow["n"] < settings.ADAPTIVE_MIN_SAMPLES or crow["first_pass"] / crow["n"] >= 0.8:
                    cls = cheaper
                    why = f"adaptive: {base} passed first time on {rate:.0%} of {row['n']} {level} items; trying {cheaper}"
        return cls, why

    # ─── tick ───────────────────────────────────────────────────────────

    def tick(self):
        self._detect_sleep()
        self._autopilot_tick()
        self._harvest()
        self._watch_logs()
        if self.paused:
            return
        self._maintenance()
        blocked = self.suspension()
        if blocked:
            self._notice("suspended", f"SERVICE SUSPENDED — {blocked}",
                         notify_event=None if self.store.kv_get("suspended_until", 0) else "suspended")
            return
        self._dispatch_items()
        self._maybe_dispatcher()
        self._maybe_ops()

    def _detect_sleep(self):
        """After the machine sleeps, shift wall-clock deadlines by the gap so
        trains don't all breach their SLA (or skip their holds) on wake."""
        now = time.time()
        gap = now - self._last_tick
        self._last_tick = now
        threshold = max(120.0, settings.TICK_INTERVAL * 20)
        if gap <= threshold:
            return
        drift = gap - settings.TICK_INTERVAL
        for item in self.store.items(ACTIVE_STATUSES + ("held",)):
            fields = {}
            if item.get("started_at"):
                fields["started_at"] = item["started_at"] + drift
            if item.get("hold_until"):
                fields["hold_until"] = item["hold_until"] + drift
            if fields:
                self.store.update_item(item["id"], **fields)
        self.store.event(None, None, "notice", f"Wake detected after {int(gap // 60)} min asleep; deadlines shifted")

    def _harvest(self):
        with self._lock:
            for key, fut in list(self.jobs.items()):
                if not fut.done():
                    continue
                del self.jobs[key]
                self.agents.pop(key, None)
                station = self.job_stations.pop(key, None)
                err = fut.exception()
                if err and not isinstance(err, ItemGone):
                    tb = "".join(traceback.format_exception(err))[-1500:]
                    log.error("job %s crashed: %s", key, tb)
                    if key.startswith("item:"):
                        item = self.store.get_item(int(key.split(":")[1]))
                        if item and item["status"] == "running":
                            try:
                                if station == "retro":  # close the journey; never loop back into retro
                                    self._finish(item, item.get("pending_status") or "failed",
                                                 item.get("outcome") or f"internal error: {err}")
                                else:
                                    self._fail(item, f"internal error: {err}")
                            except ItemGone:
                                pass

    def _submit(self, key: str, fn, *args):
        with self._lock:
            self.jobs[key] = self.pool.submit(fn, *args)

    def _dispatch_items(self):
        now = time.time()
        queued = [i for i in self.store.items(("queued",)) if (i.get("hold_until") or 0) <= now]
        queued.sort(key=lambda i: (-STATION_KEYS.index(i["station"]),  # finish trains first
                                   PRIORITY_ORDER.get(i["priority"], 1), i["id"]))
        in_use = self.trains_in_use()
        free_trains = [t for t in self.trains if t not in in_use]
        stations_busy = list(self.job_stations.values())
        planning = sum(1 for st in stations_busy if st in PLANNING_STATIONS)
        retros = sum(1 for st in stations_busy if st == "retro")
        for item in queued:
            key = f"item:{item['id']}"
            if key in self.jobs:
                continue
            station = item["station"]
            if station in PLANNING_STATIONS:
                if planning >= settings.MAX_PLANNING_JOBS:
                    continue
                planning += 1
            if station == "retro":
                if retros >= settings.MAX_RETRO_JOBS:
                    continue
                retros += 1
            fields = {"status": "running"}
            if station in LINE_STATIONS and not item.get("train"):
                if not free_trains:
                    continue
                fields.update(train=free_trains[0], started_at=item.get("started_at") or now)
            if not self.store.transition(item["id"], ("queued",), **fields):
                continue  # cancelled between the query and now
            if "train" in fields:
                free_trains.pop(0)
                self.store.event(item["id"], station, "departed", f"Boarded train set {fields['train']}")
            self.job_stations[key] = station
            self._submit(key, self._station_job, item["id"], station)

    # ─── station job wrapper & transitions ──────────────────────────────

    def _station_job(self, item_id: int, station: str):
        item = self.store.get_item(item_id)
        if not item or item["status"] != "running":
            raise ItemGone()
        snapshot = {"attempt": item.get("attempt") or 0, "service_class": item.get("service_class")}
        try:
            getattr(self, f"_station_{station}")(item)
        except LineSuspended as e:
            self._suspend(e.reason, e.seconds)
            # Not the train's fault: put it back where it was, attempt not spent.
            if self.store.transition(item_id, ("running",), status="queued", **snapshot):
                self.store.event(item_id, station, "notice", f"Held at this station: {e.reason}")
        except OverBudget as e:
            cur = self.store.get_item(item_id)
            if cur and cur["status"] == "running":
                self._fail(cur, str(e))

    def _live(self, item_id: int) -> dict:
        """Re-read an item mid-job; abort the job if it was cancelled meanwhile."""
        item = self.store.get_item(item_id)
        if not item or item["status"] != "running":
            raise ItemGone()
        return item

    def _move(self, item: dict, station: str, message: str, **fields):
        if not self.store.transition(item["id"], ("running",), station=station, status="queued", **fields):
            raise ItemGone()
        self.store.event(item["id"], station, "arrived", message)

    def _wait_at_gate(self, item: dict, gate: str, message: str):
        updated = self.store.transition(item["id"], ("running",), status="waiting", gate=gate)
        if not updated:
            raise ItemGone()
        self.store.event(item["id"], updated["station"], "gate", message)
        self._notify("gate", f"Yamanote: #{item['id']} waiting for {gate} approval", f"{item['title']} — {message}",
                     updated)

    def _finish(self, item: dict, status: str, reason: str, *, keep_branch: bool = False,
                expect: tuple[str, ...] = ("running",), announce: bool = True):
        """Terminal transition: release the train, clean the worktree (and branch)."""
        before = self.store.get_item(item["id"]) or item
        done = self.store.transition(item["id"], expect, status=status, outcome=reason[:1000], train=None,
                                     worktree=None, gate=None, finished_at=time.time(), pending_status=None)
        if not done:
            raise ItemGone()
        if before.get("worktree"):
            gitops.remove_worktree(before["project"], before["worktree"])
        if not keep_branch and before.get("branch"):
            gitops.delete_branch(before["project"], before["branch"])
        METRICS.specs_total.inc({"outcome": status})
        if announce:
            self.store.event(item["id"], before["station"], status, reason[:1000])
            self._notify_outcome(done, status, reason)

    def _notify_outcome(self, item: dict, status: str, reason: str):
        if status in ("failed", "done"):
            self._notify(status, f"Yamanote: #{item['id']} {'arrived' if status == 'done' else 'failed'}",
                         f"{item['title']} — {reason[:300]}", item)

    def _end_journey(self, item: dict, status: str, reason: str, *, keep_branch: bool = False,
                     expect: tuple[str, ...] = ("running",)):
        """A train that arrived or failed goes to JY09 Retro before its journey
        is closed, so every outcome feeds what the next train knows."""
        if not settings.RETRO_ENABLED:
            return self._finish(item, status, reason, keep_branch=keep_branch, expect=expect)
        before = self.store.get_item(item["id"]) or item
        moved = self.store.transition(item["id"], expect, station="retro", status="queued", pending_status=status,
                                      outcome=reason[:1000], train=None, worktree=None, gate=None, hold_until=None)
        if not moved:
            raise ItemGone()
        if before.get("worktree"):
            gitops.remove_worktree(before["project"], before["worktree"])
        if not keep_branch and before.get("branch"):
            gitops.delete_branch(before["project"], before["branch"])
        self.store.event(item["id"], before["station"], status, reason[:1000])
        self._notify_outcome(moved, status, reason)
        self.store.event(item["id"], "retro", "arrived", "Journey over — reflecting on what to learn from it")

    def _fail(self, item: dict, reason: str, expect: tuple[str, ...] = ("running",)):
        self._end_journey(item, "failed", reason, keep_branch=True, expect=expect)

    # ─── running agents ─────────────────────────────────────────────────

    def _check_budget(self, item: dict | None):
        if not item:
            return
        cost = (self.store.get_item(item["id"]) or item).get("cost_usd") or 0
        if cost >= settings.ITEM_BUDGET_USD:
            raise OverBudget(f"Over budget: ${cost:.2f} spent (limit ${settings.ITEM_BUDGET_USD:.2f} per item)")

    @staticmethod
    def _fatal_api_error(res: AgentResult) -> LineSuspended | None:
        code = res.error_code
        if code in (401, 402, 403):
            return LineSuspended(f"OpenRouter refused the request (HTTP {code}): out of credit or key problem",
                                 settings.CREDIT_SUSPEND_SECONDS)
        if code == 429:
            return LineSuspended("OpenRouter rate limits persisted through retries", settings.RATE_LIMIT_SUSPEND_SECONDS)
        if code == UNREACHABLE:
            return LineSuspended("OpenRouter unreachable (network down?)", 300)
        return None

    def _run_agent(self, item: dict | None, station: str, role: str, system: str, task: str, *,
                   root: str, tools: tuple, schema: dict | None, key: str | None = None) -> AgentResult:
        self._check_budget(item)
        model, cls = self.model_for(station, item)
        item_id = item["id"] if item else None
        run_id = self.store.start_run(item_id, role, station, model, cls)
        budget = settings.RUN_BUDGET_USD
        if item:
            spent = (self.store.get_item(item_id) or item).get("cost_usd") or 0
            budget = min(budget, settings.ITEM_BUDGET_USD - spent)
        agent = Agent(role=role, model=model, system=system.format(root=root, branch=(item or {}).get("branch", "")),
                      root=root, tools=tools, result_schema=schema, recorder=StoreRecorder(self.store, run_id),
                      client=self.client, fallbacks=[settings.FALLBACK_MODEL], budget_usd=budget)
        agent_key = key or (f"item:{item_id}" if item_id else role)
        self.agents[agent_key] = agent
        METRICS.agent_launches_total.inc({"agent": role})
        try:
            res = agent.run(task)
        except Exception as e:
            res = AgentResult(status="error", error=f"{type(e).__name__}: {e}")
        finally:
            self.agents.pop(agent_key, None)
        self.store.finish_run(run_id, "interrupted" if self._stopping else res.status, res.summary, res.error)
        if self._stopping:
            raise ItemGone()  # leave the item 'running'; startup recovery re-queues it
        fatal = self._fatal_api_error(res)
        if fatal:
            raise fatal
        if not res.ok:
            METRICS.agent_failures_total.inc({"agent": role})
        return res

    def _single_shot(self, item: dict | None, station: str, role: str, system: str, prompt: str, *,
                     enforce_item_budget: bool = True):
        """One JSON-answer model call. Fatal API errors suspend the line (and,
        inside a station job, hold the train)."""
        if enforce_item_budget:
            self._check_budget(item)
        model, cls = self.model_for(station, item)
        run_id = self.store.start_run(item["id"] if item else None, role, station, model, cls)
        data, res = single_shot(self.client, model, system, prompt, fallbacks=[settings.FALLBACK_MODEL],
                                recorder=StoreRecorder(self.store, run_id))
        self.store.finish_run(run_id, res.status, res.summary or "", res.error)
        fatal = self._fatal_api_error(res)
        if fatal:
            if item:
                raise fatal
            self._suspend(fatal.reason, fatal.seconds)
        return data, res

    def _record_jev(self, item_id: int | None, station: str, label: str, cost: float, tokens: int, detail: str,
                    backend: str | None = None):
        if not tokens and not cost:
            return
        st = decisions.status(backend)
        run_id = self.store.start_run(item_id, "jev", station, st.get("model") or "jev")
        self.store.add_step(run_id, "decision", label, detail, tokens_in=tokens, cost_usd=cost)
        self.store.finish_run(run_id, "ok", detail[:500])

    def _check(self, item: dict, station: str, kind: str, command: str, cwd: str, timeout: int) -> CheckResult:
        """Run a deterministic check and record it as a (free) 'ci' run."""
        run_id = self.store.start_run(item["id"], "ci", station, "shell")
        cancel = self._cancel_events.setdefault(item["id"], threading.Event())
        result = run_check(command, cwd, timeout, cancel=cancel)
        self.store.add_step(run_id, "tool", kind, f"$ {command}\n→ exit {result.exit_code} in {result.seconds:.0f}s\n"
                            + result.output[-3000:])
        self.store.finish_run(run_id, "ok" if result.ok else "error", result.summary())
        return result

    # ─── JY02 Triage ────────────────────────────────────────────────────

    def _station_triage(self, item: dict):
        project = item["project"]
        backend = settings.project_decide_backend(project)
        context = self._history_context(project, exclude_id=item["id"])
        spec_text = f"Title: {item['title']}\nKind: {item['kind']}\n\n{item['description']}"

        d = decisions.assess_spec(spec_text, context, backend)
        scores = None
        if d:
            level = decisions.difficulty_level(d)
            scores = {"useful": d.p("useful"), "ready": d.p("ready"), "duplicate": d.p("duplicate"),
                      "difficulty": round(d.score("difficulty"), 2)}
            self._record_jev(item["id"], "triage", "assess", d.cost_usd, d.tokens,
                             f"difficulty={level} " + " ".join(f"{k}={v:.2f}" for k, v in scores.items()), backend)
            cls, why = self.class_for_difficulty(level)
            item = self.store.update_item(item["id"], difficulty=level, difficulty_score=d.score("difficulty"),
                                          service_class=cls, first_class=cls)
            self.store.event(item["id"], "triage", "classified",
                             f"Jev: {level} → {settings.SERVICE_CLASSES[cls]['label']} service ({why})", scores)

        if item["source"] == "human":
            self._move(item, "spec", "Human-requested work skips the triage gate")
            return

        if scores:
            if scores["duplicate"] >= 0.85:
                return self._reject(item, "Jev: already in progress, built, or recently rejected")
            if scores["useful"] < 0.15:
                return self._reject(item, "Jev: unlikely to solve a real problem")
            if scores["ready"] < 0.10:
                return self._hold(item, "Jev: too vague to build without guesswork")
            if scores["useful"] >= 0.85 and scores["ready"] >= 0.80 and scores["duplicate"] < 0.2:
                self._move(item, "spec", "Jev fast-pass: useful, ready, not a duplicate (no LLM triage needed)")
                self._triage_passed(project)
                return

        hint = json.dumps(scores) if scores else "(classifier unavailable)"
        task = (f"Work item:\n{spec_text}\n\nClassifier scores (hint): {hint}\n\n"
                f"Project history:\n{context}\n\nRecent commits:\n{gitops.recent_log(project)}"
                + self._playbook_section(project, "triage"))
        res = self._run_agent(item, "triage", "triage", prompts.TRIAGE, task, root=project,
                              tools=READ_TOOLS, schema=prompts.TRIAGE_RESULT)
        item = self._live(item["id"])
        if not res.ok or not res.result:
            # fail-open like the original orchestrator, but say so
            self._move(item, "spec", f"Triage agent failed ({res.error or res.status}); passing by default")
            return
        verdict = str(res.result.get("verdict", "")).upper()
        reason = str(res.result.get("reason", ""))
        if verdict == "BUILD":
            self._move(item, "spec", f"BUILD — {reason}")
            self._triage_passed(project)
        elif verdict == "HOLD":
            self._hold(item, reason)
        else:
            self._reject(item, reason)

    def _triage_passed(self, project: str):
        self.store.kv_set(f"rejects:{project}", 0)

    def _reject(self, item: dict, reason: str):
        project = item["project"]
        self._finish(item, "rejected", f"REJECT — {reason}")
        n = (self.store.kv_get(f"rejects:{project}", 0) or 0) + 1
        self.store.kv_set(f"rejects:{project}", n)
        if n >= settings.MAX_CONSECUTIVE_REJECTIONS:
            self.store.kv_set(f"stall:{project}", time.time() + settings.STALL_PAUSE_SECONDS)
            self.store.kv_set(f"rejects:{project}", 0)
            msg = (f"{os.path.basename(project)}: {n} rejections in a row — dispatcher paused for "
                   f"{settings.STALL_PAUSE_SECONDS // 3600}h")
            self.store.event(None, "intake", "stalled", msg)
            self._notify("stalled", "Yamanote: project stalled", msg)

    def _hold(self, item: dict, reason: str):
        holds = (item.get("holds") or 0) + 1
        if holds > settings.MAX_HOLDS:
            return self._reject(item, f"held {holds - 1} times without becoming ready. Last reason: {reason}")
        if not self.store.transition(item["id"], ("running",), status="held", holds=holds,
                                     outcome=f"HOLD — {reason}"[:1000],
                                     hold_until=time.time() + settings.HOLD_RECYCLE_SECONDS):
            raise ItemGone()
        self.store.event(item["id"], "triage", "held",
                         f"HOLD {holds}/{settings.MAX_HOLDS} — {reason[:500]} "
                         f"(re-evaluated in {settings.HOLD_RECYCLE_SECONDS // 3600}h)")

    # ─── JY03 Spec ──────────────────────────────────────────────────────

    def _station_spec(self, item: dict):
        project = item["project"]
        backend = settings.project_decide_backend(project)
        brief = f"Title: {item['title']}\nKind: {item['kind']}\nPriority: {item['priority']}\n\n{item['description']}"
        files: list[str] = []
        ranked = decisions.relevant_files(project, brief, backend=backend)
        if ranked:
            pairs, cost = ranked
            files = [p for p, _ in pairs]
            price = 0.42e-6
            self._record_jev(item["id"], "spec", "relevant files", cost, int(cost / price) if cost else 0,
                             ", ".join(f"{p} ({s:.2f})" for p, s in pairs[:10]), backend)
            if files:
                self.store.event(item["id"], "spec", "context",
                                 f"Jev picked {len(files)} relevant files", {"files": files})
        hint = ("\n\nFiles a relevance classifier flagged (start here):\n" + "\n".join(f"- {f}" for f in files)) if files else ""
        test_cmd = settings.project_commands(project)["test"]
        if test_cmd:
            hint += f"\n\nThe project's automated test command is `{test_cmd}` (the factory runs it after each build)."
        hint += self._playbook_section(project, "spec")
        res = self._run_agent(item, "spec", "spec", prompts.SPEC, brief + hint, root=project,
                              tools=READ_TOOLS, schema=prompts.SPEC_RESULT)
        item = self._live(item["id"])
        if not res.ok or not res.result:
            return self._retry_or_fail(item, "spec", f"spec writer failed: {res.error or res.status}")
        r = res.result
        scenarios = [s for s in (r.get("scenarios") or []) if isinstance(s, dict) and s.get("name")]
        relevant = list(dict.fromkeys(files + [f for f in (r.get("relevant_files") or []) if isinstance(f, str)]))
        spec = {"acceptance_criteria": r.get("acceptance_criteria") or [], "plan": r.get("plan") or "",
                "relevant_files": relevant[:25]}
        fields = {"spec": spec, "scenarios": scenarios}
        if not item.get("service_class"):  # Jev unavailable: use the spec writer's estimate
            level = r.get("difficulty") if r.get("difficulty") in settings.DIFFICULTY_LEVELS else "moderate"
            cls, _why = self.class_for_difficulty(level)
            fields.update(difficulty=level, service_class=cls, first_class=cls)
        item = self.store.update_item(item["id"], **fields)
        self.store.event(item["id"], "spec", "specified",
                         f"{len(spec['acceptance_criteria'])} acceptance criteria, "
                         f"{len(scenarios)} holdout scenarios sealed")
        if self.gates(project)["spec"]:
            return self._wait_at_gate(item, "spec", "Signal at red — waiting for human spec approval")
        self._move(item, "build", "Spec ready; waiting for a train")

    # ─── JY04 Build ─────────────────────────────────────────────────────

    def _station_build(self, item: dict):
        project = item["project"]
        commands = settings.project_commands(project)
        branch = item.get("branch") or gitops.branch_name(item["id"], item["title"])
        worktree = item.get("worktree")
        if not (worktree and os.path.isdir(worktree) and gitops.current_branch(worktree) == branch):
            try:  # reuse the train's worktree across reworks (keeps an in-progress trunk merge)
                worktree = gitops.create_worktree(project, branch, f"item-{item['id']}")
            except RuntimeError as e:
                return self._fail(self._live(item["id"]), str(e))
        item = self.store.update_item(item["id"], branch=branch, worktree=worktree)

        # Install dependencies once per worktree (they're git-ignored, so a fresh worktree has none).
        if commands["setup"] and self.store.kv_get(f"setup:{item['id']}") != worktree:
            result = self._check(item, "build", "setup", commands["setup"], worktree, settings.SETUP_TIMEOUT_SECONDS)
            self._live(item["id"])
            if not result.ok:
                return self._retry_or_fail(item, "build", f"Setup command failed: {result.summary()}\n{result.output[-800:]}")
            self.store.kv_set(f"setup:{item['id']}", worktree)
            self.store.event(item["id"], "build", "checks", f"Setup ready: {result.summary()}")

        attempt = (item.get("attempt") or 0) + 1
        current = item.get("service_class") or "rapid"
        base = item.get("first_class") or current
        cls = current
        if base in settings.CLASS_ORDER:
            # Escalation is a function of the starting class and attempts made, so a
            # restart (which rolls back an interrupted attempt) can't escalate twice.
            steps = max(0, attempt - 1 - settings.ESCALATE_AFTER)
            cls = settings.CLASS_ORDER[min(settings.CLASS_ORDER.index(base) + steps, len(settings.CLASS_ORDER) - 1)]
            if cls != current and steps:
                self.store.event(item["id"], "build", "escalated",
                                 f"Rework #{attempt - 1}: upgraded to {settings.SERVICE_CLASSES[cls]['label']} service")
        item = self.store.update_item(item["id"], attempt=attempt, service_class=cls,
                                      first_class=item.get("first_class") or cls)

        spec = item.get("spec") or {}
        parts = [f"# Work item: {item['title']}\n\n{item['description']}"]
        if spec.get("acceptance_criteria"):
            parts.append("# Acceptance criteria\n" + "\n".join(f"- {c}" for c in spec["acceptance_criteria"]))
        if spec.get("plan"):
            parts.append("# Plan\n" + spec["plan"])
        if spec.get("relevant_files"):
            parts.append("# Relevant files\n" + "\n".join(f"- {f}" for f in spec["relevant_files"]))
        if commands["test"]:
            parts.append(f"# Test command\n`{commands['test']}` must pass; the factory runs it after you finish.")
        playbook = self._playbook_section(project, "build")
        if playbook:
            parts.append(playbook.strip())
        if item.get("feedback"):
            parts.append(f"# Rework (attempt {attempt}) — fix these problems found in the previous attempt\n"
                         + item["feedback"] + "\n\nThe previous attempt's commits are already on this branch.")
        res = self._run_agent(item, "build", "builder", prompts.BUILD, "\n\n".join(parts), root=worktree,
                              tools=READ_TOOLS + WRITE_TOOLS + RUN_TOOLS, schema=prompts.BUILD_RESULT)
        item = self._live(item["id"])

        if gitops.current_branch(worktree) != branch:
            return self._fail(item, f"builder left branch {branch}; refusing to continue")
        markers = gitops.leftover_conflict_markers(worktree)
        if markers:
            return self._rework(item, "build", "Conflict markers are still present; resolve them all:\n" + markers[:1500])
        committed = gitops.commit_all(worktree, f"{item['title']} (attempt {attempt})\n\n"
                                      f"{(res.summary or '')[:1500]}\n\nYamanote item #{item['id']}")
        has_diff = bool(gitops.diff_stat(worktree))
        if not res.ok:
            note = f"Builder stopped early ({res.status}: {res.error})."
            if has_diff:
                note += " Partial work was committed."
            return self._rework(item, "build", note + " Continue from the current state and finish the job.")
        if not has_diff:
            return self._rework(item, "build", "The previous attempt produced no changes. Implement the spec.")

        if commands["test"]:
            result = self._check(item, "build", "test", commands["test"], worktree, settings.TEST_TIMEOUT_SECONDS)
            item = self._live(item["id"])
            if not result.ok:
                return self._rework(item, "build", f"The project's tests fail after your change ({result.summary()}):\n"
                                    f"```\n{result.output[-3000:]}\n```")
            self.store.event(item["id"], "build", "checks", f"Tests passed: {result.summary()}")
        stat = gitops.diff_stat(worktree).splitlines()
        self._move(item, "inspect", f"Built (attempt {attempt}){' and committed' if committed else ''}: "
                   f"{stat[-1].strip() if stat else ''}", feedback=None)

    # ─── JY05 Inspect ───────────────────────────────────────────────────

    def _station_inspect(self, item: dict):
        worktree = item["worktree"]
        if not worktree or not os.path.isdir(worktree):
            return self._move(item, "build", "Worktree missing; rebuilding")
        ok, conflict = gitops.trial_merge(item["project"], item["branch"])
        if not ok:
            return self._conflict(item, conflict)
        diff = gitops.diff_trunk(worktree)
        if len(diff) > settings.DIFF_MAX_CHARS:
            diff = diff[:settings.DIFF_MAX_CHARS] + "\n... [diff truncated; read files for the rest]"
        spec = item.get("spec") or {}
        test_cmd = settings.project_commands(item["project"])["test"]
        tests = f"\n\n# Automated tests\n`{test_cmd}` passed on this build." if test_cmd else ""
        task = (f"# Work item: {item['title']}\n{item['description']}\n\n# Acceptance criteria\n"
                + "\n".join(f"- {c}" for c in spec.get("acceptance_criteria", [])) + tests +
                self._playbook_section(item["project"], "inspect") +
                f"\n\n# Diff against {settings.TRUNK_BRANCH}\n```diff\n{diff}\n```")
        res = self._run_agent(item, "inspect", "inspector", prompts.INSPECT, task, root=worktree,
                              tools=READ_TOOLS, schema=prompts.INSPECT_RESULT)
        item = self._live(item["id"])
        if not res.ok or not res.result:
            return self._retry_or_fail(item, "inspect", f"inspector failed: {res.error or res.status}")
        verdict = str(res.result.get("verdict", "")).upper()
        issues = res.result.get("issues") or []
        if verdict == "APPROVED":
            self._move(item, "verify", f"APPROVED by inspector — {res.summary[:300]}")
            return
        feedback = "\n".join(
            f"- {i.get('file', '')}{':' + str(i['line']) if i.get('line') else ''} {i.get('problem', '')}"
            + (f" → {i['fix']}" if i.get("fix") else "") for i in issues if isinstance(i, dict)) or res.summary
        self._rework(item, "inspect", f"Inspector requested changes:\n{feedback}")

    # ─── JY06 Verify (holdout scenarios) ────────────────────────────────

    def _station_verify(self, item: dict):
        scenarios = item.get("scenarios") or []
        worktree = item["worktree"]
        if not scenarios:
            self._move(item, "merge", "No holdout scenarios on this item; skipping verification")
            return
        task = "# Holdout scenarios to execute\n\n" + "\n\n".join(
            f"## {s['name']}\nSteps: {s.get('steps', '')}\nExpected: {s.get('expected', '')}" for s in scenarios)
        task += self._playbook_section(item["project"], "verify")
        try:
            res = self._run_agent(item, "verify", "verifier", prompts.VERIFY, task, root=worktree,
                                  tools=READ_TOOLS + RUN_TOOLS, schema=prompts.VERIFY_RESULT)
        finally:
            if os.path.isdir(worktree):
                gitops.discard_changes(worktree)
        item = self._live(item["id"])
        if not res.ok or not res.result:
            return self._retry_or_fail(item, "verify", f"verifier failed: {res.error or res.status}")
        results = [r for r in res.result.get("results") or [] if isinstance(r, dict)]
        runnable = [r for r in results if not r.get("unrunnable")]
        # A scenario the verifier proves self-contradictory is excluded — but only
        # up to half of them, so disputing can't become a way to pass.
        disputed = [r for r in runnable if not r.get("passed") and r.get("scenario_error")]
        if disputed and len(disputed) <= len(scenarios) // 2:
            runnable = [r for r in runnable if r not in disputed]
            self.store.event(item["id"], "verify", "disputed",
                             f"{len(disputed)} holdout scenario(s) excluded as self-contradictory: "
                             + "; ".join(f"{r.get('name')}: {str(r.get('evidence', ''))[:300]}" for r in disputed))
        else:
            disputed = []
        passed = [r for r in runnable if r.get("passed")]
        satisfaction = (len(passed) / len(runnable)) if runnable else None
        self.store.update_item(item["id"], satisfaction=satisfaction)
        detail = {"results": [{"name": r.get("name"), "passed": bool(r.get("passed")),
                               "unrunnable": bool(r.get("unrunnable")), "disputed": r in disputed,
                               "evidence": str(r.get("evidence", ""))[:600]} for r in results]}
        if satisfaction is None:
            self.store.event(item["id"], "verify", "verified",
                             "No scenario could be executed here; proceeding on inspector approval", detail)
            self._move(item, "merge", "Verification inconclusive (scenarios unrunnable or disputed)")
            return
        msg = f"Satisfaction {len(passed)}/{len(runnable)} ({satisfaction:.0%})"
        if disputed:
            msg += f", {len(disputed)} disputed"
        self.store.event(item["id"], "verify", "verified", msg, detail)
        if satisfaction >= settings.SATISFACTION_THRESHOLD:
            self._move(item, "merge", msg + " — holdout passed")
            return
        failures = [r for r in runnable if not r.get("passed")]
        self._rework(item, "verify", f"End-to-end checks failed ({msg}). Defects observed:\n"
                     + self._redacted_failures(item, failures))

    def _redacted_failures(self, item: dict, failures: list[dict]) -> str:
        """Describe failed holdout checks as behaviour, without revealing how
        they were run, so the builder fixes the defect instead of the check."""
        evidence = "\n".join(f"- {r.get('name', '')}: {str(r.get('evidence', ''))[:800]}" for r in failures)
        data, res = self._single_shot(item, "redact", "redactor", prompts.REDACT,
                                      f"Work item: {item['title']}\n\nVerifier observations:\n{evidence}")
        defects = [str(d).strip() for d in (data or {}).get("defects") or [] if str(d).strip()]
        if defects:
            return "\n".join(f"- {d[:500]}" for d in defects)
        return "\n".join(f"- The behaviour \"{r.get('name', 'unnamed')}\" does not work as specified." for r in failures)

    # ─── JY07 Merge (merge queue) ───────────────────────────────────────

    def _merge_lock(self, project: str) -> threading.Lock:
        with self._lock:
            return self._merge_locks.setdefault(os.path.realpath(project), threading.Lock())

    def _station_merge(self, item: dict):
        """One train per project at a time: bring the branch up to date with
        trunk, re-run the tests on the combination, then land it. Two trains
        that each pass alone can't break trunk together."""
        project = item["project"]
        if self.gates(project)["merge"] and item.get("gate") != "approved":
            return self._wait_at_gate(item, "merge", "Signal at red — waiting for human merge approval")
        lock = self._merge_lock(project)
        if not lock.acquire(blocking=False):
            if self.store.transition(item["id"], ("running",), status="queued",
                                     hold_until=time.time() + settings.MERGE_QUEUE_RETRY_SECONDS):
                self._notice(f"merge-queue:{item['id']}", "Waiting for the merge queue (another train is landing)",
                             every=600, item_id=item["id"], station="merge")
            return
        try:
            self._land(item)
        finally:
            lock.release()

    def _land(self, item: dict):
        project, worktree = item["project"], item.get("worktree")
        if not worktree or not os.path.isdir(worktree):
            return self._move(item, "build", "Worktree missing; rebuilding")
        before = gitops.head(worktree)
        clean, files = gitops.start_trunk_merge(worktree)
        if not clean:
            return self._conflict(item, "trunk merge into branch", files=files)
        if gitops.head(worktree) != before:
            self.store.event(item["id"], "merge", "integrated",
                             f"Merged the latest {settings.TRUNK_BRANCH} into the branch")
            test_cmd = settings.project_commands(project)["test"]
            if test_cmd:
                result = self._check(item, "merge", "integration test", test_cmd, worktree, settings.TEST_TIMEOUT_SECONDS)
                item = self._live(item["id"])
                if not result.ok:
                    return self._rework(item, "merge", f"After merging the latest {settings.TRUNK_BRANCH}, the tests fail "
                                        f"({result.summary()}). Make this change work with trunk's new code:\n"
                                        f"```\n{result.output[-3000:]}\n```")
                self.store.event(item["id"], "merge", "checks", f"Integration tests passed: {result.summary()}")
            else:
                self.store.event(item["id"], "merge", "notice",
                                 "Trunk changed since verification and no test command is configured; "
                                 "landing without an integration check")
        ok, info = gitops.merge(project, item["branch"],
                                f"Merge {item['branch']}: {item['title']}\n\nYamanote item #{item['id']}")
        item = self._live(item["id"])
        if not ok:
            if "uncommitted" in info or "not " + settings.TRUNK_BRANCH in info:
                if self.store.transition(item["id"], ("running",), status="queued", hold_until=time.time() + 300):
                    self._notice(f"merge-blocked:{project}", f"Merge blocked: {info}. Retrying every 5 min.",
                                 item_id=item["id"], station="merge")
                return
            return self._conflict(item, info)
        self.store.kv_set(f"rejects:{project}", 0)
        self.store.kv_set(f"stall:{project}", 0)
        self.store.kv_set("last_merge_at", time.time())
        self.store.kv_set(f"merged:{item['id']}", info)
        stat = gitops.merge_stats(project, info)
        if stat:
            self.store.kv_set(f"mergestat:{item['id']}", stat)
        self._move(item, "deploy", f"Merged to {settings.TRUNK_BRANCH} at {info[:10]}", gate=None)

    # ─── JY08 Deploy ────────────────────────────────────────────────────

    def _station_deploy(self, item: dict):
        msg = self._deploy(item["project"])
        item = self._live(item["id"])
        self.store.event(item["id"], "deploy", "deployed", msg)
        self._start_watch(item)
        self._end_journey(item, "done", f"Arrived — {msg}")

    def _deploy(self, project: str) -> str:
        if settings.RAILWAY_PROJECT:
            rc, _, err = gitops.git("push", "origin", f"{settings.TRUNK_BRANCH}:staging", cwd=project, timeout=60)
            if rc != 0:
                return f"Railway staging push failed: {err[:200]}"
            time.sleep(60)
            logs = _railway_logs(settings.RAILWAY_STAGING_ENV, project)
            bad = [s for s in ("Traceback", "FATAL", "ModuleNotFoundError", "SyntaxError", "ImportError", "panic:") if s in logs]
            if bad:
                return f"Railway staging unhealthy ({', '.join(bad)}); production deploy skipped"
            rc, _, err = gitops.git("push", "origin", settings.TRUNK_BRANCH, cwd=project, timeout=60)
            return "Railway production deploy triggered" if rc == 0 else f"Railway production push failed: {err[:200]}"
        if settings.SERVICE_RESTART_CMD:
            try:
                r = subprocess.run(shlex.split(settings.SERVICE_RESTART_CMD), capture_output=True, text=True,
                                   timeout=settings.SERVICE_RESTART_TIMEOUT)
            except (ValueError, OSError, subprocess.TimeoutExpired) as e:
                return f"service restart failed: {e}"
            return "service restarted" if r.returncode == 0 else f"service restart failed (rc={r.returncode}): {r.stderr[:200]}"
        return "no deploy method configured"

    # ─── post-deploy watch (regressions) ────────────────────────────────

    def _start_watch(self, item: dict):
        if settings.DEPLOY_WATCH_SECONDS <= 0:
            return
        project = item["project"]
        now = time.time()
        baseline = [s for s, ts in self._sig_history.get(project, {}).items() if now - ts < 86400]
        self.store.kv_set(f"watch:{project}", {"item_id": item["id"], "title": item["title"],
                                                "commit": self.store.kv_get(f"merged:{item['id']}"),
                                                "until": now + settings.DEPLOY_WATCH_SECONDS, "baseline": baseline})
        self.store.event(item["id"], "deploy", "watching",
                         f"Watching app logs for {settings.DEPLOY_WATCH_SECONDS // 60} min for regressions")

    def _check_regression(self, project: str, errors: list[str]) -> bool:
        """New error signatures during a post-deploy watch: file a linked bug
        (and optionally revert). Returns True if handled as a regression."""
        watch = self.store.kv_get(f"watch:{project}")
        if not watch or watch.get("until", 0) < time.time():
            return False
        baseline = set(watch.get("baseline") or [])
        fresh = [l for l in errors if _signature(l) not in baseline]
        if not fresh:
            return False
        self.store.kv_set(f"watch:{project}", None)  # one regression report per deploy
        origin = watch["item_id"]
        slug = _error_slug(fresh[0])
        description = (f"New errors appeared in the app logs within {settings.DEPLOY_WATCH_SECONDS // 60} minutes of "
                       f"deploying #{origin} ({watch.get('title')}, merge {str(watch.get('commit') or '')[:10]}). "
                       f"Find out whether that change caused them and fix the cause.\n\nLog lines:\n"
                       + "\n".join(fresh[:30]))
        bug = self.create_item(f"regression-after-{origin}-{slug}", description, project, kind="bug",
                               priority="high", source="signal")
        bug = self.store.update_item(bug["id"], parent_id=origin)
        self.store.event(origin, "deploy", "regression", f"Possible regression: new errors after deploy — filed #{bug['id']}",
                         {"lines": fresh[:20]})
        origin_item = self.store.get_item(origin)
        self._notify("regression", f"Yamanote: regression after #{origin}",
                     f"{watch.get('title')}: new errors after deploy; filed #{bug['id']}", origin_item,
                     {"lines": fresh[:10], "bug_id": bug["id"]})
        if (settings.AUTO_REVERT or self.autopilot_on) and watch.get("commit"):
            self._submit(f"revert:{project}", self._revert_job, project, origin, watch["commit"])
        return True

    def _revert_job(self, project: str, origin: int, commit: str):
        ok, info = gitops.revert_merge(project, commit)
        if not ok:
            self.store.event(origin, "deploy", "notice", f"Automatic revert of {commit[:10]} failed: {info}")
            return
        msg = self._deploy(project)
        self.store.event(origin, "deploy", "reverted", f"Merge {commit[:10]} reverted automatically ({info[:10]}); {msg}")
        item = self.store.get_item(origin)
        if item:
            self.store.update_item(origin, outcome=(item.get("outcome") or "") + " — reverted after a regression")
        self._notify("reverted", f"Yamanote: reverted #{origin}", f"Merge {commit[:10]} reverted; {msg}", item)

    # ─── rework / conflict / retry ──────────────────────────────────────

    def _rework(self, item: dict, station: str, feedback: str):
        item = self._live(item["id"])
        reworks = item.get("attempt") or 0  # attempts so far = reworks + 1
        if reworks > settings.MAX_REWORK_ATTEMPTS:
            return self._fail(item, f"Gave up after {reworks} build attempts. Last problem: {feedback[:600]}")
        self.store.event(item["id"], station, "rework", feedback[:2000])
        self._move(item, "build", f"Returned for rework (attempt {reworks + 1})", feedback=feedback[:6000], gate=None)

    def _conflict(self, item: dict, detail: str, files: list[str] | None = None):
        """Trunk moved underneath the branch. Merge trunk in and let the
        builder resolve any conflicts; the result is re-inspected and
        re-verified, and any earlier merge approval no longer applies."""
        item = self._live(item["id"])
        n = (item.get("conflicts") or 0) + 1
        if n > settings.MAX_CONFLICT_RETRIES:
            return self._fail(item, f"Persistent merge conflict with {settings.TRUNK_BRANCH} after {n - 1} attempts")
        worktree = item.get("worktree")
        if files is None and worktree and os.path.isdir(worktree):
            clean, files = gitops.start_trunk_merge(worktree)
            if clean:
                self.store.event(item["id"], item["station"], "conflict",
                                 f"Merged {settings.TRUNK_BRANCH} into the branch cleanly — re-inspecting")
                self._move(item, "inspect", "Re-inspecting after merging trunk", conflicts=n, gate=None)
                return
        if files:
            feedback = (f"{settings.TRUNK_BRANCH} changed while this branch was in review. A merge of "
                        f"{settings.TRUNK_BRANCH} into this branch is in progress and left conflict markers in: "
                        + ", ".join(files) + ". Resolve every conflict so both trunk's changes and this work "
                        "item's behaviour are kept, remove all markers, and re-run the tests. Do not abort the merge.")
            self.store.event(item["id"], item["station"], "conflict",
                             f"Conflicts with {settings.TRUNK_BRANCH} in {', '.join(files)} — builder will resolve "
                             f"({n}/{settings.MAX_CONFLICT_RETRIES})", {"files": files, "detail": detail[:500]})
            self._move(item, "build", "Returned to resolve merge conflicts", conflicts=n, feedback=feedback, gate=None)
            return
        # No usable worktree: start over from fresh trunk.
        gitops.remove_worktree(item["project"], worktree)
        gitops.delete_branch(item["project"], item.get("branch"))
        self.store.event(item["id"], item["station"], "conflict",
                         f"Conflicts with {settings.TRUNK_BRANCH}; rebuilding from fresh trunk ({n}/{settings.MAX_CONFLICT_RETRIES})",
                         {"detail": detail[:500]})
        self._move(item, "build", "Rebuilding on fresh trunk after conflict", conflicts=n, attempt=0,
                   branch=None, worktree=None, feedback=None, gate=None)

    def _retry_or_fail(self, item: dict, station: str, reason: str):
        key = f"retries:{item['id']}:{station}"
        n = (self.store.kv_get(key, 0) or 0) + 1
        self.store.kv_set(key, n)
        if n >= 3:
            return self._fail(item, reason)
        if not self.store.transition(item["id"], ("running",), status="queued", hold_until=time.time() + 60 * 2 ** n):
            raise ItemGone()
        self.store.event(item["id"], station, "retry", f"{reason} — retrying in {2 ** n} min ({n}/3)")

    # ─── maintenance: holds, SLA, worktree GC, retention ────────────────

    def _maintenance(self):
        now = time.time()
        for item in self.store.items(("held",)):
            if (item.get("hold_until") or 0) <= now:
                if self.store.transition(item["id"], ("held",), status="queued", station="triage", hold_until=None):
                    self.store.event(item["id"], "triage", "recycled", "HOLD expired — back to triage for re-evaluation")
        for item in self.store.items(("running", "queued")):
            started = item.get("started_at")
            if item["station"] != "retro" and started and now - started > settings.ITEM_SLA_SECONDS:
                agent = self.agents.get(f"item:{item['id']}")
                if agent:
                    agent.stop()
                try:
                    self._fail(item, f"SLA breach: {int(now - started) // 60} min on the line "
                               f"(limit {settings.ITEM_SLA_SECONDS // 60})", expect=("running", "queued"))
                except ItemGone:
                    pass
        if now - self._last_gc > 3600:
            self._last_gc = now
            keep = {os.path.realpath(i["worktree"]) for i in self.store.items(ACTIVE_STATUSES) if i.get("worktree")}
            for project in {i["project"] for i in self.store.items(limit=200)}:
                if os.path.isdir(project):
                    for path in gitops.gc_worktrees(project, keep):
                        self.store.event(None, None, "gc", f"Removed orphaned worktree {path}")
        if now - (self.store.kv_get("last_prune", 0) or 0) > 86400:
            self.store.kv_set("last_prune", now)
            self._prune(now)

    def _prune(self, now: float):
        cutoff = now - settings.LOG_RETENTION_DAYS * 86400
        removed = self.store.prune(settings.LOG_RETENTION_DAYS)
        branches = 0
        for item in self.store.items(TERMINAL_STATUSES, limit=5000):
            if item.get("branch") and (item.get("finished_at") or now) < cutoff and os.path.isdir(item["project"]):
                gitops.delete_branch(item["project"], item["branch"])
                self.store.update_item(item["id"], branch=None)
                branches += 1
        if removed["steps"] or removed["events"] or branches:
            self.store.event(None, None, "gc", f"Retention ({settings.LOG_RETENTION_DAYS}d): pruned {removed['steps']} "
                             f"steps, {removed['events']} events, {branches} old branches")

    # ─── Autopilot (dark mode) ──────────────────────────────────────────

    def autopilot_config(self) -> dict:
        """Settings-panel values; seeded from the environment until first saved."""
        cfg = self.store.kv_get("autopilot_config")
        if cfg is None:
            cfg = {"schedule_enabled": bool(settings.AUTOPILOT_ON_CRON or settings.AUTOPILOT_OFF_CRON),
                   "on_cron": settings.AUTOPILOT_ON_CRON, "off_cron": settings.AUTOPILOT_OFF_CRON,
                   "merge_without_tests": settings.AUTOPILOT_MERGE_WITHOUT_TESTS,
                   "supervised_gates": {"spec": settings.GATE_SPEC, "merge": settings.GATE_MERGE}}
        return cfg

    def save_autopilot_config(self, **changes) -> dict:
        cfg = dict(self.autopilot_config())
        for key in ("on_cron", "off_cron"):
            if key in changes:
                expr = str(changes[key] or "").strip()
                if expr:
                    cron.parse(expr)  # raises CronError (a ValueError) with a readable message
                cfg[key] = expr
        for key in ("schedule_enabled", "merge_without_tests"):
            if key in changes:
                cfg[key] = bool(changes[key])
        if "supervised_gates" in changes:
            gates = changes["supervised_gates"] or {}
            cfg["supervised_gates"] = {k: bool(gates.get(k, cfg["supervised_gates"].get(k, False)))
                                       for k in ("spec", "merge")}
        if cfg["schedule_enabled"] and not (cfg["on_cron"] and cfg["off_cron"]):
            raise ValueError("a schedule needs both an 'on' and an 'off' time")
        self.store.kv_set("autopilot_config", cfg)
        if any(k in changes for k in ("schedule_enabled", "on_cron", "off_cron")):
            # A new schedule takes effect now: the line switches to whatever it says for this moment.
            self.store.kv_set("autopilot_applied", 0)
        self.store.event(None, None, "notice", "Autopilot settings updated")
        return cfg

    def autopilot_state(self) -> dict:
        st = self.store.kv_get("autopilot")
        if st is None:
            st = {"on": settings.AUTOPILOT, "since": time.time(), "source": "default"}
            self.store.kv_set("autopilot", st)
        return st

    @property
    def autopilot_on(self) -> bool:
        return bool(self.autopilot_state()["on"])

    def gates(self, project: str) -> dict:
        """Which human gates apply to this project right now."""
        cfg = self.autopilot_config()
        gates = dict(cfg.get("supervised_gates") or {"spec": False, "merge": False})
        gates.update({k: bool(v) for k, v in (settings.project_config(project).get("gates") or {}).items()
                      if k in gates})
        if self.autopilot_on:
            untested = not settings.project_commands(project)["test"]
            gates = {"spec": False, "merge": untested and not cfg.get("merge_without_tests")}
        return gates

    def schedule_preview(self) -> dict:
        cfg = self.autopilot_config()
        now = dt.datetime.now()
        out = {"next_on": None, "next_off": None, "error": None}
        try:
            if cfg.get("on_cron"):
                n = cron.parse(cfg["on_cron"]).next(now)
                out["next_on"] = n.timestamp() if n else None
            if cfg.get("off_cron"):
                n = cron.parse(cfg["off_cron"]).next(now)
                out["next_off"] = n.timestamp() if n else None
        except cron.CronError as e:
            out["error"] = str(e)
        return out

    def set_autopilot(self, on: bool, source: str = "manual", now: float | None = None) -> dict:
        state = self.autopilot_state()
        now = now or time.time()
        if source == "manual":
            # Like a thermostat: a manual switch holds until the next scheduled change.
            self.store.kv_set("autopilot_applied", now)
        if bool(state["on"]) == bool(on):
            return state
        new = {"on": bool(on), "since": now, "source": source}
        self.store.kv_set("autopilot", new)
        who = "by schedule" if source == "schedule" else "by human"
        if on:
            self.store.event(None, None, "autopilot", f"Autopilot ON {who} — running dark: gates skipped, "
                             "proposals board themselves, regressions auto-revert")
            self._notify("autopilot", "Yamanote: autopilot on", f"Switched on {who}; the line is running dark.")
            self._release_gates()
        else:
            report = self._autopilot_report(state.get("since") or now, now)
            self.store.event(None, None, "autopilot", f"Autopilot OFF {who} — supervised again")
            self.store.event(None, None, "autopilot_report", report["message"], report)
            self._notify("autopilot", "Yamanote: while you were away", report["message"], data=report)
        return new

    def _autopilot_tick(self, now: float | None = None):
        """Apply the schedule, at most once a minute. Only scheduled times later
        than the last applied change (or manual switch) take effect."""
        if now is None:
            now = time.time()
            if now - getattr(self, "_autopilot_checked", 0) < 60:
                return
            self._autopilot_checked = now
        cfg = self.autopilot_config()
        if not (cfg.get("schedule_enabled") and cfg.get("on_cron") and cfg.get("off_cron")):
            return
        try:
            moment = dt.datetime.fromtimestamp(now)
            last_on = cron.parse(cfg["on_cron"]).prev(moment)
            last_off = cron.parse(cfg["off_cron"]).prev(moment)
        except cron.CronError as e:
            self._notice("autopilot-cron", f"Autopilot schedule is invalid: {e}")
            return
        events = [(t.timestamp(), on) for t, on in ((last_on, True), (last_off, False)) if t]
        if not events:
            return
        latest_ts, want_on = max(events)
        if latest_ts <= (self.store.kv_get("autopilot_applied", 0) or 0):
            return
        self.store.kv_set("autopilot_applied", latest_ts)
        self.set_autopilot(want_on, source="schedule", now=now)

    def _release_gates(self):
        """Autopilot starts: let waiting trains through under autopilot rules."""
        for item in self.store.items(("waiting",)):
            gate = item.get("gate")
            if gate == "board":
                fields, station, msg = {"station": "triage", "gate": None}, "triage", "Boarded by autopilot"
            elif gate == "spec":
                fields, station, msg = {"station": "build", "gate": None}, "build", "Spec gate released by autopilot"
            elif gate == "merge" and not self.gates(item["project"])["merge"]:
                fields, station, msg = {"gate": "approved"}, "merge", "Merge gate released by autopilot"
            else:
                if gate == "merge":
                    self.store.event(item["id"], "merge", "notice",
                                     "Still waiting under autopilot: this project has no test command "
                                     "(enable 'Merge without tests' to let it through)")
                continue
            if self.store.transition(item["id"], ("waiting",), status="queued", **fields):
                self.store.event(item["id"], station, "approved", msg)

    def _autopilot_report(self, start: float, end: float) -> dict:
        """'While you were away': what the line did during an autopilot window."""
        items = [i for i in self.store.items(limit=2000)
                 if (i.get("finished_at") or 0) >= start or i["created_at"] >= start]
        arrived = [i for i in items if i["status"] == "done" and (i.get("finished_at") or 0) >= start]
        failed = [i for i in items if i["status"] == "failed" and (i.get("finished_at") or 0) >= start]
        rejected = [i for i in items if i["status"] == "rejected" and (i.get("finished_at") or 0) >= start]
        waiting = [i for i in self.store.items(("waiting",))]
        events = [e for e in self.store.events(limit=2000) if e["ts"] >= start]
        regressions = sum(1 for e in events if e["kind"] == "regression")
        reverts = sum(1 for e in events if e["kind"] == "reverted")
        notes = sum(1 for e in events if e["kind"] == "playbook" and e["message"].startswith("New "))
        spend = self.store.spend_since(start)
        hours = (end - start) / 3600
        parts = [f"{len(arrived)} arrived", f"{len(failed)} failed"]
        if rejected:
            parts.append(f"{len(rejected)} not in service")
        if regressions:
            parts.append(f"{regressions} regression{'s' if regressions > 1 else ''}"
                         + (f" ({reverts} reverted)" if reverts else ""))
        if notes:
            parts.append(f"{notes} new playbook note{'s' if notes > 1 else ''}")
        message = (f"Autopilot ran {hours:.1f}h: " + ", ".join(parts) + f"; spent ${spend:.2f}."
                   + (f" {len(waiting)} train(s) now waiting for you." if waiting else ""))
        return {"message": message, "start": start, "end": end, "spend": spend,
                "arrived": [{"id": i["id"], "title": i["title"]} for i in arrived],
                "failed": [{"id": i["id"], "title": i["title"], "outcome": (i.get("outcome") or "")[:200]} for i in failed],
                "rejected": len(rejected), "regressions": regressions, "reverts": reverts, "notes": notes,
                "waiting": [{"id": i["id"], "title": i["title"], "gate": i.get("gate")} for i in waiting]}

    # ─── human actions (from the dashboard) ─────────────────────────────

    def create_item(self, title: str, description: str, project: str | None = None, *, kind: str = "feature",
                    priority: str = "medium", source: str = "human") -> dict:
        project = os.path.realpath(os.path.expanduser(project or self.default_project() or ""))
        problem = validate_project(project)
        if problem:
            raise ValueError(problem)
        title = re.sub(r"\s+", "-", title.strip().lower())[:80] or "untitled"
        item = self.store.create_item(title=title, project=project, description=description.strip(), kind=kind,
                                      source=source, priority=priority if priority in PRIORITY_ORDER else "medium")
        if source in ("dispatcher", "signal") and not self.autopilot_on:
            # Supervised: work the factory proposed itself waits at Intake for a human to board it.
            held = self.store.transition(item["id"], ("queued",), station="intake", status="waiting", gate="board")
            if held:
                self.store.event(item["id"], "intake", "gate",
                                 f"Proposed by {source} — waiting for a human to board it (Autopilot would board it)")
                self._notify("gate", f"Yamanote: #{item['id']} proposed by {source}", f"{held['title']} — board it?", held)
                item = held
        return item

    def approve(self, item_id: int) -> dict:
        item = self.store.get_item(item_id)
        if not item or item["status"] != "waiting":
            raise ValueError("item is not waiting at a gate")
        if item["gate"] == "board":
            if not self.store.transition(item_id, ("waiting",), status="queued", station="triage", gate=None):
                raise ValueError("item is no longer waiting")
            self.store.event(item_id, "triage", "approved", "Boarded by human")
        elif item["gate"] == "spec":
            if not self.store.transition(item_id, ("waiting",), status="queued", station="build", gate=None):
                raise ValueError("item is no longer waiting")
            self.store.event(item_id, "build", "approved", "Spec approved by human — waiting for a train")
        else:
            if not self.store.transition(item_id, ("waiting",), status="queued", gate="approved"):
                raise ValueError("item is no longer waiting")
            self.store.event(item_id, "merge", "approved", "Merge approved by human")
        return self.store.get_item(item_id)

    def reject(self, item_id: int, reason: str = "") -> dict:
        item = self.store.get_item(item_id)
        if not item or item["status"] != "waiting":
            raise ValueError("item is not waiting at a gate")
        try:
            label = "Declined by human (not boarded)" if item["gate"] == "board" else \
                f"Rejected by human at {item['gate']} gate"
            self._finish(item, "rejected", label + (f": {reason}" if reason else ""), expect=("waiting",))
        except ItemGone:
            raise ValueError("item is no longer waiting") from None
        return self.store.get_item(item_id)

    def cancel(self, item_id: int) -> dict:
        item = self.store.get_item(item_id)
        if not item or item["status"] not in ACTIVE_STATUSES + ("held",):
            raise ValueError("item is not active")
        agent = self.agents.get(f"item:{item_id}")
        if agent:
            agent.stop()
        self._cancel_events.setdefault(item_id, threading.Event()).set()
        if item["station"] == "retro":  # the journey already ended; just skip the retrospective
            try:
                self._finish(item, item.get("pending_status") or "failed",
                             (item.get("outcome") or "") + " (retrospective skipped by human)",
                             expect=ACTIVE_STATUSES, announce=False)
            except ItemGone:
                raise ValueError("item finished before it could be cancelled") from None
            self.store.event(item_id, "retro", "notice", "Retrospective skipped by human")
            return self.store.get_item(item_id)
        try:
            self._finish(item, "cancelled", "Cancelled by human", expect=ACTIVE_STATUSES + ("held",))
        except ItemGone:
            raise ValueError("item finished before it could be cancelled") from None
        return self.store.get_item(item_id)

    def retry(self, item_id: int) -> dict:
        """Human override: re-run a failed/rejected/held/cancelled item. A
        human retry skips the triage gate (the human has decided it's wanted)."""
        item = self.store.get_item(item_id)
        if not item or item["status"] not in ("failed", "rejected", "held", "cancelled"):
            raise ValueError("only failed, rejected, held or cancelled items can be retried")
        station = "build" if item.get("spec") else "spec"
        if not self.store.transition(item_id, ("failed", "rejected", "held", "cancelled"), status="queued",
                                     station=station, outcome=None, finished_at=None, hold_until=None, attempt=0,
                                     conflicts=0, holds=0, feedback=None, started_at=None, satisfaction=None,
                                     gate=None):
            raise ValueError("item changed; try again")
        self.store.kv_delete_prefix(f"retries:{item_id}:")
        self._cancel_events.pop(item_id, None)
        self.store.kv_set(f"setup:{item_id}", None)
        self.store.event(item_id, station, "retried", f"Re-queued at {station} by human")
        return self.store.get_item(item_id)

    def dispatch_now(self):
        self.store.kv_set("dispatcher_last", 0)
        self.store.event(None, "intake", "notice", "Dispatcher requested by human")

    # ─── JY09 Retro & the playbook (continuous learning) ────────────────

    def playbook(self, project: str) -> list[dict]:
        """Per-station notes injected into future trains' prompts for this project."""
        notes = self.store.kv_get(f"playbook:{project}")
        if notes is None:  # migrate builder lessons from before the retrospective existed
            notes = [{"id": i + 1, "station": "build", "text": l["text"], "item_id": l.get("item_id"),
                      "ts": l.get("ts", 0), "uses": 0, "wins": 0}
                     for i, l in enumerate(self.store.kv_get(f"lessons:{project}", []) or [])]
            self.store.kv_set(f"playbook:{project}", notes)
        return notes

    def _save_playbook(self, project: str, notes: list[dict]):
        self.store.kv_set(f"playbook:{project}", notes)

    def delete_note(self, project: str, note_id: int) -> list[dict]:
        notes = self.playbook(project)
        keep = [n for n in notes if n["id"] != note_id]
        if len(keep) == len(notes):
            raise ValueError("no such playbook note")
        removed = next(n for n in notes if n["id"] == note_id)
        self._save_playbook(project, keep)
        self.store.event(None, "retro", "notice", f"Playbook note removed by human ({removed['station']}): "
                         f"{removed['text'][:200]}")
        return keep

    def _playbook_section(self, project: str, station: str) -> str:
        notes = [n for n in self.playbook(project) if n["station"] == station]
        if not notes:
            return ""
        return ("\n\n# Playbook for this station (learned from previous trains on this project)\n"
                + "\n".join(f"- {n['text']}" for n in notes))

    def _journey(self, item: dict) -> str:
        """Everything the retrospective needs to know about one train's trip."""
        now = time.time()
        runs = self.store.runs(item["id"])
        per_station: dict[str, dict] = {}
        for r in runs:
            d = per_station.setdefault(r["station"] or "-", {"seconds": 0.0, "cost": 0.0, "runs": 0, "models": set()})
            d["seconds"] += (r["ended_at"] or now) - r["started_at"]
            d["cost"] += r["cost_usd"] or 0
            d["runs"] += 1
            if r["role"] not in ("ci", "jev"):
                d["models"].add(r["model"])
        table = "\n".join(f"- {st}: {d['runs']} runs, {d['seconds']:.0f}s, ${d['cost']:.3f}"
                          + (f", models {', '.join(sorted(d['models']))}" if d["models"] else "")
                          for st, d in per_station.items())
        interesting = ("classified", "rework", "conflict", "escalated", "checks", "verified", "disputed", "retry", "gate",
                       "approved", "held", "regression", "integrated", "notice", "failed", "done")
        lines = []
        for e in self.store.events(item["id"], limit=400):
            if e["kind"] not in interesting:
                continue
            line = f"- [{e['station'] or '-'}] {e['kind']}: {e['message'][:700]}"
            for r in (e.get("data") or {}).get("results") or []:
                line += f"\n    {'PASS' if r.get('passed') else 'FAIL'} {r.get('name')}: {str(r.get('evidence', ''))[:240]}"
            lines.append(line)
        parent = ""
        if item.get("parent_id"):
            p = self.store.get_item(item["parent_id"]) or {}
            parent = f"\nThis bug was filed as a regression after train #{item['parent_id']} ({p.get('title')}).\n"
        return (f"# Train #{item['id']}: {item['title']} ({item['kind']}, {item['priority']} priority, from {item['source']})\n"
                f"{item['description'][:1500]}\n{parent}\n"
                f"Outcome: {item.get('pending_status')} — {item.get('outcome') or ''}\n"
                f"Difficulty (Jev): {item.get('difficulty')}; service class: {item.get('first_class')} → "
                f"{item.get('service_class')}; build attempts: {item.get('attempt')}; conflicts: {item.get('conflicts')}; "
                f"holdout satisfaction: {item.get('satisfaction')}; cost: ${item.get('cost_usd') or 0:.3f}\n\n"
                f"# Time and cost by station\n{table or '(no runs)'}\n\n# Journey\n" + ("\n".join(lines) or "(no events)"))

    def _station_retro(self, item: dict):
        outcome = item.get("pending_status") or "failed"
        project = item["project"]
        notes = self.playbook(project)
        visited = {r["station"] for r in self.store.runs(item["id"]) if r["station"]}
        # Score the notes this train ran with: a "win" is arriving without rework.
        won = outcome == "done" and (item.get("attempt") or 0) <= 1
        started = item.get("started_at") or item["created_at"]
        for n in notes:
            if n["station"] in visited and n.get("ts", 0) < started:
                n["uses"] = n.get("uses", 0) + 1
                n["wins"] = n.get("wins", 0) + (1 if won else 0)
        retired: list[dict] = []
        for n in list(notes):
            if n.get("uses", 0) >= settings.NOTE_RETIRE_MIN_USES and \
                    n.get("wins", 0) / n["uses"] < settings.NOTE_RETIRE_MAX_WIN_RATE:
                notes.remove(n)
                retired.append({**n, "why": "auto: poor first-pass rate"})
        playbook_text = "\n".join(
            f"- id {n['id']} [{n['station']}] {n['text']} (used by {n.get('uses', 0)} trains, "
            f"{n.get('wins', 0)} passed first time)" for n in notes) or "(empty)"
        prompt = f"{self._journey(item)}\n\n# Current playbook\n{playbook_text}"
        try:
            data, res = self._single_shot(item, "retro", "retrospective", prompts.RETRO, prompt,
                                          enforce_item_budget=False)
            if not _usable_retro(data):
                # Cheap models sometimes echo the schema instead of answering; one retry a class up.
                base = self.model_for("retro", item)[1] or "rapid"
                up = settings.CLASS_ORDER[min(settings.CLASS_ORDER.index(base) + 1, len(settings.CLASS_ORDER) - 2)]
                retry_item = {**item, "service_class": up}  # retro_retry routes to the item's class
                data, res = self._single_shot(retry_item, "retro_retry", "retrospective", prompts.RETRO,
                                              prompt + "\n\nYour previous answer was empty or a copy of the schema. "
                                              "Write the actual retrospective for this journey.",
                                              enforce_item_budget=False)
                if not _usable_retro(data):
                    data, res = None, AgentResult(status="error", error="retrospective returned no usable answer")
        except LineSuspended:
            raise
        except Exception as e:  # a failed retrospective must never strand a train
            data, res = None, AgentResult(status="error", error=str(e))
        item = self._live(item["id"])
        added: list[dict] = []
        if data:
            for rid in data.get("retire") or []:
                victim = next((n for n in notes if str(n["id"]) == str(rid)), None)
                if victim:
                    notes.remove(victim)
                    retired.append({**victim, "why": "retrospective"})
            known = {n["text"].strip().lower() for n in notes}
            next_id = max([n["id"] for n in notes + retired] + [self.store.kv_get(f"playbook_seq:{project}", 0) or 0]) + 1
            for raw in (data.get("notes") or [])[:3]:
                if not isinstance(raw, dict):
                    continue
                station, text = str(raw.get("station", "")).strip(), str(raw.get("note", "")).strip()
                if station not in settings.PLAYBOOK_STATIONS or not text or text.lower() in known:
                    continue
                note = {"id": next_id, "station": station, "text": text[:280], "item_id": item["id"],
                        "ts": time.time(), "uses": 0, "wins": 0}
                next_id += 1
                notes.append(note)
                added.append(note)
                known.add(text.lower())
                # Cap each station's notes: drop the weakest (lowest win rate, then oldest).
                same = [n for n in notes if n["station"] == station]
                while len(same) > settings.MAX_NOTES_PER_STATION:
                    weakest = min((n for n in same if n is not note),
                                  key=lambda n: (n.get("wins", 0) / max(1, n.get("uses", 0)), n.get("ts", 0)))
                    notes.remove(weakest)
                    same.remove(weakest)
                    retired.append({**weakest, "why": "replaced: station playbook full"})
            self.store.kv_set(f"playbook_seq:{project}", next_id)
        self._save_playbook(project, notes)
        class_fit = (data or {}).get("class_fit") if (data or {}).get("class_fit") in (
            "right", "underpowered", "overpowered") else None
        record = {"summary": str((data or {}).get("summary", ""))[:1500],
                  "went_well": [str(x)[:300] for x in (data or {}).get("went_well") or []][:6],
                  "went_wrong": [str(x)[:300] for x in (data or {}).get("went_wrong") or []][:6],
                  "root_cause": str((data or {}).get("root_cause", ""))[:600], "class_fit": class_fit,
                  "notes_added": added, "notes_retired": retired, "error": None if data else (res.error or "no answer")}
        self.store.save_retro(item["id"], outcome, class_fit, record)
        for n in added:
            self.store.event(item["id"], "retro", "playbook", f"New {n['station']} note: {n['text']}")
        for n in retired:
            self.store.event(item["id"], "retro", "playbook", f"Retired {n['station']} note ({n['why']}): {n['text'][:200]}")
        msg = record["summary"] or ("Retrospective unavailable: " + (record["error"] or "")[:200])
        if class_fit and class_fit != "right":
            msg += f" Class fit: {class_fit}."
        self.store.event(item["id"], "retro", "retro", msg[:1500],
                         {"added": len(added), "retired": len(retired), "class_fit": class_fit})
        self._finish(item, outcome, item.get("outcome") or outcome, keep_branch=outcome == "failed", announce=False)

    # ─── Dispatcher (feeds the line) ────────────────────────────────────

    def projects(self) -> list[dict]:
        projs = settings.load_projects()
        if not projs:
            d = self.default_project()
            return [{"name": os.path.basename(d), "path": d, "priority": 0}] if d else []
        out = []
        for name, p in projs.items():
            out.append({"name": name, "path": os.path.realpath(os.path.expanduser(p.get("path", ""))),
                        "priority": p.get("priority", 999), "schedule": p.get("schedule"),
                        "paused": bool(p.get("paused"))})
        return out

    @staticmethod
    def default_project() -> str | None:
        if settings.DEFAULT_PROJECT:
            return os.path.realpath(os.path.join(settings.DEVELOPMENT_DIR, settings.DEFAULT_PROJECT))
        return None

    def pick_project(self) -> str | None:
        now_hour = time.localtime().tm_hour
        candidates = []
        for p in self.projects():
            if p.get("paused") or validate_project(p["path"]):
                continue
            if (self.store.kv_get(f"stall:{p['path']}", 0) or 0) > time.time():
                continue
            sched = p.get("schedule")
            candidates.append((0 if sched and settings.is_in_schedule_window(sched, now_hour) else
                               1 if not sched else 2, p.get("priority", 999), p["path"]))
        candidates = [c for c in candidates if c[0] < 2]
        return min(candidates)[2] if candidates else None

    def _maybe_dispatcher(self):
        if "dispatcher" in self.jobs:
            return
        waiting = [i for i in self.store.items(ACTIVE_STATUSES) if not i.get("train") and i["station"] != "retro"]
        busy_trains = len(self.trains_in_use())
        if len(waiting) >= settings.MIN_READY_ITEMS or busy_trains >= len(self.trains):
            return
        if time.time() - (self.store.kv_get("dispatcher_last", 0) or 0) < settings.DISPATCHER_INTERVAL:
            return
        project = self.pick_project()
        if not project:
            return
        self.store.kv_set("dispatcher_last", time.time())
        self._submit("dispatcher", self._dispatcher_job, project)

    def _dispatcher_job(self, project: str):
        self.store.event(None, "intake", "dispatching", f"Dispatcher surveying {os.path.basename(project)}")
        task = (f"Project: {project}\n\nRecent app log lines:\n{_app_log_tail(project) or '(none found)'}\n\n"
                f"{self._history_context(project)}\n\nRecent commits:\n{gitops.recent_log(project, 20)}\n\n"
                f"Work balance: {self._balance(project)}" + self._playbook_section(project, "dispatcher"))
        try:
            res = self._run_agent(None, "dispatcher", "dispatcher", prompts.DISPATCHER, task, root=project,
                                  tools=READ_TOOLS, schema=prompts.DISPATCHER_RESULT, key="dispatcher")
        except LineSuspended as e:
            self._suspend(e.reason, e.seconds)
            return
        if not res.ok or not res.result:
            self.store.event(None, "intake", "notice", f"Dispatcher failed: {res.error or res.status}")
            return
        r = res.result
        if not str(r.get("title", "")).strip():
            self.store.event(None, "intake", "notice", f"Dispatcher found nothing worth doing: {res.summary[:300]}")
            return
        kind = r.get("kind") if r.get("kind") in ("feature", "bug", "chore") else "feature"
        self.create_item(r["title"], r.get("description", ""), project, kind=kind,
                         priority=r.get("priority", "medium"), source="dispatcher")

    def _history_context(self, project: str, exclude_id: int | None = None) -> str:
        done = self.store.history(project, ("done",), days=30, limit=15)
        rejected = self.store.history(project, ("rejected", "failed"), days=30, limit=15)
        active = [i for i in self.store.items(ACTIVE_STATUSES + ("held",))
                  if i["project"] == project and i["id"] != exclude_id]
        in_flight = [f"- #{i['id']} {i['title']} ({i['station']}): {i['description'][:160]}" for i in active] or ["- (nothing)"]
        built = [f"- {i['title']}" for i in done] or ["- (nothing)"]
        dropped = [f"- {i['title']} [{time.strftime('%Y-%m-%d', time.localtime(i['updated_at']))}]: "
                   f"{(i.get('outcome') or '')[:160]}" for i in rejected] or ["- (nothing)"]
        return "\n".join(["Already in progress on the line (do not duplicate):", *in_flight,
                          "Recently built:", *built, "Recently rejected or failed (with reasons):", *dropped])

    def _balance(self, project: str) -> str:
        recent = self.store.history(project, ("done",), days=60, limit=20)
        if not recent:
            return "no recent merges"
        counts: dict[str, int] = {}
        for i in recent:
            counts[i["kind"]] = counts.get(i["kind"], 0) + 1
        return ", ".join(f"{k}: {v}" for k, v in counts.items()) + f" (last {len(recent)} merges)"

    # ─── Signal (log watcher) ───────────────────────────────────────────

    def _watch_logs(self):
        for p in self.projects():
            if settings.RAILWAY_PROJECT:
                new = self._read_new_railway_lines(p["path"])
            else:
                path = _find_app_log(p["path"])
                new = self._read_new_lines(path) if path else []
            errors = [l for l in new if _WATCH_PATTERN.search(l)]
            if not errors:
                continue
            METRICS.log_errors_detected_total.inc(amount=len(errors))
            now = time.time()
            # A post-deploy regression check needs no model, so it runs even when paused or suspended.
            regression = self._check_regression(p["path"], errors)
            history = self._sig_history.setdefault(p["path"], {})
            for line in errors:
                history[_signature(line)] = now
            if len(history) > 500:
                for s, _ in sorted(history.items(), key=lambda kv: kv[1])[:100]:
                    history.pop(s, None)
            if regression or self.paused or self.suspension():
                continue  # Signal's model calls respect pause and budget
            sig = _signature(errors[0])
            self._signal_seen = {k: v for k, v in self._signal_seen.items() if now - v < 3600}
            if sig in self._signal_seen or f"signal:{p['path']}" in self.jobs:
                continue
            if now - (self.store.kv_get("last_merge_at", 0) or 0) < 60:
                continue  # let a fresh deploy settle
            open_bugs = [i for i in self.store.items(ACTIVE_STATUSES + ("held",))
                         if i["source"] == "signal" and i["project"] == p["path"]]
            if len(open_bugs) >= settings.MAX_SIGNAL_OPEN_BUGS:
                continue
            self._signal_seen[sig] = now
            self._submit(f"signal:{p['path']}", self._signal_job, p["path"], errors[:80],
                         [b["title"] for b in open_bugs])

    def _read_new_railway_lines(self, project: str) -> list[str]:
        """Poll `railway logs` at most once a minute; return lines after the last one seen."""
        now = time.time()
        if now - self._railway_polled.get(project, 0) < 60:
            return []
        self._railway_polled[project] = now
        lines = _railway_logs(settings.RAILWAY_PRODUCTION_ENV, project).splitlines()
        if not lines:
            return []
        last = self._railway_last.get(project)
        self._railway_last[project] = lines[-1]
        if last is None:
            return []  # first poll: don't replay history
        return lines[lines.index(last) + 1:] if last in lines else lines

    def _read_new_lines(self, path: str) -> list[str]:
        try:
            size = os.path.getsize(path)
        except OSError:
            return []
        if path not in self._log_offsets:
            self._log_offsets[path] = size
            return []
        start = self._log_offsets[path] if size >= self._log_offsets[path] else 0
        if size == start:
            return []
        try:
            with open(path, errors="replace") as f:
                f.seek(start)
                text = f.read(2_000_000)
        except OSError:
            return []
        self._log_offsets[path] = size
        return text.splitlines()

    def _signal_job(self, project: str, lines: list[str], open_bugs: list[str]):
        METRICS.signal_triggers_total.inc()
        backend = settings.project_decide_backend(project)
        d = decisions.log_is_actionable(lines, open_bugs, backend)
        if d:
            self._record_jev(None, "intake", "signal screen", d.cost_usd, d.tokens,
                             f"actionable={d.p('actionable'):.2f} tracked={d.p('tracked'):.2f}", backend)
            if d.p("actionable") < 0.5 or d.p("tracked") >= 0.5:
                self.store.event(None, "intake", "signal",
                                 f"Signal: {len(lines)} error lines in {os.path.basename(project)} judged "
                                 f"{'already tracked' if d.p('tracked') >= 0.5 else 'not actionable'} by Jev")
                return
        prompt = ("New log lines:\n" + "\n".join(lines[:60]) + "\n\nOpen bugs:\n" +
                  ("\n".join(f"- {b}" for b in open_bugs) or "(none)"))
        data, res = self._single_shot(None, "signal", "signal", prompts.SIGNAL, prompt)
        if not data or not data.get("file") or not data.get("title"):
            self.store.event(None, "intake", "signal", f"Signal reviewed {len(lines)} error lines: nothing new to file")
            return
        item = self.create_item(data["title"], data.get("description", ""), project, kind="bug",
                                priority=data.get("priority", "high"), source="signal")
        self.store.event(item["id"], "intake", "signal", "Filed by Signal from application logs",
                         {"lines": lines[:20]})

    # ─── Ops digest ─────────────────────────────────────────────────────

    def _maybe_ops(self):
        if "ops" in self.jobs:
            return
        last = self.store.kv_get("ops_last", None)
        if last is None:
            self.store.kv_set("ops_last", time.time())
            return
        if time.time() - last < settings.OPS_INTERVAL:
            return
        self.store.kv_set("ops_last", time.time())
        events = [e for e in self.store.events(limit=150) if e["ts"] > last and e["kind"] != "ops_report"]
        if len(events) < 5:
            return
        self._submit("ops", self._ops_job, events)

    def _ops_job(self, events: list[dict]):
        stats = self.store.stats()
        timeline = "\n".join(f"{time.strftime('%H:%M', time.localtime(e['ts']))} "
                             f"{'#' + str(e['item_id']) if e['item_id'] else '  '} [{e['station'] or '-'}] "
                             f"{e['kind']}: {e['message'][:200]}" for e in events)
        data, _ = self._single_shot(None, "ops", "ops", prompts.OPS,
                                    f"Stats: {json.dumps(stats, default=str)[:3000]}\n\n{_ops_settings()}"
                                    f"\n\nTimeline:\n{timeline}")
        if data and data.get("summary"):
            self.store.event(None, None, "ops_report", data["summary"][:3000],
                             {"recommendations": data.get("recommendations") or []})


# ─── module helpers ──────────────────────────────────────────────────────────

_LOG_NOISE = re.compile(r"\d{4}-\d\d-\d\dT[\d:.]+Z?|\b(error|critical|fatal|traceback|most recent call last|"
                        r"exception|warn(ing)?|info|debug)\b", re.I)


def _error_slug(line: str) -> str:
    """Short title fragment for an error line: the exception name and its
    message if there is one ("valueerror-time-data-does-not-match"), else the
    line with log boilerplate removed."""
    m = re.search(r"\b([A-Z]\w*(?:Error|Exception|Exit|Interrupt))\b[:\s]*(.*)", line)
    text = f"{m.group(1)} {m.group(2)}" if m else _LOG_NOISE.sub(" ", line)
    words = re.findall(r"[a-z]+", re.sub(r"\d+", " ", text.lower()))
    return "-".join(words)[:44].strip("-") or "new-errors"


def _usable_retro(data) -> bool:
    """A real retrospective, not an echo of the schema."""
    if not isinstance(data, dict):
        return False
    summary = str(data.get("summary") or "").strip()
    return len(summary) >= 20 and summary.strip(". ") != "" and "..." not in summary[:10]


def _ops_settings() -> str:
    """The tunable settings, with current values, so Ops recommends real knobs."""
    knobs = {
        "AGENT_TEAM_MAX_TRAINS": settings.MAX_TRAINS, "AGENT_TEAM_DAILY_BUDGET_USD": settings.DAILY_BUDGET_USD,
        "AGENT_TEAM_ITEM_BUDGET_USD": settings.ITEM_BUDGET_USD, "AGENT_TEAM_RUN_BUDGET_USD": settings.RUN_BUDGET_USD,
        "AGENT_TEAM_GATE_SPEC": settings.GATE_SPEC, "AGENT_TEAM_GATE_MERGE": settings.GATE_MERGE,
        "AGENT_TEAM_SATISFACTION": settings.SATISFACTION_THRESHOLD, "AGENT_TEAM_TEST_CMD": settings.TEST_CMD or "(unset)",
        "AGENT_TEAM_SETUP_CMD": settings.SETUP_CMD or "(unset)", "AGENT_TEAM_ADAPTIVE_ROUTING": settings.ADAPTIVE_ROUTING,
        "AGENT_TEAM_DISPATCHER_INTERVAL": settings.DISPATCHER_INTERVAL, "AGENT_TEAM_AUTO_REVERT": settings.AUTO_REVERT,
        "AGENT_TEAM_DEPLOY_WATCH_SECONDS": settings.DEPLOY_WATCH_SECONDS,
        "models.json": "per-class / per-station model overrides",
        "projects.json": "per-project setup/test commands, gates, schedule, decide_backend",
    }
    return "AVAILABLE SETTINGS (current values):\n" + "\n".join(f"- {k} = {v}" for k, v in knobs.items())


def _signature(line: str) -> str:
    """Error identity with numbers (ids, timestamps, ports) masked."""
    return re.sub(r"\d+", "N", line)[:120]


def validate_project(path: str) -> str | None:
    """Reason a project directory can't be worked on, or None if it's fine."""
    if not path or not os.path.isdir(path):
        return f"project directory {path or '(none)'} does not exist"
    real = os.path.realpath(path)
    if real == os.path.realpath(settings.BASE_DIR):
        return "Yamanote must not work on itself"
    dev = os.path.realpath(settings.DEVELOPMENT_DIR)
    if real != dev and not real.startswith(dev + os.sep):
        return f"project must be under {settings.DEVELOPMENT_DIR}"
    if not os.path.isdir(os.path.join(real, ".git")):
        return "project is not a git repository"
    return None


def _find_app_log(project_dir: str) -> str | None:
    patterns = ([settings.APP_LOG_GLOB] if settings.APP_LOG_GLOB else []) + ["logs/*.log", "*.log"]
    for pattern in patterns:
        matches = sorted(glob.glob(os.path.join(project_dir, pattern)), key=os.path.getmtime, reverse=True)
        if matches:
            return matches[0]
    return None


def _app_log_tail(project_dir: str, lines: int = 80) -> str:
    if settings.RAILWAY_PROJECT:
        return _railway_logs(settings.RAILWAY_PRODUCTION_ENV, project_dir)[-8000:]
    path = _find_app_log(project_dir)
    if not path:
        return ""
    try:
        with open(path, errors="replace") as f:
            return "".join(f.readlines()[-lines:])[-8000:]
    except OSError:
        return ""


def _railway_logs(environment: str, cwd: str) -> str:
    try:
        proc = subprocess.run(["railway", "logs", "--environment", environment], cwd=cwd,
                              capture_output=True, text=True, timeout=8)
        return proc.stdout
    except subprocess.TimeoutExpired as e:
        return (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
    except OSError:
        return ""
