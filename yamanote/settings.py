"""Configuration for the Yamanote software factory.

Everything here can be overridden by environment variables (loaded from the
repo's .env and ~/development/.env without overriding the real environment),
and model routing can be overridden with a models.json next to this package.
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DOTENV_PATHS = [BASE_DIR / ".env", Path.home() / "development" / ".env"]


def load_dotenv(paths=DOTENV_PATHS) -> None:
    """Load KEY=VALUE lines into os.environ without overriding existing vars."""
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.removeprefix("export ").partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


load_dotenv()


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


# ─── Paths ───────────────────────────────────────────────────────────────────

DATA_DIR = Path(_env("YAMANOTE_DATA_DIR", str(BASE_DIR / "agents")))
DB_PATH = DATA_DIR / "yamanote.db"
PAUSE_FILE = DATA_DIR / "pause"  # touch to pause, rm to resume (dashboard toggles it too)
PID_FILE = DATA_DIR / "orchestrator.pid"
WEB_DIR = Path(__file__).resolve().parent / "web"
PROJECTS_CONFIG_PATH = BASE_DIR / "projects.json"
MODELS_CONFIG_PATH = BASE_DIR / "models.json"

DEVELOPMENT_DIR = os.path.expanduser(_env("AGENT_TEAM_DEV_DIR", "~/development"))
DEFAULT_PROJECT = _env("AGENT_TEAM_DEFAULT_PROJECT", "")
APP_LOG_GLOB = _env("AGENT_TEAM_APP_LOG_GLOB", "")
TRUNK_BRANCH = _env("AGENT_TEAM_TRUNK_BRANCH", "main")
SERVICE_RESTART_CMD = _env("AGENT_TEAM_SERVICE_RESTART_CMD", "")
RAILWAY_PROJECT = _env("AGENT_TEAM_RAILWAY_PROJECT", "")
RAILWAY_STAGING_ENV = _env("AGENT_TEAM_RAILWAY_STAGING_ENV", "staging")
RAILWAY_PRODUCTION_ENV = _env("AGENT_TEAM_RAILWAY_PRODUCTION_ENV", "production")

# ─── OpenRouter ──────────────────────────────────────────────────────────────

OPENROUTER_URL = _env("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions")


def openrouter_key() -> str | None:
    return os.environ.get("OPENROUTER_API_KEY")


# ─── decide / Jev ────────────────────────────────────────────────────────────

# decide is imported as an installed package; YAMANOTE_DECIDE_SRC points at a
# checkout's src/ instead (https://github.com/jeffcampbell/Decide).
DECIDE_INSTALL = "pip install git+https://github.com/jeffcampbell/Decide.git"
DECIDE_SRC = os.path.expanduser(_env("YAMANOTE_DECIDE_SRC", ""))
DECIDE_ENABLED = _env_bool("YAMANOTE_DECIDE", True)
DECIDE_BACKEND = _env("DECIDE_BACKEND", "jev")

# ─── Service classes (model tiers) ───────────────────────────────────────────
# A work item's service class is chosen from its difficulty (scored by Jev,
# or by the spec writer when Jev is unavailable) and is upgraded one class each
# time verification sends the train back for rework beyond ESCALATE_AFTER.

SERVICE_CLASSES: dict[str, dict] = {
    "local":      {"label": "Local",           "kanji": "各停",   "model": "deepseek/deepseek-v4-pro"},
    "rapid":      {"label": "Rapid",           "kanji": "快速",   "model": "minimax/minimax-m3"},
    "express":    {"label": "Limited Express", "kanji": "特急",   "model": "anthropic/claude-sonnet-5.5"},
    "shinkansen": {"label": "Shinkansen",      "kanji": "新幹線", "model": "anthropic/claude-opus-5.5"},
}
CLASS_ORDER = ["local", "rapid", "express", "shinkansen"]

# Difficulty levels (Jev score question, lowest first) → service class
DIFFICULTY_LEVELS = ["trivial", "small", "moderate", "hard", "very hard"]
DIFFICULTY_TO_CLASS = {
    "trivial": "local", "small": "local", "moderate": "rapid",
    "hard": "express", "very hard": "express",
}

# Per-station model. A value is one of:
#   a service class name       "rapid"
#   the train's class           "builder", or relative to it: "builder+1", "builder-1"
#   either of those, clamped    "builder[rapid..express]", "builder+1[..express]"
#   a literal OpenRouter model  "openai/gpt-6"
# Stations that run before a train has a class (dispatcher, triage) use a fixed class;
# a train without one yet counts as "rapid".
STATION_MODELS: dict[str, str] = {
    "dispatcher": "rapid",
    "triage":     "local",
    "spec":       "builder[local..express]",   # trivial work gets a cheap spec, hard work a strong one
    "build":      "builder",
    "inspect":    "builder+1",
    "verify":     "builder[rapid..express]",
    "signal":     "local",
    "ops":        "local",
    "redact":     "local",
    "retro":      "builder[rapid..express]",
    "retro_retry": "builder",  # set to one class above "retro" when a retrospective comes back unusable
}

_CHOICE = re.compile(r"builder(?P<delta>[+-]\d+)?(?:\[(?P<lo>\w*)\.\.(?P<hi>\w*)\])?")


def station_class(choice: str, item_class: str | None) -> str | None:
    """The service class a STATION_MODELS value resolves to for a train of
    `item_class`, or None when the value is a literal model id."""
    if choice in SERVICE_CLASSES:
        return choice
    m = _CHOICE.fullmatch(choice)
    if not m or any(b and b not in CLASS_ORDER for b in (m["lo"], m["hi"])):
        return None
    order = CLASS_ORDER
    base = order.index(item_class) if item_class in order else order.index("rapid")
    lo = order.index(m["lo"]) if m["lo"] in order else 0
    hi = order.index(m["hi"]) if m["hi"] in order else len(order) - 1
    return order[max(lo, min(hi, base + int(m["delta"] or 0)))]
FALLBACK_MODEL = _env("YAMANOTE_FALLBACK_MODEL", "openrouter/auto")

# ─── Model fleet (see fleet.py) ──────────────────────────────────────────────
# The models above are only starting points. Once a day the fleet reads
# OpenRouter's catalogue, sorts usable models into each class's price band, and
# tries challengers on a share of low-risk trains; one that does at least as
# well for no more money becomes the class's model. Classes named in
# models.json "classes" are pinned and never change.
FLEET_ENABLED = _env_bool("AGENT_TEAM_MODEL_FLEET", True)
TRIAL_RATE = _env_float("AGENT_TEAM_MODEL_TRIAL_RATE", 0.10)   # share of eligible trains that try a challenger
TRIAL_LEVELS = ("trivial", "small")      # only low-risk work runs trials
TRIAL_STATIONS = ("spec", "build")       # first build only; a rework goes back to the class's model
TRIAL_MIN_SAMPLES = 8                    # trains each side before a challenger can be promoted or benched
TRIAL_BENCH_DAYS = 30                    # a clearly worse challenger sits out this long
CHALLENGERS_PER_CLASS = 3
CATALOG_URL = _env("YAMANOTE_CATALOG_URL", "https://openrouter.ai/api/v1/models")
CATALOG_REFRESH_SECONDS = 86400
FLEET_EVALUATE_SECONDS = 3600
MIN_CONTEXT_TOKENS = 128_000
EXPIRY_MARGIN_DAYS = 30                  # replace a model this long before OpenRouter retires it
JEV_ROUTER_MODEL = "typesafe/jev-router" # tried as a challenger in every class below Shinkansen
# Upper price of each class, in USD per million input tokens with output and cache
# reads folded in at this line's measured mix (fleet.blended_price). Shinkansen is
# everything above Limited Express.
CLASS_BANDS: dict[str, float | None] = {"local": 0.15, "rapid": 0.60, "express": 1.80, "shinkansen": None}
PINNED_CLASSES: set[str] = set()
ESCALATE_AFTER = 1  # reworks on the same class before escalating


def _load_model_overrides() -> None:
    if not MODELS_CONFIG_PATH.is_file():
        return
    try:
        data = json.loads(MODELS_CONFIG_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return
    for name, model in (data.get("classes") or {}).items():
        if name in SERVICE_CLASSES and isinstance(model, str):
            SERVICE_CLASSES[name]["model"] = model
            PINNED_CLASSES.add(name)
    for name, ceiling in (data.get("bands") or {}).items():
        if name in CLASS_BANDS and (ceiling is None or isinstance(ceiling, (int, float))):
            CLASS_BANDS[name] = ceiling
    for station, choice in (data.get("stations") or {}).items():
        if isinstance(choice, str):
            if "/" not in choice and station_class(choice, None) is None:
                logging.getLogger("yamanote.settings").warning(
                    "models.json: %s → %r is neither a class, a builder choice, nor a provider/model id",
                    station, choice)
            STATION_MODELS[station] = choice


_load_model_overrides()
# What each class starts on (after models.json); the fleet moves classes on from here.
DEFAULT_CLASS_MODELS = {c: v["model"] for c, v in SERVICE_CLASSES.items()}

# ─── Capacity, timing, guardrails ────────────────────────────────────────────

TICK_INTERVAL = _env_float("AGENT_TEAM_TICK_SECONDS", 3.0)
# Concurrent trains (items between BUILD and MERGE). The old per-type counts
# are still honoured if MAX_TRAINS isn't set.
MAX_TRAINS = _env_int("AGENT_TEAM_MAX_TRAINS", max(1, sum(
    _env_int(f"AGENT_TEAM_{k}_TRAINS", d) for k, d in (("REGULAR", 0), ("STANDARD", 1), ("EXPRESS", 0)))))
MAX_PLANNING_JOBS = _env_int("AGENT_TEAM_MAX_PLANNING_JOBS", 2)  # concurrent triage/spec runs
MIN_READY_ITEMS = 1  # dispatcher tops the line up when fewer items than this are waiting

AGENT_TIMEOUT_SECONDS = _env_int("AGENT_TEAM_AGENT_TIMEOUT_SECONDS", 1200)
AGENT_MAX_STEPS = _env_int("AGENT_TEAM_AGENT_MAX_STEPS", 60)
TOOL_TIMEOUT_SECONDS = 180
DISPATCHER_INTERVAL = _env_int("AGENT_TEAM_DISPATCHER_INTERVAL", 1800)
OPS_INTERVAL = _env_int("AGENT_TEAM_OPS_INTERVAL", 3600)

DAILY_BUDGET_USD = _env_float("AGENT_TEAM_DAILY_BUDGET_USD", 20.0)
ITEM_BUDGET_USD = _env_float("AGENT_TEAM_ITEM_BUDGET_USD", 4.0)
RUN_BUDGET_USD = _env_float("AGENT_TEAM_RUN_BUDGET_USD", 2.0)
MAX_RUNS_PER_HOUR = _env_int("AGENT_TEAM_MAX_RUNS_PER_HOUR", 60)

MAX_REWORK_ATTEMPTS = 3
MAX_CONFLICT_RETRIES = 3
SATISFACTION_THRESHOLD = _env_float("AGENT_TEAM_SATISFACTION", 1.0)  # fraction of holdout scenarios that must pass
ITEM_SLA_SECONDS = _env_int("AGENT_TEAM_ITEM_SLA_SECONDS", 3 * 3600)
HOLD_RECYCLE_SECONDS = 86400
MAX_CONSECUTIVE_REJECTIONS = 5
STALL_PAUSE_SECONDS = 86400
MAX_SIGNAL_OPEN_BUGS = 3
DIFF_MAX_CHARS = 40000
GIT_TIMEOUT = 30
SERVICE_RESTART_TIMEOUT = 300
LOG_RETENTION_DAYS = 14

# Human gates (defaults; projects.json can override per project with
# "gates": {"spec": true, "merge": true})
GATE_SPEC = _env_bool("AGENT_TEAM_GATE_SPEC", False)
GATE_MERGE = _env_bool("AGENT_TEAM_GATE_MERGE", False)

# ─── Checks (deterministic CI) ───────────────────────────────────────────────
# Per-project "setup" and "test" commands live in projects.json; these env vars
# are the defaults (handy for the single-project setup).
SETUP_CMD = _env("AGENT_TEAM_SETUP_CMD", "")   # e.g. "npm ci" or "python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
TEST_CMD = _env("AGENT_TEAM_TEST_CMD", "")     # e.g. "npm test" or "python3 -m unittest -q"
SETUP_TIMEOUT_SECONDS = _env_int("AGENT_TEAM_SETUP_TIMEOUT", 900)
TEST_TIMEOUT_SECONDS = _env_int("AGENT_TEAM_TEST_TIMEOUT", 600)
CHECK_OUTPUT_CHARS = 6000

# ─── Post-deploy watch ───────────────────────────────────────────────────────
DEPLOY_WATCH_SECONDS = _env_int("AGENT_TEAM_DEPLOY_WATCH_SECONDS", 900)  # watch logs this long after a deploy
AUTO_REVERT = _env_bool("AGENT_TEAM_AUTO_REVERT", False)  # revert the merge when a regression appears

# ─── Notifications ───────────────────────────────────────────────────────────
# NOTIFY_CMD runs through bash with the event as JSON on stdin and YAMANOTE_* env
# vars; NOTIFY_WEBHOOK receives the same JSON as a POST. Either, both, or neither.
NOTIFY_CMD = _env("AGENT_TEAM_NOTIFY_CMD", "")
NOTIFY_WEBHOOK = _env("AGENT_TEAM_NOTIFY_WEBHOOK", "")
NOTIFY_EVENTS = {e.strip() for e in _env(
    "AGENT_TEAM_NOTIFY_EVENTS", "gate,failed,suspended,regression,reverted,stalled,autopilot,routing").split(",") if e.strip()}
NOTIFY_TIMEOUT_SECONDS = 30
PUBLIC_URL = _env("AGENT_TEAM_PUBLIC_URL", "").rstrip("/")  # dashboard URL used for links in notifications

# ─── Learning ────────────────────────────────────────────────────────────────
ADAPTIVE_ROUTING = _env_bool("AGENT_TEAM_ADAPTIVE_ROUTING", True)  # let history move difficulty→class
ADAPTIVE_MIN_SAMPLES = 5
# Every train ends at JY09 Retro: a retrospective that writes per-station
# playbook notes for future trains on the project, retires notes that aren't
# helping, and judges whether the model class fitted the work.
RETRO_ENABLED = _env_bool("AGENT_TEAM_RETRO", _env_bool("AGENT_TEAM_LESSONS", True))
MAX_RETRO_JOBS = 2
PLAYBOOK_STATIONS = ("dispatcher", "triage", "spec", "build", "inspect", "verify")
MAX_NOTES_PER_STATION = 6
NOTE_RETIRE_MIN_USES = 8     # auto-retire a note after this many uses ...
NOTE_RETIRE_MAX_WIN_RATE = 0.3  # ... if fewer than this share of those trains passed first time

# ─── Misc guardrails added after review ──────────────────────────────────────
MERGE_QUEUE_RETRY_SECONDS = 15     # a train finding the project's merge queue busy checks back after this
MAX_HOLDS = 2                       # a third HOLD becomes a REJECT
CREDIT_SUSPEND_SECONDS = 3600       # pause the line this long on 401/402/403 from OpenRouter
RATE_LIMIT_SUSPEND_SECONDS = 600    # ... and this long when 429s outlast the client's retries

# ─── Autopilot ───────────────────────────────────────────────────────────────
# Supervised: the gates below apply, and work the Dispatcher or Signal propose
# waits at Intake for a human "Board". Autopilot ("dark"): gates are skipped,
# proposals board themselves, regressions auto-revert. These env values seed
# the dashboard's Settings panel; once saved there, the stored values win.
AUTOPILOT = _env_bool("AGENT_TEAM_AUTOPILOT", False)            # initial mode on first start
AUTOPILOT_ON_CRON = _env("AGENT_TEAM_AUTOPILOT_ON_CRON", "")     # e.g. "0 22 * * *"
AUTOPILOT_OFF_CRON = _env("AGENT_TEAM_AUTOPILOT_OFF_CRON", "")   # e.g. "0 7 * * 1-5"
AUTOPILOT_MERGE_WITHOUT_TESTS = _env_bool("AGENT_TEAM_AUTOPILOT_MERGE_WITHOUT_TESTS", False)

DASHBOARD_PORT = _env_int("AGENT_TEAM_DASHBOARD_PORT", 0)
DASHBOARD_HOST = _env("AGENT_TEAM_DASHBOARD_HOST", "127.0.0.1")  # 0.0.0.0 to expose on the LAN
DASHBOARD_TOKEN = _env("AGENT_TEAM_DASHBOARD_TOKEN", "")  # if set, required for actions


# ─── Projects ────────────────────────────────────────────────────────────────

def load_projects() -> dict:
    """Project definitions from projects.json (empty dict if absent)."""
    if not PROJECTS_CONFIG_PATH.is_file():
        return {}
    try:
        return json.loads(PROJECTS_CONFIG_PATH.read_text()).get("projects", {})
    except (OSError, json.JSONDecodeError):
        return {}


def project_config(project_dir: str) -> dict:
    """projects.json entry for this directory ({} if none)."""
    for proj in load_projects().values():
        if os.path.realpath(os.path.expanduser(proj.get("path", ""))) == os.path.realpath(project_dir):
            return proj
    return {}


def project_gates(project_dir: str) -> dict:
    gates = {"spec": GATE_SPEC, "merge": GATE_MERGE}
    gates.update({k: bool(v) for k, v in (project_config(project_dir).get("gates") or {}).items() if k in gates})
    return gates


def project_commands(project_dir: str) -> dict:
    """{"setup": str, "test": str} for a project ('' when not configured)."""
    proj = project_config(project_dir)
    return {"setup": proj.get("setup", SETUP_CMD) or "", "test": proj.get("test", TEST_CMD) or ""}


def project_decide_backend(project_dir: str) -> str:
    """'jev' (hosted), 'ollama' (local, for confidential code) or 'off'."""
    return project_config(project_dir).get("decide_backend") or DECIDE_BACKEND


def is_in_schedule_window(schedule: str | None, now_hour: int) -> bool:
    """'9-17' or '22-2' (inclusive, wraps midnight). None/empty/malformed = always active."""
    if not schedule:
        return True
    try:
        start_s, end_s = schedule.split("-", 1)
        start, end = int(start_s), int(end_s)
    except ValueError:
        return True
    if start <= end:
        return start <= now_hour <= end
    return now_hour >= start or now_hour <= end
