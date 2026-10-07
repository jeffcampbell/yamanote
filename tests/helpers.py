"""Offline test scaffolding: a scripted OpenRouter stand-in and temp repos."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

os.environ.setdefault("YAMANOTE_DECIDE", "0")  # tests never call Jev unless they patch it in

from yamanote import decisions, settings  # noqa: E402
from yamanote.llm import Completion  # noqa: E402

# The real .env (repo or ~/development) is loaded when settings is imported. Tests
# must not inherit its side effects: a dashboard token, notification hooks that
# message a real phone, restart/deploy commands. Tests that need one set it.
for _name, _value in {"DASHBOARD_TOKEN": "", "NOTIFY_CMD": "", "NOTIFY_WEBHOOK": "", "PUBLIC_URL": "",
                      "SERVICE_RESTART_CMD": "", "APP_LOG_GLOB": "", "RAILWAY_PROJECT": "",
                      "SETUP_CMD": "", "TEST_CMD": "", "DECIDE_SRC": "", "AUTOPILOT": False,
                      "AUTOPILOT_ON_CRON": "", "AUTOPILOT_OFF_CRON": "",
                      "AUTOPILOT_MERGE_WITHOUT_TESTS": False}.items():
    setattr(settings, _name, _value)


def tool_call(name: str, args: dict, n: int = 0) -> dict:
    return {"id": f"call_{name}_{n}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def reply(*calls: dict, text: str = "", cost: float = 0.001) -> Completion:
    msg = {"role": "assistant", "content": text}
    if calls:
        msg["tool_calls"] = list(calls)
    return Completion(message=msg, model="fake/model", tokens_in=100, tokens_out=20, cost_usd=cost)


def role_of(messages: list[dict]) -> str:
    system = messages[0]["content"]
    for role in ("Redactor", "Retrospective", "Dispatcher", "Triage", "Spec writer", "Builder", "Inspector",
                 "Verifier", "Signal", "Operations"):
        if role in system:
            return role.split()[0].lower()
    return "unknown"


class FakeClient:
    """Routes each chat() to a per-role script. A script is a callable
    (messages, call_number) -> Completion; turns already answered by a tool
    result are counted per role so multi-step agents can be scripted."""

    def __init__(self, scripts: dict):
        self.scripts = scripts
        self.calls: list[tuple[str, str]] = []  # (role, model)
        self.turns: dict[str, int] = {}

    def chat(self, model, messages, tools=None, fallbacks=None, max_tokens=8000,
             temperature=None, response_format=None):
        role = role_of(messages)
        self.calls.append((role, model))
        # Count turns in this conversation: assistant messages so far
        turn = sum(1 for m in messages if m.get("role") == "assistant")
        return self.scripts[role](messages, turn)


def finish(summary: str = "done", result: dict | None = None) -> Completion:
    args = {"summary": summary}
    if result is not None:
        args["result"] = result
    return reply(tool_call("finish", args))


def default_scripts(**overrides) -> dict:
    """A happy-path factory: triage BUILD, spec with 1 scenario, builder writes
    hello.py, inspector approves, verifier passes."""

    def build(messages, turn):
        if turn == 0:
            return reply(tool_call("write_file", {"path": "hello.py", "content": "print('hello')\n"}))
        return finish("wrote hello.py", {"verified_with": "python3 hello.py"})

    scripts = {
        "triage": lambda m, t: finish("ok", {"verdict": "BUILD", "reason": "useful"}),
        "spec": lambda m, t: finish("spec", {
            "acceptance_criteria": ["hello.py prints hello"], "plan": "add hello.py",
            "relevant_files": [], "difficulty": "small",
            "scenarios": [{"name": "prints hello", "steps": "python3 hello.py", "expected": "hello"}]}),
        "builder": build,
        "inspector": lambda m, t: finish("lgtm", {"verdict": "APPROVED", "issues": []}),
        "verifier": lambda m, t: finish("ran", {"results": [{"name": "prints hello", "passed": True,
                                                             "evidence": "printed hello"}]}),
        "redactor": lambda m, t: reply(text='{"defects": ["the program misbehaves"]}'),
        "retrospective": lambda m, t: reply(text='{"summary": "Clean run: arrived first time with no problems.", "went_well": [], "went_wrong": [],'
                                                 ' "root_cause": "", "class_fit": "right", "notes": [], "retire": []}'),
        "dispatcher": lambda m, t: finish("idea", {"title": "add-greeting", "kind": "feature",
                                                   "priority": "medium", "description": "Add a greeting script."}),
    }
    scripts.update(overrides)
    return scripts


class TempFactoryEnv:
    """Temp dev dir with a git project, temp data dir, settings patched."""

    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.dev = root / "dev"
        self.project = self.dev / "proj"
        self.project.mkdir(parents=True)
        (self.project / "README.md").write_text("# proj\n")
        sh("git init -q -b main && git add -A && git -c user.email=t@t -c user.name=t commit -qm init",
           self.project)
        self._saved = {k: getattr(settings, k) for k in
                       ("DEVELOPMENT_DIR", "DEFAULT_PROJECT", "DATA_DIR", "PAUSE_FILE", "PROJECTS_CONFIG_PATH",
                        "GATE_SPEC", "GATE_MERGE", "TICK_INTERVAL", "DISPATCHER_INTERVAL", "MAX_TRAINS")}
        settings.DEVELOPMENT_DIR = str(self.dev)
        settings.DEFAULT_PROJECT = "proj"
        settings.DATA_DIR = root / "data"
        settings.PAUSE_FILE = settings.DATA_DIR / "pause"
        settings.PROJECTS_CONFIG_PATH = root / "projects.json"
        settings.GATE_SPEC = settings.GATE_MERGE = False
        settings.DISPATCHER_INTERVAL = 10 ** 12  # tests opt in to the dispatcher explicitly
        decisions._engines.clear()
        decisions._errors.clear()
        decisions._errors[settings.DECIDE_BACKEND] = "disabled in tests"

    def close(self):
        for k, v in self._saved.items():
            setattr(settings, k, v)
        self.tmp.cleanup()


def sh(cmd: str, cwd) -> str:
    return subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, check=True).stdout


def run_until(factory, predicate, max_ticks: int = 200):
    import time
    for _ in range(max_ticks):
        factory.tick()
        if predicate():
            factory.tick()  # harvest
            return True
        time.sleep(0.02)
    return False
