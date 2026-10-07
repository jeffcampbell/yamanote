"""Deterministic checks: a project's own setup and test commands.

These run without any model: setup once per worktree (install dependencies),
tests after every build and again when trunk is merged in before landing.
They're free, fast and can't be talked into passing, so they run before any
reviewer tokens are spent.
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass

from . import settings
from .agent import _SECRET_ENV, _kill_group


@dataclass
class CheckResult:
    command: str
    ok: bool
    exit_code: int | str
    seconds: float
    output: str  # tail of combined stdout/stderr

    def summary(self) -> str:
        return f"`{self.command}` {'passed' if self.ok else 'FAILED'} (exit {self.exit_code}, {self.seconds:.0f}s)"


def run_check(command: str, cwd: str, timeout: int, cancel=None) -> CheckResult:
    env = {k: v for k, v in os.environ.items() if not _SECRET_ENV.search(k) and k != "CLAUDECODE"}
    env.update(GIT_TERMINAL_PROMPT="0", CI="1")
    start = time.monotonic()
    with tempfile.TemporaryFile(mode="w+", errors="replace") as out:
        proc = subprocess.Popen(["bash", "-c", command], cwd=cwd, env=env, text=True, stdout=out,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        rc: int | str
        while True:
            try:
                rc = proc.wait(timeout=0.25)
                break
            except subprocess.TimeoutExpired:
                if cancel is not None and cancel.is_set():
                    rc = "cancelled"
                    break
                if time.monotonic() - start > timeout:
                    rc = f"timeout after {timeout}s"
                    break
        _kill_group(proc.pid)  # also reaps anything the command left running
        proc.wait()
        out.seek(0)
        text = out.read()
    tail = text[-settings.CHECK_OUTPUT_CHARS:]
    if len(text) > len(tail):
        tail = f"[... {len(text) - len(tail)} earlier characters omitted]\n" + tail
    return CheckResult(command, rc == 0, rc, time.monotonic() - start, _strip_ansi(tail))


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)
