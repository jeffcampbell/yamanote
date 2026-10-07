"""Cheap typed decisions via `decide` (Jev System One model by default).

Jev answers yes/no, choice and score questions about text for a fraction of a
cent, so the factory asks it first and only spends LLM tokens where it is
unsure. Every call here degrades to None when decide or its API key is
unavailable; callers then fall back to an LLM agent.

Used for:
  - difficulty → service class (model tier) for each work item
  - triage pre-screen (useful / ready / already done)
  - relevant-file selection, so builders don't spend tokens exploring
  - signal: is a burst of log errors actionable?
"""
from __future__ import annotations

import logging
import os
import sys
import threading
from dataclasses import dataclass

from . import settings

log = logging.getLogger("yamanote.decide")

_lock = threading.Lock()
_engines: dict[str, object] = {}
_backends: dict[str, object] = {}
_errors: dict[str, str] = {}


@dataclass
class Decision:
    answers: dict           # question name -> answer object (see decide.questions)
    tokens: int
    cost_usd: float

    def p(self, name: str) -> float:
        """P(yes) for a yes/no question."""
        return float(self.answers.get(name, {}).get("noul", 0.0))

    def score(self, name: str) -> float:
        return float(self.answers.get(name, {}).get("score", 0.0))


def _load(backend: str | None = None):
    """The decide Engine for `backend` (default: DECIDE_BACKEND), or None."""
    name = backend or settings.DECIDE_BACKEND
    with _lock:
        if name in _engines:
            return _engines[name]
        if name in _errors:
            return None
        if not settings.DECIDE_ENABLED or name == "off":
            _errors[name] = "disabled"
            return None
        if settings.DECIDE_SRC and settings.DECIDE_SRC not in sys.path:
            if os.path.isdir(settings.DECIDE_SRC):
                sys.path.insert(0, settings.DECIDE_SRC)
            else:
                log.warning("YAMANOTE_DECIDE_SRC=%s is not a directory", settings.DECIDE_SRC)
        try:
            from decide.cache import Cache
            from decide.config import get_backend
            from decide.engine import Engine
            _backends[name] = get_backend(name)
            _engines[name] = Engine(_backends[name], Cache())
            return _engines[name]
        except ImportError:
            _errors[name] = (f"decide isn't installed ({settings.DECIDE_INSTALL}, "
                             "or set YAMANOTE_DECIDE_SRC to a checkout's src directory)")
        except SystemExit as e:  # decide reports a missing key this way
            _errors[name] = str(e)
        except Exception as e:
            _errors[name] = f"{type(e).__name__}: {e}"
        log.warning("decide backend %s unavailable: %s", name, _errors[name])
        return None


def available(backend: str | None = None) -> bool:
    return _load(backend) is not None


def status(backend: str | None = None) -> dict:
    name = backend or settings.DECIDE_BACKEND
    _load(name)
    b = _backends.get(name)
    return {"available": name in _engines, "backend": name, "model": getattr(b, "model", None),
            "error": _errors.get(name)}


def _questions():
    from decide.questions import noul, score  # noqa: deferred until decide is on sys.path
    return noul, score


def ask(documents: list[tuple[str, str]], questions: dict, backend: str | None = None) -> list[Decision] | None:
    """Ask every question about every (label, text) document."""
    engine = _load(backend)
    if engine is None:
        return None
    from decide.engine import Document
    try:
        results = engine.run([Document(label, text) for label, text in documents], questions)
    except Exception as e:
        log.warning("decide call failed: %s", e)
        return None
    price = getattr(engine.backend, "usd_per_mtok", 0.0)
    out = []
    for r in results:
        if r.error or not r.answers:
            log.warning("decide error for %s: %s", r.label, r.error)
            return None
        out.append(Decision(r.answers, r.input_tokens, r.input_tokens * price / 1e6))
    return out


# ─── Factory questions ──────────────────────────────────────────────────────

def assess_spec(spec_text: str, context: str = "", backend: str | None = None) -> Decision | None:
    """Difficulty + triage pre-screen for a proposed work item."""
    if not available(backend):
        return None
    noul, score = _questions()
    qs = {
        "difficulty": score("How much engineering effort would implementing this software change take "
                            "for a competent engineer familiar with the codebase?", settings.DIFFICULTY_LEVELS),
        "useful": noul("Would this change plausibly solve a real problem for users or maintainers of the "
                       "product (not just a neat idea, dashboard, or meta-tooling)?"),
        "ready": noul("Is this specific enough to implement without guessing what is wanted "
                      "(clear scope and a checkable outcome)?"),
        "duplicate": noul("Does the CONTEXT section show this same change (or one that substantially overlaps it) "
                          "already in progress, already built, or rejected recently for reasons that still apply?"),
    }
    text = f"# PROPOSED CHANGE\n{spec_text}\n\n# CONTEXT\n{context or '(none)'}"
    res = ask([("work item", text)], qs, backend)
    return res[0] if res else None


def difficulty_level(decision: Decision) -> str:
    idx = int(round(decision.score("difficulty")))
    return settings.DIFFICULTY_LEVELS[max(0, min(idx, len(settings.DIFFICULTY_LEVELS) - 1))]


HEAD_CHARS = 6000  # per file: imports, definitions and docstrings are enough to judge relevance


def relevant_files(repo_dir: str, question: str, *, max_files: int = 150, threshold: float = 0.35,
                   top: int = 15, backend: str | None = None) -> tuple[list[tuple[str, float]], float] | None:
    """Rank the repo's text files by relevance to a change. Each file is sent
    as its path plus its first HEAD_CHARS characters, which keeps the cost to a
    few cents even on large repos. Returns ([(path, P(relevant))...], cost) or
    None when unavailable."""
    if not available(backend):
        return None
    from decide.files import discover
    noul, _ = _questions()
    try:
        files, _skipped = discover([repo_dir])
    except Exception as e:
        log.warning("decide discover failed: %s", e)
        return None
    files = [f for f in files if ".worktrees" not in f.parts]
    if not files or len(files) > max_files:
        return None
    docs = []
    for f in files:
        with open(f, errors="replace") as fh:
            head = fh.read(HEAD_CHARS + 1)
        if len(head) > HEAD_CHARS:
            head = head[:HEAD_CHARS] + "\n[... file continues]"
        docs.append((os.path.relpath(f, repo_dir), head))
    res = ask(docs, {"relevant": noul(
        "Would an engineer need to read or modify this file to make the following change?\n" + question)}, backend)
    if res is None:
        return None
    ranked = sorted(((label, d.p("relevant")) for (label, _), d in zip(docs, res)),
                    key=lambda x: x[1], reverse=True)
    return [r for r in ranked if r[1] >= threshold][:top], sum(d.cost_usd for d in res)


def log_is_actionable(lines: list[str], open_bugs: list[str], backend: str | None = None) -> Decision | None:
    if not available(backend):
        return None
    noul, _ = _questions()
    text = "# NEW LOG LINES\n" + "\n".join(lines[:80]) + "\n\n# ALREADY-TRACKED BUGS\n" + \
           ("\n".join(f"- {b}" for b in open_bugs) or "(none)")
    res = ask([("log excerpt", text)], {
        "actionable": noul("Do the NEW LOG LINES show an application defect that a code change should fix "
                           "(not transient network noise, expected warnings, or user error)?"),
        "tracked": noul("Is the problem in the NEW LOG LINES already covered by one of the ALREADY-TRACKED BUGS?"),
    }, backend)
    return res[0] if res else None

