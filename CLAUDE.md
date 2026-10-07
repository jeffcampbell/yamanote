# CLAUDE.md — yamanote

Yamanote is a software factory: work items ("trains") travel stations JY01–JY09
(intake → triage → spec → build → inspect → verify → merge → deploy → retro), with
OpenRouter agents at each station and Jev (via `~/development/decide`) for cheap
decisions. User-facing docs: `README.md`; guided install for agents: `SETUP.md`.

## Layout (`yamanote/`)

| Module | Role |
|---|---|
| `factory.py` | Tick loop; one `_station_<name>` method per station; retrospectives and playbooks; Signal, Dispatcher, Ops |
| `agent.py` | Tool-calling loop over OpenRouter; sandboxed file tools; `run` tool (temp-file output, process-group cleanup) |
| `llm.py` | OpenRouter client: exact cost from `usage.cost`, fallbacks, Anthropic cache breakpoints |
| `stats.py` | Stats view numbers, computed from the store per request |
| `telemetry.py` | OpenTelemetry export: train traces (GenAI conventions) + metrics over OTLP/HTTP JSON |
| `cron.py` | Five-field cron parser (`prev`/`next`) for the autopilot schedule |
| `store.py` | SQLite: items, events, runs, steps, retros, kv. `Store.transition()` is compare-and-set — use it for status changes from jobs |
| `prompts.py` | Station system prompts + JSON result schemas |
| `decisions.py` | Jev questions (difficulty, triage screen, file relevance, log screen); every call returns None when decide is unavailable |
| `checks.py`, `gitops.py`, `notify.py`, `dashboard.py`, `settings.py`, `web/` | Deterministic checks, git, hooks, API/SSE, config, UI |

## Conventions

- Stdlib only (Python 3.11+); the UI is vanilla JS/CSS with no build step or CDN.
- Status changes inside station jobs go through `_move`, `_wait_at_gate`, `_end_journey`
  (arrived/failed → JY09 Retro) or `_finish` (terminal) so cancels and the SLA reaper
  can't be overwritten. Raise `ItemGone` when a transition loses the race.
- Every model call is recorded as a run with steps; spend must flow through `Store.add_step`.
- Ask `factory.gates(project)` which human gates apply — never read `settings` gates
  directly — so Autopilot, the Settings panel and projects.json all stay in charge.
- Never weaken the guardrails in README "Guardrails" without the user asking.
- UI changes: check phone (390px), tablet (768px) and laptop (1280px) widths; headless
  Chrome can't go below 500px, so load the page in an iframe of the target width. Use
  `--timeout` ≥ 25000 and a fresh `--user-data-dir`: a short timeout cancels in-flight
  fetches and captures a half-loaded page that looks like an app bug.
- Exported OpenTelemetry counters must never decrease: derive them from append-only
  data (events, finished runs, kv merge stats), not from current item/run status.
- Tests finish journeys in milliseconds; anything that compares timestamps needs a test
  with a stretched timeline (see `test_every_run_appears_when_the_journey_takes_real_time`).
- Charts follow the dataviz rules (validated palette, thin marks, hover + table view,
  one axis); check new class names don't collide with existing ones (`.stats-grid` is
  the drawer's tile grid).

## Tests

```bash
python3 -m unittest discover -s tests -t .
```

Offline: `tests/helpers.py` has a `FakeClient` that routes each model call to a
per-role script by matching words in the station's system prompt (`role_of`). A new
station or single-shot role needs an entry there and a default script. Factory tests run
against real temporary git repos; `run_until(factory, predicate)` ticks until done.

To try changes for real, run against a scratch project (see README "Getting started")
with a small `AGENT_TEAM_DAILY_BUDGET_USD`.

## Papercuts log (global, all projects)

`~/development/papercuts.md` is a shared log of anything that has slowed down
development — broken tooling, misleading errors, environment quirks.

- **Check it first when tooling fails mysteriously.** The fix may already be recorded.
- **Append to it when you lose time to a papercut mid-session**, one line, newest at the bottom:
  `- YYYY-MM-DD · symptom · fix · project`
- The `project` field is the repo/directory you were working in, or `global` if
  the papercut wasn't project-specific.
- Keep these same instructions in every `CLAUDE.md` under `~/development`.
