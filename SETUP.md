# AI-Assisted Setup

This file is written for an AI coding agent to follow. It configures Yamanote to work
on the user's project.

**Human:** tell your coding agent to "follow SETUP.md".

---

## Step 1: Check prerequisites

```bash
python3 --version   # need 3.11+
git --version
```

Check for an OpenRouter key without printing it:

```bash
grep -l '^OPENROUTER_API_KEY=' .env ~/development/.env 2>/dev/null || echo "missing"
```

If it's missing, ask the user to create one at https://openrouter.ai/keys and add
`OPENROUTER_API_KEY=...` to `.env` themselves. Never echo key values. Recommend they
also set a credit limit on the key in OpenRouter's settings as a backstop to
Yamanote's own budgets.

Jev is optional. It's available if `~/development/decide/src` exists and
`TYPESAFE_API_KEY` is set. Without it, LLMs make its decisions instead.

## Step 2: Choose the project

```bash
find ~/development -maxdepth 2 -name .git -type d 2>/dev/null | sed 's|/.git$||'
```

Ask the user:
1. **Which project(s) should Yamanote work on?** (git repos under the dev dir). For more
   than one, use `projects.json` (copy `projects.json.example`) instead of
   `AGENT_TEAM_DEFAULT_PROJECT`; setup/test commands, gates and schedules then go there
   per project.
2. **Restart a service after merging?** If so, which command (e.g. `sudo systemctl restart my-app.service`)?
3. **Human gates?** Approve specs before building, and/or approve merges? (Recommend the merge gate to start.)
4. **Daily budget in USD?** (default 20)
5. **Dashboard port?** (default 8080). Ask whether it must be reachable from other devices.
   If yes, it needs `AGENT_TEAM_DASHBOARD_HOST=0.0.0.0` plus a token.
6. **How are the project's dependencies installed and tests run?** Look for package.json,
   pyproject.toml, requirements.txt, Makefile, etc. and propose a `setup` and `test`
   command; confirm them with the user. Strongly recommended: the test command gates
   every build and the merge queue.
7. **Notifications?** Ask if they have a command or webhook that should receive alerts
   (gate waits, failures, regressions, suspensions). If yes, set `AGENT_TEAM_NOTIFY_CMD`
   or `AGENT_TEAM_NOTIFY_WEBHOOK` and `AGENT_TEAM_PUBLIC_URL` (see README "Notifications").

## Step 3: Write `.env`

`cp .env.example .env`, then set:

```
AGENT_TEAM_DEV_DIR=~/development
AGENT_TEAM_DEFAULT_PROJECT=my-app        # directory name only
AGENT_TEAM_SERVICE_RESTART_CMD=          # or the command from step 2
AGENT_TEAM_GATE_MERGE=1                  # per the user's answer
AGENT_TEAM_DAILY_BUDGET_USD=20
AGENT_TEAM_DASHBOARD_PORT=8080
AGENT_TEAM_SETUP_CMD=npm ci              # from question 6 (or per project in projects.json)
AGENT_TEAM_TEST_CMD=npm test
```

If the dashboard is exposed, generate a token (`python3 -c "import secrets; print(secrets.token_urlsafe(24))"`),
set `AGENT_TEAM_DASHBOARD_HOST=0.0.0.0` and `AGENT_TEAM_DASHBOARD_TOKEN=<token>`, and
tell the user to keep the token somewhere safe. Don't print it again.

## Step 4: Validate

Run the test command from question 6 once in the project itself and confirm it passes
on the current trunk. A test command that already fails would send every train back
for rework:

```bash
cd ~/development/my-app && npm test; echo "exit $?"   # use the real command; expect exit 0
```

If it fails, tell the user and either fix the command or leave it unset for now.

Then check the configuration:

```bash
python3 -m unittest discover -s tests -t . 2>&1 | tail -1     # expect OK
set -a; source .env; set +a; python3 -c "
from yamanote import settings, decisions
from yamanote.factory import Factory, validate_project
import os
p = os.path.realpath(os.path.join(settings.DEVELOPMENT_DIR, settings.DEFAULT_PROJECT))
print('project:', p, '->', validate_project(p) or 'OK')
print('openrouter key:', 'set' if settings.openrouter_key() else 'MISSING')
print('jev:', decisions.status())
"
```

Fix anything that isn't OK before continuing.

## Step 5: Run

Ask whether to run it now or install it as a service.

- **Now:** `./start.sh` (the dashboard port comes from `.env`)
- **systemd:** copy `agent-team.service`, set `User`, `WorkingDirectory`, `EnvironmentFile`
  and the `ExecStart` path, then `sudo systemctl daemon-reload && sudo systemctl enable --now agent-team`.
  If the restart command uses sudo, add a sudoers rule for exactly that command.

## Step 6: Hand off

Tell the user:
- the dashboard URL (it works on a phone too);
- that "+ New request" adds work by hand (it skips triage), and "Pause line" stops new
  departures;
- that with the merge gate on, trains stop at JY07 with a red signal until they click
  Approve, and that the merge queue re-runs the tests on branch + latest trunk before
  anything lands;
- that every train ends at JY09 Retro, whose notes build up a per-station playbook for
  the project; bad notes can be removed with × on the "Retrospectives & playbook" card;
- that spend is visible in the header bar and capped at the daily budget;
- that the shell tool is not a hard sandbox, so Yamanote should run as an unprivileged
  user.
