# Yamanote

![Yamanote](img/yamanote_banner.png)

A multi-agent orchestrator that coordinates six Claude Code agent personas — **Dispatcher**, **Conductor**, **Inspector**, **Signal**, **Station Manager**, and **Operations** — to autonomously develop and maintain a software project. Agents communicate through a folder-based message bus and follow a structured spec-driven development pipeline.

Built to run unattended on a Raspberry Pi (or any Linux machine) as a systemd service.

## Architecture

The orchestrator runs a tick loop (every 10 seconds by default) that evaluates phases in order:

```
service_recovery  →  rework  →  dispatcher  →  conductor  →  inspector  →  signal  →  entropy_check  →  station_manager_check
```

Each phase decides whether to launch its agent based on the current state of the pipeline. Only one instance of each agent runs at a time.

### Agents

| Agent | Model | Role |
|---|---|---|
| **Dispatcher** | Haiku | Analyzes the codebase and app logs to write feature specs when the backlog is empty |
| **Conductor** | Sonnet | Implements specs on feature branches, one at a time |
| **Inspector** | Haiku | Reviews diffs against `main`. Merges acceptable work or requests changes |
| **Signal** | Haiku | Monitors `app.log` for errors and files bug tickets into the backlog |
| **Station Manager** | Haiku | Resets branches when Conductor gets stuck in edit loops |
| **Operations** | Sonnet | Analyzes orchestrator activity and implements small operational improvements |

### Pipeline flow

```
Dispatcher creates spec
       ↓
Conductor implements on feature branch
       ↓
Inspector reviews diff
      ↙         ↘
  MERGED     CHANGES_REQUESTED
     ↓              ↓
 Service restart   Conductor rework (up to 3 attempts)
                     ↓
                 Re-review
```

### Directory structure

```
agent-team/
├── orchestrator.py       # Main orchestration loop
├── config.py             # All configuration and agent prompts
├── dashboard.py          # Optional web dashboard server
├── dashboard.html        # Dashboard UI (single-page, dark theme)
├── SETUP.md              # AI-agent-friendly setup instructions
├── agent-team.service    # systemd unit file
├── .env.example          # Template for environment variables
├── .gitignore
└── agents/               # Runtime data (gitignored)
    ├── backlog/          # JSON spec files (features and bugs)
    ├── review/           # Inspector feedback files
    ├── logs/             # Stdout/stderr from each agent run
    └── activity.log      # Human-readable event log
```

## Getting started

> **Quick setup with an AI agent:** Open this repo in Claude Code (or any AI coding tool) and say "follow SETUP.md". It will detect your project, write the config, and set up the service for you.

### Prerequisites

- **Python 3.11+**
- **Claude Code CLI** — installed and authenticated (`claude` must be on your PATH). See [Claude Code docs](https://docs.anthropic.com/en/docs/claude-code) for setup.
- **Git** — the target project must be a git repository
- **Linux with systemd** (for running as a service; manual invocation works anywhere)

### 1. Clone the repository

```bash
git clone https://github.com/jeffcampbell/agent-team.git
cd agent-team
```

### 2. Configure your target project

Copy the example environment file and edit it:

```bash
cp .env.example .env
```

Set the environment variables for your project:

```bash
# Path to the parent directory containing your project(s)
AGENT_TEAM_DEV_DIR=~/Development

# Name of the default project directory to manage
AGENT_TEAM_DEFAULT_PROJECT=my-app

# Command to restart your app after a merge (leave empty to skip)
AGENT_TEAM_SERVICE_RESTART_CMD=sudo systemctl restart my-app.service

```

### 3. Run manually

Load the environment and start the orchestrator:

```bash
source .env && python3 orchestrator.py
```

The orchestrator creates `agents/backlog/`, `agents/review/`, and `agents/logs/` on first run. Press `Ctrl+C` to gracefully shut down all agents.

### 4. Run as a systemd service

Copy and adapt the included unit file:

```bash
sudo cp agent-team.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable agent-team
sudo systemctl start agent-team
```

The unit file includes an `EnvironmentFile` directive that loads your `.env` automatically. Edit the `[Service]` section paths to match your setup:

- `WorkingDirectory` — path to the cloned `agent-team` repo
- `ExecStart` — path to `orchestrator.py`
- `EnvironmentFile` — path to your `.env` file
- `User` — the user to run as

If your `AGENT_TEAM_SERVICE_RESTART_CMD` uses `sudo`, ensure the service user has passwordless sudo for that command:

```bash
# /etc/sudoers.d/agent-team
pi ALL=(ALL) NOPASSWD: /usr/bin/systemctl restart your-app.service
```

### 5. Monitor

```bash
# Service status
systemctl status agent-team

# Live activity log
tail -f agents/activity.log

# Agent subprocess logs
ls -lt agents/logs/ | head
```

### 6. Web dashboard (optional)

![Dashboard](img/yamanote_dashboard.png)

A locally-hosted web dashboard gives an at-a-glance view of agent status, pipeline progress, backlog, and recent activity — accessible from any device on the LAN. Disabled by default.

**Enable via CLI flag:**
```bash
python3 orchestrator.py --dashboard              # port 8080
python3 orchestrator.py --dashboard-port 9090    # custom port
```

**Enable via environment variable** (recommended for systemd):
```bash
# Add to .env
AGENT_TEAM_DASHBOARD_PORT=8080
```

Then open `http://<host>:8080/` in a browser. The page auto-refreshes every 10 seconds.

The dashboard shows:
- **Agent cards** — status (running/idle/cooldown), PID, elapsed time, model
- **Pipeline** — current stage (Spec, Engineer, Review, Rework, Merged)
- **Stats** — launches per hour, sleep mode indicator
- **Backlog** — queued specs with priority
- **Activity feed** — color-coded event log
- **Configuration** — collapsible current settings

A JSON API is also available at `GET /api/status` for programmatic access.

## Adding work manually

Drop a JSON spec file into `agents/backlog/`:

```json
{
  "title": "short-kebab-title",
  "description": "What to build, acceptance criteria, and constraints.",
  "priority": "high",
  "created_by": "manual",
  "working_dir": "/path/to/your/project"
}
```

Conductor picks up the highest-priority spec first (`high` > `medium` > `low`), then oldest within the same priority. The Dispatcher also generates specs automatically when the backlog is empty.

## Configuration reference

All settings are in `config.py`. Key settings can be overridden via environment variables (see `.env.example`).

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `AGENT_TEAM_DEV_DIR` | `~/Development` | Parent directory containing your project(s) |
| `AGENT_TEAM_DEFAULT_PROJECT` | `quote-bot` | Default project directory name under `AGENT_TEAM_DEV_DIR` |
| `AGENT_TEAM_SERVICE_RESTART_CMD` | *(empty — skip restart)* | Shell command to restart your app after a merge |
| `AGENT_TEAM_DASHBOARD_PORT` | `0` *(disabled)* | Port for the web dashboard (`0` = off) |

### Timing

| Setting | Default | Description |
|---|---|---|
| `TICK_INTERVAL` | 10s | Seconds between orchestration ticks |
| `AGENT_TIMEOUT_SECONDS` | 600s (10 min) | Max runtime per agent subprocess before termination |
| `SLEEP_MODE_DURATION` | 3600s (1 hr) | How long to sleep when fare limit triggers |

### Agent models

| Setting | Default | Description |
|---|---|---|
| `AGENT_MODELS` | See below | Claude model ID per agent |

```python
AGENT_MODELS = {
    "dispatcher":      "claude-haiku-4-5-20251001",
    "conductor":       "claude-sonnet-4-5-20250929",
    "inspector":       "claude-haiku-4-5-20251001",
    "signal":          "claude-haiku-4-5-20251001",
    "station_manager": "claude-haiku-4-5-20251001",
    "ops":             "claude-sonnet-4-5-20250929",
}
```

Conductor uses Sonnet (the most capable coding model); all other agents use Haiku to minimize cost. Operations also uses Sonnet as it modifies the orchestrator itself.

### Agent throttling

| Setting | Default | Description |
|---|---|---|
| `AGENT_MIN_INTERVALS` | See below | Minimum seconds between consecutive launches of each agent |

```python
AGENT_MIN_INTERVALS = {
    "dispatcher":      900,    # 15 minutes
    "conductor":       0,      # on-demand
    "inspector":       0,      # on-demand
    "signal":          300,    # 5 minutes
    "station_manager": 0,      # on-demand
    "ops":             3600,   # 1 hour
}
```

### Guardrails

| Setting | Default | Description |
|---|---|---|
| `MAX_AGENT_LAUNCHES_PER_HOUR` | 30 | Triggers sleep mode when exceeded |
| `AGENT_ERROR_COOLDOWN` | 120s | Base cooldown after an agent exits non-zero |
| `MAX_ERROR_BACKOFF` | 3600s | Cap for exponential backoff on repeated failures |
| `ENTROPY_FIX_COMMIT_THRESHOLD` | 5 | "fix"/"update" commits on a branch before Conductor is fired and the branch is reset |
| `MAX_ENG_EDITS_BEFORE_RESET` | 3 | File edit cycles before Station Manager resets the branch |
| `MAX_REWORK_ATTEMPTS` | 3 | Inspector change requests before the spec is abandoned |
| `MAX_SPEC_TIMEOUTS` | 2 | Conductor timeouts on a spec before it is dropped |
| `SELF_PROJECT_DIR` | `BASE_DIR` | Prevents agents from modifying the orchestrator itself |

### Git

| Setting | Default | Description |
|---|---|---|
| `TRUNK_BRANCH` | `main` | Branch that Conductor branches from and Inspector merges to |

## How the Dispatcher uses app logs

The Dispatcher agent receives the last 100 lines of the target project's `app.log` (if it exists). It uses this data to:

- Identify which features are used most frequently
- Check whether recently shipped features are seeing adoption
- Spot recurring errors or user friction points
- Prioritize refinements to popular features

When no `app.log` exists, the Dispatcher falls back to codebase-only analysis.

## Safety features

- **Self-protection** — agents cannot create specs targeting the orchestrator's own codebase
- **Fare limit** — enters sleep mode for 1 hour after 30 launches in a rolling hour
- **Error cooldown** — exponential backoff (120s base, 1hr cap) on agent failures
- **Entropy detection** — if a branch accumulates 5+ "fix"/"update" commits, the branch is deleted and the spec re-queued with a fresh start
- **Timeout enforcement** — agents are terminated after 10 minutes; timeouts trigger exponential cooldown and specs are dropped after 2 consecutive timeouts
- **Orphan recovery** — on startup, any `.in_progress` specs from a previous crash are restored to the backlog
- **Working directory validation** — specs must target a directory under `DEVELOPMENT_DIR`

## License

MIT
