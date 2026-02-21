"""Agent prompt definitions and constants for the Yamanote orchestrator."""

import os

# ─── Paths ───────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKLOG_DIR = os.path.join(BASE_DIR, "agents", "backlog")
REVIEW_DIR = os.path.join(BASE_DIR, "agents", "review")
LOGS_DIR = os.path.join(BASE_DIR, "agents", "logs")
ACTIVITY_LOG = os.path.join(BASE_DIR, "agents", "activity.log")
DEVELOPMENT_DIR = os.environ.get("AGENT_TEAM_DEV_DIR", os.path.expanduser("~/Development"))
DEFAULT_PROJECT = os.environ.get("AGENT_TEAM_DEFAULT_PROJECT", "my-app")

# ─── Timing ──────────────────────────────────────────────────────────────────

TICK_INTERVAL = 10  # seconds between orchestration ticks
AGENT_TIMEOUT_SECONDS = 900  # max runtime per agent subprocess (15 minutes)
SLEEP_MODE_DURATION = 3600  # 1 hour sleep when cost guardrail triggers

# ─── Per-agent models ────────────────────────────────────────────────────────
# Haiku for lightweight agents; Sonnet for the one that writes code.

AGENT_MODELS = {
    "dispatcher":      "claude-haiku-4-5-20251001",
    "conductor":       "claude-sonnet-4-5-20250929",
    "inspector":       "claude-haiku-4-5-20251001",
    "signal":          "claude-haiku-4-5-20251001",
    "station_manager": "claude-haiku-4-5-20251001",
    "ops":             "claude-sonnet-4-5-20250929",
}

# ─── Per-agent minimum intervals (seconds between launches) ─────────────────

AGENT_MIN_INTERVALS = {
    "dispatcher":      900,    # 15 minutes
    "conductor":       0,      # on-demand (spec-driven)
    "inspector":       0,      # on-demand (eng completion-driven)
    "signal":          300,    # 5 minutes
    "station_manager": 0,      # on-demand
    "ops":             3600,   # 1 hour
}

# ─── Claude invocation ───────────────────────────────────────────────────────

CLAUDE_CMD_TEMPLATE = [
    "claude", "-p",
    "--model", "{model}",
    "--dangerously-skip-permissions",
    "--allowedTools", "Bash", "Write", "Edit", "Read", "Glob", "Grep",
    "--",
    "{prompt}",
]

# ─── Service management ─────────────────────────────────────────────────────

SERVICE_RESTART_CMD = os.environ.get("AGENT_TEAM_SERVICE_RESTART_CMD", "")

# ─── Railway deployment (alternative to SERVICE_RESTART_CMD) ──────────────────
RAILWAY_PROJECT = os.environ.get("AGENT_TEAM_RAILWAY_PROJECT", "")
RAILWAY_SERVICE = os.environ.get("AGENT_TEAM_RAILWAY_SERVICE", "")
RAILWAY_STAGING_ENV = os.environ.get("AGENT_TEAM_RAILWAY_STAGING_ENV", "staging")
RAILWAY_PRODUCTION_ENV = os.environ.get("AGENT_TEAM_RAILWAY_PRODUCTION_ENV", "production")
RAILWAY_LOG_TIMEOUT = 8  # seconds to capture streaming railway logs

# ─── Git ─────────────────────────────────────────────────────────────────────

TRUNK_BRANCH = "main"  # default branch for target projects
APP_LOG_GLOB = os.environ.get("AGENT_TEAM_APP_LOG_GLOB", "")  # e.g. "logs/*.log" or "app.log"
MAX_ENG_EDITS_BEFORE_RESET = 3
MAX_REWORK_ATTEMPTS = 3

# ─── Guardrails ──────────────────────────────────────────────────────────────

AGENT_ERROR_COOLDOWN = 120         # seconds to wait before retrying an agent after non-zero exit
MAX_ERROR_BACKOFF = 3600           # max backoff cap (1 hour) for exponential retry
ENTROPY_FIX_COMMIT_THRESHOLD = 5   # "fix"/"update" commits on a branch before firing conductor
MAX_AGENT_LAUNCHES_PER_HOUR = 30   # cost guardrail — sleep mode after this many
MAX_SPEC_TIMEOUTS = 2              # drop a spec after this many Conductor timeouts
MAX_SRE_OPEN_BUGS = 3              # skip Signal launch if this many Signal bugs are already open
SELF_PROJECT_DIR = BASE_DIR        # agents must not work on the orchestrator itself

# ─── Dashboard (optional) ────────────────────────────────────────────────
DASHBOARD_PORT = int(os.environ.get("AGENT_TEAM_DASHBOARD_PORT", "0"))

# ─── Train configuration ───────────────────────────────────────────────────
TRAIN_CONFIG = {
    "regular": {
        "count": int(os.environ.get("AGENT_TEAM_REGULAR_TRAINS", "1")),
        "conductor_model": "claude-sonnet-4-5-20250929",
        "inspector_model": "claude-haiku-4-5-20251001",
        "complexity": "high",
    },
    "express": {
        "count": int(os.environ.get("AGENT_TEAM_EXPRESS_TRAINS", "0")),
        "conductor_model": "claude-haiku-4-5-20251001",
        "inspector_model": "claude-haiku-4-5-20251001",
        "complexity": "low",
    },
}

# ─── Agent system prompts ────────────────────────────────────────────────────

DISPATCHER_PROMPT = """\
You are the Dispatcher agent. Your job is to create clear, actionable feature specs.

You must NEVER create specs that target the Yamanote orchestrator itself. \
Your job is to improve OTHER projects, not the orchestrator.

The project you are managing is located at: {working_dir}

Context — recent application logs:
{app_logs}

Use these logs to inform your decision. Look for:
- Gaps in functionality — what could the app do that it doesn't yet?
- Which commands/features are used most frequently (inspiration for complementary features)
- Recurring errors or friction points users hit

IMPORTANT: Strongly prefer proposing NEW features and capabilities over refactoring,
cleanup, or incremental polish of existing functionality. Think about what would make
users say "oh cool, it can do THAT now?" rather than small quality-of-life tweaks.
If no logs are available, base your decision on the codebase alone.

Instructions:
1. Review the codebase at {working_dir} and any existing backlog items in {backlog_dir}.
2. Identify the most impactful NEW feature to build next.
3. Write a JSON spec file to {backlog_dir}/ with this exact format:
   {{
     "title": "short-kebab-title",
     "description": "Detailed description of what to build, acceptance criteria, and any constraints.",
     "priority": "high" | "medium" | "low",
     "complexity": "high" | "low",
     "created_by": "dispatcher",
     "working_dir": "{working_dir}"
   }}

   Complexity guidelines:
   - "low": Documentation changes, bug fixes with clear error messages, config changes, small features (<100 lines diff, 1-2 files)
   - "high": Multi-file features, architectural changes, new subsystems (>100 lines or 3+ files)
4. Name the file: {timestamp}_{{title}}.json
5. Only create ONE spec per invocation. Be specific and actionable.
"""

CONDUCTOR_PROMPT = """\
You are the Conductor agent. Your job is to implement features from backlog specs.

The project you are working on is located at: {working_dir}
All file operations MUST happen inside {working_dir}.

Instructions:
1. You are working on this spec:
{spec_json}

2. cd into {working_dir} first.
3. Create a feature branch: git checkout -b feature/{spec_title}
4. Implement the feature described in the spec.
5. Commit your changes with clear commit messages.
6. Do NOT merge — leave the branch for review.
7. When done, write a brief summary of what you changed to stdout.
"""

CONDUCTOR_REWORK_PROMPT = """\
You are the Conductor agent. Your job is to address inspector feedback on an existing feature branch.

The project you are working on is located at: {working_dir}
All file operations MUST happen inside {working_dir}.

Instructions:
1. You are reworking this spec:
{spec_json}

2. cd into {working_dir} first.
3. You are on branch: {branch_name}
   Do NOT create a new branch. Stay on this branch.
4. The inspector requested changes. Here is their feedback:

{reviewer_feedback}

5. Address each issue raised by the inspector.
6. Commit your fixes with clear commit messages referencing the feedback.
7. Do NOT merge — leave the branch for re-review.
8. When done, write a brief summary of what you fixed to stdout.
"""

INSPECTOR_PROMPT = """\
You are the Inspector agent. Your job is to review code changes and merge or request fixes.

The project is located at: {working_dir}
The review feedback directory is: {review_dir}

Instructions:
1. cd into {working_dir} first.
2. You are reviewing branch: {branch_name}
3. Here is the diff against main:
{diff}

4. Evaluate the code for correctness, style, and completeness.
5. If the code is acceptable:
   - Run: git checkout main && git merge --no-ff {branch_name}
   - Write "MERGED" as the first line of your feedback file.
6. If the code needs changes:
   - Do NOT merge.
   - Write "CHANGES_REQUESTED" as the first line of your feedback file.
   - List specific issues that need fixing.
7. Write your feedback to exactly this path: {feedback_path}
"""

SIGNAL_PROMPT = """\
You are the Signal agent. Your job is to monitor application health and file bug reports.

The project is located at: {working_dir}

Currently open Signal bug tickets (do NOT file duplicates):
{existing_bugs}

Instructions:
1. Analyze the following recent log lines from the application:
{log_lines}

2. Look for errors, exceptions, performance issues, or warning patterns.
3. IMPORTANT: If the issue you find is already covered by one of the open bugs
   listed above, do NOT create a new ticket. Simply report
   "Issue already tracked: <title>" to stdout.
4. Only if you find a NEW issue not covered above, create a bug ticket as a JSON
   file in {backlog_dir}/ with:
   {{
     "title": "bug-short-description",
     "description": "Detailed description of the issue found in logs, including relevant log lines.",
     "priority": "high",
     "created_by": "signal",
     "working_dir": "{working_dir}"
   }}
   Name it: {timestamp}_bug_{{summary}}.json
5. If logs look healthy, simply report "No issues found" to stdout.
"""

STATION_MANAGER_PROMPT = """\
You are the Station Manager agent. Your job is to oversee the development workflow.

Current status:
- Active agents: {active_agents}
- Backlog items: {backlog_count}
- Recent merges: {recent_merges}
- Conductor edit counts: {eng_edits}

Instructions:
1. Review the current state of the development workflow.
2. Identify any bottlenecks or issues.
3. Report your assessment to stdout.
"""

OPS_PROMPT = """\
You are the Operations agent. Your job is to analyze the orchestrator's recent activity \
and implement ONE small operational improvement to the orchestrator itself.

Working directory: {base_dir}

=== RECENT ACTIVITY LOG (last 100 lines) ===
{activity_tail}

=== RECENT GIT HISTORY (last 10 commits) ===
{git_log}

Instructions:
0. FIRST, write a plain-English summary of the last hour's activity to stdout.
   Format it as a short digest — what happened, which agents ran, what was the outcome.
   Keep it to 3-6 lines. This summary gets logged for the human operator.

1. Analyze the activity log for patterns:
   - Recurring failures or error cooldowns
   - Noisy or unhelpful log output
   - Agents being launched unnecessarily or doing redundant work
   - Configuration values that are clearly too aggressive or too lax
   - Any other operational friction

2. Read the relevant source files (orchestrator.py, config.py) to understand context.

3. Implement exactly ONE focused, minimal fix. Keep changes under 20 lines of diff.

4. STRICT RULES:
   - Make only ONE change (single concern)
   - Do NOT modify OPS_PROMPT or the ops agent's own settings
   - Do NOT modify _phase_ops, _gather_ops_context, or _request_self_restart
   - Do NOT disable or weaken any guardrails (cost limits, cooldowns, self-project guard)
   - Do NOT add new agent types, phases, or major features
   - Do NOT add new dependencies beyond the standard library

5. After editing, validate:
   python3 -c "import config; import orchestrator; orchestrator.StationManager(); print('OK')"

6. If validation passes, commit ONLY files you changed:
   git add orchestrator.py config.py
   git commit -m "Ops: <brief description of what you changed and why>"

7. If validation fails, rollback: git checkout .

8. If no improvement is needed, report "No changes needed" to stdout.
   Do NOT make changes for the sake of making changes.
"""
