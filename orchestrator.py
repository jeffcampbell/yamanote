#!/usr/bin/env python3
"""Yamanote — multi-agent orchestrator for Claude Code agent personas."""

import glob
import json
import logging
import os
import re
import subprocess
import sys
import time
from collections import deque

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("orchestrator")


# ─── Activity log ────────────────────────────────────────────────────────────

def activity(msg: str):
    """Append a pretty-printed line to the activity log and also log it."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}]  {msg}\n"
    log.info(msg)
    try:
        with open(config.ACTIVITY_LOG, "a") as f:
            f.write(line)
    except OSError:
        pass


class AgentProcess:
    """Thin wrapper around a single Claude subprocess."""

    def __init__(self, name: str, prompt: str, cwd: str | None = None, model: str | None = None):
        self.name = name
        self.prompt = prompt
        self.cwd = cwd
        self.model = model or config.AGENT_MODELS.get(name, "claude-sonnet-4-5-20250929")
        self.proc: subprocess.Popen | None = None
        self.start_time: float | None = None
        self._output: str | None = None
        self._stderr: str | None = None

    def start(self) -> subprocess.Popen:
        cmd = [arg.format(prompt=self.prompt, model=self.model)
               for arg in config.CLAUDE_CMD_TEMPLATE]
        log.info("Starting agent %s (cwd=%s)", self.name, self.cwd or "default")
        # Strip CLAUDECODE env var so nested claude sessions are allowed
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            cwd=self.cwd,
        )
        self.start_time = time.time()
        log.info("Agent %s started with PID %d", self.name, self.proc.pid)
        return self.proc

    def poll(self) -> bool:
        """Return True if the process has finished."""
        if self.proc is None:
            return True
        return self.proc.poll() is not None

    def is_timed_out(self) -> bool:
        if self.proc is None or self.start_time is None:
            return False
        return time.time() - self.start_time > config.AGENT_TIMEOUT_SECONDS

    def get_output(self) -> str:
        """Read stdout after completion. Blocks if still running."""
        if self._output is not None:
            return self._output
        if self.proc is None:
            return ""
        stdout, stderr = self.proc.communicate()
        self._output = stdout
        self._stderr = stderr
        return self._output

    def get_stderr(self) -> str:
        if self._stderr is not None:
            return self._stderr
        if self.proc is None:
            return ""
        self.get_output()
        return self._stderr or ""

    def save_log(self, marker: str = ""):
        ts = time.strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(config.LOGS_DIR, f"{self.name}_{ts}.log")
        with open(log_path, "w") as f:
            if marker:
                f.write(f"{marker}\n")
            f.write(f"=== Agent: {self.name} ===\n")
            f.write(f"=== CWD: {self.cwd or 'default'} ===\n")
            f.write(f"=== Started: {time.ctime(self.start_time)} ===\n")
            f.write(f"=== Return code: {self.proc.returncode if self.proc else 'N/A'} ===\n\n")
            f.write("--- STDOUT ---\n")
            f.write(self.get_output())
            f.write("\n--- STDERR ---\n")
            f.write(self.get_stderr())
        log.info("Saved log for %s to %s", self.name, log_path)
        return log_path


class StationManager:
    """Main orchestration loop managing 6 agent personas."""

    def __init__(self):
        # Ensure folder structure exists
        for d in (config.BACKLOG_DIR, config.REVIEW_DIR, config.LOGS_DIR):
            os.makedirs(d, exist_ok=True)

        self.active_agents: dict[str, AgentProcess | None] = {
            "dispatcher": None,
            "conductor": None,
            "inspector": None,
            "signal": None,
            "station_manager": None,
            "ops": None,
        }
        self.conductor_file_edits: dict[str, int] = {}
        self.last_merge_commit: str | None = None
        self.current_conductor_spec: str | None = None
        self.current_conductor_branch: str | None = None
        self.current_working_dir: str | None = None
        self.rework_counts: dict[str, int] = {}
        self.spec_timeout_counts: dict[str, int] = {}  # spec path → timeout streak
        self._dispatcher_skip_logged_branch: str | None = None

        # Cost guardrail: track agent launches in a rolling window
        self.launch_times: deque[float] = deque()
        self.sleep_until: float = 0.0

        # Error cooldown: don't retry agents immediately after failures
        self.agent_cooldowns: dict[str, float] = {}  # agent name → earliest retry time
        self.consecutive_failures: dict[str, int] = {}  # agent name → failure streak
        self.last_launch_times: dict[str, float] = {}  # agent name → last launch timestamp
        self._conductor_edits_tallied: bool = False  # True once edits counted for current Conductor run

        # Signal high-water mark: only analyze new log lines since last run
        self.sre_log_offsets: dict[str, int] = {}  # project_dir → byte offset in app.log
        self._sre_prev_offsets: dict[str, int] = {}  # offset before last Signal read (for rollback on failure)

        # Ops agent: track HEAD before ops launch to detect new commits
        self._ops_head_before: str | None = None

        # Uptime tracking (used by dashboard)
        self.start_time: float = time.time()

        # Don't run Dispatcher or Ops immediately on startup — wait for activity to accumulate
        self.last_launch_times["dispatcher"] = time.time()
        self.last_launch_times["ops"] = time.time()

        # Recover orphaned .in_progress specs from previous runs
        self._recover_orphaned_specs()

    # ─── Helpers ─────────────────────────────────────────────────────────

    def _feedback_path(self, branch: str) -> str:
        """Return the path to an inspector feedback file for the given branch.

        Tries the canonical name first, then falls back to any *_feedback.md
        in the review dir (only one branch is in review at a time).
        """
        canonical = os.path.join(
            config.REVIEW_DIR,
            f"{branch.replace('/', '_')}_feedback.md",
        )
        if os.path.exists(canonical):
            return canonical
        matches = glob.glob(os.path.join(config.REVIEW_DIR, "*_feedback.md"))
        if len(matches) == 1:
            return matches[0]
        return canonical  # fall back to canonical (may not exist)

    def _recover_orphaned_specs(self):
        """On startup, rename any .in_progress specs back to .json so they re-enter the pipeline."""
        pattern = os.path.join(config.BACKLOG_DIR, "*.json.in_progress")
        orphaned = glob.glob(pattern)
        for path in orphaned:
            original = path.removesuffix(".in_progress")
            os.rename(path, original)
            activity(f"RECOVERED orphaned spec: {os.path.basename(original)}")

    def _backlog_specs(self) -> list[str]:
        """Return backlog specs sorted by priority (high first), then oldest first."""
        PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}
        specs = sorted(glob.glob(os.path.join(config.BACKLOG_DIR, "*.json")))
        def sort_key(path: str) -> tuple[int, str]:
            try:
                with open(path) as f:
                    data = json.load(f)
                return (PRIORITY_ORDER.get(data.get("priority", "medium"), 1), path)
            except (json.JSONDecodeError, OSError):
                return (1, path)
        return sorted(specs, key=sort_key)

    def _signal_open_bugs(self) -> list[dict]:
        """Return spec dicts for open Signal-authored bugs (both .json and .json.in_progress)."""
        bugs = []
        for pattern in ("*.json", "*.json.in_progress"):
            for path in glob.glob(os.path.join(config.BACKLOG_DIR, pattern)):
                try:
                    with open(path) as f:
                        data = json.load(f)
                    if data.get("created_by") in ("sre", "signal"):
                        bugs.append(data)
                except (json.JSONDecodeError, OSError):
                    continue
        return bugs

    def _is_agent_active(self, name: str) -> bool:
        agent = self.active_agents.get(name)
        if agent is None:
            return False
        if agent.poll():
            agent.save_log()
            rc = agent.proc.returncode if agent.proc else "?"
            summary = agent.get_output()[:200].replace("\n", " ").strip()
            activity(f"ARRIVED  [{agent.name}] rc={rc} — {summary or '(no output)'}")
            self.active_agents[name] = None
            # Set cooldown on non-zero exit with exponential backoff
            if rc != 0:
                # Detect API rate-limit responses and enter sleep mode to avoid
                # hammering a quota wall with repeated retries (seen as "out of extra usage")
                agent_output = agent.get_output() + agent.get_stderr()
                if "out of extra usage" in agent_output or "rate limit" in agent_output.lower():
                    self.sleep_until = time.time() + config.SLEEP_MODE_DURATION
                    activity(
                        f"RATE LIMIT [{name}] — API quota exhausted, "
                        f"entering SERVICE SUSPENDED for {config.SLEEP_MODE_DURATION}s"
                    )
                    return False
                self.consecutive_failures[name] = self.consecutive_failures.get(name, 0) + 1
                backoff = min(
                    config.AGENT_ERROR_COOLDOWN * (2 ** self.consecutive_failures[name]),
                    config.MAX_ERROR_BACKOFF,
                )
                cooldown_until = time.time() + backoff
                self.agent_cooldowns[name] = cooldown_until
                activity(f"DELAY [{name}] — failure #{self.consecutive_failures[name]}, retry after {backoff}s")
                # Signal failed — roll back log offsets so the same lines are retried next run
                if name == "signal":
                    self.sre_log_offsets.update(self._sre_prev_offsets)
            else:
                self.consecutive_failures.pop(name, None)
            self._sre_prev_offsets.clear()
            return False
        if agent.is_timed_out():
            self._kill_timed_out_agent(name, agent)
            return False
        return True

    def _kill_timed_out_agent(self, name: str, agent: AgentProcess):
        elapsed = time.time() - (agent.start_time or 0)
        activity(f"OVERDUE [{name}] after {elapsed:.0f}s — terminating")
        if agent.proc and agent.proc.poll() is None:
            agent.proc.terminate()
            try:
                agent.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                agent.proc.kill()
                agent.proc.wait()
        agent.save_log(marker="[OVERDUE]")
        self.active_agents[name] = None

        # Treat timeouts as failures for cooldown purposes
        self.consecutive_failures[name] = self.consecutive_failures.get(name, 0) + 1
        backoff = min(
            config.AGENT_ERROR_COOLDOWN * (2 ** self.consecutive_failures[name]),
            config.MAX_ERROR_BACKOFF,
        )
        self.agent_cooldowns[name] = time.time() + backoff
        activity(f"DELAY [{name}] — overdue #{self.consecutive_failures[name]}, retry after {backoff}s")

        # Signal timed out — roll back log offsets so those lines are retried next run
        if name == "signal":
            self.sre_log_offsets.update(self._sre_prev_offsets)
        self._sre_prev_offsets.clear()

        if name == "conductor" and self.current_conductor_spec:
            spec_path = self.current_conductor_spec
            self.spec_timeout_counts[spec_path] = self.spec_timeout_counts.get(spec_path, 0) + 1
            timeouts = self.spec_timeout_counts[spec_path]

            if timeouts >= config.MAX_SPEC_TIMEOUTS:
                activity(
                    f"TERMINATED spec after {timeouts} overdue: {os.path.basename(spec_path)}"
                )
                # Reset conductor failure counter so the next spec doesn't inherit this
                # spec's timeout backoff — each spec deserves a fresh start.
                self.consecutive_failures.pop("conductor", None)
                self.agent_cooldowns.pop("conductor", None)
                in_progress = spec_path + ".in_progress"
                if os.path.exists(in_progress):
                    os.remove(in_progress)
                # Clean up the branch if it exists
                cwd = self.current_working_dir
                branch = self.current_conductor_branch
                if cwd and branch and self._git_has_branch(branch, cwd=cwd):
                    self._git("checkout", config.TRUNK_BRANCH, cwd=cwd)
                    self._git("branch", "-D", branch, cwd=cwd)
                # Delete any stale inspector feedback for this branch
                if branch:
                    fb = self._feedback_path(branch)
                    if os.path.exists(fb):
                        os.remove(fb)
                del self.spec_timeout_counts[spec_path]
            else:
                in_progress = spec_path + ".in_progress"
                if os.path.exists(in_progress):
                    os.rename(in_progress, spec_path)
                    activity(f"RE-ROUTED spec after Conductor overdue ({timeouts}/{config.MAX_SPEC_TIMEOUTS}): {os.path.basename(spec_path)}")

            self.current_conductor_branch = None
            self.current_conductor_spec = None
            self.current_working_dir = None
            self.conductor_file_edits.clear()

    def _launch_agent(self, name: str, prompt: str, cwd: str | None = None) -> AgentProcess | None:
        # Error cooldown check
        now = time.time()
        if name in self.agent_cooldowns and now < self.agent_cooldowns[name]:
            remaining = int(self.agent_cooldowns[name] - now)
            log.info("Agent %s in cooldown (%ds remaining), skipping launch", name, remaining)
            return None

        # Clear expired cooldown
        self.agent_cooldowns.pop(name, None)

        # Minimum interval throttle
        min_interval = config.AGENT_MIN_INTERVALS.get(name, 0)
        if min_interval > 0:
            last_launch = self.last_launch_times.get(name, 0)
            elapsed = now - last_launch
            if elapsed < min_interval:
                remaining = int(min_interval - elapsed)
                log.info("Agent %s throttled (%ds until next allowed launch)", name, remaining)
                return None

        # Cost guardrail check
        self.launch_times.append(now)
        # Prune launches older than 1 hour
        while self.launch_times and self.launch_times[0] < now - 3600:
            self.launch_times.popleft()

        if len(self.launch_times) > config.MAX_AGENT_LAUNCHES_PER_HOUR:
            self.sleep_until = now + config.SLEEP_MODE_DURATION
            activity(
                f"FARE LIMIT — {len(self.launch_times)} launches in the last hour "
                f"(limit {config.MAX_AGENT_LAUNCHES_PER_HOUR}). "
                f"Entering SERVICE SUSPENDED until {time.ctime(self.sleep_until)}"
            )
            self.launch_times.clear()
            return None  # type: ignore[return-value]

        model = config.AGENT_MODELS.get(name, "claude-sonnet-4-5-20250929")
        agent = AgentProcess(name, prompt, cwd=cwd, model=model)
        agent.start()
        self.active_agents[name] = agent
        self.last_launch_times[name] = now
        activity(f"DEPARTED [{name}] PID {agent.proc.pid} model={model} cwd={cwd or 'default'}")
        return agent

    def _git(self, *args: str, cwd: str | None = None) -> str:
        """Run a git command and return stdout."""
        result = subprocess.run(
            ["git"] + list(args),
            capture_output=True, text=True,
            cwd=cwd or config.BASE_DIR,
        )
        return result.stdout.strip()

    def _git_has_branch(self, branch: str, cwd: str | None = None) -> bool:
        result = self._git("branch", "--list", branch, cwd=cwd)
        return bool(result.strip())

    def _git_diff_trunk(self, branch: str, cwd: str | None = None) -> str:
        return self._git("diff", f"{config.TRUNK_BRANCH}..{branch}", cwd=cwd)

    def _git_last_commit(self, cwd: str | None = None) -> str:
        return self._git("rev-parse", "HEAD", cwd=cwd)

    def _find_app_log(self, project_dir: str) -> str | None:
        """Resolve the project's application log file.

        Priority: AGENT_TEAM_APP_LOG_GLOB env var → common log file names.
        Returns the most-recently-modified match, or None.
        """
        patterns = []
        if config.APP_LOG_GLOB:
            patterns.append(config.APP_LOG_GLOB)
        # Fallback: common conventions
        patterns += ["logs/*.log", "*.log"]
        for pattern in patterns:
            matches = sorted(
                glob.glob(os.path.join(project_dir, pattern)),
                key=lambda p: os.path.getmtime(p),
                reverse=True,
            )
            if matches:
                return matches[0]
        return None

    def _fetch_railway_logs(self, environment: str) -> str:
        """Fetch recent logs from Railway via CLI. Streams for RAILWAY_LOG_TIMEOUT seconds."""
        project_dir = os.path.join(config.DEVELOPMENT_DIR, config.DEFAULT_PROJECT)
        cmd = [
            "railway", "logs",
            "-e", environment,
            "-s", config.RAILWAY_SERVICE,
        ]
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                cwd=project_dir,
                start_new_session=True,  # own process group so we can kill the tree
            )
            stdout, _ = proc.communicate(timeout=config.RAILWAY_LOG_TIMEOUT)
            return stdout
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, 9)  # SIGKILL the entire process group
            stdout, _ = proc.communicate()
            return stdout
        except (OSError, FileNotFoundError) as e:
            log.warning("Railway CLI failed: %s", e)
            return ""

    def _read_app_log_tail(self, project_dir: str, lines: int = 100) -> str:
        if config.RAILWAY_PROJECT:
            output = self._fetch_railway_logs(config.RAILWAY_PRODUCTION_ENV)
            if output:
                return "\n".join(output.splitlines()[-lines:])
            return ""

        log_path = self._find_app_log(project_dir)
        if not log_path:
            return ""
        result = subprocess.run(
            ["tail", "-n", str(lines), log_path],
            capture_output=True, text=True,
        )
        return result.stdout

    def _read_new_log_lines(self, project_dir: str) -> str:
        """Read only log lines written since the last Signal run (high-water mark)."""
        if config.RAILWAY_PROJECT:
            return self._read_new_railway_logs(project_dir)

        log_path = self._find_app_log(project_dir)
        if not log_path:
            return ""

        if project_dir not in self.sre_log_offsets:
            # First run (or after restart): set high-water mark to current EOF.
            # Don't re-analyze logs that were already seen before the restart.
            try:
                self.sre_log_offsets[project_dir] = os.path.getsize(log_path)
            except OSError:
                pass
            return ""

        stored_offset = self.sre_log_offsets[project_dir]
        try:
            file_size = os.path.getsize(log_path)
        except OSError:
            return ""

        # Log rotation: file shrank below stored offset → reset to start
        if file_size < stored_offset:
            stored_offset = 0

        if file_size == stored_offset:
            return ""  # No new content

        with open(log_path, "r") as f:
            f.seek(stored_offset)
            new_content = f.read()

        self._sre_prev_offsets[project_dir] = stored_offset
        self.sre_log_offsets[project_dir] = file_size
        return new_content

    def _read_new_railway_logs(self, project_dir: str) -> str:
        """Fetch Railway production logs and return only lines not seen before."""
        key = "_railway_"
        output = self._fetch_railway_logs(config.RAILWAY_PRODUCTION_ENV)
        if not output:
            return ""

        lines = output.splitlines()
        if not lines:
            return ""

        if key not in self.sre_log_offsets:
            # First run: set high-water mark, return empty (same semantics as local mode)
            self._sre_prev_offsets[key] = None
            self.sre_log_offsets[key] = lines[-1]
            return ""

        last_seen = self.sre_log_offsets[key]
        # Find where the last-seen line is in the new output
        try:
            idx = lines.index(last_seen)
            new_lines = lines[idx + 1:]
        except ValueError:
            # Last-seen line not found (log rotated or too much new output) — return all
            new_lines = lines

        if not new_lines:
            return ""

        self._sre_prev_offsets[key] = last_seen
        self.sre_log_offsets[key] = new_lines[-1]
        return "\n".join(new_lines)

    def _gather_ops_context(self) -> tuple[str, str]:
        """Collect diagnostic data for the ops agent."""
        activity_tail = ""
        if os.path.exists(config.ACTIVITY_LOG):
            result = subprocess.run(
                ["tail", "-n", "100", config.ACTIVITY_LOG],
                capture_output=True, text=True,
            )
            activity_tail = result.stdout
        git_log = self._git("log", "--oneline", "-10", cwd=config.BASE_DIR)
        return activity_tail, git_log

    def _request_self_restart(self):
        """Gracefully terminate all agents and exit for systemd to restart."""
        activity("OPS RESTART — new commits detected, restarting orchestrator...")
        for name, agent in self.active_agents.items():
            if agent and agent.proc and agent.proc.poll() is None:
                activity(f"Terminating {name} (PID {agent.proc.pid})")
                agent.proc.terminate()
                try:
                    agent.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    agent.proc.kill()
                agent.save_log()
        activity("All agents stopped. Exiting for restart.")
        sys.exit(0)

    def _is_self_project(self, working_dir: str | None) -> bool:
        """Return True if the spec targets the Yamanote orchestrator itself."""
        if not working_dir:
            return False
        return os.path.realpath(working_dir) == os.path.realpath(config.SELF_PROJECT_DIR)

    # ─── Entropy check ───────────────────────────────────────────────────

    def _count_fix_commits_on_branch(self, branch: str, cwd: str | None = None) -> int:
        """Count commits on branch (not on trunk) whose message contains 'fix' or 'update'."""
        log_output = self._git(
            "log", f"{config.TRUNK_BRANCH}..{branch}",
            "--oneline", cwd=cwd,
        )
        if not log_output:
            return 0
        count = 0
        for line in log_output.splitlines():
            lower = line.lower()
            if "fix" in lower or "update" in lower:
                count += 1
        return count

    def _fire_conductor_entropy(self, branch: str, cwd: str | None = None):
        """Fire the Conductor agent — nuke the branch and re-queue the spec."""
        activity(
            f"DERAILED [conductor] — branch {branch} has too many fix/update commits. "
            f"Clearing branch and re-queuing spec."
        )
        # Kill conductor if still running
        conductor = self.active_agents.get("conductor")
        if conductor and conductor.proc and conductor.proc.poll() is None:
            conductor.proc.terminate()
            try:
                conductor.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                conductor.proc.kill()
                conductor.proc.wait()
            conductor.save_log(marker="[DERAILED — ENTROPY]")
        self.active_agents["conductor"] = None

        # Delete any stale inspector feedback for this branch
        fb = self._feedback_path(branch)
        if os.path.exists(fb):
            os.remove(fb)

        # Reset branch
        self._git("checkout", config.TRUNK_BRANCH, cwd=cwd)
        self._git("branch", "-D", branch, cwd=cwd)

        # Re-queue spec
        if self.current_conductor_spec:
            in_progress = self.current_conductor_spec + ".in_progress"
            if os.path.exists(in_progress):
                os.rename(in_progress, self.current_conductor_spec)
                activity(f"RE-ROUTED spec: {os.path.basename(self.current_conductor_spec)}")

        self.conductor_file_edits.clear()
        self.current_conductor_branch = None
        self.current_conductor_spec = None
        self.current_working_dir = None

    # ─── Phases ──────────────────────────────────────────────────────────

    def _phase_dispatcher(self):
        """If backlog is empty and no Dispatcher running, launch Dispatcher agent."""
        if self._is_agent_active("dispatcher"):
            return
        # Don't log intent if Dispatcher is in cooldown — _launch_agent would silently skip
        if "dispatcher" in self.agent_cooldowns and time.time() < self.agent_cooldowns["dispatcher"]:
            return
        if self._backlog_specs():
            return

        # Default to configured project if no specific project context
        default_dir = os.path.join(config.DEVELOPMENT_DIR, config.DEFAULT_PROJECT)
        if not os.path.isdir(default_dir):
            default_dir = config.DEVELOPMENT_DIR

        # Don't generate new specs while conductor→inspector→merge pipeline is active
        if self.current_conductor_branch:
            if self._dispatcher_skip_logged_branch != self.current_conductor_branch:
                activity(f"Dispatcher — skipped, conductor pipeline active on {self.current_conductor_branch}")
                self._dispatcher_skip_logged_branch = self.current_conductor_branch
            return

        ts = time.strftime("%Y%m%d_%H%M%S")
        app_logs = self._read_app_log_tail(default_dir) or "(no app.log found)"
        prompt = config.DISPATCHER_PROMPT.format(
            timestamp=ts,
            working_dir=default_dir,
            backlog_dir=config.BACKLOG_DIR,
            app_logs=app_logs,
        )
        agent = self._launch_agent("dispatcher", prompt, cwd=default_dir)
        if agent is not None:
            activity(f"Dispatcher — backlog empty, generating spec for {default_dir}")

    def _phase_conductor(self):
        """If backlog has specs and no Conductor running, pick oldest spec and launch Conductor."""
        if self._is_agent_active("conductor"):
            return
        # Don't log intent if Conductor is in cooldown — _launch_agent would silently skip
        if "conductor" in self.agent_cooldowns and time.time() < self.agent_cooldowns["conductor"]:
            return

        # Track file edits once after Conductor finishes (not every tick)
        if not self._conductor_edits_tallied and self.active_agents.get("conductor") is None and self.current_conductor_branch:
            cwd = self.current_working_dir
            if self._git_has_branch(self.current_conductor_branch, cwd=cwd):
                diff_stat = self._git(
                    "diff", "--name-only",
                    f"{config.TRUNK_BRANCH}..{self.current_conductor_branch}",
                    cwd=cwd,
                )
                for fname in diff_stat.splitlines():
                    fname = fname.strip()
                    if fname:
                        self.conductor_file_edits[fname] = self.conductor_file_edits.get(fname, 0) + 1
            self._conductor_edits_tallied = True

        # Don't pick up a new spec while a branch is still in the review pipeline
        if self.current_conductor_branch:
            return

        specs = self._backlog_specs()
        if not specs:
            return

        spec_path = specs[0]
        try:
            with open(spec_path) as f:
                spec_data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            log.warning("Bad spec file %s: %s", spec_path, e)
            return

        # Read working_dir from spec, default to configured project
        working_dir = spec_data.get("working_dir")
        if not working_dir:
            working_dir = os.path.join(config.DEVELOPMENT_DIR, config.DEFAULT_PROJECT)

        # "Don't reinvent ourselves" guardrail
        if self._is_self_project(working_dir):
            activity(f"RESTRICTED spec {os.path.basename(spec_path)} — targets Yamanote itself. Removing.")
            os.remove(spec_path)
            return

        # Validate working_dir exists and is under /home/pi/Development
        if not os.path.isdir(working_dir):
            activity(f"RESTRICTED spec {os.path.basename(spec_path)} — working_dir {working_dir} does not exist. Removing.")
            os.remove(spec_path)
            return
        if not os.path.realpath(working_dir).startswith(os.path.realpath(config.DEVELOPMENT_DIR)):
            activity(f"RESTRICTED spec {os.path.basename(spec_path)} — working_dir outside {config.DEVELOPMENT_DIR}. Removing.")
            os.remove(spec_path)
            return

        spec_title = spec_data.get("title", "untitled")
        branch_name = f"feature/{spec_title}"
        self.conductor_file_edits.clear()  # Reset edit counts for the new spec
        self.current_conductor_spec = spec_path
        self.current_conductor_branch = branch_name
        self.current_working_dir = working_dir

        # If the feature branch already exists with changes (e.g. orphaned spec from
        # a prior run that actually completed), skip Conductor and go straight to inspector.
        if self._git_has_branch(branch_name, cwd=working_dir) and self._git_diff_trunk(branch_name, cwd=working_dir):
            activity(f"Conductor — branch {branch_name} already has changes, routing to inspector (orphan recovery)")
            os.rename(spec_path, spec_path + ".in_progress")
            return

        spec_desc = spec_data.get("description", "")
        activity(f"Conductor — starting spec '{spec_title}' in {working_dir}")
        spec_summary = spec_desc.split("\n")[0][:120].strip()
        activity(f"  SPEC: {spec_summary}")
        prompt = config.CONDUCTOR_PROMPT.format(
            spec_json=json.dumps(spec_data, indent=2),
            spec_title=spec_title,
            working_dir=working_dir,
        )
        agent = self._launch_agent("conductor", prompt, cwd=working_dir)
        if agent is None:
            # Launch was blocked (cooldown or cost guardrail) — don't move the spec
            self.current_conductor_spec = None
            self.current_conductor_branch = None
            self.current_working_dir = None
            return
        self._conductor_edits_tallied = False
        os.rename(spec_path, spec_path + ".in_progress")

    def _phase_inspector(self):
        """If Conductor finished and branch exists with changes, launch Inspector."""
        if self._is_agent_active("inspector"):
            return
        if self._is_agent_active("conductor"):
            return
        # Don't log intent if Inspector is in cooldown — _launch_agent would silently skip
        if "inspector" in self.agent_cooldowns and time.time() < self.agent_cooldowns["inspector"]:
            return

        branch = self.current_conductor_branch
        cwd = self.current_working_dir
        if not branch or not cwd:
            return
        if not self._git_has_branch(branch, cwd=cwd):
            return

        diff = self._git_diff_trunk(branch, cwd=cwd)
        if not diff:
            activity(f"Inspector — no diff on branch {branch}, cleaning up empty branch")
            self._git("checkout", config.TRUNK_BRANCH, cwd=cwd)
            self._git("branch", "-D", branch, cwd=cwd)
            if self.current_conductor_spec:
                in_progress = self.current_conductor_spec + ".in_progress"
                if os.path.exists(in_progress):
                    os.remove(in_progress)
            self.current_conductor_branch = None
            self.current_conductor_spec = None
            self.current_working_dir = None
            return

        feedback_path = os.path.join(
            config.REVIEW_DIR,
            f"{branch.replace('/', '_')}_feedback.md",
        )  # canonical path passed to Inspector prompt
        activity(f"Inspector — reviewing branch {branch} in {cwd}")
        prompt = config.INSPECTOR_PROMPT.format(
            branch_name=branch,
            diff=diff[:8000],
            working_dir=cwd,
            review_dir=config.REVIEW_DIR,
            feedback_path=feedback_path,
        )
        self._launch_agent("inspector", prompt, cwd=cwd)

    def _phase_rework(self):
        """If inspector requested changes and both Conductor and Inspector are idle, re-launch Conductor."""
        if self._is_agent_active("conductor") or self._is_agent_active("inspector"):
            return

        branch = self.current_conductor_branch
        spec_path = self.current_conductor_spec
        cwd = self.current_working_dir
        if not branch or not spec_path:
            return

        feedback_path = self._feedback_path(branch)
        if not os.path.exists(feedback_path):
            return

        try:
            with open(feedback_path) as f:
                first_line = f.readline().strip()
                if first_line != "CHANGES_REQUESTED":
                    return
                reviewer_feedback = first_line + "\n" + f.read()
        except OSError:
            return

        rework_key = spec_path
        self.rework_counts[rework_key] = self.rework_counts.get(rework_key, 0) + 1
        if self.rework_counts[rework_key] > config.MAX_REWORK_ATTEMPTS:
            activity(f"CANCELLED spec after {config.MAX_REWORK_ATTEMPTS} rework attempts — branch {branch}")
            self._git("checkout", config.TRUNK_BRANCH, cwd=cwd)
            self._git("branch", "-D", branch, cwd=cwd)
            in_progress = spec_path + ".in_progress"
            if os.path.exists(in_progress):
                os.remove(in_progress)
            # Clean up the feedback file so it doesn't accumulate
            if os.path.exists(feedback_path):
                os.remove(feedback_path)
            self.conductor_file_edits.clear()
            self.current_conductor_branch = None
            self.current_conductor_spec = None
            self.current_working_dir = None
            del self.rework_counts[rework_key]
            return

        in_progress_path = spec_path + ".in_progress"
        spec_read_path = in_progress_path if os.path.exists(in_progress_path) else spec_path
        try:
            with open(spec_read_path) as f:
                spec_data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            log.warning("Cannot read spec for rework %s: %s", spec_read_path, e)
            return

        activity(
            f"RETURN [{self.rework_counts[rework_key]}/{config.MAX_REWORK_ATTEMPTS}] "
            f"— Conductor re-addressing feedback on {branch}"
        )
        prompt = config.CONDUCTOR_REWORK_PROMPT.format(
            spec_json=json.dumps(spec_data, indent=2),
            spec_title=spec_data.get("title", "untitled"),
            branch_name=branch,
            reviewer_feedback=reviewer_feedback,
            working_dir=cwd,
        )
        self._launch_agent("conductor", prompt, cwd=cwd)
        self._conductor_edits_tallied = False
        os.remove(feedback_path)

    def _phase_signal(self):
        """If a log file exists in the current project, launch Signal to analyze."""
        if self._is_agent_active("signal"):
            return

        # Check the current working project, or default to configured project
        project_dir = self.current_working_dir
        if not project_dir:
            project_dir = os.path.join(config.DEVELOPMENT_DIR, config.DEFAULT_PROJECT)

        # Layer 1: hard cap on open Signal bugs
        open_bugs = self._signal_open_bugs()
        if len(open_bugs) >= config.MAX_SRE_OPEN_BUGS:
            log.debug(
                "Signal skipped — %d open Signal bugs (cap %d)",
                len(open_bugs), config.MAX_SRE_OPEN_BUGS,
            )
            return

        log_lines = self._read_new_log_lines(project_dir)
        if not log_lines:
            return

        # Layer 2: pass existing bug titles into prompt for LLM dedup
        if open_bugs:
            existing_bugs_text = "\n".join(
                f"- {bug.get('title', '(untitled)')}" for bug in open_bugs
            )
        else:
            existing_bugs_text = "(none)"

        ts = time.strftime("%Y%m%d_%H%M%S")
        log.debug("Signal launching — analyzing logs in %s (%d open Signal bugs)", project_dir, len(open_bugs))
        prompt = config.SIGNAL_PROMPT.format(
            log_lines=log_lines,
            timestamp=ts,
            working_dir=project_dir,
            backlog_dir=config.BACKLOG_DIR,
            existing_bugs=existing_bugs_text,
        )
        self._launch_agent("signal", prompt, cwd=project_dir)

    def _phase_entropy_check(self):
        """If branch has too many fix/update commits, fire Conductor and restart."""
        branch = self.current_conductor_branch
        cwd = self.current_working_dir
        if not branch or not cwd:
            return
        if not self._git_has_branch(branch, cwd=cwd):
            return

        fix_count = self._count_fix_commits_on_branch(branch, cwd=cwd)
        if fix_count >= config.ENTROPY_FIX_COMMIT_THRESHOLD:
            self._fire_conductor_entropy(branch, cwd=cwd)

    def _phase_station_manager_check(self):
        """If Conductor edited same files >= 3 times without merge, reset branch and re-queue."""
        if self._is_agent_active("inspector"):
            return
        branch = self.current_conductor_branch
        cwd = self.current_working_dir
        if not branch:
            return

        max_edits = max(self.conductor_file_edits.values()) if self.conductor_file_edits else 0
        if max_edits < config.MAX_ENG_EDITS_BEFORE_RESET:
            return

        activity(f"SIGNAL CHANGE — {max_edits} edits without merge on {branch}")

        if self.current_conductor_spec and os.path.exists(self.current_conductor_spec + ".in_progress"):
            os.rename(self.current_conductor_spec + ".in_progress", self.current_conductor_spec)
            activity(f"RE-ROUTED spec: {os.path.basename(self.current_conductor_spec)}")

        # Delete any stale inspector feedback for this branch
        fb = self._feedback_path(branch)
        if os.path.exists(fb):
            os.remove(fb)

        self._git("checkout", config.TRUNK_BRANCH, cwd=cwd)
        self._git("branch", "-D", branch, cwd=cwd)

        self.conductor_file_edits.clear()
        self.current_conductor_branch = None
        self.current_conductor_spec = None
        self.current_working_dir = None

    def _deploy_to_railway(self):
        """Deploy to Railway via git push: staging branch first, then main if healthy."""
        cwd = self.current_working_dir
        crash_indicators = ("Traceback", "FATAL", "ModuleNotFoundError", "SyntaxError", "ImportError", "panic:")

        # 1. Push to staging branch → triggers Railway staging deploy
        activity(f"RAILWAY pushing to staging branch...")
        result = subprocess.run(
            ["git", "push", "origin", f"{config.TRUNK_BRANCH}:staging"],
            capture_output=True, text=True, timeout=60, cwd=cwd,
        )
        if result.returncode != 0:
            activity(f"RAILWAY staging push failed (rc={result.returncode}): {result.stderr[:200]}")
            return

        # 2. Wait for Railway to build and start the service
        activity("RAILWAY staging pushed, waiting 60s for build + startup...")
        time.sleep(60)

        # 3. Check staging health
        staging_logs = self._fetch_railway_logs(config.RAILWAY_STAGING_ENV)
        unhealthy = [ind for ind in crash_indicators if ind in staging_logs]
        if unhealthy:
            activity(f"RAILWAY staging UNHEALTHY — found: {', '.join(unhealthy)}. Skipping production deploy.")
            return

        # 4. Push to main branch → triggers Railway production deploy
        activity(f"RAILWAY staging healthy, pushing to main branch...")
        result = subprocess.run(
            ["git", "push", "origin", config.TRUNK_BRANCH],
            capture_output=True, text=True, timeout=60, cwd=cwd,
        )
        if result.returncode != 0:
            activity(f"RAILWAY production push failed (rc={result.returncode}): {result.stderr[:200]}")
            return

        activity("RAILWAY production deploy triggered")

    def _phase_service_recovery(self):
        """If Inspector merged to trunk, restart the service."""
        if self._is_agent_active("inspector"):
            return
        if not self.current_conductor_branch:
            return

        cwd = self.current_working_dir
        feedback_path = self._feedback_path(self.current_conductor_branch)
        if not os.path.exists(feedback_path):
            return

        try:
            with open(feedback_path) as f:
                first_line = f.readline().strip()
        except OSError:
            return

        if first_line != "MERGED":
            return

        current_head = self._git_last_commit(cwd=cwd)
        if current_head == self.last_merge_commit:
            return

        activity(f"TERMINUS — branch {self.current_conductor_branch} merged to trunk.")
        self.last_merge_commit = current_head

        # Delete the feature branch now that it's merged
        if self._git_has_branch(self.current_conductor_branch, cwd=cwd):
            self._git("checkout", config.TRUNK_BRANCH, cwd=cwd)
            self._git("branch", "-D", self.current_conductor_branch, cwd=cwd)

        if config.RAILWAY_PROJECT:
            self._deploy_to_railway()
        elif config.SERVICE_RESTART_CMD:
            rc = os.system(config.SERVICE_RESTART_CMD)
            if rc == 0:
                activity("SERVICE restarted successfully")
            else:
                activity(f"SERVICE restart failed (rc={rc})")
        else:
            activity("SERVICE restart skipped (no deployment method configured)")

        if self.current_conductor_spec and os.path.exists(self.current_conductor_spec + ".in_progress"):
            os.remove(self.current_conductor_spec + ".in_progress")

        # Clean up the feedback file so it doesn't accumulate and can't falsely
        # re-trigger service recovery if the same branch name is reused later.
        if os.path.exists(feedback_path):
            os.remove(feedback_path)

        self.conductor_file_edits.clear()
        self.current_conductor_branch = None
        self.current_conductor_spec = None
        self.current_working_dir = None

    def _log_ops_summary(self, output: str):
        """Extract and log the ops agent's activity summary with visual breakers."""
        if not output or not output.strip():
            return
        # Use the full output as the summary — ops is instructed to lead with it
        lines = output.strip().splitlines()
        # Cap at 15 lines to keep the log readable
        summary_lines = lines[:15]
        activity("*" * 60)
        activity("OPS REPORT — last hour")
        for line in summary_lines:
            activity(f"  {line}")
        activity("*" * 60)

    def _phase_ops(self):
        """Periodically analyze orchestrator activity and implement small improvements."""
        ops_agent = self.active_agents.get("ops")
        if self._is_agent_active("ops"):
            return

        # If ops just completed, log summary and check for new commits → restart
        if self._ops_head_before is not None:
            if ops_agent is not None:
                self._log_ops_summary(ops_agent.get_output())
            current_head = self._git_last_commit(cwd=config.BASE_DIR)
            if current_head != self._ops_head_before:
                self._ops_head_before = None
                self._request_self_restart()
            self._ops_head_before = None
            return

        # Skip if in cooldown
        if "ops" in self.agent_cooldowns and time.time() < self.agent_cooldowns["ops"]:
            return

        activity_tail, git_log = self._gather_ops_context()
        prompt = config.OPS_PROMPT.format(
            base_dir=config.BASE_DIR,
            activity_tail=activity_tail or "(no activity log)",
            git_log=git_log or "(no commits)",
        )
        agent = self._launch_agent("ops", prompt, cwd=config.BASE_DIR)
        if agent is not None:
            self._ops_head_before = self._git_last_commit(cwd=config.BASE_DIR)

    # ─── Main loop ───────────────────────────────────────────────────────

    def run(self):
        activity("=" * 60)
        activity("YAMANOTE LINE OPEN")
        activity(f"  Tick interval: {config.TICK_INTERVAL}s")
        activity(f"  Backlog: {config.BACKLOG_DIR}")
        activity(f"  Activity log: {config.ACTIVITY_LOG}")
        activity(f"  Agent timeout: {config.AGENT_TIMEOUT_SECONDS}s")
        activity(f"  Fare limit: {config.MAX_AGENT_LAUNCHES_PER_HOUR} launches/hr")
        activity(f"  Entropy threshold: {config.ENTROPY_FIX_COMMIT_THRESHOLD} fix commits")
        for agent_name, model in config.AGENT_MODELS.items():
            interval = config.AGENT_MIN_INTERVALS.get(agent_name, 0)
            activity(f"  {agent_name}: model={model}  min_interval={interval}s")
        activity("=" * 60)

        try:
            while True:
                # Sleep mode check
                if time.time() < self.sleep_until:
                    remaining = int(self.sleep_until - time.time())
                    if remaining % 300 < config.TICK_INTERVAL:  # log every ~5 min
                        activity(f"SERVICE SUSPENDED — {remaining}s remaining (fare limit)")
                    time.sleep(config.TICK_INTERVAL)
                    continue

                self._phase_service_recovery()
                self._phase_rework()
                self._phase_dispatcher()
                self._phase_conductor()
                self._phase_inspector()
                self._phase_signal()
                self._phase_entropy_check()
                self._phase_station_manager_check()
                self._phase_ops()

                # Tick summary
                active = [n for n, a in self.active_agents.items() if a is not None]
                specs = self._backlog_specs()
                if active or specs:
                    log.info(
                        "Tick: active=[%s] backlog=%d",
                        ", ".join(active) if active else "none",
                        len(specs),
                    )

                time.sleep(config.TICK_INTERVAL)
        except KeyboardInterrupt:
            activity("LAST TRAIN — terminating active agents...")
            for name, agent in self.active_agents.items():
                if agent and agent.proc and agent.proc.poll() is None:
                    activity(f"Terminating {name} (PID {agent.proc.pid})")
                    agent.proc.terminate()
                    try:
                        agent.proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        agent.proc.kill()
                    agent.save_log()
            activity("All agents stopped. Goodbye.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Yamanote — multi-agent orchestrator")
    parser.add_argument("--dashboard", action="store_true",
                        help="Enable web dashboard on port 8080")
    parser.add_argument("--dashboard-port", type=int, default=0, metavar="PORT",
                        help="Enable web dashboard on a specific port")
    args = parser.parse_args()

    # Priority: --dashboard-port > --dashboard (8080) > env var > disabled
    dash_port = args.dashboard_port or (8080 if args.dashboard else config.DASHBOARD_PORT)

    station_manager = StationManager()

    if dash_port:
        from dashboard import start_dashboard
        start_dashboard(station_manager, dash_port)

    station_manager.run()
