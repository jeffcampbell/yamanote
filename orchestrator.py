#!/usr/bin/env python3
"""Multi-agent orchestrator for Claude Code agent personas."""

import glob
import json
import logging
import os
import re
import subprocess
import sys
import time
import urllib.request
import urllib.error
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


class Supervisor:
    """Main orchestration loop managing 5 agent personas."""

    def __init__(self):
        # Ensure folder structure exists
        for d in (config.BACKLOG_DIR, config.REVIEW_DIR, config.LOGS_DIR):
            os.makedirs(d, exist_ok=True)

        self.active_agents: dict[str, AgentProcess | None] = {
            "pm": None,
            "eng": None,
            "reviewer": None,
            "sre": None,
            "supervisor": None,
        }
        self.eng_file_edits: dict[str, int] = {}
        self.last_merge_commit: str | None = None
        self.current_eng_spec: str | None = None
        self.current_eng_branch: str | None = None
        self.current_working_dir: str | None = None
        self.rework_counts: dict[str, int] = {}

        # Cost guardrail: track agent launches in a rolling window
        self.launch_times: deque[float] = deque()
        self.sleep_until: float = 0.0

        # Error cooldown: don't retry agents immediately after failures
        self.agent_cooldowns: dict[str, float] = {}  # agent name → earliest retry time
        self.consecutive_failures: dict[str, int] = {}  # agent name → failure streak
        self.last_launch_times: dict[str, float] = {}  # agent name → last launch timestamp
        self._eng_edits_tallied: bool = False  # True once edits counted for current Eng run

        # SRE high-water mark: only analyze new log lines since last run
        self.sre_log_offsets: dict[str, int] = {}  # project_dir → byte offset in app.log

        # Recover orphaned .in_progress specs from previous runs
        self._recover_orphaned_specs()

    # ─── Helpers ─────────────────────────────────────────────────────────

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

    def _sre_open_bugs(self) -> list[dict]:
        """Return spec dicts for open SRE-authored bugs (both .json and .json.in_progress)."""
        bugs = []
        for pattern in ("*.json", "*.json.in_progress"):
            for path in glob.glob(os.path.join(config.BACKLOG_DIR, pattern)):
                try:
                    with open(path) as f:
                        data = json.load(f)
                    if data.get("created_by") == "sre":
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
            activity(f"DONE  [{agent.name}] rc={rc} — {summary or '(no output)'}")
            self.active_agents[name] = None
            # Set cooldown on non-zero exit with exponential backoff
            if rc != 0:
                self.consecutive_failures[name] = self.consecutive_failures.get(name, 0) + 1
                backoff = min(
                    config.AGENT_ERROR_COOLDOWN * (2 ** self.consecutive_failures[name]),
                    config.MAX_ERROR_BACKOFF,
                )
                cooldown_until = time.time() + backoff
                self.agent_cooldowns[name] = cooldown_until
                activity(f"COOLDOWN [{name}] — failure #{self.consecutive_failures[name]}, retry after {backoff}s")
            else:
                self.consecutive_failures.pop(name, None)
            return False
        if agent.is_timed_out():
            self._kill_timed_out_agent(name, agent)
            return False
        return True

    def _kill_timed_out_agent(self, name: str, agent: AgentProcess):
        elapsed = time.time() - (agent.start_time or 0)
        activity(f"TIMEOUT [{name}] after {elapsed:.0f}s — terminating")
        if agent.proc and agent.proc.poll() is None:
            agent.proc.terminate()
            try:
                agent.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                agent.proc.kill()
                agent.proc.wait()
        agent.save_log(marker="[TIMEOUT]")
        self.active_agents[name] = None

        if name == "eng" and self.current_eng_spec:
            in_progress = self.current_eng_spec + ".in_progress"
            if os.path.exists(in_progress):
                os.rename(in_progress, self.current_eng_spec)
                activity(f"RE-QUEUED spec after Eng timeout: {os.path.basename(self.current_eng_spec)}")
            self.current_eng_branch = None
            self.current_eng_spec = None
            self.current_working_dir = None

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
                f"COST GUARDRAIL — {len(self.launch_times)} launches in the last hour "
                f"(limit {config.MAX_AGENT_LAUNCHES_PER_HOUR}). "
                f"Entering SLEEP MODE until {time.ctime(self.sleep_until)}"
            )
            self.launch_times.clear()
            return None  # type: ignore[return-value]

        model = config.AGENT_MODELS.get(name, "claude-sonnet-4-5-20250929")
        agent = AgentProcess(name, prompt, cwd=cwd, model=model)
        agent.start()
        self.active_agents[name] = agent
        self.last_launch_times[name] = now
        activity(f"START [{name}] PID {agent.proc.pid} model={model} cwd={cwd or 'default'}")
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

    def _read_app_log_tail(self, project_dir: str, lines: int = 100) -> str:
        log_path = os.path.join(project_dir, "app.log")
        if not os.path.exists(log_path):
            return ""
        result = subprocess.run(
            ["tail", "-n", str(lines), log_path],
            capture_output=True, text=True,
        )
        return result.stdout

    def _read_new_log_lines(self, project_dir: str) -> str:
        """Read only log lines written since the last SRE run (high-water mark)."""
        log_path = os.path.join(project_dir, "app.log")
        if not os.path.exists(log_path):
            return ""

        if project_dir not in self.sre_log_offsets:
            # First run: fall back to tail, then store current EOF offset
            tail = self._read_app_log_tail(project_dir, 100)
            try:
                self.sre_log_offsets[project_dir] = os.path.getsize(log_path)
            except OSError:
                pass
            return tail

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

        self.sre_log_offsets[project_dir] = file_size
        return new_content

    def _is_self_project(self, working_dir: str | None) -> bool:
        """Return True if the spec targets the agent-team orchestrator itself."""
        if not working_dir:
            return False
        return os.path.realpath(working_dir) == os.path.realpath(config.SELF_PROJECT_DIR)

    # ─── Discord notifications ───────────────────────────────────────────

    def _notify_discord(self, spec_data: dict, branch: str):
        """Post a merge notification to Discord. Fails silently."""
        url = config.DISCORD_WEBHOOK_URL
        if not url:
            return
        try:
            title = spec_data.get("title", "unknown")
            description = spec_data.get("description", "")
            payload = json.dumps({
                "embeds": [{
                    "title": f"Merged: {title}",
                    "description": description[:2048],
                    "color": 0x57F287,  # green
                    "fields": [{"name": "Branch", "value": branch, "inline": True}],
                }]
            }).encode()
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=10)
            log.info("Discord notification sent for %s", title)
        except Exception as exc:
            log.warning("Discord notification failed: %s", exc)

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

    def _fire_eng_entropy(self, branch: str, cwd: str | None = None):
        """Fire the Eng agent — nuke the branch and re-queue the spec."""
        activity(
            f"ENTROPY FIRED ENG — branch {branch} has too many fix/update commits. "
            f"Clearing branch and re-queuing spec."
        )
        # Kill eng if still running
        eng = self.active_agents.get("eng")
        if eng and eng.proc and eng.proc.poll() is None:
            eng.proc.terminate()
            try:
                eng.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                eng.proc.kill()
                eng.proc.wait()
            eng.save_log(marker="[FIRED — ENTROPY]")
        self.active_agents["eng"] = None

        # Reset branch
        self._git("checkout", config.TRUNK_BRANCH, cwd=cwd)
        self._git("branch", "-D", branch, cwd=cwd)

        # Re-queue spec
        if self.current_eng_spec:
            in_progress = self.current_eng_spec + ".in_progress"
            if os.path.exists(in_progress):
                os.rename(in_progress, self.current_eng_spec)
                activity(f"RE-QUEUED spec: {os.path.basename(self.current_eng_spec)}")

        self.eng_file_edits.clear()
        self.current_eng_branch = None
        self.current_eng_spec = None
        self.current_working_dir = None

    # ─── Phases ──────────────────────────────────────────────────────────

    def _has_unmerged_feature_branches(self, cwd: str) -> bool:
        """Return True if any feature/* branches exist in the given repo."""
        result = self._git("branch", "--list", "feature/*", cwd=cwd)
        return bool(result.strip())

    def _phase_pm(self):
        """If backlog is empty and no PM running, launch PM agent."""
        if self._is_agent_active("pm"):
            return
        # Don't log intent if PM is in cooldown — _launch_agent would silently skip
        if "pm" in self.agent_cooldowns and time.time() < self.agent_cooldowns["pm"]:
            return
        if self._backlog_specs():
            return

        # Default to configured project if no specific project context
        default_dir = os.path.join(config.DEVELOPMENT_DIR, config.DEFAULT_PROJECT)
        if not os.path.isdir(default_dir):
            default_dir = config.DEVELOPMENT_DIR

        # Don't generate new specs while feature branches are still in flight
        if self._has_unmerged_feature_branches(default_dir):
            return

        ts = time.strftime("%Y%m%d_%H%M%S")
        app_logs = self._read_app_log_tail(default_dir) or "(no app.log found)"
        activity(f"PM — backlog empty, generating spec for {default_dir}")
        prompt = config.PM_PROMPT.format(
            timestamp=ts,
            working_dir=default_dir,
            backlog_dir=config.BACKLOG_DIR,
            app_logs=app_logs,
        )
        self._launch_agent("pm", prompt, cwd=default_dir)

    def _phase_eng(self):
        """If backlog has specs and no Eng running, pick oldest spec and launch Eng."""
        if self._is_agent_active("eng"):
            return
        # Don't log intent if Eng is in cooldown — _launch_agent would silently skip
        if "eng" in self.agent_cooldowns and time.time() < self.agent_cooldowns["eng"]:
            return

        # Track file edits once after Eng finishes (not every tick)
        if not self._eng_edits_tallied and self.active_agents.get("eng") is None and self.current_eng_branch:
            cwd = self.current_working_dir
            if self._git_has_branch(self.current_eng_branch, cwd=cwd):
                diff_stat = self._git(
                    "diff", "--name-only",
                    f"{config.TRUNK_BRANCH}..{self.current_eng_branch}",
                    cwd=cwd,
                )
                for fname in diff_stat.splitlines():
                    fname = fname.strip()
                    if fname:
                        self.eng_file_edits[fname] = self.eng_file_edits.get(fname, 0) + 1
            self._eng_edits_tallied = True

        # Don't pick up a new spec while a branch is still in the review pipeline
        if self.current_eng_branch:
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
            activity(f"BLOCKED spec {os.path.basename(spec_path)} — targets agent-team itself. Removing.")
            os.remove(spec_path)
            return

        # Validate working_dir exists and is under /home/pi/Development
        if not os.path.isdir(working_dir):
            activity(f"BLOCKED spec {os.path.basename(spec_path)} — working_dir {working_dir} does not exist. Removing.")
            os.remove(spec_path)
            return
        if not os.path.realpath(working_dir).startswith(os.path.realpath(config.DEVELOPMENT_DIR)):
            activity(f"BLOCKED spec {os.path.basename(spec_path)} — working_dir outside {config.DEVELOPMENT_DIR}. Removing.")
            os.remove(spec_path)
            return

        spec_title = spec_data.get("title", "untitled")
        branch_name = f"feature/{spec_title}"
        self.eng_file_edits.clear()  # Reset edit counts for the new spec
        self.current_eng_spec = spec_path
        self.current_eng_branch = branch_name
        self.current_working_dir = working_dir

        spec_desc = spec_data.get("description", "")
        activity(f"ENG — starting spec '{spec_title}' in {working_dir}")
        activity(f"  SPEC: {spec_desc}")
        prompt = config.ENG_PROMPT.format(
            spec_json=json.dumps(spec_data, indent=2),
            spec_title=spec_title,
            working_dir=working_dir,
        )
        agent = self._launch_agent("eng", prompt, cwd=working_dir)
        if agent is None:
            # Launch was blocked (cooldown or cost guardrail) — don't move the spec
            self.current_eng_spec = None
            self.current_eng_branch = None
            self.current_working_dir = None
            return
        self._eng_edits_tallied = False
        os.rename(spec_path, spec_path + ".in_progress")

    def _phase_reviewer(self):
        """If Eng finished and branch exists with changes, launch Reviewer."""
        if self._is_agent_active("reviewer"):
            return
        if self._is_agent_active("eng"):
            return

        branch = self.current_eng_branch
        cwd = self.current_working_dir
        if not branch or not cwd:
            return
        if not self._git_has_branch(branch, cwd=cwd):
            return

        diff = self._git_diff_trunk(branch, cwd=cwd)
        if not diff:
            log.info("No diff on branch %s — skipping review", branch)
            return

        feedback_path = os.path.join(
            config.REVIEW_DIR,
            f"{branch.replace('/', '_')}_feedback.md",
        )
        activity(f"REVIEWER — reviewing branch {branch} in {cwd}")
        prompt = config.REVIEWER_PROMPT.format(
            branch_name=branch,
            diff=diff[:8000],
            working_dir=cwd,
            review_dir=config.REVIEW_DIR,
            feedback_path=feedback_path,
        )
        self._launch_agent("reviewer", prompt, cwd=cwd)

    def _phase_rework(self):
        """If reviewer requested changes and both Eng and Reviewer are idle, re-launch Eng."""
        if self._is_agent_active("eng") or self._is_agent_active("reviewer"):
            return

        branch = self.current_eng_branch
        spec_path = self.current_eng_spec
        cwd = self.current_working_dir
        if not branch or not spec_path:
            return

        feedback_path = os.path.join(
            config.REVIEW_DIR,
            f"{branch.replace('/', '_')}_feedback.md",
        )
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
            activity(f"ABANDONED spec after {config.MAX_REWORK_ATTEMPTS} rework attempts — branch {branch}")
            self._git("checkout", config.TRUNK_BRANCH, cwd=cwd)
            self._git("branch", "-D", branch, cwd=cwd)
            in_progress = spec_path + ".in_progress"
            if os.path.exists(in_progress):
                os.remove(in_progress)
            self.eng_file_edits.clear()
            self.current_eng_branch = None
            self.current_eng_spec = None
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
            f"REWORK [{self.rework_counts[rework_key]}/{config.MAX_REWORK_ATTEMPTS}] "
            f"— Eng re-addressing feedback on {branch}"
        )
        prompt = config.ENG_REWORK_PROMPT.format(
            spec_json=json.dumps(spec_data, indent=2),
            spec_title=spec_data.get("title", "untitled"),
            branch_name=branch,
            reviewer_feedback=reviewer_feedback,
            working_dir=cwd,
        )
        self._launch_agent("eng", prompt, cwd=cwd)
        self._eng_edits_tallied = False
        os.rename(feedback_path, feedback_path + ".addressed")

    def _phase_sre(self):
        """If app.log exists in the current project, launch SRE to analyze."""
        if self._is_agent_active("sre"):
            return

        # Check the current working project, or default to configured project
        project_dir = self.current_working_dir
        if not project_dir:
            project_dir = os.path.join(config.DEVELOPMENT_DIR, config.DEFAULT_PROJECT)

        # Layer 1: hard cap on open SRE bugs
        open_bugs = self._sre_open_bugs()
        if len(open_bugs) >= config.MAX_SRE_OPEN_BUGS:
            log.debug(
                "SRE skipped — %d open SRE bugs (cap %d)",
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
        log.debug("SRE launching — analyzing app.log in %s (%d open SRE bugs)", project_dir, len(open_bugs))
        prompt = config.SRE_PROMPT.format(
            log_lines=log_lines,
            timestamp=ts,
            working_dir=project_dir,
            backlog_dir=config.BACKLOG_DIR,
            existing_bugs=existing_bugs_text,
        )
        self._launch_agent("sre", prompt, cwd=project_dir)

    def _phase_entropy_check(self):
        """If branch has too many fix/update commits, fire Eng and restart."""
        branch = self.current_eng_branch
        cwd = self.current_working_dir
        if not branch or not cwd:
            return
        if not self._git_has_branch(branch, cwd=cwd):
            return

        fix_count = self._count_fix_commits_on_branch(branch, cwd=cwd)
        if fix_count >= config.ENTROPY_FIX_COMMIT_THRESHOLD:
            self._fire_eng_entropy(branch, cwd=cwd)

    def _phase_supervisor_check(self):
        """If Eng edited same files >= 3 times without merge, reset branch and re-queue."""
        if self._is_agent_active("reviewer"):
            return
        branch = self.current_eng_branch
        cwd = self.current_working_dir
        if not branch:
            return

        max_edits = max(self.eng_file_edits.values()) if self.eng_file_edits else 0
        if max_edits < config.MAX_ENG_EDITS_BEFORE_RESET:
            return

        activity(f"SUPERVISOR RESET — {max_edits} edits without merge on {branch}")

        if self.current_eng_spec and os.path.exists(self.current_eng_spec + ".in_progress"):
            os.rename(self.current_eng_spec + ".in_progress", self.current_eng_spec)
            activity(f"RE-QUEUED spec: {os.path.basename(self.current_eng_spec)}")

        self._git("checkout", config.TRUNK_BRANCH, cwd=cwd)
        self._git("branch", "-D", branch, cwd=cwd)

        self.eng_file_edits.clear()
        self.current_eng_branch = None
        self.current_eng_spec = None
        self.current_working_dir = None

    def _phase_service_recovery(self):
        """If Reviewer merged to trunk, restart the service."""
        if self._is_agent_active("reviewer"):
            return
        if not self.current_eng_branch:
            return

        cwd = self.current_working_dir
        feedback_path = os.path.join(
            config.REVIEW_DIR,
            f"{self.current_eng_branch.replace('/', '_')}_feedback.md",
        )
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

        activity(f"MERGED — branch {self.current_eng_branch} merged to trunk.")
        self.last_merge_commit = current_head

        # Notify Discord before cleaning up state
        if self.current_eng_spec:
            spec_read = self.current_eng_spec + ".in_progress"
            if not os.path.exists(spec_read):
                spec_read = self.current_eng_spec
            try:
                with open(spec_read) as f:
                    spec_data = json.load(f)
                self._notify_discord(spec_data, self.current_eng_branch)
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("Could not read spec for Discord notification: %s", exc)

        if not config.SERVICE_RESTART_CMD:
            activity("SERVICE restart skipped (no SERVICE_RESTART_CMD configured)")
        else:
            rc = os.system(config.SERVICE_RESTART_CMD)
            if rc == 0:
                activity("SERVICE restarted successfully")
            else:
                activity(f"SERVICE restart failed (rc={rc})")

        if self.current_eng_spec and os.path.exists(self.current_eng_spec + ".in_progress"):
            os.remove(self.current_eng_spec + ".in_progress")

        self.eng_file_edits.clear()
        self.current_eng_branch = None
        self.current_eng_spec = None
        self.current_working_dir = None

    # ─── Main loop ───────────────────────────────────────────────────────

    def run(self):
        activity("=" * 60)
        activity("ORCHESTRATOR STARTING")
        activity(f"  Tick interval: {config.TICK_INTERVAL}s")
        activity(f"  Backlog: {config.BACKLOG_DIR}")
        activity(f"  Activity log: {config.ACTIVITY_LOG}")
        activity(f"  Agent timeout: {config.AGENT_TIMEOUT_SECONDS}s")
        activity(f"  Cost limit: {config.MAX_AGENT_LAUNCHES_PER_HOUR} launches/hr")
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
                        activity(f"SLEEPING — {remaining}s remaining (cost guardrail)")
                    time.sleep(config.TICK_INTERVAL)
                    continue

                self._phase_service_recovery()
                self._phase_rework()
                self._phase_pm()
                self._phase_eng()
                self._phase_reviewer()
                self._phase_sre()
                self._phase_entropy_check()
                self._phase_supervisor_check()

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
            activity("SHUTTING DOWN — terminating active agents...")
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
    supervisor = Supervisor()
    supervisor.run()
