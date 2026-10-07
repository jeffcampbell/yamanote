"""Git plumbing for trains: worktrees, commits, diffs, merges."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading

from . import settings

_repo_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def repo_lock(repo: str) -> threading.Lock:
    """Serialise operations that touch a repo's main checkout (merges, worktree add/remove)."""
    key = os.path.realpath(repo)
    with _locks_guard:
        return _repo_locks.setdefault(key, threading.Lock())


def git(*args: str, cwd: str, timeout: int = settings.GIT_TIMEOUT) -> tuple[int, str, str]:
    try:
        r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=timeout,
                           env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except subprocess.TimeoutExpired:
        return 1, "", f"git {' '.join(args)} timed out"
    except OSError as e:
        return 1, "", str(e)


def out(*args: str, cwd: str) -> str:
    return git(*args, cwd=cwd)[1]


def branch_name(item_id: int, title: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", title.lower()).strip("-")[:48].strip("-") or "work"
    return f"yamanote/{item_id}-{slug}"


def has_branch(repo: str, branch: str) -> bool:
    return bool(out("branch", "--list", branch, cwd=repo))


def head(repo: str) -> str:
    return out("rev-parse", "HEAD", cwd=repo)


def recent_log(repo: str, n: int = 15) -> str:
    return out("log", "--oneline", f"-{n}", cwd=repo)


def ensure_ignored(repo: str, entry: str = ".worktrees/") -> None:
    """Ignore via .git/info/exclude so the tracked .gitignore (and the main
    checkout's cleanliness) is left alone."""
    common = out("rev-parse", "--git-common-dir", cwd=repo) or ".git"
    path = os.path.join(repo, common, "info", "exclude")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        with open(path) as f:
            if any(line.strip() == entry for line in f):
                return
    with open(path, "a") as f:
        f.write(f"\n{entry}\n")


def create_worktree(repo: str, branch: str, name: str) -> str:
    """Worktree at <repo>/.worktrees/<name> on `branch` (created from trunk if new)."""
    with repo_lock(repo):
        base = os.path.join(repo, ".worktrees")
        os.makedirs(base, exist_ok=True)
        ensure_ignored(repo)
        path = os.path.join(base, name)
        if os.path.isdir(path):
            git("worktree", "remove", "--force", path, cwd=repo)
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
        git("worktree", "prune", cwd=repo)
        if has_branch(repo, branch):
            rc, _, err = git("worktree", "add", path, branch, cwd=repo)
        else:
            rc, _, err = git("worktree", "add", "-b", branch, path, settings.TRUNK_BRANCH, cwd=repo)
        if rc != 0 or not os.path.isdir(path):
            raise RuntimeError(f"worktree add failed for {branch}: {err[:300]}")
        return path


def remove_worktree(repo: str | None, path: str | None) -> None:
    if not repo or not path:
        return
    with repo_lock(repo):
        if os.path.isdir(path):
            rc, _, _ = git("worktree", "remove", "--force", path, cwd=repo)
            if rc != 0 and os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
        git("worktree", "prune", cwd=repo)


def delete_branch(repo: str, branch: str | None) -> None:
    if branch and has_branch(repo, branch):
        git("branch", "-D", branch, cwd=repo)


def current_branch(worktree: str) -> str:
    return out("rev-parse", "--abbrev-ref", "HEAD", cwd=worktree)


def commit_all(worktree: str, message: str) -> bool:
    """Stage and commit everything in the worktree. True if a commit was made."""
    git("add", "-A", cwd=worktree)
    merging = git("rev-parse", "-q", "--verify", "MERGE_HEAD", cwd=worktree)[0] == 0
    if not merging and not out("status", "--porcelain", cwd=worktree):
        return False
    rc, _, _ = git("-c", "user.name=Yamanote", "-c", "user.email=yamanote@localhost",
                   "commit", "-q", "-m", message, cwd=worktree)
    return rc == 0


def discard_changes(worktree: str) -> None:
    """Throw away uncommitted changes (used after verification runs)."""
    git("reset", "-q", "--hard", cwd=worktree)
    git("clean", "-qfd", cwd=worktree)


def start_trunk_merge(worktree: str) -> tuple[bool, list[str]]:
    """Merge trunk into the feature branch inside its worktree. Returns
    (clean, conflicted_files). On conflict the merge is left in progress with
    markers in the files, for the builder to resolve; committing concludes it."""
    rc, _, _ = git("-c", "user.name=Yamanote", "-c", "user.email=yamanote@localhost",
                   "merge", "--no-edit", settings.TRUNK_BRANCH, cwd=worktree)
    if rc == 0:
        return True, []
    files = [f for f in out("diff", "--name-only", "--diff-filter=U", cwd=worktree).splitlines() if f]
    if not files:  # failed for another reason
        git("merge", "--abort", cwd=worktree)
    return False, files


def leftover_conflict_markers(worktree: str) -> str:
    """`git diff --check` output for staged conflict markers ('' if none)."""
    git("add", "-A", cwd=worktree)
    report = out("diff", "--cached", "--check", cwd=worktree)
    return "\n".join(l for l in report.splitlines() if "conflict marker" in l)


def diff_trunk(worktree: str) -> str:
    return out("diff", f"{settings.TRUNK_BRANCH}...HEAD", cwd=worktree)


def diff_stat(worktree: str) -> str:
    return out("diff", "--stat", f"{settings.TRUNK_BRANCH}...HEAD", cwd=worktree)


def changed_files(worktree: str) -> list[str]:
    return [l for l in out("diff", "--name-only", f"{settings.TRUNK_BRANCH}...HEAD", cwd=worktree).splitlines() if l]


def trial_merge(repo: str, branch: str) -> tuple[bool, str]:
    """Would `branch` merge cleanly into trunk? Uses merge-tree so the main
    checkout is never touched."""
    rc, stdout, stderr = git("merge-tree", "--write-tree", "--name-only", settings.TRUNK_BRANCH, branch, cwd=repo)
    if rc == 0:
        return True, ""
    if rc == 1:
        return False, stdout[:500]
    # very old git without --write-tree: fall back to a no-commit merge
    with repo_lock(repo):
        rc, _, err = git("merge", "--no-commit", "--no-ff", branch, cwd=repo)
        git("merge", "--abort", cwd=repo)
    return rc == 0, err[:500]


def merge(repo: str, branch: str, message: str) -> tuple[bool, str]:
    """Merge `branch` into trunk in the main checkout. Reverts if conflict markers slip in."""
    with repo_lock(repo):
        cur = current_branch(repo)
        if cur != settings.TRUNK_BRANCH:
            return False, f"main checkout is on '{cur}', not {settings.TRUNK_BRANCH}"
        if out("status", "--porcelain", "--untracked-files=no", cwd=repo):
            return False, "main checkout has uncommitted changes"
        rc, stdout, stderr = git("-c", "user.name=Yamanote", "-c", "user.email=yamanote@localhost",
                                 "merge", "--no-ff", "-m", message, branch, cwd=repo)
        if rc != 0:
            git("merge", "--abort", cwd=repo)
            return False, (stderr or stdout)[:500]
        if out("diff", "--check", "HEAD~1..HEAD", cwd=repo):
            git("-c", "user.name=Yamanote", "-c", "user.email=yamanote@localhost",
                "revert", "--no-edit", "-m", "1", "HEAD", cwd=repo)
            return False, "conflict markers detected after merge; reverted"
        return True, head(repo)


def merge_stats(repo: str, merge_commit: str) -> dict | None:
    """What a merge brought into trunk: files, lines added/removed, and how many
    commits the branch carried. None if the commit isn't in this repo."""
    rc, short, _ = git("diff", "--shortstat", f"{merge_commit}^1", merge_commit, cwd=repo)
    if rc != 0:
        return None
    nums = {k: 0 for k in ("files", "insertions", "deletions")}
    for n, word in re.findall(r"(\d+) (file|insertion|deletion)", short):
        nums[{"file": "files", "insertion": "insertions", "deletion": "deletions"}[word]] = int(n)
    rc, count, _ = git("rev-list", "--count", f"{merge_commit}^1..{merge_commit}^2", cwd=repo)
    nums["commits"] = int(count) if rc == 0 and count.isdigit() else 0
    return nums


def revert_merge(repo: str, merge_commit: str) -> tuple[bool, str]:
    """Revert a merge commit on trunk in the main checkout."""
    with repo_lock(repo):
        if current_branch(repo) != settings.TRUNK_BRANCH:
            return False, f"main checkout is not on {settings.TRUNK_BRANCH}"
        rc, _, err = git("-c", "user.name=Yamanote", "-c", "user.email=yamanote@localhost",
                         "revert", "--no-edit", "-m", "1", merge_commit, cwd=repo)
        if rc != 0:
            git("revert", "--abort", cwd=repo)
            return False, err[:300]
        return True, head(repo)


def gc_worktrees(repo: str, keep: set[str]) -> list[str]:
    """Remove worktrees under <repo>/.worktrees not in `keep` (absolute paths)."""
    base = os.path.join(repo, ".worktrees")
    removed = []
    if not os.path.isdir(base):
        return removed
    for entry in os.listdir(base):
        path = os.path.join(base, entry)
        if os.path.isdir(path) and os.path.realpath(path) not in keep:
            remove_worktree(repo, path)
            removed.append(path)
    return removed
