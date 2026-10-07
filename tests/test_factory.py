"""End-to-end factory tests against a real temp git repo and a scripted LLM."""
from __future__ import annotations

import os
import unittest

from tests.helpers import (FakeClient, TempFactoryEnv, default_scripts, finish, reply, run_until, sh,
                           tool_call)
from yamanote import decisions, gitops, settings
from yamanote.factory import Factory
from yamanote.store import Store


class FactoryTestCase(unittest.TestCase):
    def setUp(self):
        self.env = TempFactoryEnv()
        self.store = Store(":memory:")

    def tearDown(self):
        if hasattr(self, "factory"):
            for agent in list(self.factory.agents.values()):
                agent.stop()
            self.factory.pool.shutdown(wait=True)
        self.store.close()
        self.env.close()

    def make(self, **script_overrides) -> Factory:
        self.client = FakeClient(default_scripts(**script_overrides))
        self.factory = Factory(self.store, client=self.client)
        return self.factory

    def item(self, item_id):
        return self.store.get_item(item_id)

    def kinds(self, item_id):
        return [e["kind"] for e in self.store.events(item_id)]


class HappyPathTest(FactoryTestCase):
    def test_human_item_travels_the_whole_line_and_merges(self):
        f = self.make()
        item = f.create_item("Add hello", "Create hello.py that prints hello")
        self.assertTrue(run_until(f, lambda: self.item(item["id"])["status"] == "done"))
        done = self.item(item["id"])
        self.assertEqual(done["station"], "retro", "every journey ends at JY09 Retro")
        self.assertEqual(self.store.get_retro(item["id"])["outcome"], "done")
        self.assertEqual(done["satisfaction"], 1.0)
        self.assertIsNone(done["train"])
        self.assertTrue((self.env.project / "hello.py").exists(), "merged into the main checkout")
        self.assertIn("Merge yamanote/", sh("git log -1 --format=%s", self.env.project))
        # no leftover worktree or branch
        self.assertFalse(os.listdir(self.env.project / ".worktrees"))
        self.assertEqual(sh("git branch --list 'yamanote/*'", self.env.project).strip(), "")
        # human items skip the triage LLM; every other station ran an agent
        roles = [r for r, _ in self.client.calls]
        self.assertNotIn("triage", roles)
        for role in ("spec", "builder", "inspector", "verifier"):
            self.assertIn(role, roles)
        # costs roll up from steps → runs → item
        self.assertGreater(done["cost_usd"], 0)
        self.assertEqual(round(sum(r["cost_usd"] for r in self.store.runs(item["id"])), 6),
                         round(done["cost_usd"], 6))
        self.assertIn("departed", self.kinds(item["id"]))

    def test_holdout_scenarios_are_never_shown_to_the_builder(self):
        seen = []

        def build(messages, turn):
            seen.append(messages[1]["content"])
            if turn == 0:
                return reply(tool_call("write_file", {"path": "hello.py", "content": "print('hello')\n"}))
            return finish("ok", {"verified_with": "ran it"})

        f = self.make(builder=build)
        item = f.create_item("Add hello", "Create hello.py")
        run_until(f, lambda: self.item(item["id"])["status"] == "done")
        self.assertTrue(seen)
        self.assertNotIn("python3 hello.py", seen[0])  # the scenario's steps
        self.assertIn("hello.py prints hello", seen[0])  # acceptance criteria are shown


class ReworkTest(FactoryTestCase):
    def test_inspector_changes_requested_sends_train_back_with_feedback(self):
        verdicts = iter(["CHANGES_REQUESTED", "APPROVED"])
        prompts_seen = []

        def inspector(messages, turn):
            v = next(verdicts)
            return finish(v, {"verdict": v, "issues": [] if v == "APPROVED" else
                              [{"file": "hello.py", "line": 1, "problem": "missing newline handling"}]})

        def build(messages, turn):
            if turn == 0:
                prompts_seen.append(messages[1]["content"])
                n = len(prompts_seen)
                return reply(tool_call("write_file", {"path": "hello.py", "content": f"print('hello {n}')\n"}))
            return finish("ok", {"verified_with": "x"})

        f = self.make(inspector=inspector, builder=build)
        item = f.create_item("Add hello", "Create hello.py")
        self.assertTrue(run_until(f, lambda: self.item(item["id"])["status"] == "done"))
        self.assertEqual(self.item(item["id"])["attempt"], 2)
        self.assertIn("missing newline handling", prompts_seen[1])
        self.assertIn("rework", self.kinds(item["id"]))

    def test_failed_holdout_reworks_then_escalates_service_class(self):
        outcomes = iter([False, False, True])

        def verifier(messages, turn):
            ok = next(outcomes)
            return finish("ran", {"results": [{"name": "prints hello", "passed": ok,
                                               "evidence": "printed nothing" if not ok else "ok"}]})

        f = self.make(verifier=verifier)
        item = f.create_item("Add hello", "Create hello.py")
        self.assertTrue(run_until(f, lambda: self.item(item["id"])["status"] == "done"))
        done = self.item(item["id"])
        self.assertEqual(done["attempt"], 3)
        self.assertEqual(done["service_class"], "rapid")  # small → local, escalated once on attempt 3
        self.assertIn("escalated", self.kinds(item["id"]))
        models = [m for r, m in self.client.calls if r == "builder"]
        self.assertEqual(models[0], settings.SERVICE_CLASSES["local"]["model"])
        self.assertEqual(models[-1], settings.SERVICE_CLASSES["rapid"]["model"])

    def test_gives_up_after_max_rework_attempts(self):
        def verifier(messages, turn):
            return finish("ran", {"results": [{"name": "s", "passed": False, "evidence": "broken"}]})

        f = self.make(verifier=verifier)
        item = f.create_item("Add hello", "Create hello.py")
        self.assertTrue(run_until(f, lambda: self.item(item["id"])["status"] == "failed", max_ticks=400))
        failed = self.item(item["id"])
        self.assertIn("Gave up", failed["outcome"])
        self.assertIsNone(failed["train"])
        # failed branches are kept for forensics; the worktree is not
        self.assertTrue(sh("git branch --list 'yamanote/*'", self.env.project).strip())

    def test_builder_without_changes_is_reworked(self):
        calls = []

        def build(messages, turn):
            calls.append(turn)
            if len(calls) <= 1:
                return finish("did nothing", {"verified_with": "-"})
            if turn == 0:
                return reply(tool_call("write_file", {"path": "hello.py", "content": "print(1)\n"}))
            return finish("ok", {"verified_with": "x"})

        f = self.make(builder=build)
        item = f.create_item("Add hello", "Create hello.py")
        self.assertTrue(run_until(f, lambda: self.item(item["id"])["status"] == "done"))
        self.assertEqual(self.item(item["id"])["attempt"], 2)


class GateTest(FactoryTestCase):
    def test_merge_gate_waits_for_human_approval(self):
        settings.GATE_MERGE = True
        f = self.make()
        item = f.create_item("Add hello", "Create hello.py")
        self.assertTrue(run_until(f, lambda: self.item(item["id"])["status"] == "waiting"))
        self.assertEqual(self.item(item["id"])["gate"], "merge")
        self.assertFalse((self.env.project / "hello.py").exists())
        f.approve(item["id"])
        self.assertTrue(run_until(f, lambda: self.item(item["id"])["status"] == "done"))
        self.assertTrue((self.env.project / "hello.py").exists())

    def test_spec_gate_reject_cleans_up(self):
        settings.GATE_SPEC = True
        f = self.make()
        item = f.create_item("Add hello", "Create hello.py")
        self.assertTrue(run_until(f, lambda: self.item(item["id"])["status"] == "waiting"))
        self.assertEqual(self.item(item["id"])["gate"], "spec")
        f.reject(item["id"], "not now")
        self.assertEqual(self.item(item["id"])["status"], "rejected")
        self.assertIn("not now", self.item(item["id"])["outcome"])


class ConflictTest(FactoryTestCase):
    def test_conflict_after_approval_goes_back_to_builder_and_needs_reapproval(self):
        settings.GATE_MERGE = True
        prompts_seen = []

        def build(messages, turn):
            if turn == 0:
                prompts_seen.append(messages[1]["content"])
                return reply(tool_call("write_file", {"path": "hello.py",
                                                      "content": "print('hello')\nprint('trunk too')\n"}))
            return finish("ok", {"verified_with": "x"})

        f = self.make(builder=build)
        item = f.create_item("Add hello", "Create hello.py")
        self.assertTrue(run_until(f, lambda: self.item(item["id"])["status"] == "waiting"))
        # trunk moves underneath the waiting train, touching the same file
        (self.env.project / "hello.py").write_text("print('trunk')\n")
        sh("git add -A && git -c user.email=t@t -c user.name=t commit -qm trunk-change", self.env.project)
        f.approve(item["id"])
        self.assertTrue(run_until(f, lambda: self.item(item["id"])["status"] == "waiting"
                                  and self.item(item["id"])["conflicts"] == 1))
        it = self.item(item["id"])
        self.assertEqual(it["gate"], "merge", "re-built code must be approved again")
        self.assertIn("conflict markers in: hello.py", prompts_seen[-1])
        self.assertEqual(it["branch"], gitops.branch_name(item["id"], "add-hello"), "branch kept, not rebuilt")
        f.approve(item["id"])
        self.assertTrue(run_until(f, lambda: self.item(item["id"])["status"] == "done"))
        self.assertIn("trunk too", (self.env.project / "hello.py").read_text())
        log = sh("git log --oneline", self.env.project)
        self.assertIn("trunk-change", log)

    def test_leftover_markers_are_detected(self):
        wt = self.env.project
        (wt / "x.txt").write_text("<<<<<<< HEAD\na\n=======\nb\n>>>>>>> main\n")
        self.assertIn("conflict marker", gitops.leftover_conflict_markers(str(wt)))


class TriageTest(FactoryTestCase):
    def _patch_jev(self, **scores):
        class D:
            answers = {}
            tokens, cost_usd = 400, 0.0002

            def p(self, name):
                return scores[name]

            def score(self, name):
                return scores["difficulty"]

        orig = decisions.assess_spec
        decisions.assess_spec = lambda text, ctx="", backend=None: D()
        self.addCleanup(setattr, decisions, "assess_spec", orig)

    def test_jev_rejects_obvious_duplicates_without_llm(self):
        self._patch_jev(useful=0.9, ready=0.9, duplicate=0.95, difficulty=1.0)
        f = self.make()
        item = f.create_item("dup", "Already built", source="dispatcher")
        self.assertTrue(run_until(f, lambda: self.item(item["id"])["status"] == "rejected"))
        self.assertNotIn("triage", [r for r, _ in self.client.calls])
        jev_runs = [r for r in self.store.runs(item["id"]) if r["role"] == "jev"]
        self.assertEqual(len(jev_runs), 1)

    def test_jev_fast_pass_skips_llm_triage_and_sets_class(self):
        self._patch_jev(useful=0.95, ready=0.9, duplicate=0.05, difficulty=3.2)
        f = self.make()
        item = f.create_item("big", "Hard but clear", source="dispatcher")
        self.assertTrue(run_until(f, lambda: self.item(item["id"])["station"] == "build"
                                  or self.item(item["id"])["status"] == "done"))
        it = self.item(item["id"])
        self.assertEqual(it["difficulty"], "hard")
        self.assertEqual(it["service_class"], "express")
        self.assertNotIn("triage", [r for r, _ in self.client.calls])

    def test_unsure_jev_defers_to_llm_triage(self):
        self._patch_jev(useful=0.5, ready=0.5, duplicate=0.3, difficulty=2.0)
        f = self.make(triage=lambda m, t: finish("no", {"verdict": "HOLD", "reason": "needs detail"}))
        item = f.create_item("meh", "Something", source="dispatcher")
        self.assertTrue(run_until(f, lambda: self.item(item["id"])["status"] == "held"))
        self.assertIn("triage", [r for r, _ in self.client.calls])
        self.assertGreater(self.item(item["id"])["hold_until"], 0)

    def test_triage_context_lists_other_in_flight_work_but_not_itself(self):
        seen = []
        orig = decisions.assess_spec

        def fake(text, ctx="", backend=None):
            seen.append(ctx)
            return None
        decisions.assess_spec = fake
        self.addCleanup(setattr, decisions, "assess_spec", orig)
        f = self.make(triage=lambda m, t: finish("no", {"verdict": "REJECT", "reason": "dup"}))
        a = f.create_item("first-thing", "x", source="dispatcher")
        self.store.update_item(a["id"], station="build", status="waiting", gate="spec")  # parked
        b = f.create_item("second-thing", "y", source="dispatcher")
        self.assertTrue(run_until(f, lambda: self.item(b["id"])["status"] == "rejected"))
        self.assertIn("first-thing", seen[-1])
        self.assertNotIn("second-thing", seen[-1])

    def test_consecutive_rejections_stall_the_project(self):
        settings_max = settings.MAX_CONSECUTIVE_REJECTIONS
        settings.MAX_CONSECUTIVE_REJECTIONS = 2
        self.addCleanup(setattr, settings, "MAX_CONSECUTIVE_REJECTIONS", settings_max)
        f = self.make(triage=lambda m, t: finish("no", {"verdict": "REJECT", "reason": "nope"}))
        a = f.create_item("a", "x", source="dispatcher")
        b = f.create_item("b", "y", source="dispatcher")
        self.assertTrue(run_until(f, lambda: self.item(a["id"])["status"] == "rejected"
                                  and self.item(b["id"])["status"] == "rejected"))
        self.assertIsNone(f.pick_project())


class ControlTest(FactoryTestCase):
    def test_cancel_stops_a_running_builder(self):
        import threading
        started = threading.Event()

        def slow_build(messages, turn):
            started.set()
            return reply(tool_call("run", {"command": "sleep 30"}))

        f = self.make(builder=slow_build)
        item = f.create_item("Add hello", "Create hello.py")
        self.assertTrue(run_until(f, started.is_set))
        f.cancel(item["id"])
        self.assertEqual(self.item(item["id"])["status"], "cancelled")
        run_until(f, lambda: not f.jobs, max_ticks=100)
        self.assertFalse(f.jobs)
        self.assertEqual(self.item(item["id"])["status"], "cancelled")

    def test_pause_blocks_departures(self):
        f = self.make()
        f.set_paused(True)
        item = f.create_item("Add hello", "Create hello.py")
        for _ in range(5):
            f.tick()
        self.assertEqual(self.item(item["id"])["status"], "queued")
        f.set_paused(False)
        self.assertTrue(run_until(f, lambda: self.item(item["id"])["status"] == "done"))

    def test_daily_budget_suspends_launches(self):
        f = self.make()
        orig = settings.DAILY_BUDGET_USD
        settings.DAILY_BUDGET_USD = 0.0
        self.addCleanup(setattr, settings, "DAILY_BUDGET_USD", orig)
        item = f.create_item("Add hello", "Create hello.py")
        f.tick()
        self.assertEqual(self.item(item["id"])["status"], "queued")
        self.assertIn("daily budget", f.suspension())

    def test_retry_requeues_failed_item(self):
        f = self.make()
        item = f.create_item("Add hello", "Create hello.py")
        self.store.update_item(item["id"], status="failed", outcome="boom")
        f.retry(item["id"])
        self.assertTrue(run_until(f, lambda: self.item(item["id"])["status"] == "done"))

    def test_rejects_projects_outside_dev_dir_and_self(self):
        f = self.make()
        with self.assertRaises(ValueError):
            f.create_item("x", "y", project="/tmp")
        with self.assertRaises(ValueError):
            f.create_item("x", "y", project=str(settings.BASE_DIR))

    def test_capacity_one_train_runs_items_one_at_a_time(self):
        settings.MAX_TRAINS = 1
        f = self.make()
        self.assertEqual(len(f.trains), 1)
        a = f.create_item("Add hello", "Create hello.py")
        b = f.create_item("Add hello two", "Create hello.py again")
        max_on_line = 0
        for _ in range(400):
            f.tick()
            on_line = [i for i in (self.item(a["id"]), self.item(b["id"])) if i["train"]]
            max_on_line = max(max_on_line, len(on_line))
            if all(self.item(x["id"])["status"] in ("done", "failed") for x in (a, b)):
                break
            import time
            time.sleep(0.02)
        self.assertEqual(max_on_line, 1)
        self.assertEqual(self.item(a["id"])["status"], "done")


class DispatcherTest(FactoryTestCase):
    def test_dispatcher_feeds_empty_line(self):
        f = self.make()
        settings.DISPATCHER_INTERVAL = 0
        self.assertTrue(run_until(f, lambda: any(i["source"] == "dispatcher" for i in self.store.items())))
        item = [i for i in self.store.items() if i["source"] == "dispatcher"][0]
        self.assertEqual(item["title"], "add-greeting")
        self.assertEqual(item["project"], os.path.realpath(self.env.project))


class SignalTest(FactoryTestCase):
    def test_new_log_errors_file_a_bug(self):
        bug = '{"file": true, "title": "crash-on-done", "priority": "high", "description": "IndexError in done"}'
        f = self.make(signal=lambda m, t: reply(text=bug), triage=lambda m, t: finish("x", {"verdict": "HOLD", "reason": "wait"}))
        log = self.env.project / "app.log"
        log.write_text("INFO boot\n")
        f.tick()  # first sight of the log: remember the offset, don't replay history
        with log.open("a") as fh:
            fh.write("ERROR Traceback (most recent call last): IndexError: list index out of range\n")
        self.assertTrue(run_until(f, lambda: any(i["source"] == "signal" for i in self.store.items())))
        item = next(i for i in self.store.items() if i["source"] == "signal")
        self.assertEqual((item["title"], item["kind"], item["priority"]), ("crash-on-done", "bug", "high"))
        # the same error signature doesn't trigger again within the hour
        with log.open("a") as fh:
            fh.write("ERROR Traceback (most recent call last): IndexError: list index out of range\n")
        for _ in range(5):
            f.tick()
        self.assertEqual(sum(1 for r, _ in self.client.calls if r == "signal"), 1)

    def test_old_log_history_is_not_replayed(self):
        (self.env.project / "app.log").write_text("ERROR old problem\n")
        f = self.make(signal=lambda m, t: self.fail("signal should not run"))
        for _ in range(3):
            f.tick()
        self.assertFalse(self.store.items())


class RecoveryTest(FactoryTestCase):
    def test_restart_requeues_running_items(self):
        item = self.store.create_item(title="x", project=str(self.env.project))
        self.store.update_item(item["id"], status="running", station="build")
        run = self.store.start_run(item["id"], "builder", "build", "m")
        self.make()
        self.assertEqual(self.item(item["id"])["status"], "queued")
        self.assertEqual(self.store.get_run(run)["status"], "interrupted")


if __name__ == "__main__":
    unittest.main()
