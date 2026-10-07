"""Regression tests for the issues found in the October review, and for the
features added with them (checks, merge queue, notifications, post-deploy
watch, redaction, lessons, adaptive routing)."""
from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import threading
import time
import unittest

from tests.helpers import FakeClient, default_scripts, finish, reply, run_until, sh, tool_call
from tests.test_factory import FactoryTestCase
from yamanote import gitops, settings
from yamanote.factory import ItemGone
from yamanote.llm import LLMError
from yamanote.store import Store


def patch(test, obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    test.addCleanup(setattr, obj, name, old)


class BudgetAndSuspensionTest(FactoryTestCase):
    def test_item_over_budget_fails_without_spending_more(self):
        f = self.make()
        item = f.create_item("Add hello", "Create hello.py")
        self.store.add_item_cost(item["id"], 50.0, 0, 0)
        self.assertTrue(run_until(f, lambda: self.item(item["id"])["status"] == "failed"))
        self.assertIn("Over budget", self.item(item["id"])["outcome"])
        self.assertEqual([r for r, _ in self.client.calls], ["retrospective"],
                         "only the (cheap) retrospective runs; no station work")

    def test_out_of_credit_suspends_line_and_keeps_attempts(self):
        class Broke(FakeClient):
            def chat(self, *a, **k):
                self.calls.append(("x", "m"))
                raise LLMError("OpenRouter HTTP 402: insufficient credits", 402)

        self.factory = f = __import__("yamanote.factory", fromlist=["Factory"]).Factory(self.store, client=Broke({}))
        item = f.create_item("Add hello", "Create hello.py")
        self.store.update_item(item["id"], station="build", spec={"plan": "p"})
        for _ in range(30):
            f.tick()
            time.sleep(0.02)
        it = self.item(item["id"])
        self.assertEqual((it["status"], it["station"], it["attempt"]), ("queued", "build", 0))
        self.assertIn("HTTP 402", f.suspension())
        self.assertEqual(len(f.client.calls), 1, "no further calls while suspended")
        self.assertIn("suspended", [e["kind"] for e in self.store.events()])
        f.set_paused(False)  # a human resume clears the API suspension
        self.assertIsNone(f.suspension())

    def test_signal_respects_budget_suspension(self):
        f = self.make(signal=lambda m, t: reply(text='{"file": true, "title": "x", "description": "d"}'))
        patch(self, settings, "DAILY_BUDGET_USD", 0.0)
        log = self.env.project / "app.log"
        log.write_text("ok\n")
        f.tick()
        with log.open("a") as fh:
            fh.write("ERROR boom\n")
        for _ in range(10):
            f.tick()
            time.sleep(0.02)
        self.assertNotIn("signal", [r for r, _ in self.client.calls])


class LifecycleTest(FactoryTestCase):
    def test_third_hold_becomes_a_reject(self):
        patch(self, settings, "HOLD_RECYCLE_SECONDS", 0)
        f = self.make(triage=lambda m, t: finish("x", {"verdict": "HOLD", "reason": "vague"}))
        item = f.create_item("meh", "x", source="dispatcher")
        self.assertTrue(run_until(f, lambda: self.item(item["id"])["status"] == "rejected", max_ticks=300))
        self.assertEqual([e["kind"] for e in self.store.events(item["id"])].count("held"), settings.MAX_HOLDS)
        self.assertIn("held 2 times", self.item(item["id"])["outcome"])

    def test_retry_resets_counters_and_skips_triage(self):
        f = self.make(triage=lambda m, t: finish("no", {"verdict": "REJECT", "reason": "nope"}))
        item = f.create_item("thing", "x", source="dispatcher")
        self.assertTrue(run_until(f, lambda: self.item(item["id"])["status"] == "rejected"))
        self.store.kv_set(f"retries:{item['id']}:spec", 2)
        f.retry(item["id"])
        self.assertIsNone(self.store.kv_get(f"retries:{item['id']}:spec"))
        self.assertEqual(self.item(item["id"])["station"], "spec", "human retry overrides the triage gate")
        self.assertTrue(run_until(f, lambda: self.item(item["id"])["status"] == "done"))

    def test_cancel_wins_over_a_finishing_job(self):
        f = self.make()
        item = f.create_item("x", "y")
        self.store.update_item(item["id"], status="running", station="inspect")
        f.cancel(item["id"])
        with self.assertRaises(ItemGone):
            f._move(item, "verify", "a job finishing late")
        self.assertEqual(self.item(item["id"])["status"], "cancelled")

    def test_sleep_shifts_deadlines(self):
        f = self.make()
        item = f.create_item("x", "y")
        started = time.time() - 600
        self.store.update_item(item["id"], status="waiting", gate="merge", started_at=started)
        f._last_tick = time.time() - 8 * 3600  # the laptop slept overnight
        f.tick()
        self.assertGreater(self.item(item["id"])["started_at"], started + 7 * 3600)
        self.assertEqual(self.item(item["id"])["status"], "waiting")

    def test_recovery_discards_verifier_junk(self):
        f = self.make()
        item = f.create_item("Add hello", "Create hello.py")
        wt = gitops.create_worktree(str(self.env.project), "yamanote/x", "item-x")
        (open(os.path.join(wt, "scratch_check.py"), "w")).write("junk")
        self.store.update_item(item["id"], status="running", station="verify", worktree=wt, branch="yamanote/x")
        self.factory.pool.shutdown()
        from yamanote.factory import Factory
        self.factory = Factory(self.store, client=self.client)
        self.assertFalse(os.path.exists(os.path.join(wt, "scratch_check.py")))


class ChecksTest(FactoryTestCase):
    def test_setup_runs_once_and_failing_tests_send_the_train_back(self):
        builds = []

        def build(messages, turn):
            if turn == 0:
                builds.append(messages[1]["content"])
                ok = len(builds) > 1
                return reply(tool_call("write_file", {"path": "hello.py",
                                                      "content": "print('hello')\n" if ok else "raise SystemExit(3)\n"}))
            return finish("done", {"verified_with": "x"})

        patch(self, settings, "SETUP_CMD", "echo setup >> setup.log")
        patch(self, settings, "TEST_CMD", "python3 hello.py")
        f = self.make(builder=build)
        item = f.create_item("Add hello", "Create hello.py")
        self.assertTrue(run_until(f, lambda: self.item(item["id"])["status"] == "done", max_ticks=400))
        self.assertIn("`python3 hello.py` must pass", builds[0])
        self.assertIn("tests fail after your change", builds[1])
        ci = [r for r in self.store.runs(item["id"]) if r["role"] == "ci"]
        self.assertEqual([self.store.steps(r["id"])[0]["name"] for r in ci][:3], ["setup", "test", "test"])
        self.assertEqual(sum(1 for r in ci if self.store.steps(r["id"])[0]["name"] == "setup"), 1)
        roles = [r for r, _ in self.client.calls]
        self.assertEqual(roles.count("inspector"), 1, "failing tests never reach the (paid) inspector")


class MergeQueueTest(FactoryTestCase):
    def test_trains_that_pass_alone_cannot_break_trunk_together(self):
        (self.env.project / "calc.py").write_text("def add(a, b):\n    return a + b\n")
        (self.env.project / "use.py").write_text("print('placeholder')\n")
        sh("git add -A && git -c user.email=t@t -c user.name=t commit -qm calc", self.env.project)
        fixed = []

        def build(messages, turn):
            task = messages[1]["content"]
            if turn == 1 and "tests fail" in task and "rename" in task:
                # the rename landed second: its callers on trunk need updating too
                return reply(tool_call("write_file", {"path": "use.py", "content": "from calc import plus\nprint(plus(2, 3))\n"}))
            if turn > 0:
                return finish("ok", {"verified_with": "ran"})
            if "tests fail" in task:
                fixed.append(task)
                if "rename" in task:
                    return reply(tool_call("write_file", {"path": "calc.py", "content": "def plus(a, b):\n    return a + b\n"}))
                return reply(tool_call("write_file", {"path": "use.py", "content": "from calc import plus\nprint(plus(2, 3))\n"}))
            if "rename" in task:
                return reply(tool_call("write_file", {"path": "calc.py", "content": "def plus(a, b):\n    return a + b\n"}))
            return reply(tool_call("write_file", {"path": "use.py", "content": "from calc import add\nprint(add(2, 3))\n"}))

        patch(self, settings, "TEST_CMD", "python3 use.py")
        patch(self, settings, "MAX_TRAINS", 2)
        patch(self, settings, "MERGE_QUEUE_RETRY_SECONDS", 0)
        f = self.make(builder=build)
        a = f.create_item("rename add to plus", "rename add to plus in calc.py")
        b = f.create_item("add use script", "use.py prints add(2,3)")
        self.assertTrue(run_until(f, lambda: all(self.item(x["id"])["status"] == "done" for x in (a, b)), max_ticks=600))
        out = sh("python3 use.py", self.env.project)
        self.assertEqual(out.strip(), "5", "trunk works after both trains land")
        self.assertTrue(fixed, "whichever train landed second was sent back when the combination failed its tests")
        kinds = [e["kind"] for e in self.store.events()]
        self.assertIn("integrated", kinds)


class NotifyTest(FactoryTestCase):
    def test_gate_and_failure_hooks_receive_json(self):
        out = os.path.join(self.env.tmp.name, "notes.jsonl")
        patch(self, settings, "NOTIFY_CMD", f"cat >> {out}; echo >> {out}")
        patch(self, settings, "PUBLIC_URL", "http://pi:8080")
        patch(self, settings, "GATE_MERGE", True)
        f = self.make()
        item = f.create_item("Add hello", "Create hello.py")
        self.assertTrue(run_until(f, lambda: self.item(item["id"])["status"] == "waiting"))
        f.cancel(item["id"])  # "cancelled" is not a notifying event
        deadline = time.time() + 5
        while time.time() < deadline and not (os.path.exists(out) and open(out).read().strip()):
            time.sleep(0.05)
        notes = [json.loads(l) for l in open(out).read().splitlines() if l.strip()]
        self.assertEqual([n["event"] for n in notes], ["gate"])
        self.assertEqual(notes[0]["item"]["id"], item["id"])
        self.assertEqual(notes[0]["url"], f"http://pi:8080/#item-{item['id']}")

    def test_failing_hook_is_reported_not_raised(self):
        patch(self, settings, "NOTIFY_CMD", "exit 7")
        from yamanote import notify
        errors = notify._deliver(notify.payload("failed", "t", "m"))
        self.assertIn("exited 7", errors[0])


class DeployWatchTest(FactoryTestCase):
    def test_new_errors_after_deploy_file_a_linked_regression_and_can_revert(self):
        patch(self, settings, "AUTO_REVERT", True)
        nothing = '{"file": false}'
        log = self.env.project / "app.log"
        log.write_text("INFO boot\nERROR old known problem 1\n")
        f = self.make(signal=lambda m, t: reply(text=nothing))
        f.tick()  # first sight of the log
        with log.open("a") as fh:
            fh.write("ERROR old known problem 2\n")
        f.tick()  # baseline signature recorded
        item = f.create_item("Add hello", "Create hello.py")
        self.assertTrue(run_until(f, lambda: self.item(item["id"])["status"] == "done"))
        self.assertTrue((self.env.project / "hello.py").exists())
        with log.open("a") as fh:
            fh.write("ERROR old known problem 3\n")
        f.tick()
        self.assertFalse([i for i in self.store.items() if i.get("parent_id")], "known errors are not regressions")
        with log.open("a") as fh:
            fh.write("ERROR Traceback: KeyError 'greeting' in hello handler\n")
        self.assertTrue(run_until(f, lambda: any(i.get("parent_id") == item["id"] for i in self.store.items())))
        bug = next(i for i in self.store.items() if i.get("parent_id") == item["id"])
        self.assertEqual((bug["kind"], bug["priority"]), ("bug", "high"))
        self.assertIn("KeyError", bug["description"])
        self.assertTrue(run_until(f, lambda: "reverted" in [e["kind"] for e in self.store.events(item["id"])]))
        self.assertFalse((self.env.project / "hello.py").exists(), "merge reverted on trunk")


class RedactionTest(FactoryTestCase):
    def test_builder_sees_behaviour_not_the_hidden_check(self):
        outcomes = iter([False, True])
        prompts_seen = []

        def verifier(m, t):
            ok = next(outcomes)
            return finish("ran", {"results": [{"name": "prints hello", "passed": ok,
                                               "evidence": "ran `python3 hello.py > /tmp/o; grep -q MARKER_OK /tmp/o` exit 1"}]})

        def build(messages, turn):
            if turn == 0:
                prompts_seen.append(messages[1]["content"])
                return reply(tool_call("write_file", {"path": "hello.py", "content": f"print({len(prompts_seen)})\n"}))
            return finish("ok", {"verified_with": "x"})

        f = self.make(verifier=verifier, builder=build,
                      redactor=lambda m, t: reply(text='{"defects": ["Running the program prints nothing useful; it should print hello."]}'))
        item = f.create_item("Add hello", "Create hello.py")
        self.assertTrue(run_until(f, lambda: self.item(item["id"])["status"] == "done"))
        rework = prompts_seen[1]
        self.assertIn("it should print hello", rework)
        self.assertNotIn("MARKER_OK", rework)
        self.assertNotIn("/tmp/o", rework)


class DisputeTest(FactoryTestCase):
    def _two_scenarios(self, m, t):
        return finish("spec", {"acceptance_criteria": ["works"], "plan": "p", "relevant_files": [], "difficulty": "small",
                               "scenarios": [{"name": "a", "steps": "x", "expected": "y"},
                                             {"name": "b", "steps": "x", "expected": "y"}]})

    def test_self_contradictory_scenario_is_excluded_not_reworked(self):
        verifier = lambda m, t: finish("ran", {"results": [
            {"name": "a", "passed": True, "evidence": "ok"},
            {"name": "b", "passed": False, "scenario_error": True, "evidence": "expects #4 but only 3 tasks are added"}]})
        f = self.make(spec=self._two_scenarios, verifier=verifier)
        item = f.create_item("Add hello", "Create hello.py")
        self.assertTrue(run_until(f, lambda: self.item(item["id"])["status"] == "done"))
        it = self.item(item["id"])
        self.assertEqual((it["attempt"], it["satisfaction"]), (1, 1.0))
        self.assertIn("disputed", [e["kind"] for e in self.store.events(item["id"])])

    def test_disputing_everything_does_not_pass(self):
        verifier = lambda m, t: finish("ran", {"results": [
            {"name": "a", "passed": False, "scenario_error": True, "evidence": "meh"},
            {"name": "b", "passed": False, "scenario_error": True, "evidence": "meh"}]})
        f = self.make(spec=self._two_scenarios, verifier=verifier)
        item = f.create_item("Add hello", "Create hello.py")
        self.assertTrue(run_until(f, lambda: "rework" in [e["kind"] for e in self.store.events(item["id"])]))
        self.assertNotIn("disputed", [e["kind"] for e in self.store.events(item["id"])])


class LearningTest(FactoryTestCase):
    def test_retro_notes_reach_only_their_station_on_later_trains(self):
        verdicts = iter(["CHANGES_REQUESTED", "APPROVED", "APPROVED"])
        seen: dict[str, list[str]] = {"spec": [], "builder": [], "inspector": []}

        def inspector(m, t):
            seen["inspector"].append(m[1]["content"])
            v = next(verdicts)
            return finish(v, {"verdict": v, "issues": [] if v == "APPROVED" else [{"problem": "errors must go to stderr"}]})

        def build(messages, turn):
            if turn == 0:
                seen["builder"].append(messages[1]["content"])
                return reply(tool_call("write_file", {"path": f"f{len(seen['builder'])}.py", "content": "x = 1\n"}))
            return finish("ok", {"verified_with": "x"})

        def spec(m, t):
            seen["spec"].append(m[1]["content"])
            return default_scripts()["spec"](m, t)

        retro = ('{"summary": "Needed a rework for stderr.", "went_well": ["tests"], "went_wrong": ["stdout errors"],'
                 ' "root_cause": "spec did not say where errors go", "class_fit": "right", "retire": [],'
                 ' "notes": [{"station": "build", "note": "Write CLI errors to stderr and exit 1."},'
                 ' {"station": "spec", "note": "State where error messages are printed."},'
                 ' {"station": "merge", "note": "not a playbook station"}]}')
        f = self.make(inspector=inspector, builder=build, spec=spec, retrospective=lambda m, t: reply(text=retro))
        a = f.create_item("first", "x")
        self.assertTrue(run_until(f, lambda: self.item(a["id"])["status"] == "done"))
        r = self.store.get_retro(a["id"])
        self.assertEqual(r["data"]["root_cause"], "spec did not say where errors go")
        self.assertEqual([n["station"] for n in r["data"]["notes_added"]], ["build", "spec"], "unknown stations dropped")
        b = f.create_item("second", "y")
        self.assertTrue(run_until(f, lambda: self.item(b["id"])["status"] == "done"))
        self.assertIn("Write CLI errors to stderr", seen["builder"][-1])
        self.assertIn("State where error messages are printed", seen["spec"][-1])
        self.assertNotIn("Write CLI errors to stderr", seen["inspector"][-1], "notes only go to their station")
        self.assertNotIn("Write CLI errors", seen["builder"][0], "the train that taught it didn't see it")

    def test_failed_train_reflects_then_stays_failed(self):
        def verifier(m, t):
            return finish("ran", {"results": [{"name": "s", "passed": False, "evidence": "broken"}]})

        retro = '{"summary": "Gave up after four builds; checks kept failing.", "class_fit": "underpowered", "notes": [], "retire": []}'
        f = self.make(verifier=verifier, retrospective=lambda m, t: reply(text=retro))
        item = f.create_item("Add hello", "Create hello.py")
        self.assertTrue(run_until(f, lambda: self.item(item["id"])["status"] == "failed", max_ticks=400))
        it = self.item(item["id"])
        self.assertEqual(it["station"], "retro")
        self.assertIn("Gave up", it["outcome"])
        self.assertEqual(self.store.get_retro(item["id"])["class_fit"], "underpowered")
        kinds = [e["kind"] for e in self.store.events(item["id"])]
        self.assertEqual(kinds.count("failed"), 1, "the failure is announced once, before the retro")
        self.assertLess(kinds.index("failed"), kinds.index("retro"))

    def test_schema_echo_is_retried_a_class_up(self):
        answers = iter(['{"summary": "...", "notes": []}',
                        '{"summary": "Clean first-time arrival; nothing to change.", "class_fit": "right", "notes": []}'])
        f = self.make(retrospective=lambda m, t: reply(text=next(answers)))
        item = f.create_item("Add hello", "Create hello.py")
        self.assertTrue(run_until(f, lambda: self.item(item["id"])["status"] == "done"))
        self.assertEqual(self.store.get_retro(item["id"])["data"]["summary"], "Clean first-time arrival; nothing to change.")
        models = [m for r, m in self.client.calls if r == "retrospective"]
        self.assertEqual(models, [settings.SERVICE_CLASSES["rapid"]["model"], settings.SERVICE_CLASSES["express"]["model"]])

    def test_restart_does_not_spend_or_escalate_an_interrupted_attempt(self):
        f = self.make()
        item = f.create_item("Add hello", "Create hello.py")
        self.store.update_item(item["id"], station="build", status="running", attempt=2, service_class="local",
                               first_class="local", spec={"plan": "p"})
        self.factory.pool.shutdown()
        from yamanote.factory import Factory
        self.factory = f = Factory(self.store, client=self.client)
        self.assertEqual(self.item(item["id"])["attempt"], 1)
        self.assertTrue(run_until(f, lambda: self.item(item["id"])["status"] == "done"))
        self.assertEqual(self.item(item["id"])["attempt"], 2)
        self.assertNotIn("escalated", [e["kind"] for e in self.store.events(item["id"])])

    def test_broken_retrospective_still_closes_the_journey(self):
        f = self.make(retrospective=lambda m, t: reply(text="not json at all"))
        item = f.create_item("Add hello", "Create hello.py")
        self.assertTrue(run_until(f, lambda: self.item(item["id"])["status"] == "done"))
        self.assertTrue(self.store.get_retro(item["id"])["data"]["error"])

    def test_cancel_during_retro_keeps_the_outcome(self):
        f = self.make()
        item = f.create_item("x", "y")
        self.store.update_item(item["id"], station="retro", status="queued", pending_status="done", outcome="Arrived")
        f.cancel(item["id"])
        it = self.item(item["id"])
        self.assertEqual((it["status"], it["station"]), ("done", "retro"))
        self.assertIn("retrospective skipped", it["outcome"])

    def test_notes_are_scored_and_weak_ones_retired(self):
        f = self.make()
        project = os.path.realpath(self.env.project)
        f._save_playbook(project, [
            {"id": 1, "station": "build", "text": "never helps", "item_id": 0, "ts": 0, "uses": 7, "wins": 0},
            {"id": 2, "station": "build", "text": "helps", "item_id": 0, "ts": 0, "uses": 7, "wins": 6},
            {"id": 3, "station": "spec", "text": "retire me", "item_id": 0, "ts": 0, "uses": 0, "wins": 0},
        ])
        verdicts = iter(["CHANGES_REQUESTED", "APPROVED"])
        f.client.scripts["inspector"] = lambda m, t: (lambda v: finish(v, {"verdict": v, "issues": []}))(next(verdicts))
        f.client.scripts["retrospective"] = lambda m, t: reply(
            text='{"summary": "One rework; note 3 proved misleading here.", "class_fit": "right", "notes": [], "retire": [3]}')
        item = f.create_item("Add hello", "Create hello.py")
        self.assertTrue(run_until(f, lambda: self.item(item["id"])["status"] == "done"))
        notes = {n["id"]: n for n in f.playbook(project)}
        self.assertEqual(set(notes), {2}, "1 auto-retired (0/8 first-pass), 3 retired by the retro")
        self.assertEqual((notes[2]["uses"], notes[2]["wins"]), (8, 6), "a rework is a use but not a win")
        retired = {n["id"]: n["why"] for n in self.store.get_retro(item["id"])["data"]["notes_retired"]}
        self.assertEqual(retired, {1: "auto: poor first-pass rate", 3: "retrospective"})

    def test_station_playbook_is_capped(self):
        f = self.make()
        project = os.path.realpath(self.env.project)
        f._save_playbook(project, [{"id": i, "station": "build", "text": f"note {i}", "item_id": 0, "ts": i,
                                    "uses": 4, "wins": 4 if i != 2 else 0} for i in range(1, 7)])
        f.client.scripts["retrospective"] = lambda m, t: reply(
            text='{"summary": "Arrived; one durable build lesson worth keeping.", "notes": [{"station": "build", "note": "brand new"}], "retire": []}')
        item = f.create_item("Add hello", "Create hello.py")
        self.assertTrue(run_until(f, lambda: self.item(item["id"])["status"] == "done"))
        texts = [n["text"] for n in f.playbook(project) if n["station"] == "build"]
        self.assertEqual(len(texts), settings.MAX_NOTES_PER_STATION)
        self.assertIn("brand new", texts)
        self.assertNotIn("note 2", texts, "the weakest note made room")

    def test_old_lessons_migrate_to_the_build_playbook(self):
        f = self.make()
        self.store.kv_set("lessons:/p", [{"text": "old lesson", "item_id": 4, "ts": 1}])
        self.assertEqual([(n["station"], n["text"]) for n in f.playbook("/p")], [("build", "old lesson")])

    def test_retro_disabled_finishes_at_deploy(self):
        patch(self, settings, "RETRO_ENABLED", False)
        f = self.make()
        item = f.create_item("Add hello", "Create hello.py")
        self.assertTrue(run_until(f, lambda: self.item(item["id"])["status"] == "done"))
        self.assertEqual(self.item(item["id"])["station"], "deploy")
        self.assertNotIn("retrospective", [r for r, _ in self.client.calls])

    def test_class_fit_votes_move_routing(self):
        patch(self, settings, "ADAPTIVE_ROUTING", True)
        f = self.make()
        for i in range(6):  # moderate work on Rapid passes first time, but retros call Rapid too weak
            it = self.store.create_item(title=f"m{i}", project=str(self.env.project))
            self.store.update_item(it["id"], status="done", difficulty="moderate", first_class="rapid",
                                   attempt=1 if i % 2 else 2, finished_at=time.time())
            self.store.save_retro(it["id"], "done", "underpowered" if i < 4 else "right", {})
        cls, why = f.class_for_difficulty("moderate")
        self.assertEqual(cls, "express")
        self.assertIn("underpowered", why)

    def test_adaptive_routing_moves_a_struggling_level_up(self):
        patch(self, settings, "ADAPTIVE_ROUTING", True)
        f = self.make()
        for i in range(6):  # small items on Local keep needing rework
            it = self.store.create_item(title=f"s{i}", project=str(self.env.project))
            self.store.update_item(it["id"], status="done", difficulty="small", first_class="local",
                                   attempt=3 if i < 4 else 1, finished_at=time.time())
        cls, why = f.class_for_difficulty("small")
        self.assertEqual(cls, "rapid")
        self.assertIn("33%", why)
        self.assertEqual(f.class_for_difficulty("moderate")[0], "rapid", "no history: default")
        patch(self, settings, "ADAPTIVE_ROUTING", False)
        self.assertEqual(f.class_for_difficulty("small")[0], "local")


class ErrorSlugTest(unittest.TestCase):
    def test_titles_lead_with_the_error(self):
        from yamanote.factory import _error_slug
        self.assertEqual(_error_slug("ERROR Traceback (most recent call last): ValueError: time data '2026-13-40' bad"),
                         "valueerror-time-data-bad")
        self.assertEqual(_error_slug("2026-10-06T17:01:02Z ERROR database connection refused on port 5432"),
                         "database-connection-refused-on-port")
        self.assertEqual(_error_slug("panic: runtime error: index out of range [3] with length 3"),
                         "panic-runtime-index-out-of-range-with-length")


class StoreMaintenanceTest(unittest.TestCase):
    def test_old_database_is_migrated(self):
        path = os.path.join(tempfile.mkdtemp(), "old.db")
        Store(path).close()
        db = sqlite3.connect(path)  # turn it back into a first-release database
        for table, col in (("items", "parent_id"), ("items", "holds"), ("items", "first_class"), ("runs", "cached_tokens")):
            db.execute(f"ALTER TABLE {table} DROP COLUMN {col}")
        db.commit()
        db.close()
        s = Store(path)
        cols = {r[1] for r in s._db.execute("PRAGMA table_info(items)")}
        self.assertTrue({"parent_id", "holds", "first_class"} <= cols)
        self.assertIn("cached_tokens", {r[1] for r in s._db.execute("PRAGMA table_info(runs)")})
        s.close()

    def test_prune_drops_detail_of_old_items_only(self):
        s = Store(":memory:")
        old = s.create_item(title="old", project="/p")
        new = s.create_item(title="new", project="/p")
        for it in (old, new):
            run = s.start_run(it["id"], "builder", "build", "m")
            s.add_step(run, "tool", "run", "x" * 100)
        s.update_item(old["id"], status="done", finished_at=time.time() - 40 * 86400)
        s.update_item(new["id"], status="done", finished_at=time.time())
        removed = s.prune(14)
        self.assertEqual(removed["steps"], 1)
        self.assertEqual(len(s.steps(s.runs(new["id"])[0]["id"])), 1)
        self.assertTrue(any(e["kind"] == "created" for e in s.events(old["id"])), "milestones kept")
        s.close()


if __name__ == "__main__":
    unittest.main()
