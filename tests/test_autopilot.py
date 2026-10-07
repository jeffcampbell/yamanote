"""Autopilot: supervised vs dark mode, the schedule, releasing gates, the report."""
from __future__ import annotations

import datetime as dt
import json
import os
import time

from tests.helpers import run_until, sh
from tests.test_factory import FactoryTestCase
from yamanote import settings


def at(day: int, hour: int, minute: int = 0) -> float:
    return dt.datetime(2026, 10, day, hour, minute).timestamp()


class ModeTest(FactoryTestCase):
    def test_supervised_holds_proposals_until_boarded(self):
        f = self.make()
        item = f.create_item("idea", "from the dispatcher", source="dispatcher")
        self.assertEqual((item["station"], item["status"], item["gate"]), ("intake", "waiting", "board"))
        for _ in range(5):
            f.tick()
        self.assertEqual(self.item(item["id"])["status"], "waiting", "nothing spent on unboarded work")
        self.assertEqual(self.client.calls, [])
        f.approve(item["id"])
        self.assertTrue(run_until(f, lambda: self.item(item["id"])["status"] == "done"))

    def test_declining_a_proposal(self):
        f = self.make()
        item = f.create_item("idea", "x", source="signal")
        f.reject(item["id"], "not now")
        self.assertEqual(self.item(item["id"])["status"], "rejected")
        self.assertIn("Declined by human", self.item(item["id"])["outcome"])

    def test_autopilot_boards_proposals_itself(self):
        f = self.make()
        f.set_autopilot(True)
        item = f.create_item("idea", "x", source="dispatcher")
        self.assertEqual((item["station"], item["status"]), ("triage", "queued"))

    def test_human_requests_never_wait_for_boarding(self):
        f = self.make()
        item = f.create_item("mine", "x")
        self.assertEqual(item["status"], "queued")

    def test_supervised_gates_come_from_settings_and_autopilot_skips_them(self):
        f = self.make()
        settings.TEST_CMD = "true"
        self.addCleanup(setattr, settings, "TEST_CMD", "")
        f.save_autopilot_config(supervised_gates={"spec": True, "merge": True})
        project = str(self.env.project)
        self.assertEqual(f.gates(project), {"spec": True, "merge": True})
        f.set_autopilot(True)
        self.assertEqual(f.gates(project), {"spec": False, "merge": False})

    def test_projects_json_gates_override_supervised_settings(self):
        f = self.make()
        settings.PROJECTS_CONFIG_PATH.write_text(json.dumps({"projects": {"p": {
            "path": str(self.env.project), "gates": {"merge": False}}}}))
        f.save_autopilot_config(supervised_gates={"spec": False, "merge": True})
        self.assertFalse(f.gates(str(self.env.project))["merge"])

    def test_untested_projects_wait_at_merge_unless_allowed(self):
        f = self.make()
        f.set_autopilot(True)
        project = str(self.env.project)
        self.assertTrue(f.gates(project)["merge"], "no test command → still gated in autopilot")
        f.save_autopilot_config(merge_without_tests=True)
        self.assertFalse(f.gates(project)["merge"])

    def test_autopilot_auto_reverts_regressions(self):
        f = self.make()
        self.assertFalse(settings.AUTO_REVERT)
        f.set_autopilot(True)
        f.store.kv_set(f"watch:{self.env.project}", {"item_id": 1, "title": "t", "commit": "abc",
                                                     "until": time.time() + 600, "baseline": []})
        submitted = []
        f._submit = lambda key, fn, *args: submitted.append(key)
        f._check_regression(str(self.env.project), ["ERROR new failure"])
        self.assertTrue(any(k.startswith("revert:") for k in submitted))


class ReleaseTest(FactoryTestCase):
    def test_turning_autopilot_on_releases_waiting_trains(self):
        f = self.make()
        settings.TEST_CMD = "true"
        self.addCleanup(setattr, settings, "TEST_CMD", "")
        proposal = f.create_item("idea", "x", source="dispatcher")
        spec_wait = f.create_item("a", "x")
        self.store.update_item(spec_wait["id"], station="spec", status="waiting", gate="spec")
        merge_wait = f.create_item("b", "x")
        self.store.update_item(merge_wait["id"], station="merge", status="waiting", gate="merge")
        f.set_autopilot(True)
        self.assertEqual((self.item(proposal["id"])["station"], self.item(proposal["id"])["status"]), ("triage", "queued"))
        self.assertEqual(self.item(spec_wait["id"])["station"], "build")
        self.assertEqual((self.item(merge_wait["id"])["status"], self.item(merge_wait["id"])["gate"]), ("queued", "approved"))

    def test_untested_merge_wait_stays_with_a_note(self):
        f = self.make()
        item = f.create_item("b", "x")
        self.store.update_item(item["id"], station="merge", status="waiting", gate="merge")
        f.set_autopilot(True)
        self.assertEqual(self.item(item["id"])["status"], "waiting")
        self.assertIn("no test command", self.store.events(item["id"])[-1]["message"])


class ScheduleTest(FactoryTestCase):
    def test_schedule_switches_and_manual_override_holds_until_next_change(self):
        f = self.make()
        f.save_autopilot_config(schedule_enabled=True, on_cron="0 22 * * *", off_cron="0 7 * * *")
        f._autopilot_tick(now=at(7, 23))           # in the night window → on
        self.assertTrue(f.autopilot_on)
        self.assertEqual(f.autopilot_state()["source"], "schedule")
        f.set_autopilot(False, now=at(7, 23, 30))  # human takes over
        f._autopilot_tick(now=at(7, 23, 45))
        self.assertFalse(f.autopilot_on, "manual switch holds")
        f._autopilot_tick(now=at(8, 7, 5))         # morning off: already off, stays off
        self.assertFalse(f.autopilot_on)
        f._autopilot_tick(now=at(8, 22, 1))        # next night → on again
        self.assertTrue(f.autopilot_on)
        f._autopilot_tick(now=at(9, 7, 0))
        self.assertFalse(f.autopilot_on)

    def test_new_schedule_applies_to_the_current_moment(self):
        f = self.make()
        f.set_autopilot(False)
        f.save_autopilot_config(schedule_enabled=True, on_cron="0 22 * * *", off_cron="0 7 * * *")
        f._autopilot_tick(now=at(8, 2))  # 2am: the latest scheduled event was 22:00 → on
        self.assertTrue(f.autopilot_on)

    def test_disabled_schedule_does_nothing(self):
        f = self.make()
        f.save_autopilot_config(on_cron="0 22 * * *", off_cron="0 7 * * *", schedule_enabled=False)
        f._autopilot_tick(now=at(7, 23))
        self.assertFalse(f.autopilot_on)

    def test_validation(self):
        f = self.make()
        with self.assertRaises(ValueError):
            f.save_autopilot_config(on_cron="25 * * * * *")
        with self.assertRaises(ValueError):
            f.save_autopilot_config(schedule_enabled=True, on_cron="0 22 * * *", off_cron="")
        self.assertEqual(f.schedule_preview()["next_on"], None)


class ReportTest(FactoryTestCase):
    def test_while_you_were_away_report(self):
        out = os.path.join(self.env.tmp.name, "notes.jsonl")
        settings.NOTIFY_CMD = f"cat >> {out}; echo >> {out}"
        self.addCleanup(setattr, settings, "NOTIFY_CMD", "")
        settings.TEST_CMD = "true"  # tested project: autopilot may merge it unattended
        self.addCleanup(setattr, settings, "TEST_CMD", "")
        f = self.make()
        f.set_autopilot(True)
        item = f.create_item("Add hello", "Create hello.py")
        self.assertTrue(run_until(f, lambda: self.item(item["id"])["status"] == "done"))
        waiting = f.create_item("later", "x", source="dispatcher")  # boards itself under autopilot
        self.store.update_item(waiting["id"], status="waiting", gate="spec", station="spec")
        f.set_autopilot(False)
        report = next(e for e in reversed(self.store.events()) if e["kind"] == "autopilot_report")
        self.assertEqual([a["id"] for a in report["data"]["arrived"]], [item["id"]])
        self.assertIn("1 arrived", report["message"])
        self.assertIn("1 train(s) now waiting", report["message"])
        deadline = time.time() + 5
        while time.time() < deadline and "while you were away" not in (open(out).read() if os.path.exists(out) else ""):
            time.sleep(0.05)
        events = [json.loads(l)["event"] for l in open(out).read().splitlines() if l.strip()]
        self.assertEqual(events, ["autopilot", "autopilot"])
