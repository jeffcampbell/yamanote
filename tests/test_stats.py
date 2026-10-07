"""Stats view computations."""
from __future__ import annotations

import time
import unittest

from tests.helpers import run_until
from tests.test_factory import FactoryTestCase
from yamanote import stats
from yamanote.store import Store

DAY = 86400


def finished(store, *, status="done", ago=3600, journey=600, cost=0.2, attempt=1, project="/p", satisfaction=1.0):
    now = time.time()
    it = store.create_item(title="t", project=project)
    store.update_item(it["id"], status=status, attempt=attempt, started_at=now - ago - journey,
                      finished_at=now - ago, satisfaction=satisfaction)
    store.add_item_cost(it["id"], cost, 0, 0)
    return it


class ComputeTest(unittest.TestCase):
    def setUp(self):
        self.s = Store(":memory:")

    def tearDown(self):
        self.s.close()

    def test_kpis_and_previous_period(self):
        finished(self.s, ago=3600, journey=300)
        finished(self.s, ago=7200, journey=900, attempt=2)
        finished(self.s, status="failed", ago=3600)
        finished(self.s, ago=8 * DAY)  # previous 7-day period
        d = stats.compute(self.s, 7)
        k = d["kpis"]
        self.assertEqual((k["arrived"], k["failed"]), (2, 1))
        self.assertAlmostEqual(k["failure_rate"], 1 / 3)
        self.assertAlmostEqual(k["first_pass"], 1 / 3)
        self.assertEqual(k["journey_median"], 600)
        self.assertEqual(d["previous"]["arrived"], 1)
        self.assertEqual(d["bucket"], "day")
        self.assertEqual(sum(b["done"] for b in d["series"]), 2)

    def test_hourly_buckets_for_short_ranges(self):
        finished(self.s, ago=1800)
        d = stats.compute(self.s, 1)
        self.assertEqual(d["bucket"], "hour")
        self.assertGreaterEqual(len(d["series"]), 24)
        self.assertEqual(d["series"][-1]["done"] + d["series"][-2]["done"], 1)

    def test_spend_and_models(self):
        it = finished(self.s)
        now = time.time()
        for model, role, cost, cached in (("m/a", "builder", 0.30, 500), ("m/a", "inspector", 0.10, 0),
                                          ("m/b", "spec", 0.05, 0), ("shell", "ci", 0.0, 0), ("jev-latest", "jev", 0.001, 0)):
            r = self.s.start_run(it["id"], role, "build", model)
            self.s.add_step(r, "model", model, "x", tokens_in=1000, tokens_out=50, cost_usd=cost, cached_tokens=cached)
            self.s.finish_run(r, "ok")
        d = stats.compute(self.s, 7)
        self.assertAlmostEqual(d["kpis"]["spend"], 0.451, places=6)
        self.assertEqual([m["model"] for m in d["models"]], ["m/a", "m/b", "jev-latest"], "ci checks aren't a model")
        a = d["models"][0]
        self.assertEqual((a["runs"], a["roles"]), (2, ["builder", "inspector"]))
        self.assertAlmostEqual(a["cache_rate"], 0.25)
        self.assertAlmostEqual(sum(m["share"] for m in d["models"]), 1.0)
        self.assertEqual(d["kpis"]["jev_decisions"], 1)
        self.assertAlmostEqual(d["kpis"]["cache_rate"], 500 / 3000, msg="excludes Jev and checks")
        self.assertLess(now - d["now"], 5)

    def test_station_time_splits_working_and_waiting(self):
        now = time.time()
        it = self.s.create_item(title="t", project="/p")
        self.s.update_item(it["id"], status="done", started_at=now - 1000, finished_at=now - 100)
        # events are timestamped now; rewrite them onto a known timeline
        self.s._exec("DELETE FROM events")
        for ts, station in ((now - 1000, "build"), (now - 400, "verify")):
            self.s._exec("INSERT INTO events (item_id, ts, station, kind, message) VALUES (?,?,?,?,?)",
                         (it["id"], ts, station, "arrived", "x"))
        r = self.s.start_run(it["id"], "builder", "build", "m")
        self.s._exec("UPDATE runs SET started_at=?, ended_at=? WHERE id=?", (now - 900, now - 600, r))
        rows = {x["station"]: x for x in stats.compute(self.s, 7)["stations"]}
        self.assertAlmostEqual(rows["build"]["working_min"], 5.0)   # 300s run
        self.assertAlmostEqual(rows["build"]["waiting_min"], 5.0)   # 600s at build minus 300s working
        self.assertAlmostEqual(rows["verify"]["waiting_min"], 5.0)  # 300s, no runs

    def test_project_filter(self):
        finished(self.s, project="/a")
        finished(self.s, project="/b")
        self.assertEqual(stats.compute(self.s, 7, project="/a")["kpis"]["arrived"], 1)


class MergeStatTest(FactoryTestCase):
    def test_lines_and_commits_are_recorded_at_merge(self):
        f = self.make()
        item = f.create_item("Add hello", "Create hello.py")
        self.assertTrue(run_until(f, lambda: self.item(item["id"])["status"] == "done"))
        stat = self.store.kv_get(f"mergestat:{item['id']}")
        self.assertEqual((stat["files"], stat["insertions"], stat["commits"]), (1, 1, 1))
        d = stats.compute(self.store, 1)
        self.assertEqual((d["kpis"]["lines_added"], d["kpis"]["commits"], d["kpis"]["merges"]), (1, 1, 1))

    def test_old_trains_are_backfilled_from_git(self):
        f = self.make()
        item = f.create_item("Add hello", "Create hello.py")
        self.assertTrue(run_until(f, lambda: self.item(item["id"])["status"] == "done"))
        self.store._exec("DELETE FROM kv WHERE key=?", (f"mergestat:{item['id']}",))  # as if merged before this feature
        self.assertEqual(stats.merge_stat(self.store, self.item(item["id"]))["insertions"], 1)
        self.assertIsNotNone(self.store.kv_get(f"mergestat:{item['id']}"), "cached after the first lookup")


if __name__ == "__main__":
    unittest.main()


class SweepStatsTest(FactoryTestCase):
    def test_backfill_finds_merges_made_before_tracking(self):
        f = self.make()
        item = f.create_item("Add hello", "Create hello.py")
        self.assertTrue(run_until(f, lambda: self.item(item["id"])["status"] == "done"))
        self.store._exec("DELETE FROM kv WHERE key IN (?, ?)", (f"mergestat:{item['id']}", f"merged:{item['id']}"))
        self.assertEqual(stats.merge_stat(self.store, self.item(item["id"]))["insertions"], 1)

    def test_efficiency_counts_respect_range_and_project(self):
        s = self.store
        mine = s.create_item(title="a", project="/a")
        other = s.create_item(title="b", project="/b")
        s.event(mine["id"], "spec", "arrived", "Jev fast-pass: useful")
        s.event(other["id"], "spec", "arrived", "Jev fast-pass: useful")
        s.event(mine["id"], "triage", "held", "HOLD 1/2 — Jev: too vague")
        s._exec("UPDATE events SET ts = ts - 40*86400 WHERE item_id = ? AND kind = 'held'", (mine["id"],))
        for _ in range(6000):  # far more than any fixed recent-events window
            s.event(None, None, "notice", "noise")
        e = stats.compute(s, 7, project="/a")["efficiency"]
        self.assertEqual((e["jev_fast_pass"], e["jev_holds"]), (1, 0), "other project and old events excluded")
        self.assertEqual(stats.compute(s, 90, project="/a")["efficiency"]["jev_holds"], 1)
