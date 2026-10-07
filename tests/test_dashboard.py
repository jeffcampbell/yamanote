"""Dashboard HTTP API tests against a live server on an ephemeral port."""
from __future__ import annotations

import json
import unittest
import urllib.error
import urllib.request

from tests.helpers import FakeClient, TempFactoryEnv, default_scripts, run_until
from yamanote import settings
from yamanote.dashboard import start_dashboard
from yamanote.factory import Factory
from yamanote.store import Store


class DashboardTest(unittest.TestCase):
    def setUp(self):
        self.env = TempFactoryEnv()
        self.factory = Factory(Store(":memory:"), client=FakeClient(default_scripts()))
        self.server = start_dashboard(self.factory, 0, "127.0.0.1")
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.factory.pool.shutdown(wait=True)
        self.factory.store.close()
        self.env.close()

    def get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=5) as r:
            body = r.read()
            return json.loads(body) if r.headers.get_content_type() == "application/json" else body.decode()

    def post(self, path, body=None, headers=None):
        h = {"Content-Type": "application/json", "X-Yamanote": "1", **(headers or {})}
        req = urllib.request.Request(self.base + path, json.dumps(body or {}).encode(), h, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_index_and_static(self):
        self.assertIn("YAMANOTE", self.get("/"))
        self.assertIn("EventSource", self.get("/static/app.js"))
        with self.assertRaises(urllib.error.HTTPError):
            self.get("/static/../settings.py")

    def test_state_shape(self):
        s = self.get("/api/state")
        self.assertEqual([st["code"] for st in s["stations"]][:2], ["JY01", "JY02"])
        self.assertEqual(len(s["trains"]), settings.MAX_TRAINS)
        for key in ("items", "arrivals", "budget", "stats", "service_classes", "jev", "projects"):
            self.assertIn(key, s)

    def test_create_item_requires_csrf_header(self):
        code, _ = self.post("/api/items", {"title": "x", "description": "y"}, headers={"X-Yamanote": ""})
        self.assertEqual(code, 403)

    def test_token_enforced_when_configured(self):
        settings.DASHBOARD_TOKEN = "s3cret"
        self.addCleanup(setattr, settings, "DASHBOARD_TOKEN", "")
        code, _ = self.post("/api/pause")
        self.assertEqual(code, 401)
        code, _ = self.post("/api/pause", headers={"Authorization": "Bearer s3cret"})
        self.assertEqual(code, 200)
        self.factory.set_paused(False)

    def test_token_protects_reads_too(self):
        settings.DASHBOARD_TOKEN = "s3cret"
        self.addCleanup(setattr, settings, "DASHBOARD_TOKEN", "")
        self.assertIn("YAMANOTE", self.get("/"), "the page shell loads so it can ask for the token")
        for path in ("/api/state", "/api/items/1", "/api/stream", "/metrics"):
            with self.assertRaises(urllib.error.HTTPError) as cm:
                self.get(path)
            self.assertEqual(cm.exception.code, 401, path)
        self.assertIn("items", self.get("/api/state?token=s3cret"))
        req = urllib.request.Request(self.base + "/api/state", headers={"Cookie": "yamanote_token=s3cret"})
        with urllib.request.urlopen(req, timeout=5) as r:
            self.assertEqual(r.status, 200)

    def test_diagram_retro_and_playbook_endpoints(self):
        code, r = self.post("/api/items", {"title": "Add hello", "description": "Create hello.py"})
        item_id = r["item"]["id"]
        self.assertTrue(run_until(self.factory, lambda: self.factory.store.get_item(item_id)["status"] == "done"))
        d = self.get("/api/diagram?hours=1")
        train = next(t for t in d["trains"] if t["id"] == item_id)
        stations = [p[1] for p in train["points"]]
        self.assertEqual(stations[0], "intake")
        self.assertIn("verify", stations)
        self.assertEqual(train["status"], "done")
        self.assertEqual(stations[-1], "retro", "the diagram follows trains into JY09")
        project = self.factory.store.get_item(item_id)["project"]
        self.factory._save_playbook(project, [{"id": 7, "station": "spec", "text": "a", "item_id": 1, "ts": 0,
                                               "uses": 0, "wins": 0}])
        state = self.get("/api/state")
        self.assertEqual(state["playbook"][project][0]["text"], "a")
        self.assertEqual(state["retros"][0]["item_id"], item_id)
        self.assertEqual(self.get(f"/api/items/{item_id}")["retro"]["outcome"], "done")
        code, r = self.post("/api/playbook/delete", {"project": project, "id": 7})
        self.assertEqual((code, r["playbook"]), (200, []))
        code, _ = self.post("/api/playbook/delete", {"project": project, "id": 99})
        self.assertEqual(code, 400)

    def test_item_lifecycle_through_api(self):
        code, r = self.post("/api/items", {"title": "Add hello", "description": "Create hello.py", "priority": "high"})
        self.assertEqual(code, 200)
        item_id = r["item"]["id"]
        self.assertTrue(run_until(self.factory, lambda: self.factory.store.get_item(item_id)["status"] == "done"))
        d = self.get(f"/api/items/{item_id}")
        self.assertEqual(d["item"]["status"], "done")
        self.assertTrue(d["item"]["scenarios"], "detail view shows scenarios to humans")
        self.assertTrue(any(e["kind"] == "verified" for e in d["events"]))
        run = next(r for r in d["runs"] if r["role"] == "builder")
        steps = self.get(f"/api/runs/{run['id']}/steps")["steps"]
        self.assertTrue(any(s["kind"] == "tool" and s["name"] == "write_file" for s in steps))
        arrivals = self.get("/api/state")["arrivals"]
        self.assertEqual(arrivals[0]["id"], item_id)
        self.assertNotIn("scenarios", arrivals[0], "list views only expose a scenario count")
        self.assertEqual(arrivals[0]["scenario_count"], 1)

    def test_autopilot_endpoints(self):
        state = self.get("/api/state")["autopilot"]
        self.assertFalse(state["on"])
        self.assertIn("supervised_gates", state["config"])
        code, r = self.post("/api/autopilot", {"on": True})
        self.assertEqual((code, r["autopilot"]["on"], r["autopilot"]["source"]), (200, True, "manual"))
        code, r = self.post("/api/autopilot/settings", {"schedule_enabled": True, "on_cron": "0 22 * * *",
                                                        "off_cron": "0 7 * * 1-5", "merge_without_tests": True})
        self.assertEqual(code, 200)
        self.assertTrue(r["preview"]["next_on"] and r["preview"]["next_off"])
        code, r = self.post("/api/autopilot/settings", {"on_cron": "61 * * * *"})
        self.assertEqual(code, 400)
        self.assertIn("out of range", r["error"])
        code, r = self.post("/api/autopilot", {"on": False})
        self.assertFalse(r["autopilot"]["on"])
        self.assertTrue(self.get("/api/state")["autopilot_report"]["message"].startswith("Autopilot ran"))

    def test_bad_requests(self):
        code, r = self.post("/api/items", {"title": "x", "description": "y", "project": "/etc"})
        self.assertEqual(code, 400)
        code, r = self.post("/api/items/999/approve")
        self.assertEqual(code, 400)

    def test_metrics(self):
        text = self.get("/metrics")
        self.assertIn("yamanote_spend_24h_usd", text)
        self.assertIn("yamanote_trains_active", text)


if __name__ == "__main__":
    unittest.main()
