"""OpenTelemetry export: payload shape and delivery against a fake OTLP endpoint."""
from __future__ import annotations

import json
import os
import re
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tests.helpers import run_until
from tests.test_factory import FactoryTestCase
from yamanote import telemetry
from yamanote.store import Store


class FakeCollector:
    """Records OTLP/HTTP JSON posts; can be told to fail."""

    def __init__(self):
        self.posts: list[tuple[str, dict, dict]] = []
        self.fail = False
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                if outer.fail:
                    self.send_response(503)
                    self.end_headers()
                    return
                outer.posts.append((self.path, json.loads(body), dict(self.headers)))
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b"{}")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def spans(self):
        return [s for path, body, _ in self.posts if path == "/v1/traces"
                for rs in body["resourceSpans"] for ss in rs["scopeSpans"] for s in ss["spans"]]

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def attr(obj, key):
    for a in obj["attributes"]:
        if a["key"] == key:
            return next(iter(a["value"].values()))
    return None


class ConfigTest(unittest.TestCase):
    def env(self, **kv):
        saved = {k: os.environ.get(k) for k in kv}
        os.environ.update({k: v for k, v in kv.items() if v is not None})
        self.addCleanup(lambda: [os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
                                 for k, v in saved.items()])

    def test_standard_env_vars(self):
        self.env(OTEL_EXPORTER_OTLP_ENDPOINT="http://collector:4318/", OTEL_SERVICE_NAME="yamanote-pi",
                 OTEL_EXPORTER_OTLP_HEADERS="Authorization=Bearer%20abc,x-team=ops",
                 OTEL_RESOURCE_ATTRIBUTES="deployment.environment=home", OTEL_METRIC_EXPORT_INTERVAL="30000")
        c = telemetry.config()
        self.assertTrue(c["enabled"])
        self.assertEqual((c["traces"], c["metrics"]), ("http://collector:4318/v1/traces", "http://collector:4318/v1/metrics"))
        self.assertEqual(c["headers"], {"Authorization": "Bearer abc", "x-team": "ops"})
        self.assertEqual((c["service"], c["interval"]), ("yamanote-pi", 30.0))
        self.assertEqual(c["resource"], {"deployment.environment": "home"})

    def test_off_by_default_and_unsupported_protocol(self):
        self.env(OTEL_EXPORTER_OTLP_ENDPOINT=None)
        os.environ.pop("OTEL_EXPORTER_OTLP_ENDPOINT", None)
        self.assertFalse(telemetry.config()["enabled"])
        self.env(OTEL_EXPORTER_OTLP_ENDPOINT="http://c:4318", OTEL_EXPORTER_OTLP_PROTOCOL="grpc")
        c = telemetry.config()
        self.assertFalse(c["enabled"])
        self.assertIn("http/json", c["problem"])


class TraceShapeTest(FactoryTestCase):
    def test_train_trace_follows_otlp_and_genai_conventions(self):
        f = self.make()
        item = f.create_item("Add hello", "Create hello.py")
        self.assertTrue(run_until(f, lambda: self.item(item["id"])["status"] == "done"))
        spans = telemetry.train_trace(self.store, self.item(item["id"]))
        ids = {s["spanId"] for s in spans}
        trace_ids = {s["traceId"] for s in spans}
        self.assertEqual(len(trace_ids), 1)
        self.assertRegex(next(iter(trace_ids)), r"^[0-9a-f]{32}$")
        for s in spans:
            self.assertRegex(s["spanId"], r"^[0-9a-f]{16}$")
            self.assertRegex(s["startTimeUnixNano"], r"^\d{19}$")
            self.assertLessEqual(int(s["startTimeUnixNano"]), int(s["endTimeUnixNano"]))
            if "parentSpanId" in s:
                self.assertIn(s["parentSpanId"], ids, "every parent exists in the trace")
        roots = [s for s in spans if "parentSpanId" not in s]
        self.assertEqual(len(roots), 1)
        self.assertEqual(attr(roots[0], "yamanote.item.status"), "done")
        names = [s["name"] for s in spans]
        self.assertTrue(any(n.startswith("station JY04") for n in names))
        agent = next(s for s in spans if s["name"] == "invoke_agent builder")
        self.assertEqual(attr(agent, "gen_ai.operation.name"), "invoke_agent")
        self.assertEqual(attr(agent, "gen_ai.provider.name"), "openrouter")
        self.assertEqual(attr(agent, "gen_ai.usage.input_tokens"), "200")
        self.assertTrue(any(s["name"] == "execute_tool write_file" and attr(s, "gen_ai.tool.name") == "write_file"
                            for s in spans))
        self.assertTrue(any(s["name"].startswith("chat ") and attr(s, "gen_ai.operation.name") == "chat" for s in spans))
        self.assertEqual(spans, telemetry.train_trace(self.store, self.item(item["id"])), "deterministic ids")

    def test_failed_train_has_error_status(self):
        f = self.make(verifier=lambda m, t: __import__("tests.helpers", fromlist=["finish"]).finish(
            "ran", {"results": [{"name": "s", "passed": False, "evidence": "broken"}]}))
        item = f.create_item("Add hello", "Create hello.py")
        self.assertTrue(run_until(f, lambda: self.item(item["id"])["status"] == "failed", max_ticks=400))
        root = next(s for s in telemetry.train_trace(self.store, self.item(item["id"])) if "parentSpanId" not in s)
        self.assertEqual(root["status"]["code"], telemetry.STATUS_ERROR)

    def test_metrics_payload(self):
        f = self.make()
        item = f.create_item("Add hello", "Create hello.py")
        self.assertTrue(run_until(f, lambda: self.item(item["id"])["status"] == "done"))
        m = telemetry.metrics_payload(telemetry.config() | {"service": "y", "resource": {}}, self.store, f)
        metrics = {x["name"]: x for x in m["resourceMetrics"][0]["scopeMetrics"][0]["metrics"]}
        for name in ("yamanote.trains.finished", "yamanote.llm.tokens", "yamanote.llm.cost", "yamanote.agent.runs",
                     "yamanote.train.journey.duration", "yamanote.merges", "yamanote.lines.changed",
                     "yamanote.trains.active", "yamanote.spend.today", "yamanote.autopilot"):
            self.assertIn(name, metrics)
        self.assertTrue(metrics["yamanote.trains.finished"]["sum"]["isMonotonic"])
        self.assertEqual(metrics["yamanote.trains.finished"]["sum"]["aggregationTemporality"], 2)
        h = metrics["yamanote.train.journey.duration"]["histogram"]["dataPoints"][0]
        self.assertEqual(len(h["bucketCounts"]), len(h["explicitBounds"]) + 1)
        self.assertEqual(h["count"], "1")


class ExporterTest(FactoryTestCase):
    def setUp(self):
        super().setUp()
        self.collector = FakeCollector()
        self.addCleanup(self.collector.close)
        self.cfg = {"traces": self.collector.base + "/v1/traces", "metrics": self.collector.base + "/v1/metrics",
                    "headers": {"x-token": "t"}, "service": "yamanote-test", "resource": {}, "interval": 60,
                    "enabled": True, "problem": None}

    def test_exports_finished_trains_once_and_survives_an_outage(self):
        f = self.make()
        exp = telemetry.Exporter(self.store, f, self.cfg)
        self.store.kv_set("otel_cursor", {"items": 0, "runs": 0})
        item = f.create_item("Add hello", "Create hello.py")
        self.assertTrue(run_until(f, lambda: self.item(item["id"])["status"] == "done"))
        self.collector.fail = True
        with self.assertRaises(RuntimeError):
            exp.flush()
        self.assertEqual(self.collector.spans(), [], "nothing accepted while the collector is down")
        self.collector.fail = False
        exp.flush(force_metrics=True)
        roots = [s for s in self.collector.spans() if "parentSpanId" not in s]
        self.assertEqual([attr(r, "yamanote.item.id") for r in roots], [str(item["id"])])
        exp.flush()
        self.assertEqual(len([s for s in self.collector.spans() if "parentSpanId" not in s]), 1, "not exported twice")
        paths = {p for p, _, _ in self.collector.posts}
        self.assertEqual(paths, {"/v1/traces", "/v1/metrics"})
        headers = {k.lower(): v for k, v in self.collector.posts[0][2].items()}  # HTTP header names are case-insensitive
        self.assertEqual((headers.get("x-token"), headers.get("content-type")), ("t", "application/json"))
        res = self.collector.posts[0][1]["resourceSpans"][0]["resource"]
        self.assertEqual(attr(res, "service.name"), "yamanote-test")

    def test_first_start_does_not_replay_history(self):
        f = self.make()
        item = f.create_item("Add hello", "Create hello.py")
        self.assertTrue(run_until(f, lambda: self.item(item["id"])["status"] == "done"))
        exp = telemetry.Exporter(self.store, f, self.cfg)
        exp.start()
        exp.stop()
        exp.flush()
        self.assertEqual(self.collector.spans(), [])


if __name__ == "__main__":
    unittest.main()


def _metric_sum(payload, name, **want):
    for m in payload["resourceMetrics"][0]["scopeMetrics"][0]["metrics"]:
        if m["name"] == name:
            total = 0
            for p in m["sum"]["dataPoints"]:
                a = {x["key"]: next(iter(x["value"].values())) for x in p["attributes"]}
                if all(a.get(k) == v for k, v in want.items()):
                    total += int(p.get("asInt", 0))
            return total
    return None


class SweepRegressionTest(FactoryTestCase):
    """Bugs found in the sweep of the stats/observability feature."""

    def _failing_then_passing(self):
        outcomes = {"fail": True}

        def verifier(m, t):
            return __import__("tests.helpers", fromlist=["finish"]).finish(
                "ran", {"results": [{"name": "s", "passed": not outcomes["fail"], "evidence": "b"}]})
        return outcomes, verifier

    def test_counters_never_decrease(self):
        outcomes, verifier = self._failing_then_passing()
        f = self.make(verifier=verifier)
        item = f.create_item("Add hello", "Create hello.py")
        self.assertTrue(run_until(f, lambda: self.item(item["id"])["status"] == "failed", max_ticks=400))
        cfg = {"service": "t", "resource": {}}
        before = _metric_sum(telemetry.metrics_payload(cfg, self.store), "yamanote.trains.finished",
                             **{"yamanote.item.status": "failed"})
        outcomes["fail"] = False
        f.retry(item["id"])
        self.assertTrue(run_until(f, lambda: self.item(item["id"])["status"] == "done", max_ticks=400))
        after = telemetry.metrics_payload(cfg, self.store)
        self.assertEqual(_metric_sum(after, "yamanote.trains.finished", **{"yamanote.item.status": "failed"}), before)
        self.assertEqual(_metric_sum(after, "yamanote.trains.finished", **{"yamanote.item.status": "done"}), 1)
        run = self.store.start_run(None, "dispatcher", "dispatcher", "m")
        self.assertEqual(_metric_sum(telemetry.metrics_payload(cfg, self.store), "yamanote.agent.runs",
                                     **{"yamanote.run.status": "running"}), 0, "in-progress runs aren't counted")
        self.store.finish_run(run, "ok")

    def test_retried_train_gets_a_fresh_trace_of_only_its_new_journey(self):
        outcomes, verifier = self._failing_then_passing()
        f = self.make(verifier=verifier)
        item = f.create_item("Add hello", "Create hello.py")
        self.assertTrue(run_until(f, lambda: self.item(item["id"])["status"] == "failed", max_ticks=400))
        first = telemetry.train_trace(self.store, self.item(item["id"]))
        outcomes["fail"] = False
        f.retry(item["id"])
        retried_at = max(e["ts"] for e in self.store.events(item["id"]) if e["kind"] == "retried")
        self.assertTrue(run_until(f, lambda: self.item(item["id"])["status"] == "done", max_ticks=400))
        second = telemetry.train_trace(self.store, self.item(item["id"]))
        self.assertNotEqual(first[0]["traceId"], second[0]["traceId"])
        self.assertTrue(all(int(s["startTimeUnixNano"]) >= int(retried_at * 1e9) - 1_000_000_000 for s in second))

    def test_redactor_spans_sit_under_verify(self):
        outcomes = iter([False, True])
        f = self.make(verifier=lambda m, t: __import__("tests.helpers", fromlist=["finish"]).finish(
            "ran", {"results": [{"name": "s", "passed": next(outcomes), "evidence": "b"}]}))
        item = f.create_item("Add hello", "Create hello.py")
        self.assertTrue(run_until(f, lambda: self.item(item["id"])["status"] == "done", max_ticks=400))
        spans = telemetry.train_trace(self.store, self.item(item["id"]))
        by_id = {s["spanId"]: s for s in spans}
        red = next(s for s in spans if attr(s, "yamanote.run.role") == "redactor")
        self.assertTrue(by_id[red["parentSpanId"]]["name"].startswith("station JY06"))

    def test_cursor_does_not_skip_ties_across_batches(self):
        s, now = self.store, time.time()
        for i in range(22):
            it = s.create_item(title=f"t{i}", project="/p")
            s.update_item(it["id"], status="done", finished_at=now - 100 if i >= 19 else now - 200 + i)
        s.kv_set("otel_cursor", {"items": 0, "runs": 0})
        sent = []
        exp = telemetry.Exporter(s, None, {"traces": "x", "metrics": "", "headers": {}, "service": "t", "resource": {},
                                           "interval": 60, "enabled": True, "problem": None})
        exp._post = lambda url, body: sent.extend(
            sp for rs in body["resourceSpans"] for ss in rs["scopeSpans"] for sp in ss["spans"] if "parentSpanId" not in sp)
        for _ in range(3):
            exp.flush()
        self.assertEqual(len(sent), 22)

    def test_every_run_appears_when_the_journey_takes_real_time(self):
        """Test journeys finish in milliseconds; real ones take minutes. Spread the
        timeline out so time-window logic is exercised the way real data does."""
        f = self.make()
        item = f.create_item("Add hello", "Create hello.py")
        self.assertTrue(run_until(f, lambda: self.item(item["id"])["status"] == "done"))
        stretch = """UPDATE {t} SET {c} = {c} + ({c} - (SELECT created_at FROM items WHERE id = {i})) * 600
                     WHERE {w}"""
        i = item["id"]
        self.store._exec(stretch.format(t="events", c="ts", i=i, w=f"item_id = {i}"))
        for col in ("started_at", "ended_at"):
            self.store._exec(stretch.format(t="runs", c=col, i=i, w=f"item_id = {i} AND {col} IS NOT NULL"))
        self.store._exec(stretch.format(t="steps", c="ts", i=i, w=f"run_id IN (SELECT id FROM runs WHERE item_id = {i})"))
        self.store._exec(stretch.format(t="items", c="finished_at", i=i, w=f"id = {i}"))
        spans = telemetry.train_trace(self.store, self.item(i))
        run_spans = [s for s in spans if attr(s, "yamanote.run.id") is not None or s["name"].startswith("check ")]
        self.assertEqual(len(run_spans), len(self.store.runs(i)), "one span per run, none dropped")

    def test_status_keeps_urls_and_counts_apart(self):
        exp = telemetry.Exporter(self.store, None, {"traces": "http://c/v1/traces", "metrics": "http://c/v1/metrics",
                                                    "headers": {}, "service": "t", "resource": {}, "interval": 60,
                                                    "enabled": True, "problem": None})
        st = exp.status()
        self.assertEqual((st["traces_url"], st["traces_sent"]), ("http://c/v1/traces", 0))
