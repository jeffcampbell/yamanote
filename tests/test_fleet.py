"""Model fleet: catalogue filtering, price bands, trials, promotion, replacement."""
from __future__ import annotations

import time
import unittest

from tests.helpers import run_until
from tests.test_factory import FactoryTestCase
from tests.test_review_fixes import patch
from yamanote import fleet, notify, settings
from yamanote.store import Store

TOOLS = ["tools", "response_format", "max_tokens"]


def model(mid, prompt, completion, *, cache=None, params=TOOLS, context=200_000, expires=None, **extra):
    pricing = {"prompt": str(prompt / 1e6), "completion": str(completion / 1e6)}
    if cache is not None:
        pricing["input_cache_read"] = str(cache / 1e6)
    return {"id": mid, "name": mid, "pricing": pricing, "supported_parameters": params,
            "context_length": context, "expiration_date": expires,
            "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]}, **extra}


def catalog(*models, ranked=()):
    return lambda: {"fetched_at": time.time(), "models": list(models), "programming": list(ranked)}


# At the default mix (5% output, half of input cached) a model priced p/c with no cache
# price blends to p + 0.05c per million input tokens.
CHEAP = model("cheap/default", 0.10, 0.20)           # 0.11  → Local
CHEAP_B = model("cheap/challenger", 0.08, 0.20)      # 0.09  → Local
CHEAP_C = model("cheap/unranked", 0.05, 0.10)        # 0.055 → Local, not ranked
MID = model("mid/default", 0.30, 1.00)               # 0.35  → Rapid
TOP = model("top/default", 1.00, 4.00)               # 1.20  → Limited Express
BEST = model("best/default", 3.00, 10.00)            # 3.50  → Shinkansen
JEV = model("typesafe/jev-router", -1, -1, params=["tools"])


class CatalogTest(unittest.TestCase):
    def test_eligibility(self):
        self.assertIsNone(fleet.ineligible(CHEAP))
        self.assertIsNone(fleet.ineligible(JEV), "Jev Router is opted in despite having no fixed price")
        cases = {
            "no tool calls": model("a/b", 1, 1, params=["response_format"]),
            "no structured output": model("a/b", 1, 1, params=["tools"]),
            "context too small": model("a/b", 1, 1, context=32_000),
            "variant (free, batch, …)": model("a/b:free", 0, 0),
            "moving alias": model("~a/latest", 1, 1),
            "router": model("openrouter/auto", -1, -1),
            "free": model("a/free", 0, 0),
        }
        for reason, m in cases.items():
            self.assertEqual(fleet.ineligible(m), reason, m["id"])
        soon = time.strftime("%Y-%m-%d", time.localtime(time.time() + 10 * 86400))
        later = time.strftime("%Y-%m-%d", time.localtime(time.time() + 90 * 86400))
        self.assertIn("retiring", fleet.ineligible(model("a/b", 1, 1, expires=soon)))
        self.assertIsNone(fleet.ineligible(model("a/b", 1, 1, expires=later)))

    def test_blended_price_and_bands(self):
        m = model("a/b", 1.0, 10.0, cache=0.1)
        self.assertAlmostEqual(fleet.blended_price(m, (0.05, 0.5)), 0.5 + 0.05 + 0.5)
        self.assertAlmostEqual(fleet.blended_price(model("a/b", 1.0, 10.0), (0.05, 0.5)), 1.5, msg="no cache price")
        self.assertEqual([fleet.band_of(p) for p in (0.1, 0.5, 1.2, 9.0, None)],
                         ["local", "rapid", "express", "shinkansen", None])

    def test_token_mix_comes_from_real_runs(self):
        s = Store(":memory:")
        self.addCleanup(s.close)
        self.assertEqual(fleet.token_mix(s), (0.05, 0.5), "defaults before there is history")
        it = s.create_item(title="t", project="/p")
        r = s.start_run(it["id"], "builder", "build", "m")
        s.add_step(r, "model", "m", "x", tokens_in=1_000_000, tokens_out=40_000, cost_usd=0.1, cached_tokens=600_000)
        s.finish_run(r, "ok")
        out, cached = fleet.token_mix(s)
        self.assertAlmostEqual(out, 0.04)
        self.assertAlmostEqual(cached, 0.6)


class FleetTest(unittest.TestCase):
    def setUp(self):
        self.store = Store(":memory:")
        self.addCleanup(self.store.close)
        patch(self, settings, "FLEET_ENABLED", True)
        patch(self, settings, "TRIAL_RATE", 1.0)
        patch(self, settings, "PINNED_CLASSES", set())
        self.saved_models = {c: dict(v) for c, v in settings.SERVICE_CLASSES.items()}
        self.addCleanup(lambda: [settings.SERVICE_CLASSES[c].update(v) for c, v in self.saved_models.items()])
        patch(self, settings, "DEFAULT_CLASS_MODELS", {"local": CHEAP["id"], "rapid": MID["id"],
                                                       "express": TOP["id"], "shinkansen": BEST["id"]})

    def make(self, *models, ranked=()):
        f = fleet.Fleet(self.store, fetch=catalog(*models, ranked=ranked), rng=lambda: 0.0)
        f.refresh()
        return f

    def finish(self, f, model_id, n, *, first_pass, cost=0.10, trial=True, level="small", cls="local"):
        for i in range(n):
            it = self.store.create_item(title=f"{model_id}-{i}", project="/p")
            attempt = 1 if i < first_pass else 2
            self.store.update_item(it["id"], status="done", difficulty=level, first_class=cls, service_class=cls,
                                   attempt=attempt, finished_at=time.time(), trial_model=model_id if trial else None)
            self.store.add_item_cost(it["id"], cost, 0, 0)
            r = self.store.start_run(it["id"], "builder", "build", model_id)
            self.store.finish_run(r, "ok")

    def test_challengers_are_ranked_models_in_the_band_plus_jev_router(self):
        f = self.make(CHEAP, CHEAP_B, CHEAP_C, MID, TOP, BEST, JEV, ranked=[CHEAP_B["id"], MID["id"]])
        self.assertEqual(f.challengers("local"), [CHEAP_B["id"], JEV["id"]], "unranked models aren't tried")
        self.assertEqual(f.challengers("shinkansen"), [], "Jev Router is never a Shinkansen challenger")
        self.assertEqual(settings.SERVICE_CLASSES["local"]["model"], CHEAP["id"])

    def test_trials_only_on_low_risk_work_and_least_tried_first(self):
        f = self.make(CHEAP, CHEAP_B, JEV, ranked=[CHEAP_B["id"]])
        item = {"service_class": "local", "difficulty": "small"}
        self.assertEqual(f.assign_trial(item), CHEAP_B["id"])
        self.assertIsNone(f.assign_trial({**item, "difficulty": "moderate"}))
        self.finish(f, CHEAP_B["id"], 2, first_pass=2)
        self.assertEqual(f.assign_trial(item), JEV["id"], "the least-tried challenger goes next")
        settings.TRIAL_RATE = 0.10
        f.rng = lambda: 0.5
        self.assertIsNone(f.assign_trial(item), "only TRIAL_RATE of trains")
        f.rng = lambda: 0.05
        self.assertIsNotNone(f.assign_trial(item))
        settings.PINNED_CLASSES.add("local")
        f.rng = lambda: 0.0
        self.assertIsNone(f.assign_trial(item), "pinned classes don't run trials")

    def test_trial_model_applies_to_spec_and_first_build_only(self):
        item = {"trial_model": "x/y", "first_class": "local", "service_class": "local", "attempt": 1}
        self.assertEqual(fleet.Fleet.trial_model(item, "build", "local"), "x/y")
        self.assertEqual(fleet.Fleet.trial_model({**item, "attempt": 0}, "spec", "local"), "x/y")
        self.assertIsNone(fleet.Fleet.trial_model({**item, "attempt": 2}, "build", "local"), "rework → class model")
        self.assertIsNone(fleet.Fleet.trial_model(item, "verify", "rapid"), "never the verifier")
        self.assertIsNone(fleet.Fleet.trial_model(item, "inspect", "local"))
        self.assertIsNone(fleet.Fleet.trial_model({**item, "service_class": "rapid"}, "build", "rapid"),
                          "an escalated train is off the trial")

    def test_promotes_a_challenger_that_is_as_good_for_less(self):
        f = self.make(CHEAP, CHEAP_B, ranked=[CHEAP_B["id"]])
        self.finish(f, CHEAP["id"], 8, first_pass=6, cost=0.20, trial=False)
        self.finish(f, CHEAP_B["id"], 7, first_pass=7, cost=0.10)
        self.assertEqual(f.evaluate(), [], "not enough trials yet")
        self.finish(f, CHEAP_B["id"], 1, first_pass=1, cost=0.10)
        [change] = f.evaluate()
        self.assertEqual((change["kind"], change["model"], change["previous"]), ("promoted", CHEAP_B["id"], CHEAP["id"]))
        self.assertEqual(settings.SERVICE_CLASSES["local"]["model"], CHEAP_B["id"])
        self.assertEqual(fleet.Fleet(self.store, fetch=catalog()).champions()["local"], CHEAP_B["id"],
                         "survives a restart")

    def test_keeps_the_class_model_when_the_challenger_costs_more_and_benches_a_loser(self):
        f = self.make(CHEAP, CHEAP_B, JEV, ranked=[CHEAP_B["id"]])
        self.finish(f, CHEAP["id"], 8, first_pass=7, cost=0.10, trial=False)
        self.finish(f, CHEAP_B["id"], 8, first_pass=8, cost=0.30)
        self.finish(f, JEV["id"], 8, first_pass=3, cost=0.05)
        changes = f.evaluate()
        self.assertEqual([(c["kind"], c["model"]) for c in changes], [("benched", JEV["id"])])
        self.assertEqual(settings.SERVICE_CLASSES["local"]["model"], CHEAP["id"])
        self.assertNotIn(JEV["id"], f.challengers("local"))

    def test_replaces_a_retired_model_with_the_best_ranked_one(self):
        f = self.make(CHEAP, MID, TOP, BEST, ranked=[])
        f.fetch = catalog(CHEAP_B, CHEAP_C, MID, TOP, BEST, ranked=[CHEAP_B["id"]])
        [change] = f.refresh()
        self.assertEqual((change["kind"], change["model"]), ("replaced", CHEAP_B["id"]))
        self.assertIn("no longer on OpenRouter", change["message"])
        self.assertEqual(settings.SERVICE_CLASSES["local"]["model"], CHEAP_B["id"])

    def test_no_ranked_replacement_keeps_the_model_and_says_so_once(self):
        soon = time.strftime("%Y-%m-%d", time.localtime(time.time() + 5 * 86400))
        f = self.make(model(CHEAP["id"], 0.1, 0.2, expires=soon), CHEAP_C, MID, TOP, BEST)
        self.assertEqual(settings.SERVICE_CLASSES["local"]["model"], CHEAP["id"])
        self.assertEqual(f.refresh(), [], "reported on the first refresh, not every day")
        history = self.store.kv_get("fleet:history")
        self.assertEqual([(h["kind"], h["class"]) for h in history], [("stuck", "local")])

    def test_repricing_out_of_the_band(self):
        f = self.make(CHEAP, CHEAP_B, MID, TOP, BEST, ranked=[CHEAP_B["id"]])
        f.fetch = catalog(model(CHEAP["id"], 0.17, 0.2), CHEAP_B, MID, TOP, BEST, ranked=[CHEAP_B["id"]])
        self.assertEqual(f.refresh(), [], "a little over the ceiling is tolerated")
        f.fetch = catalog(model(CHEAP["id"], 0.5, 1.0), CHEAP_B, MID, TOP, BEST, ranked=[CHEAP_B["id"]])
        [change] = f.refresh()
        self.assertIn("repriced", change["message"])

    def test_pinned_classes_never_change(self):
        settings.PINNED_CLASSES.add("local")
        f = self.make(CHEAP_B, MID, TOP, BEST, ranked=[CHEAP_B["id"]])
        self.assertEqual(settings.SERVICE_CLASSES["local"]["model"], CHEAP["id"])
        self.assertEqual(f.challengers("local"), [])

    def test_catalogue_errors_keep_the_last_good_copy(self):
        f = self.make(CHEAP, CHEAP_B, MID, TOP, BEST, ranked=[CHEAP_B["id"]])
        def down():
            raise OSError("network down")
        f.fetch = down
        self.assertEqual(f.refresh(), [])
        self.assertIn("network down", f.status()["error"])
        self.assertEqual(f.challengers("local"), [CHEAP_B["id"]])


class FleetInFactoryTest(FactoryTestCase):
    def test_a_trial_train_builds_on_the_challenger_and_reports_routing_changes(self):
        patch(self, settings, "FLEET_ENABLED", True)
        patch(self, settings, "TRIAL_RATE", 1.0)
        local = settings.SERVICE_CLASSES["local"]["model"]
        challenger = model("cheap/challenger", 0.02, 0.05)
        f = self.make()
        f.fleet.fetch = catalog(model(local, 0.1, 0.2), challenger, ranked=[challenger["id"]])
        f.fleet.rng = lambda: 0.0
        f.fleet.refresh()
        sent = []
        patch(self, notify, "send", lambda event, title, message, *a, **k: sent.append((event, message)))
        item = f.create_item("Add hello", "Create hello.py")  # the fake spec writer calls it small → Local
        self.assertTrue(run_until(f, lambda: self.item(item["id"])["status"] == "done"))
        self.assertEqual(self.item(item["id"])["trial_model"], challenger["id"])
        models = {role: m for role, m in self.client.calls}
        self.assertEqual(models["builder"], challenger["id"])
        self.assertNotEqual(models["verifier"], challenger["id"])
        self.assertTrue(any(e["kind"] == "trial" for e in self.store.events(item["id"])))
        # the job runs on the tick: make it due and give it something to report
        self.store.kv_set("fleet:refreshed", 0)
        f.fleet.fetch = catalog(challenger, ranked=[challenger["id"]])  # the Local model was retired
        self.assertTrue(run_until(f, lambda: any(e == "routing" for e, _ in sent)))
        self.assertTrue(any(e["kind"] == "routing" for e in self.store.events(limit=50)))


if __name__ == "__main__":
    unittest.main()
