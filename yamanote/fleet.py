"""The model fleet: which OpenRouter model each service class runs on.

Static picks go stale: new models arrive weekly and old ones are retired. The
fleet keeps each class current without anyone editing a table:

1. Catalogue. Once a day, OpenRouter's public model list is read and filtered
   to models this line can use (tool calls, structured output, enough context,
   not a free variant, not about to be retired). OpenRouter's programming
   ranking, which orders models by how much developers use them, decides
   which candidates are worth trying first.
2. Price bands. Each class is a price ceiling (settings.CLASS_BANDS), applied
   to a blended price at this line's real mix of input, cached input and
   output tokens. New models land in a band; repriced ones move.
3. Trials. A share of low-risk trains (settings.TRIAL_RATE of trivial/small
   work) try a challenger from their class's band for their spec and first
   build. A rework goes back to the class's model.
4. Promotion. A challenger with enough trials that arrived first time at least
   as often as the class's model, for no more per arrival, becomes the
   class's model. One that does clearly worse sits out for a while.
5. Replacement. A class whose model leaves the catalogue, nears retirement or
   is repriced well above its band gets the best-ranked model in the band.

Classes pinned in models.json never change. Every change is recorded as a
'routing' event and sent through the notification hooks.
"""
from __future__ import annotations

import json
import logging
import random
import time
import urllib.request

from . import settings
from .store import Store

log = logging.getLogger("yamanote.fleet")

ROUTERS = ("openrouter/",)  # meta-models that pick another model; never candidates (Jev Router is opted in)


# ─── catalogue ──────────────────────────────────────────────────────────────

def _get(url: str, timeout: float = 30) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "yamanote"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def fetch_catalog(url: str | None = None) -> dict:
    """OpenRouter's model list plus its programming ranking (no API key needed)."""
    url = url or settings.CATALOG_URL
    models = _get(url)["data"]
    try:
        ranked = [m["id"] for m in _get(url + "?category=programming")["data"]]
    except Exception as e:  # the ranking is a nice-to-have
        log.warning("programming ranking unavailable: %s", e)
        ranked = []
    return {"fetched_at": time.time(), "models": models, "programming": ranked}


def _price(pricing: dict, key: str) -> float | None:
    try:
        v = float(pricing.get(key))
    except (TypeError, ValueError):
        return None
    return v * 1e6 if v >= 0 else None  # per million tokens; routers report -1


def ineligible(m: dict, now: float | None = None) -> str | None:
    """Why a catalogue entry can't serve this line, or None if it can."""
    mid = m.get("id") or ""
    if mid == settings.JEV_ROUTER_MODEL:
        return None if "tools" in (m.get("supported_parameters") or []) else "no tool calls"
    if ":" in mid:
        return "variant (free, batch, …)"
    if mid.startswith("~"):
        return "moving alias"
    if mid.startswith(ROUTERS):
        return "router"
    params = m.get("supported_parameters") or []
    if "tools" not in params:
        return "no tool calls"
    if not {"response_format", "structured_outputs"} & set(params):
        return "no structured output"
    arch = m.get("architecture") or {}
    if "text" not in (arch.get("output_modalities") or ["text"]) or "text" not in (arch.get("input_modalities") or ["text"]):
        return "not a text model"
    if (m.get("context_length") or 0) < settings.MIN_CONTEXT_TOKENS:
        return "context too small"
    pricing = m.get("pricing") or {}
    if _price(pricing, "prompt") is None or _price(pricing, "completion") is None:
        return "no price"
    if not (_price(pricing, "prompt") or _price(pricing, "completion")):
        return "free"
    if (exp := m.get("expiration_date")) and _expires_soon(exp, now):
        return f"retiring {exp}"
    return None


def _expires_soon(date: str, now: float | None = None) -> bool:
    try:
        t = time.mktime(time.strptime(str(date)[:10], "%Y-%m-%d"))
    except ValueError:
        return False
    return t - (now or time.time()) < settings.EXPIRY_MARGIN_DAYS * 86400


def token_mix(store: Store, days: int = 30) -> tuple[float, float]:
    """(output tokens per input token, share of input served from cache) for
    this line's model runs; sensible defaults before there is history."""
    row = store._one("SELECT SUM(tokens_in) AS i, SUM(tokens_out) AS o, SUM(cached_tokens) AS c FROM runs"
                     " WHERE role NOT IN ('ci','jev') AND started_at > ?", (time.time() - days * 86400,))
    i = (row or {}).get("i") or 0
    if i < 200_000:
        return 0.05, 0.5
    return (row["o"] or 0) / i, min(1.0, (row["c"] or 0) / i)


def blended_price(m: dict, mix: tuple[float, float]) -> float | None:
    """USD per million input tokens, with output and cache reads folded in."""
    pricing = m.get("pricing") or {}
    prompt, completion = _price(pricing, "prompt"), _price(pricing, "completion")
    if prompt is None or completion is None:
        return None
    cache_read = _price(pricing, "input_cache_read")
    out_ratio, cached = mix
    read = cache_read if cache_read is not None else prompt
    return prompt * (1 - cached) + read * cached + completion * out_ratio


def band_of(price: float | None) -> str | None:
    if price is None:
        return None
    for cls in settings.CLASS_ORDER:
        ceiling = settings.CLASS_BANDS.get(cls)
        if ceiling is None or price <= ceiling:
            return cls
    return None


# ─── the fleet ──────────────────────────────────────────────────────────────

class Fleet:
    """Per-class model choice, trials and promotion. State lives in the
    store's kv table so it survives restarts and is shared with the dashboard."""

    def __init__(self, store: Store, fetch=fetch_catalog, rng=random.random):
        self.store = store
        self.fetch = fetch
        self.rng = rng
        self.defaults = dict(settings.DEFAULT_CLASS_MODELS)
        self.apply()

    # state -------------------------------------------------------------

    def champions(self) -> dict[str, str]:
        saved = self.store.kv_get("fleet:champions", {}) or {}
        return {c: (self.defaults[c] if c in settings.PINNED_CLASSES else saved.get(c) or self.defaults[c])
                for c in settings.CLASS_ORDER}

    def apply(self) -> None:
        """Point each service class at its current model."""
        for cls, model in self.champions().items():
            settings.SERVICE_CLASSES[cls]["model"] = model

    def _set_champion(self, cls: str, model: str) -> None:
        saved = self.store.kv_get("fleet:champions", {}) or {}
        saved[cls] = model
        self.store.kv_set("fleet:champions", saved)
        since = self.store.kv_get("fleet:since", {}) or {}
        since[cls] = time.time()
        self.store.kv_set("fleet:since", since)
        self.apply()

    def catalog(self) -> dict:
        return self.store.kv_get("fleet:catalog", {}) or {}

    def candidates(self, cls: str) -> list[dict]:
        """Eligible catalogue models in this class's band, best-ranked first.
        Each is {id, name, price, rank}; rank is None when unranked."""
        cat = self.catalog()
        ranked = {mid: i for i, mid in enumerate(cat.get("programming") or [])}
        out = []
        for m in cat.get("models") or []:
            # the catalogue was filtered when it was stored; only retirement dates move on
            if m["id"] == settings.JEV_ROUTER_MODEL or (m.get("expires") and _expires_soon(m["expires"])):
                continue
            price = m.get("price")
            if band_of(price) == cls:
                out.append({"id": m["id"], "name": m.get("name") or m["id"], "price": price,
                            "rank": ranked.get(m["id"])})
        return sorted(out, key=lambda c: (c["rank"] is None, c["rank"] if c["rank"] is not None else 0,
                                          -(c["price"] or 0)))

    def challengers(self, cls: str) -> list[str]:
        if cls in settings.PINNED_CLASSES or not self.catalog():
            return []
        champion = self.champions()[cls]
        benched = self.store.kv_get("fleet:benched", {}) or {}
        now = time.time()
        out = [c["id"] for c in self.candidates(cls)
               if c["rank"] is not None and c["id"] != champion and benched.get(f"{cls}|{c['id']}", 0) < now]
        out = out[:settings.CHALLENGERS_PER_CLASS]
        jev = any(m["id"] == settings.JEV_ROUTER_MODEL for m in self.catalog().get("models") or [])
        if (jev and cls != "shinkansen" and champion != settings.JEV_ROUTER_MODEL
                and benched.get(f"{cls}|{settings.JEV_ROUTER_MODEL}", 0) < now):
            out.append(settings.JEV_ROUTER_MODEL)
        return out

    # trials ------------------------------------------------------------

    def assign_trial(self, item: dict) -> str | None:
        """Maybe pick a challenger for this train; the least-tried one first."""
        cls = item.get("service_class")
        if (not settings.FLEET_ENABLED or item.get("trial_model") or cls is None
                or item.get("difficulty") not in settings.TRIAL_LEVELS or self.rng() >= settings.TRIAL_RATE):
            return None
        options = self.challengers(cls)
        if not options:
            return None
        counts = {r["model"]: r["n"] for r in self.trial_stats(cls)}
        return min(options, key=lambda m: (counts.get(m, 0), options.index(m)))

    @staticmethod
    def trial_model(item: dict | None, station: str, cls: str | None) -> str | None:
        """The challenger this run should use instead of the class's model, if any."""
        if (not item or not item.get("trial_model") or station not in settings.TRIAL_STATIONS
                or cls != item.get("first_class") or (item.get("attempt") or 0) > 1):
            return None
        return item["trial_model"]

    def _outcomes(self, cls: str, days: int = 60) -> list[dict]:
        marks = ",".join("?" * len(settings.TRIAL_LEVELS))
        return self.store._all(
            "SELECT i.trial_model, (SELECT r.model FROM runs r WHERE r.item_id=i.id AND r.role='builder'"
            " ORDER BY r.id LIMIT 1) AS builder, i.status, i.attempt, i.cost_usd FROM items i"
            f" WHERE i.first_class=? AND i.difficulty IN ({marks}) AND i.status IN ('done','failed')"
            " AND i.attempt>0 AND i.finished_at>?",
            (cls, *settings.TRIAL_LEVELS, time.time() - days * 86400))

    @staticmethod
    def _summarize(rows: list[dict]) -> dict:
        n = len(rows)
        arrived = [r for r in rows if r["status"] == "done"]
        first = sum(1 for r in arrived if (r["attempt"] or 0) <= 1)
        cost = sum(r["cost_usd"] or 0 for r in rows)
        return {"n": n, "first_pass": first, "rate": first / n if n else None,
                "cost_per_arrival": cost / len(arrived) if arrived else None}

    def trial_stats(self, cls: str) -> list[dict]:
        rows = self._outcomes(cls)
        by: dict[str, list] = {}
        for r in rows:
            if r["trial_model"]:
                by.setdefault(r["trial_model"], []).append(r)
        return [{"model": m, **self._summarize(rs)} for m, rs in by.items()]

    def champion_stats(self, cls: str) -> dict:
        champion = self.champions()[cls]
        return self._summarize([r for r in self._outcomes(cls) if not r["trial_model"] and r["builder"] == champion])

    # daily refresh and hourly evaluation ---------------------------------

    def refresh(self) -> list[dict]:
        """Re-read the catalogue and replace any class model that can no longer
        serve. Returns the changes made (also recorded for the dashboard)."""
        try:
            raw = self.fetch()
        except Exception as e:
            cat = self.catalog()
            cat["error"] = f"{type(e).__name__}: {e}"
            cat["error_at"] = time.time()
            self.store.kv_set("fleet:catalog", cat)
            log.warning("model catalogue unavailable: %s", e)
            return []
        mix = token_mix(self.store)
        models = []
        for m in raw["models"]:
            reason = ineligible(m)
            if reason is None:
                models.append({"id": m["id"], "name": m.get("name"), "price": blended_price(m, mix),
                               "context": m.get("context_length"), "created": m.get("created"),
                               "expires": m.get("expiration_date"), "pricing": m.get("pricing")})
        known = {m["id"]: m for m in raw["models"]}
        self.store.kv_set("fleet:catalog", {"fetched_at": raw["fetched_at"], "models": models,
                                            "programming": raw.get("programming") or [], "mix": list(mix),
                                            "total": len(raw["models"]), "error": None})
        changes = []
        champions = self.champions()
        for cls in settings.CLASS_ORDER:
            if cls in settings.PINNED_CLASSES:
                continue
            model = champions[cls]
            entry = known.get(model)
            if entry is None:
                why = "no longer on OpenRouter"
            elif (reason := ineligible(entry)) is not None:
                why = reason
            else:
                price, ceiling = blended_price(entry, mix), settings.CLASS_BANDS.get(cls)
                why = (f"repriced to ${price:.2f}/M, above the {settings.SERVICE_CLASSES[cls]['label']} band"
                       if price is not None and ceiling is not None and price > ceiling * 1.25 else None)
            if not why:
                continue
            # only a model developers actually use takes over unattended
            best = next((c for c in self.candidates(cls) if c["id"] != model and c["rank"] is not None), None)
            if not best:
                stuck = self.store.kv_get("fleet:stuck", {}) or {}
                if stuck.get(cls) == model:
                    continue  # already reported
                stuck[cls] = model
                self.store.kv_set("fleet:stuck", stuck)
                changes.append({"kind": "stuck", "class": cls, "model": model,
                                "message": f"{settings.SERVICE_CLASSES[cls]['label']}: {model} is {why}, and no "
                                           "programming-ranked model is in its price band; keeping it — set one "
                                           "in models.json"})
                continue
            self._set_champion(cls, best["id"])
            changes.append({"kind": "replaced", "class": cls, "model": best["id"], "previous": model,
                            "message": f"{settings.SERVICE_CLASSES[cls]['label']}: {model} is {why}; "
                                       f"now running {best['id']} (${best['price']:.2f}/M"
                                       f"{', #' + str(best['rank'] + 1) + ' in programming' if best['rank'] is not None else ''})"})
        self._record(changes)
        return changes

    def evaluate(self) -> list[dict]:
        """Promote challengers that beat their class's model; bench clear losers."""
        changes = []
        for cls in settings.CLASS_ORDER:
            if cls in settings.PINNED_CLASSES:
                continue
            champ = self.champion_stats(cls)
            if champ["n"] < settings.TRIAL_MIN_SAMPLES or champ["rate"] is None:
                continue
            label = settings.SERVICE_CLASSES[cls]["label"]
            best = None
            for t in sorted(self.trial_stats(cls), key=lambda t: (-(t["rate"] or 0), t["cost_per_arrival"] or 1e9)):
                if t["model"] not in self.challengers(cls) or t["n"] < settings.TRIAL_MIN_SAMPLES:
                    continue
                cheaper = (champ["cost_per_arrival"] is None or
                           (t["cost_per_arrival"] is not None and t["cost_per_arrival"] <= champ["cost_per_arrival"]))
                if t["rate"] >= champ["rate"] and cheaper and best is None:
                    best = t
                elif t["rate"] < champ["rate"] - 0.25:
                    benched = self.store.kv_get("fleet:benched", {}) or {}
                    benched[f"{cls}|{t['model']}"] = time.time() + settings.TRIAL_BENCH_DAYS * 86400
                    self.store.kv_set("fleet:benched", benched)
                    changes.append({"kind": "benched", "class": cls, "model": t["model"],
                                    "message": f"{label}: benched {t['model']} for {settings.TRIAL_BENCH_DAYS} days — "
                                               f"first time on {t['rate']:.0%} of {t['n']} trial trains vs "
                                               f"{champ['rate']:.0%} for {self.champions()[cls]}"})
            if best:
                previous = self.champions()[cls]
                self._set_champion(cls, best["model"])
                cost = lambda v: "—" if v is None else f"${v:.2f}"  # noqa: E731
                changes.append({"kind": "promoted", "class": cls, "model": best["model"], "previous": previous,
                                "message": f"{label}: promoted {best['model']} over {previous} — first time on "
                                           f"{best['rate']:.0%} of {best['n']} trains ({cost(best['cost_per_arrival'])}"
                                           f"/arrival) vs {champ['rate']:.0%} of {champ['n']} "
                                           f"({cost(champ['cost_per_arrival'])})"})
        self._record(changes)
        return changes

    def _record(self, changes: list[dict]) -> None:
        if not changes:
            return
        history = (self.store.kv_get("fleet:history", []) or []) + [{**c, "ts": time.time()} for c in changes]
        self.store.kv_set("fleet:history", history[-50:])

    # dashboard -----------------------------------------------------------

    def status(self) -> dict:
        cat = self.catalog()
        champions = self.champions()
        since = self.store.kv_get("fleet:since", {}) or {}
        prices = {m["id"]: m.get("price") for m in cat.get("models") or []}
        classes = []
        for cls in settings.CLASS_ORDER:
            trials = {t["model"]: t for t in self.trial_stats(cls)}
            classes.append({
                "class": cls, "model": champions[cls], "pinned": cls in settings.PINNED_CLASSES,
                "since": since.get(cls), "price": prices.get(champions[cls]), "band": settings.CLASS_BANDS.get(cls),
                "stats": self.champion_stats(cls),
                "challengers": [{"model": m, "price": prices.get(m), **(trials.get(m) or {"n": 0, "rate": None,
                                                                                         "cost_per_arrival": None})}
                                for m in self.challengers(cls)],
            })
        return {"enabled": settings.FLEET_ENABLED, "trial_rate": settings.TRIAL_RATE,
                "min_samples": settings.TRIAL_MIN_SAMPLES, "levels": list(settings.TRIAL_LEVELS),
                "fetched_at": cat.get("fetched_at"), "error": cat.get("error"), "eligible": len(cat.get("models") or []),
                "total": cat.get("total"), "mix": cat.get("mix"), "classes": classes,
                "history": list(reversed((self.store.kv_get("fleet:history", []) or [])[-10:]))}
