"""Prometheus-compatible metrics registry for Yamanote.

Thread-safe counters and gauges, rendered on demand as Prometheus text format.
No third-party dependencies — stdlib only so Ops can read and modify this file.

Usage in orchestrator.py:
    from metrics import METRICS
    METRICS.agent_launches_total.inc({"agent": "conductor"})
    METRICS.specs_total.inc({"outcome": "merged"})

The /metrics HTTP endpoint in dashboard.py renders the full registry for
Prometheus scraping. Point Grafana at http://<host>:<port>/metrics.

To add a new metric:
  1. Define a _Counter or _Gauge field on the Metrics dataclass below.
  2. Call .inc() / .set() at the relevant event site in orchestrator.py.
  3. It will appear automatically in /metrics output — no other changes needed.
"""
import threading
from dataclasses import dataclass, field
from typing import Dict, Tuple


class _Counter:
    """Monotonically increasing counter with optional label dimensions."""

    def __init__(self, name: str, help_text: str):
        self.name = name
        self.help_text = help_text
        self._values: Dict[Tuple, float] = {}
        self._lock = threading.Lock()

    def inc(self, labels: dict | None = None, amount: float = 1.0) -> None:
        key = tuple(sorted((labels or {}).items()))
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def collect(self) -> list[tuple[dict, float]]:
        with self._lock:
            return [(dict(k), v) for k, v in sorted(self._values.items())]


class _Gauge:
    """Current-value metric with optional label dimensions."""

    def __init__(self, name: str, help_text: str):
        self.name = name
        self.help_text = help_text
        self._values: Dict[Tuple, float] = {}
        self._lock = threading.Lock()

    def set(self, value: float, labels: dict | None = None) -> None:
        key = tuple(sorted((labels or {}).items()))
        with self._lock:
            self._values[key] = value

    def inc(self, labels: dict | None = None, amount: float = 1.0) -> None:
        key = tuple(sorted((labels or {}).items()))
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def collect(self) -> list[tuple[dict, float]]:
        with self._lock:
            return [(dict(k), v) for k, v in sorted(self._values.items())]


@dataclass
class Metrics:
    """All Yamanote metrics in one place. Fields appear in /metrics output automatically."""

    # ── Spec lifecycle ──────────────────────────────────────────────────────
    # outcome label values: merged, rejected, sla_breach, entropy, conflict
    specs_total: _Counter = field(
        default_factory=lambda: _Counter(
            "yamanote_specs_total",
            "Total specs processed, by outcome.",
        )
    )

    # ── Agent lifecycle ─────────────────────────────────────────────────────
    agent_launches_total: _Counter = field(
        default_factory=lambda: _Counter(
            "yamanote_agent_launches_total",
            "Total agent subprocess launches, by agent name.",
        )
    )
    agent_failures_total: _Counter = field(
        default_factory=lambda: _Counter(
            "yamanote_agent_failures_total",
            "Total agent non-zero exits (including rate-limit), by agent name.",
        )
    )

    # ── Log watcher ─────────────────────────────────────────────────────────
    log_errors_detected_total: _Counter = field(
        default_factory=lambda: _Counter(
            "yamanote_log_errors_detected_total",
            "Total ERROR/WARNING lines detected by the log watcher across all sources.",
        )
    )
    signal_triggers_total: _Counter = field(
        default_factory=lambda: _Counter(
            "yamanote_signal_triggers_total",
            "Total times the Signal agent was triggered reactively by the log watcher.",
        )
    )

    # ── Live gauges (refreshed on each /metrics scrape) ─────────────────────
    backlog_size: _Gauge = field(
        default_factory=lambda: _Gauge(
            "yamanote_backlog_size",
            "Current number of specs in the backlog queue.",
        )
    )
    trains_active: _Gauge = field(
        default_factory=lambda: _Gauge(
            "yamanote_trains_active",
            "Number of trains currently assigned to a spec.",
        )
    )
    launches_last_hour: _Gauge = field(
        default_factory=lambda: _Gauge(
            "yamanote_launches_last_hour",
            "Agent launches in the past 60 minutes (cost-guardrail window).",
        )
    )
    sleep_mode_active: _Gauge = field(
        default_factory=lambda: _Gauge(
            "yamanote_sleep_mode_active",
            "1 if the orchestrator is in rate-limit sleep mode, else 0.",
        )
    )
    spend_24h_usd: _Gauge = field(
        default_factory=lambda: _Gauge(
            "yamanote_spend_24h_usd",
            "OpenRouter + Jev spend over the past 24 hours, in USD.",
        )
    )
    uptime_seconds: _Gauge = field(
        default_factory=lambda: _Gauge(
            "yamanote_uptime_seconds",
            "Seconds since the orchestrator process started.",
        )
    )


METRICS = Metrics()

_COUNTERS = [
    "specs_total",
    "agent_launches_total",
    "agent_failures_total",
    "log_errors_detected_total",
    "signal_triggers_total",
]
_GAUGES = [
    "backlog_size",
    "trains_active",
    "launches_last_hour",
    "sleep_mode_active",
    "spend_24h_usd",
    "uptime_seconds",
]


def _label_str(labels: dict) -> str:
    if not labels:
        return ""
    pairs = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
    return "{" + pairs + "}"


def render_prometheus(metrics: Metrics | None = None) -> str:
    """Render all metrics in Prometheus text exposition format (version 0.0.4).

    Returns a UTF-8 string ending with a newline. Suitable for Content-Type:
    text/plain; version=0.0.4; charset=utf-8.
    """
    if metrics is None:
        metrics = METRICS

    lines: list[str] = []

    for attr, metric_type in (
        [(name, "counter") for name in _COUNTERS]
        + [(name, "gauge") for name in _GAUGES]
    ):
        metric = getattr(metrics, attr)
        lines.append(f"# HELP {metric.name} {metric.help_text}")
        lines.append(f"# TYPE {metric.name} {metric_type}")
        samples = metric.collect()
        if samples:
            for labels, value in samples:
                fmt = f"{value:.0f}" if metric_type == "counter" else f"{value:.6g}"
                lines.append(f"{metric.name}{_label_str(labels)} {fmt}")
        else:
            lines.append(f"{metric.name} 0")

    return "\n".join(lines) + "\n"
