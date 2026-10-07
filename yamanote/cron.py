"""Five-field cron expressions (minute hour day-of-month month day-of-week),
evaluated in local time. Supports `*`, lists, ranges, steps and names:

    0 22 * * *        every day at 22:00
    30 7 * * 1-5      weekdays at 07:30
    0 */4 * * *       every four hours
    0 9 * * mon,wed   Mondays and Wednesdays at 09:00

As in standard cron, when both day-of-month and day-of-week are restricted, a
day matches if EITHER does. Day-of-week 0 and 7 are both Sunday.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
DAYS = {d: i for i, d in enumerate(["sun", "mon", "tue", "wed", "thu", "fri", "sat"])}
FIELDS = [("minute", 0, 59, {}), ("hour", 0, 23, {}), ("day of month", 1, 31, {}),
          ("month", 1, 12, MONTHS), ("day of week", 0, 7, DAYS)]


class CronError(ValueError):
    pass


@dataclass(frozen=True)
class Cron:
    expr: str
    minutes: frozenset
    hours: frozenset
    days: frozenset
    months: frozenset
    weekdays: frozenset  # 0 = Sunday
    dom_restricted: bool
    dow_restricted: bool

    def day_matches(self, d: dt.date) -> bool:
        if d.month not in self.months:
            return False
        dom_ok = d.day in self.days
        dow_ok = (d.isoweekday() % 7) in self.weekdays
        if self.dom_restricted and self.dow_restricted:
            return dom_ok or dow_ok
        return dom_ok and dow_ok

    def matches(self, t: dt.datetime) -> bool:
        return self.day_matches(t.date()) and t.hour in self.hours and t.minute in self.minutes

    def prev(self, before: dt.datetime, max_days: int = 400) -> dt.datetime | None:
        """Latest matching minute at or before `before` (seconds ignored)."""
        t = before.replace(second=0, microsecond=0)
        day = t.date()
        for i in range(max_days):
            if self.day_matches(day):
                for h in sorted(self.hours, reverse=True):
                    if i == 0 and h > t.hour:
                        continue
                    for m in sorted(self.minutes, reverse=True):
                        if i == 0 and h == t.hour and m > t.minute:
                            continue
                        return dt.datetime.combine(day, dt.time(h, m))
            day -= dt.timedelta(days=1)
        return None

    def next(self, after: dt.datetime, max_days: int = 400) -> dt.datetime | None:
        """Earliest matching minute strictly after `after`."""
        t = after.replace(second=0, microsecond=0) + dt.timedelta(minutes=1)
        day = t.date()
        for i in range(max_days):
            if self.day_matches(day):
                for h in sorted(self.hours):
                    if i == 0 and h < t.hour:
                        continue
                    for m in sorted(self.minutes):
                        if i == 0 and h == t.hour and m < t.minute:
                            continue
                        return dt.datetime.combine(day, dt.time(h, m))
            day += dt.timedelta(days=1)
        return None


def _field(text: str, name: str, lo: int, hi: int, names: dict) -> frozenset:
    values: set[int] = set()
    for part in text.lower().split(","):
        if not part:
            raise CronError(f"empty item in {name}")
        rng, _, step_s = part.partition("/")
        try:
            step = int(step_s) if step_s else 1
        except ValueError:
            raise CronError(f"bad step '{step_s}' in {name}") from None
        if step < 1:
            raise CronError(f"step must be at least 1 in {name}")
        if rng == "*":
            start, end = lo, hi
        else:
            a, dash, b = rng.partition("-")
            start = _value(a, name, lo, hi, names)
            end = _value(b, name, lo, hi, names) if dash else (hi if step_s else start)
            if end < start:
                raise CronError(f"range {rng} runs backwards in {name}")
        values.update(range(start, end + 1, step))
    return frozenset(values)


def _value(text: str, name: str, lo: int, hi: int, names: dict) -> int:
    if text in names:
        return names[text]
    try:
        v = int(text)
    except ValueError:
        raise CronError(f"'{text}' is not valid in {name}") from None
    if not lo <= v <= hi:
        raise CronError(f"{v} is out of range for {name} ({lo}-{hi})")
    return v


def parse(expr: str) -> Cron:
    parts = (expr or "").split()
    if len(parts) != 5:
        raise CronError("a cron expression needs 5 fields: minute hour day-of-month month day-of-week")
    sets = [_field(p, *spec) for p, spec in zip(parts, FIELDS)]
    weekdays = frozenset(0 if d == 7 else d for d in sets[4])
    return Cron(expr.strip(), sets[0], sets[1], sets[2], sets[3], weekdays,
                dom_restricted=parts[2] != "*", dow_restricted=parts[4] != "*")


def describe_next(expr: str, now: dt.datetime | None = None) -> str | None:
    """'Tue 22:00' style label for the next run, or None if it never runs."""
    nxt = parse(expr).next(now or dt.datetime.now())
    return nxt.strftime("%a %d %b %H:%M") if nxt else None
