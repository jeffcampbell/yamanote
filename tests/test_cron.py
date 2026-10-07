"""Cron parser tests (pure, no time dependence)."""
from __future__ import annotations

import datetime as dt
import unittest

from yamanote.cron import CronError, parse

T = dt.datetime


class CronTest(unittest.TestCase):
    def test_daily(self):
        c = parse("0 22 * * *")
        self.assertEqual(c.next(T(2026, 10, 7, 9, 0)), T(2026, 10, 7, 22, 0))
        self.assertEqual(c.next(T(2026, 10, 7, 22, 0)), T(2026, 10, 8, 22, 0), "strictly after")
        self.assertEqual(c.prev(T(2026, 10, 7, 21, 59)), T(2026, 10, 6, 22, 0))
        self.assertEqual(c.prev(T(2026, 10, 7, 22, 0, 30)), T(2026, 10, 7, 22, 0), "at or before")

    def test_weekdays_and_names(self):
        c = parse("30 7 * * mon-fri")
        self.assertEqual(c.next(T(2026, 10, 9, 8, 0)), T(2026, 10, 12, 7, 30))  # Fri → Mon
        self.assertEqual(parse("0 9 * * MON,wed").weekdays, frozenset({1, 3}))
        self.assertEqual(parse("0 0 * * 7").weekdays, frozenset({0}), "7 is Sunday")

    def test_steps_ranges_lists(self):
        self.assertEqual(sorted(parse("*/15 * * * *").minutes), [0, 15, 30, 45])
        self.assertEqual(sorted(parse("0 9-17/4 * * *").hours), [9, 13, 17])
        self.assertEqual(sorted(parse("5,10 * * * *").minutes), [5, 10])
        self.assertEqual(sorted(parse("0 1/6 * * *").hours), [1, 7, 13, 19])

    def test_dom_or_dow_when_both_restricted(self):
        c = parse("0 12 1 * mon")  # the 1st OR any Monday
        self.assertTrue(c.matches(T(2026, 10, 1, 12, 0)))   # Thursday the 1st
        self.assertTrue(c.matches(T(2026, 10, 5, 12, 0)))   # a Monday
        self.assertFalse(c.matches(T(2026, 10, 6, 12, 0)))

    def test_impossible_date_never_runs(self):
        self.assertIsNone(parse("0 0 30 2 *").next(T(2026, 1, 1)))

    def test_errors(self):
        for bad in ("", "* * * *", "60 * * * *", "* 24 * * *", "0 0 0 * *", "*/0 * * * *",
                    "5-1 * * * *", "0 0 * * funday", "a b c d e"):
            with self.assertRaises(CronError, msg=bad):
                parse(bad)


if __name__ == "__main__":
    unittest.main()
