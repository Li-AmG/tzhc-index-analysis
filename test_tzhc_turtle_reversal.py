import unittest

import pandas as pd

from tzhc_turtle_reversal import GRADE_A, score_turtle_reversal_events


class TurtleReversalTests(unittest.TestCase):
    def test_pianzaihuang_event_scores_as_grade_a(self):
        prices = pd.read_csv("F:/tzhc/data/a_share_prices.csv", dtype={"symbol": str})
        events = pd.read_csv("F:/tzhc/reports/turtle_analytics/events_latest.csv", dtype={"symbol": str})
        event = events.loc[(events["symbol"].astype(str).str.zfill(6) == "600436") & (events["signal_date"] == "2026-07-10")]
        self.assertEqual(1, len(event))
        scored = score_turtle_reversal_events(prices, event)
        self.assertEqual(5, int(scored.iloc[0]["反转评分"]))
        self.assertEqual(GRADE_A, scored.iloc[0]["研究等级"])

    def test_nonmatching_event_does_not_receive_grade_a(self):
        prices = pd.read_csv("F:/tzhc/data/a_share_prices.csv", dtype={"symbol": str})
        events = pd.read_csv("F:/tzhc/reports/turtle_analytics/events_latest.csv", dtype={"symbol": str})
        scored = score_turtle_reversal_events(prices, events.head(1))
        self.assertLess(int(scored.iloc[0]["反转评分"]), 5)


if __name__ == "__main__":
    unittest.main()
