import unittest

import numpy as np
import pandas as pd

from tzhc_turtle_reversal import GRADE_A, score_turtle_reversal_events


class TurtleReversalTests(unittest.TestCase):
    @staticmethod
    def reversal_fixture():
        dates = pd.date_range("2026-01-01", periods=70, freq="B")
        closes = list(np.linspace(100.0, 50.0, 69)) + [60.0]
        prices = pd.DataFrame(
            {
                "date": dates,
                "symbol": "600436",
                "open": [value - 0.4 for value in closes[:-1]] + [50.0],
                "high": [value + 0.6 for value in closes[:-1]] + [61.0],
                "low": [value - 0.8 for value in closes[:-1]] + [49.0],
                "close": closes,
                "volume": [1_000_000] * 69 + [2_000_000],
            }
        )
        events = pd.DataFrame(
            [
                {
                    "symbol": "600436",
                    "signal_date": dates[-1].strftime("%Y-%m-%d"),
                    "signal_close": 60.0,
                    "volume_ratio20": 2.0,
                }
            ]
        )
        return prices, events

    def test_pianzaihuang_event_scores_as_grade_a(self):
        prices, events = self.reversal_fixture()
        scored = score_turtle_reversal_events(prices, events)
        self.assertEqual(5, int(scored.iloc[0]["反转评分"]))
        self.assertEqual(GRADE_A, scored.iloc[0]["研究等级"])

    def test_nonmatching_event_does_not_receive_grade_a(self):
        prices, events = self.reversal_fixture()
        events.loc[0, "volume_ratio20"] = 1.0
        scored = score_turtle_reversal_events(prices, events)
        self.assertLess(int(scored.iloc[0]["反转评分"]), 5)


if __name__ == "__main__":
    unittest.main()
