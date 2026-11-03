from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import pandas as pd

from tzhc_dashboard_signals import calculate_signals, scan_watchlist_signals


class DashboardSignalTest(unittest.TestCase):
    def test_signal_columns_and_state_are_created(self) -> None:
        rows = []
        for index in range(90):
            close = 10 + index * 0.03
            volume = 1000
            if index == 70:
                close += 1.0
                volume = 2300
            rows.append({"date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=index), "open": close * 0.98, "high": close * 1.01, "low": close * 0.97, "close": close, "volume": volume})
        result = calculate_signals(pd.DataFrame(rows))
        self.assertTrue({"raw_double_high", "buy_candidate", "trade_action", "trade_reason", "holding_state"}.issubset(result.columns))
        self.assertTrue((result["signal_score"] >= 0).all())

    def test_watchlist_scan_writes_structured_csv(self) -> None:
        dates = pd.date_range("2026-01-01", periods=90, freq="D")
        prices = pd.DataFrame({
            "date": dates,
            "symbol": "000001",
            "open": [10 + index * 0.02 for index in range(90)],
            "high": [10.2 + index * 0.02 for index in range(90)],
            "low": [9.8 + index * 0.02 for index in range(90)],
            "close": [10.1 + index * 0.02 for index in range(90)],
            "volume": [1000] * 90,
        })
        watchlist = pd.DataFrame([{"code": "000001", "name": "测试", "tags": "观察"}])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "signals.csv"
            scan_watchlist_signals(prices, watchlist, output)
            self.assertTrue(output.exists())
            saved = pd.read_csv(output)
            self.assertIn("signal", saved.columns)
            self.assertIn("reason", saved.columns)


if __name__ == "__main__":
    unittest.main()
