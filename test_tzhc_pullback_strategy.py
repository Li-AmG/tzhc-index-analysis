from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tzhc_pullback_strategy import (
    PullbackIgnitionRules,
    calculate_pullback_signals,
    scan_pullback_signals,
)


def signal_frame(symbol: str = "000001") -> pd.DataFrame:
    rows = []
    dates = pd.bdate_range("2025-01-01", periods=101)
    platform = [20.5, 20.6, 20.7, 20.8, 20.9, 21.0, 21.1, 21.2, 21.3, 21.4]
    for index, date in enumerate(dates):
        if index < 80:
            close = 10.0 + index * (11.5 / 79)
        elif index == 80:
            close = 21.5
        elif index < 90:
            close = 21.5 + (index - 80) * 0.02
        elif index < 100:
            close = platform[index - 90]
        else:
            close = 22.2
        opened, high, low, volume = close - 0.08, close + 0.15, close - 0.15, 1_000_000
        if index == 80:
            high = 23.0
        if 90 <= index < 100:
            opened, low, volume = close - 0.03, close - 0.20, 500_000
        if index == 100:
            opened, high, low, volume = 21.4, 22.3, 21.35, 2_000_000
        rows.append(
            {
                "date": date,
                "symbol": symbol,
                "open": opened,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )
    return pd.DataFrame(rows)


class PullbackIgnitionTest(unittest.TestCase):
    def test_strict_pullback_ignition_signal(self) -> None:
        result = calculate_pullback_signals(signal_frame(), False, PullbackIgnitionRules())
        self.assertTrue(bool(result.iloc[-1]["pullback_ignition"]))
        self.assertGreaterEqual(float(result.iloc[-1]["pullback_pct"]), 10.0)
        self.assertLessEqual(float(result.iloc[-1]["pullback_volume_ratio"]), 0.70)
        self.assertGreaterEqual(float(result.iloc[-1]["volume_ratio20"]), 1.8)

    def test_signal_fails_without_trigger_volume(self) -> None:
        data = signal_frame()
        data.loc[data.index[-1], "volume"] = 600_000
        result = calculate_pullback_signals(data, False, PullbackIgnitionRules())
        self.assertFalse(bool(result.iloc[-1]["pullback_ignition"]))

    def test_scan_excludes_st_and_writes_strategy_risk(self) -> None:
        prices = pd.concat([signal_frame("000001"), signal_frame("000002")], ignore_index=True)
        watchlist = pd.DataFrame(
            [
                {"code": "000001", "name": "测试股份"},
                {"code": "000002", "name": "ST测试"},
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "scan.csv"
            result = scan_pullback_signals(prices, watchlist, output, {}, recent_days=10)
            self.assertTrue(output.exists())
            self.assertEqual(["000001"], result["symbol"].tolist())
            self.assertEqual(4.0, float(result.iloc[0]["initial_stop_pct"]))
            self.assertEqual(5.0, float(result.iloc[0]["trailing_stop_pct"]))


if __name__ == "__main__":
    unittest.main()
