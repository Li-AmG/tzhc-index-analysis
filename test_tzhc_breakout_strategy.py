from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tzhc_breakout_strategy import (
    BreakoutRules,
    analyze_breakout_strategy,
    calculate_breakout_signals,
    save_breakout_bundle,
)


class BreakoutStrategyTest(unittest.TestCase):
    def frame(self) -> pd.DataFrame:
        dates = pd.bdate_range("2025-01-02", periods=115)
        closes = [10.0 + index * 0.05 for index in range(80)]
        closes.extend([14.0 + (index % 2) * 0.02 for index in range(10)])
        closes.append(14.75)
        closes.extend([14.8 + index * 0.08 for index in range(len(dates) - len(closes))])
        volumes = [200000] * 80 + [100000] * 10 + [250000] + [150000] * (len(dates) - 91)
        opens = list(closes)
        opens[90] = 14.05
        highs = [value * 1.006 for value in closes]
        lows = [value * 0.994 for value in closes]
        highs[90] = 14.85
        lows[90] = 13.98
        return pd.DataFrame(
            {
                "date": dates,
                "symbol": "000001",
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": volumes,
            }
        )

    def config(self) -> dict:
        return {
            "initial_capital": 100000,
            "max_position_pct": 18,
            "lot_size": 100,
            "commission_rate": 0,
            "min_commission": 0,
            "stamp_tax_rate": 0,
            "slippage_pct": 0,
            "price_limit_pct": 10,
            "breakout_consolidation_days": 10,
            "breakout_consolidation_range_pct": 8,
            "breakout_volume_contraction_ratio": 0.85,
            "breakout_min_volume_ratio": 1.5,
            "breakout_max_volume_ratio": 4,
            "breakout_main_min_gain_pct": 4,
            "breakout_main_max_gain_pct": 9.5,
            "breakout_growth_min_gain_pct": 6,
            "breakout_growth_max_gain_pct": 15,
            "breakout_min_turnover_amount": 50_000_000,
        }

    def test_strong_breakout_requires_trend_contraction_volume_and_candle(self) -> None:
        result = calculate_breakout_signals(self.frame(), False, BreakoutRules())
        signal = result.iloc[90]
        self.assertTrue(signal.trend_ok)
        self.assertTrue(signal.consolidation_ok)
        self.assertTrue(signal.volume_ok)
        self.assertTrue(signal.candle_ok)
        self.assertTrue(signal.breakout_20)
        self.assertTrue(signal.breakout_60)

    def test_st_name_is_excluded(self) -> None:
        result = analyze_breakout_strategy(
            self.frame(), self.config(), names={"000001": "*ST测试"}, rules=BreakoutRules(start_date="2025-01-01")
        )
        self.assertTrue(result.events.empty)

    def test_analysis_and_report_are_written(self) -> None:
        result = analyze_breakout_strategy(
            self.frame(), self.config(), names={"000001": "测试股"}, rules=BreakoutRules(start_date="2025-01-01")
        )
        self.assertEqual(2, len(result.events))
        self.assertTrue(result.events["return_10d_pct"].notna().all())
        with tempfile.TemporaryDirectory() as directory:
            paths = save_breakout_bundle(result, Path(directory))
            self.assertTrue(paths["report"].exists())
            self.assertIn("未伪造50-80亿元市值过滤", paths["report"].read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
