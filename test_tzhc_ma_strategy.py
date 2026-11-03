from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tzhc_ma_strategy import (
    COMBINED,
    MaStrategyRules,
    analyze_ma_strategy,
    calculate_ma_entry_signals,
    save_ma_strategy_bundle,
)


class MaStrategyTest(unittest.TestCase):
    def frame(self, symbol: str = "000001") -> pd.DataFrame:
        dates = pd.bdate_range("2026-01-01", periods=55)
        closes = [10 + index * 0.1 for index in range(31)]
        closes.extend([13.0, 13.05, 13.0, 12.95, 12.9, 13.3])
        closes.extend([13.4 + index * 0.1 for index in range(len(dates) - len(closes))])
        return pd.DataFrame(
            {
                "date": dates,
                "symbol": symbol,
                "open": closes,
                "high": [value * 1.02 for value in closes],
                "low": [value * 0.98 for value in closes],
                "close": closes,
                "volume": [1000] * len(dates),
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
            "ma_consolidation_days": 5,
            "ma_consolidation_range_pct": 8,
        }

    def test_pullback_requires_hold_above_ma20_and_reclaim_ma5(self) -> None:
        result = calculate_ma_entry_signals(self.frame(), MaStrategyRules(start_date="2026-01-01"))
        pullbacks = result.loc[result["pullback_reclaim"]]
        self.assertFalse(pullbacks.empty)
        event = pullbacks.iloc[0]
        self.assertTrue(event.close_cross_ma5)
        self.assertLessEqual(event.consolidation_range_pct, 8.0)

    def test_combined_is_union_of_three_entry_rules(self) -> None:
        result = calculate_ma_entry_signals(self.frame())
        expected = result["ma5_cross_ma20"] | result["close_cross_ma5"] | result["pullback_reclaim"]
        self.assertTrue((result["combined_entry"] == expected).all())

    def test_analysis_and_bundle_are_traceable(self) -> None:
        prices = pd.concat([self.frame("000001"), self.frame("000002")], ignore_index=True)
        result = analyze_ma_strategy(
            prices,
            self.config(),
            names={"000001": "测试一", "000002": "测试二"},
            rules=MaStrategyRules(start_date="2026-01-01"),
        )
        self.assertFalse(result.events.empty)
        self.assertIn(COMBINED, set(result.summary["strategy"]))
        self.assertIn("return_10d_pct", result.events.columns)
        with tempfile.TemporaryDirectory() as directory:
            paths = save_ma_strategy_bundle(result, Path(directory))
            self.assertTrue(paths["report"].exists())
            self.assertIn("5.0%暂停", paths["report"].read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
