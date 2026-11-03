from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from tzhc_engine import Candle, DEFAULT_CONFIG
from tzhc_lifecycle_strategy import (
    LifecycleRules,
    calculate_lifecycle_signals,
    scan_lifecycle_signals,
)
from tzhc_paper import (
    PaperPaths,
    add_paper_watch_symbol,
    advance_paper_account,
    initialize_paper_account,
    queue_paper_buy_signal,
    _lifecycle_action,
)


def lifecycle_frame(symbol: str = "000001") -> pd.DataFrame:
    count = 130
    closes = list(np.linspace(20.0, 18.0, count)) + [19.08, 19.25]
    volumes = [1000.0] * count + [1300.0, 2200.0]
    opens = [value * 0.995 for value in closes]
    highs = [value * 1.01 for value in closes]
    lows = [value * 0.99 for value in closes]
    opens[-2] = 18.40
    lows[-2] = 18.20
    highs[-2] = 19.20
    return pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=len(closes), freq="B"),
            "symbol": symbol,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    )


class LifecycleSignalTest(unittest.TestCase):
    def test_woman_day_two_generates_95_percent_candidate(self) -> None:
        result = calculate_lifecycle_signals(
            lifecycle_frame(),
            LifecycleRules(min_amount=0.0),
        )
        self.assertTrue(bool(result.iloc[-2]["woman_day1"]))
        self.assertTrue(bool(result.iloc[-1]["raw_double_high"]))
        self.assertTrue(bool(result.iloc[-1]["eligible"]))

    def test_scan_excludes_st(self) -> None:
        prices = pd.concat([lifecycle_frame("000001"), lifecycle_frame("000002")], ignore_index=True)
        watchlist = pd.DataFrame(
            [
                {"code": "000001", "name": "测试股份"},
                {"code": "000002", "name": "ST测试"},
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            result = scan_lifecycle_signals(
                prices,
                watchlist,
                Path(directory) / "scan.csv",
                {"ghost_lifecycle_min_amount": 0.0},
                recent_days=10,
            )
        self.assertEqual(["000001"], result["symbol"].tolist())

    def test_real_lifecycle_action_reduces_at_ten_percent(self) -> None:
        frame = lifecycle_frame()
        rows = [
            Candle(
                row.date.strftime("%Y-%m-%d"), row.symbol, float(row.open), float(row.high),
                float(row.low), float(row.close), float(row.volume),
            )
            for row in frame.itertuples()
        ]
        position = {
            "entry_price": 17.0,
            "peak_price": 19.25,
            "previous_close": 19.0,
            "strategy": "ghost_lifecycle",
        }
        action = _lifecycle_action(DEFAULT_CONFIG, rows, len(rows) - 1, position)
        self.assertIsNotNone(action)
        self.assertEqual("sell", action["kind"])
        self.assertEqual(0.5, action["fraction"])
        self.assertIn("10%", action["reason"])


class LifecyclePaperTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.paths = PaperPaths.for_root(Path(self.temporary.name))
        self.config = {
            "initial_capital": 100000,
            "soft_drawdown_pct": 50,
            "hard_drawdown_pct": 80,
            "max_positions": 5,
            "max_position_pct": 100,
            "lot_size": 100,
            "commission_rate": 0,
            "min_commission": 0,
            "stamp_tax_rate": 0,
            "slippage_pct": 0,
            "price_limit_pct": 20,
            "ghost_lifecycle_min_amount": 0,
        }
        self.rows = [
            Candle("2026-01-05", "000001", 10, 10.1, 9.9, 10, 1000),
            Candle("2026-01-06", "000001", 10, 10.1, 9.9, 10, 1000),
            Candle("2026-01-07", "000001", 11, 11.1, 10.9, 11, 1000),
        ]
        initialize_paper_account(self.paths, self.config, "2026-01-05")
        add_paper_watch_symbol(self.paths, "000001", "测试股票")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @patch("tzhc_lifecycle_strategy.latest_signal_is_valid", return_value=True)
    def test_lifecycle_order_uses_five_percent_initial_limit(self, _validated) -> None:
        with self.assertRaisesRegex(ValueError, "单股风控上限"):
            queue_paper_buy_signal(
                self.paths, {"000001": self.rows}, self.config, "000001",
                target_value=6000, confirmed=True, strategy="ghost_lifecycle",
            )
        state = queue_paper_buy_signal(
            self.paths, {"000001": self.rows}, self.config, "000001",
            target_value=5000, confirmed=True, strategy="ghost_lifecycle",
        )
        self.assertEqual("ghost_lifecycle", state["pending_entries"]["000001"]["strategy"])

    @patch("tzhc_paper._lifecycle_action")
    @patch("tzhc_lifecycle_strategy.latest_signal_is_valid", return_value=True)
    def test_ten_percent_action_can_sell_half_next_open(self, _validated, lifecycle_action) -> None:
        lifecycle_action.side_effect = [
            {"kind": "sell", "fraction": 0.5, "reason": "盈利达到10%，减仓一半"},
            None,
        ]
        queue_paper_buy_signal(
            self.paths, {"000001": self.rows}, self.config, "000001",
            target_value=5000, confirmed=True, strategy="ghost_lifecycle",
        )
        state = advance_paper_account(self.paths, {"000001": self.rows}, self.config, "2026-01-06")
        initial_shares = state["positions"]["000001"]["shares"]
        self.assertIn("000001", state["pending_exits"])
        state = advance_paper_account(self.paths, {"000001": self.rows}, self.config, "2026-01-07")
        sold_shares = int(initial_shares * 0.5 // 100) * 100
        self.assertEqual(initial_shares - sold_shares, state["positions"]["000001"]["shares"])
        self.assertEqual("ghost_lifecycle", state["trades"][0]["strategy"])
        self.assertIn("盈利达到10%", state["trades"][0]["reason"])


if __name__ == "__main__":
    unittest.main()
