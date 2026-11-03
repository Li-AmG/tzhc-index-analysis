from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tzhc_engine import Candle
from tzhc_paper import (
    PaperPaths,
    RISK_HARD_LOCK,
    RISK_NORMAL,
    RISK_SOFT_LOCK,
    add_paper_watch_symbol,
    advance_paper_account,
    initialize_paper_account,
    load_paper_account,
    queue_paper_buy_signal,
    remove_paper_watch_symbol,
    unlock_soft_risk,
)


def candle(date: str, close: float, open_price: float | None = None) -> Candle:
    opened = close if open_price is None else open_price
    return Candle(date, "000001", opened, max(opened, close) * 1.01, min(opened, close) * 0.99, close, 1000)


class PaperAccountTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.paths = PaperPaths.for_root(self.root)
        self.config = {
            "initial_capital": 100000,
            "entry_strategy": "ma_cross",
            "soft_drawdown_pct": 5,
            "hard_drawdown_pct": 8,
            "max_positions": 1,
            "max_position_pct": 100,
            "stop_loss_pct": 99,
            "take_profit_pct": 99,
            "min_volume_ratio": 0,
            "volume_window": 2,
            "ma_fast": 2,
            "ma_trend": 2,
            "ma_exit": 50,
            "lot_size": 100,
            "commission_rate": 0,
            "min_commission": 0,
            "stamp_tax_rate": 0,
            "slippage_pct": 0,
            "price_limit_pct": 20,
        }
        self.rows = [
            candle("2026-01-01", 10),
            candle("2026-01-02", 9),
            candle("2026-01-05", 11),
            candle("2026-01-06", 11, 11),
            candle("2026-01-07", 10.38),
            candle("2026-01-08", 9.95),
            candle("2026-01-09", 9.90),
        ]
        initialize_paper_account(self.paths, self.config, "2025-12-31")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_repeated_update_is_idempotent_and_persistent(self) -> None:
        state = advance_paper_account(self.paths, {"000001": self.rows}, self.config, "2026-01-06")
        counts = (len(state["equity_history"]), len(state["events"]), len(state["trades"]))
        state = advance_paper_account(self.paths, {"000001": self.rows}, self.config, "2026-01-06")
        self.assertEqual(counts, (len(state["equity_history"]), len(state["events"]), len(state["trades"])))
        self.assertIn("000001", state["positions"])
        self.assertEqual(state["cash"], load_paper_account(self.paths)["cash"])

    def test_soft_lock_survives_restart_and_requires_manual_review(self) -> None:
        state = advance_paper_account(self.paths, {"000001": self.rows}, self.config, "2026-01-07")
        self.assertEqual(RISK_SOFT_LOCK, state["risk_lock"]["status"])
        self.assertEqual(RISK_SOFT_LOCK, load_paper_account(self.paths)["risk_lock"]["status"])
        with self.assertRaisesRegex(ValueError, "确认"):
            unlock_soft_risk(self.paths, "已核对持仓和行情", False)
        state = unlock_soft_risk(self.paths, "已核对持仓和行情", True)
        self.assertEqual(RISK_NORMAL, state["risk_lock"]["status"])
        self.assertEqual("SOFT_UNLOCK", state["events"][-1]["action"])

    def test_hard_lock_persists_and_liquidates_next_open(self) -> None:
        state = advance_paper_account(self.paths, {"000001": self.rows}, self.config, "2026-01-08")
        self.assertEqual(RISK_HARD_LOCK, state["risk_lock"]["status"])
        self.assertIn("000001", state["pending_exits"])
        state = advance_paper_account(self.paths, {"000001": self.rows}, self.config, "2026-01-09")
        self.assertEqual(RISK_HARD_LOCK, state["risk_lock"]["status"])
        self.assertFalse(state["positions"])
        self.assertEqual(1, len(state["trades"]))
        self.assertIn("持久熔断", state["trades"][0]["reason"])

    def test_daily_report_and_csv_exports_are_created(self) -> None:
        advance_paper_account(self.paths, {"000001": self.rows}, self.config, "2026-01-06")
        report = self.paths.reports / "paper_daily_2026-01-06.md"
        self.assertTrue(report.exists())
        self.assertIn("模拟账户日报", report.read_text(encoding="utf-8"))
        self.assertTrue(self.paths.equity.exists())
        self.assertTrue(self.paths.events.exists())
        self.assertTrue(self.paths.trades.exists())

    def test_search_watchlist_signal_can_queue_target_market_value(self) -> None:
        initialize_paper_account(self.paths, self.config, "2026-01-05", overwrite=True)
        state = add_paper_watch_symbol(self.paths, "000001", "测试股票")
        self.assertEqual("000001", state["paper_watchlist"][0]["symbol"])
        state = queue_paper_buy_signal(
            self.paths,
            {"000001": self.rows},
            self.config,
            "000001",
            target_value=5000,
            confirmed=True,
        )
        self.assertEqual(5000, state["pending_entries"]["000001"]["target_value"])
        state = advance_paper_account(self.paths, {"000001": self.rows}, self.config, "2026-01-06")
        self.assertEqual(400, state["positions"]["000001"]["shares"])
        state = remove_paper_watch_symbol(self.paths, "000001")
        self.assertFalse(state["paper_watchlist"])

    def test_only_current_high_quality_signal_can_be_queued(self) -> None:
        initialize_paper_account(self.paths, self.config, "2026-01-07", overwrite=True)
        add_paper_watch_symbol(self.paths, "000001", "测试股票")
        with self.assertRaisesRegex(ValueError, "高质量买入信号"):
            queue_paper_buy_signal(
                self.paths,
                {"000001": self.rows},
                self.config,
                "000001",
                target_value=5000,
                confirmed=True,
            )

    @patch("tzhc_pullback_strategy.latest_signal_is_valid", return_value=True)
    def test_pullback_order_keeps_its_own_risk_plan(self, _validated) -> None:
        initialize_paper_account(self.paths, self.config, "2026-01-05", overwrite=True)
        add_paper_watch_symbol(self.paths, "000001", "测试股票")
        state = queue_paper_buy_signal(
            self.paths,
            {"000001": self.rows},
            self.config,
            "000001",
            target_value=5000,
            confirmed=True,
            strategy="pullback_ignition",
        )
        pending = state["pending_entries"]["000001"]
        self.assertEqual("pullback_ignition", pending["strategy"])
        self.assertEqual(4.0, pending["initial_stop_pct"])
        self.assertEqual(5.0, pending["trailing_stop_pct"])

        state = advance_paper_account(self.paths, {"000001": self.rows}, self.config, "2026-01-06")
        self.assertEqual("pullback_ignition", state["positions"]["000001"]["strategy"])
        state = advance_paper_account(self.paths, {"000001": self.rows}, self.config, "2026-01-07")
        self.assertIn("初始止损 4%", state["pending_exits"]["000001"]["reason"])
        state = advance_paper_account(self.paths, {"000001": self.rows}, self.config, "2026-01-08")
        self.assertEqual("pullback_ignition", state["trades"][0]["strategy"])


if __name__ == "__main__":
    unittest.main()
