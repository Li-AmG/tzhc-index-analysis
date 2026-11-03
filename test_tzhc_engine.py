from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tzhc_engine import Candle, load_prices_csv, run_backtest


def candle(date: str, close: float, open_price: float | None = None, volume: float = 1000) -> Candle:
    opened = close if open_price is None else open_price
    return Candle(date, "000001", opened, max(opened, close) * 1.01, min(opened, close) * 0.99, close, volume)


class BacktestRulesTest(unittest.TestCase):
    def base_config(self) -> dict:
        return {
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

    def test_signal_fills_next_open_in_board_lots(self) -> None:
        rows = [
            candle("2026-01-01", 10),
            candle("2026-01-02", 9),
            candle("2026-01-05", 11),
            candle("2026-01-06", 11.2, 10.5),
            candle("2026-01-07", 11.3),
        ]
        result = run_backtest({"000001": rows}, self.base_config())
        buy_signal = next(s for s in result.signals if s.action == "BUY_SIGNAL")
        buy_fill = next(s for s in result.signals if s.action == "BUY_FILLED")
        self.assertEqual("2026-01-05", buy_signal.date)
        self.assertEqual("2026-01-06", buy_fill.date)
        self.assertEqual(10.5, buy_fill.close)
        self.assertTrue(all(trade.shares % 100 == 0 for trade in result.trades))

    def test_five_percent_pause_and_eight_percent_halt(self) -> None:
        rows = [
            candle("2026-01-01", 10),
            candle("2026-01-02", 9),
            candle("2026-01-05", 11),
            candle("2026-01-06", 11, 11),
            candle("2026-01-07", 10.38),
            candle("2026-01-08", 9.95),
            candle("2026-01-09", 9.90),
        ]
        result = run_backtest({"000001": rows}, self.base_config())
        pause = next(s for s in result.signals if s.action == "PAUSE_BUY")
        halt = next(s for s in result.signals if s.action == "HALT")
        hard_exit = next(t for t in result.trades if "硬回撤" in t.reason)
        self.assertEqual("2026-01-07", pause.date)
        self.assertEqual("2026-01-08", halt.date)
        self.assertEqual("2026-01-09", hard_exit.exit_date)
        self.assertTrue(result.halted)
        self.assertGreaterEqual(result.max_drawdown_pct, 8)

    def test_invalid_ohlc_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            path.write_text("date,symbol,open,high,low,close,volume\n2026-01-01,X,10,9,8,11,100\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "OHLC"):
                load_prices_csv(path)


if __name__ == "__main__":
    unittest.main()
