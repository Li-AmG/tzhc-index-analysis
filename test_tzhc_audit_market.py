from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tzhc_audit import append_backtest_audit, verify_audit_chain
from tzhc_market import (
    apply_security_names,
    latest_completed_market_date,
    market_code,
    normalize_bars,
    update_benchmark_prices,
    update_watchlist_prices,
)
from tzhc_watchlist import WatchItem


class AuditAndMarketTest(unittest.TestCase):
    def test_market_bars_are_normalized(self) -> None:
        bars = [
            {"datetime": "2026-07-10 15:00", "open": 10, "high": 11, "low": 9, "close": 10.5, "vol": 1234},
            {"datetime": "2026-07-14 11:00", "open": 10, "high": 12, "low": 9, "close": 11.5, "vol": 999},
        ]
        rows = normalize_bars("000001", bars, "2026-07-13")
        self.assertEqual("2026-07-10", rows[0]["date"])
        self.assertEqual(1234.0, rows[0]["volume"])
        self.assertEqual(1, len(rows))
        self.assertEqual(0, market_code("000001"))
        self.assertEqual(1, market_code("600000"))

    def test_completed_market_date_excludes_intraday_and_weekend(self) -> None:
        self.assertEqual("2026-07-13", latest_completed_market_date(datetime(2026, 7, 14, 11, 0)))
        self.assertEqual("2026-07-14", latest_completed_market_date(datetime(2026, 7, 14, 15, 20)))
        self.assertEqual("2026-07-10", latest_completed_market_date(datetime(2026, 7, 13, 9, 0)))

    def test_security_names_replace_codes_but_preserve_manual_names(self) -> None:
        rows, updated = apply_security_names(
            [
                WatchItem("300487", "300487", "观察", "导入"),
                WatchItem("301526", "国际复材", "重点", "手工"),
            ],
            {"300487": "蓝晓科技", "301526": "错误覆盖"},
        )
        self.assertEqual(1, updated)
        self.assertEqual("蓝晓科技", rows[0].name)
        self.assertEqual("国际复材", rows[1].name)
        self.assertEqual("观察", rows[0].tags)

    @patch("tzhc_market._connect")
    def test_benchmark_history_is_saved_separately(self, connect) -> None:
        start = datetime(2026, 4, 1)
        bars = [
            {
                "datetime": (start + timedelta(days=index)).strftime("%Y-%m-%d 15:00"),
                "open": 100 + index,
                "high": 101 + index,
                "low": 99 + index,
                "close": 100.5 + index,
                "vol": 1000,
            }
            for index in range(60)
        ]

        class FakeApi:
            def get_index_bars(self, _category, _market, _code, _start, _count):
                return bars

            def disconnect(self):
                return None

        connect.return_value = (FakeApi(), "test:7709")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = update_benchmark_prices(
                root / "benchmarks.csv",
                root / "benchmark_report.json",
                now=datetime(2026, 7, 14, 15, 20),
            )
            saved = (root / "benchmarks.csv").read_text(encoding="utf-8")
            self.assertIn("399006,创业板指", saved)
            self.assertEqual(4, result.succeeded)
            self.assertEqual(240, result.rows)

    @patch("tzhc_market._connect")
    def test_failed_symbol_keeps_existing_history(self, connect) -> None:
        start = datetime(2026, 4, 1)
        bars = [
            {
                "datetime": (start + timedelta(days=index)).strftime("%Y-%m-%d 15:00"),
                "open": 10,
                "high": 11,
                "low": 9,
                "close": 10.5,
                "vol": 1000,
            }
            for index in range(60)
        ]

        class FakeApi:
            def get_security_bars(self, _category, _market, code, _start, _count):
                if code == "000002":
                    raise RuntimeError("temporary failure")
                return bars

            def disconnect(self):
                return None

        connect.return_value = (FakeApi(), "test:7709")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "prices.csv"
            output.write_text(
                "date,symbol,open,high,low,close,volume\n2026-05-01,000002,8,9,7,8.5,500\n",
                encoding="utf-8",
            )
            result = update_watchlist_prices(
                [WatchItem("000001", "A", "", "test"), WatchItem("000002", "B", "", "test")],
                output,
                root / "report.json",
                now=datetime(2026, 7, 14, 11, 0),
            )
            saved = output.read_text(encoding="utf-8")
            self.assertIn("000002", saved)
            self.assertEqual(1, result.failed)
            self.assertEqual("2026-07-13", result.cutoff_date)

    def test_audit_records_form_hash_chain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "prices.csv"
            output = root / "report.txt"
            audit = root / "audit.jsonl"
            source.write_text("prices", encoding="utf-8")
            output.write_text("report", encoding="utf-8")
            result = SimpleNamespace(
                config={"soft_drawdown_pct": 5, "hard_drawdown_pct": 8},
                final_equity=101000,
                total_return_pct=1,
                max_drawdown_pct=0.5,
                trade_count=2,
                sharpe_ratio=1.2,
                halted=False,
            )
            append_backtest_audit(result, source, [str(output)], audit)
            append_backtest_audit(result, source, [str(output)], audit)
            self.assertEqual((True, 2, "审计链完整，共 2 条记录"), verify_audit_chain(audit))


if __name__ == "__main__":
    unittest.main()
