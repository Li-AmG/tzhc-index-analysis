from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import pandas as pd

from tzhc_turtle import TurtleRules, analyze_turtle_events, board_benchmark, save_turtle_bundle


class TurtleAnalyticsTest(unittest.TestCase):
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
            "turtle_index_drop_pct": -2,
            "turtle_shanghai_drop_pct": -1,
            "turtle_ordinary_gain_pct": 5,
            "turtle_growth_gain_pct": 10,
            "turtle_min_volume_ratio": 1.2,
            "turtle_max_volume_ratio": 2.0,
        }

    def frames(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        dates = pd.bdate_range(datetime(2026, 1, 2), periods=35)
        event_index = 21
        index_close = [100.0] * len(dates)
        index_close[event_index] = 97.0
        for index in range(event_index + 1, len(dates)):
            index_close[index] = 97.0 + (index - event_index) * 0.2
        benchmark_rows = []
        for code, name in (("000001", "上证指数"), ("399001", "深证成指"), ("399006", "创业板指")):
            for date, close in zip(dates, index_close):
                benchmark_rows.append(
                    {
                        "date": date,
                        "benchmark_code": code,
                        "benchmark_name": name,
                        "open": close,
                        "high": close,
                        "low": close,
                        "close": close,
                        "volume": 1000,
                    }
                )
        price_rows = []
        closes = {"000001": [10.0] * len(dates), "300001": [10.0] * len(dates)}
        closes["000001"][event_index] = 10.6
        closes["300001"][event_index] = 11.1
        for index in range(event_index + 1, len(dates)):
            closes["000001"][index] = 10.6 + (index - event_index) * 0.1
            closes["300001"][index] = 11.1 + (index - event_index) * 0.1
        for symbol, values in closes.items():
            for index, (date, close) in enumerate(zip(dates, values)):
                volume = 1000
                if index == event_index:
                    volume = 1600 if symbol == "000001" else 2500
                price_rows.append(
                    {
                        "date": date,
                        "symbol": symbol,
                        "open": close,
                        "high": close * 1.02,
                        "low": close * 0.98,
                        "close": close,
                        "volume": volume,
                    }
                )
        return pd.DataFrame(price_rows), pd.DataFrame(benchmark_rows)

    def test_board_mapping_uses_matching_index_and_threshold(self) -> None:
        rules = TurtleRules(index_drop_pct=-2, ordinary_gain_pct=5, growth_gain_pct=10)
        self.assertEqual(("深证成指", 5), (board_benchmark("000001", rules).benchmark_name, board_benchmark("000001", rules).gain_threshold_pct))
        self.assertEqual(("创业板指", 10), (board_benchmark("300001", rules).benchmark_name, board_benchmark("300001", rules).gain_threshold_pct))
        self.assertEqual("科创50", board_benchmark("688001", rules).benchmark_name)
        self.assertIsNone(board_benchmark("830001", rules))

    def test_only_qualified_contrarian_stock_is_marked(self) -> None:
        prices, benchmarks = self.frames()
        result = analyze_turtle_events(prices, benchmarks, self.config(), names={"000001": "测试主板", "300001": "测试创业"})
        self.assertEqual(1, len(result.events))
        event = result.events.iloc[0]
        self.assertEqual("000001", event.symbol)
        self.assertEqual("测试主板", event["name"])
        self.assertAlmostEqual(6.0, event.stock_return_pct)
        self.assertAlmostEqual(-3.0, event.benchmark_return_pct)
        self.assertAlmostEqual(9.0, event.relative_strength_pct)
        self.assertAlmostEqual(1.6, event.volume_ratio20)
        self.assertTrue(pd.notna(event.return_10d_pct))
        self.assertTrue(pd.notna(event.excess_return_10d_pct))

    def test_index_without_crash_does_not_create_event(self) -> None:
        prices, benchmarks = self.frames()
        event_date = sorted(benchmarks["date"].unique())[21]
        benchmarks.loc[benchmarks["date"] == event_date, "close"] = 99.5
        result = analyze_turtle_events(prices, benchmarks, self.config())
        self.assertTrue(result.events.empty)

    def test_shanghai_drop_can_trigger_growth_board_turtle(self) -> None:
        prices, benchmarks = self.frames()
        event_date = sorted(benchmarks["date"].unique())[21]
        benchmarks.loc[benchmarks["date"] == event_date, "close"] = 99.5
        benchmarks.loc[
            (benchmarks["date"] == event_date) & (benchmarks["benchmark_code"] == "000001"),
            "close",
        ] = 98.5
        prices.loc[
            (prices["date"] == event_date) & (prices["symbol"] == "300001"),
            "volume",
        ] = 1600
        result = analyze_turtle_events(prices, benchmarks, self.config())
        growth = result.events.loc[result.events["symbol"] == "300001"]
        self.assertEqual(1, len(growth))
        self.assertEqual("上证指数", growth.iloc[0]["market_trigger"])
        self.assertAlmostEqual(-1.5, growth.iloc[0]["shanghai_return_pct"])

    def test_volume_ratio_outside_range_is_not_turtle(self) -> None:
        prices, benchmarks = self.frames()
        event_date = sorted(prices["date"].unique())[21]
        prices.loc[(prices["symbol"] == "000001") & (prices["date"] == event_date), "volume"] = 2100
        result = analyze_turtle_events(prices, benchmarks, self.config())
        self.assertTrue(result.events.empty)

    def test_st_name_is_excluded(self) -> None:
        prices, benchmarks = self.frames()
        result = analyze_turtle_events(
            prices,
            benchmarks,
            self.config(),
            names={"000001": "*ST测试", "300001": "ST测试"},
        )
        self.assertTrue(result.events.empty)

    def test_bundle_writes_research_outputs(self) -> None:
        prices, benchmarks = self.frames()
        result = analyze_turtle_events(prices, benchmarks, self.config())
        with tempfile.TemporaryDirectory() as directory:
            paths = save_turtle_bundle(result, Path(directory))
            self.assertTrue(paths["events"].exists())
            self.assertTrue(paths["validation"].exists())
            self.assertIn("不等同于经典海龟交易法", paths["report"].read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
