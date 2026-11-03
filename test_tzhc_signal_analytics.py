from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from tzhc_signal_analytics import (
    ORDINARY,
    ROCKET,
    SignalAnalyticsResult,
    build_stability_report,
    build_walkforward_validation,
    classify_market_regimes,
    evaluate_signal_frame,
    save_analytics_bundle,
    summarize_events,
)


class SignalAnalyticsTest(unittest.TestCase):
    def config(self) -> dict:
        return {
            "initial_capital": 100000,
            "max_position_pct": 18,
            "lot_size": 100,
            "commission_rate": 0,
            "min_commission": 0,
            "stamp_tax_rate": 0,
            "slippage_pct": 0,
            "price_limit_pct": 20,
        }

    def signal_frame(self) -> pd.DataFrame:
        rows = []
        start = datetime(2026, 1, 1)
        closes = [9, 9.5, 10, 11, 10.5, 12, 11.5, 13, 12.5, 14, 13.5, 15, 14.5, 16, 15.5]
        for index, close in enumerate(closes):
            rows.append(
                {
                    "date": start + timedelta(days=index),
                    "open": 10 if index == 3 else close,
                    "high": close * 1.05,
                    "low": close * 0.95,
                    "close": close,
                    "volume": 1000,
                    "raw_double_high": index in (2, 4),
                    "buy_candidate": index == 4,
                    "signal_score": 100 if index == 4 else 45,
                    "volume_ratio": 2.0,
                    "extension_pct": 3.0,
                }
            )
        return pd.DataFrame(rows)

    def test_forward_returns_and_mfe_mae_are_traceable(self) -> None:
        stock = self.signal_frame()
        regimes = {pd.Timestamp(row.date): "牛市" for row in stock.itertuples()}
        events = evaluate_signal_frame(stock, "000001", regimes, self.config(), 10000)
        self.assertEqual([ORDINARY, ROCKET], [item["signal_type"] for item in events])
        self.assertEqual(10.0, events[0]["return_1d_pct"])
        self.assertTrue(events[0]["mfe_10d_pct"] > 50)
        self.assertTrue(events[0]["mae_10d_pct"] < 0)
        self.assertTrue(events[1]["overlaps_prior_10d"])

    def test_summary_contains_win_loss_and_payoff(self) -> None:
        events = pd.DataFrame(
            {
                "signal_type": [ORDINARY, ORDINARY, ROCKET],
                "return_1d_pct": [10.0, -5.0, 2.0],
                "return_3d_pct": [10.0, -5.0, 2.0],
                "return_5d_pct": [10.0, -5.0, 2.0],
                "return_10d_pct": [10.0, -5.0, 2.0],
                "mfe_10d_pct": [12.0, 2.0, 5.0],
                "mae_10d_pct": [-2.0, -8.0, -1.0],
            }
        )
        summary = summarize_events(events)
        ordinary = summary.loc[(summary.signal_type == ORDINARY) & (summary.horizon_days == 10)].iloc[0]
        self.assertEqual(50.0, ordinary.win_rate_pct)
        self.assertEqual(2.0, ordinary.payoff_ratio)
        self.assertEqual(2.5, ordinary.avg_return_pct)

    def test_market_regime_and_stability_report(self) -> None:
        rows = []
        start = datetime(2025, 1, 1)
        for day in range(90):
            for symbol, base in (("000001", 10), ("000002", 20), ("000003", 30)):
                rows.append({"date": start + timedelta(days=day), "symbol": symbol, "close": base * (1 + day * 0.003)})
        regimes = classify_market_regimes(pd.DataFrame(rows))
        self.assertEqual("牛市", regimes.iloc[-1].regime)

        events = pd.DataFrame(
            {
                "signal_type": [ORDINARY] * 4 + [ROCKET] * 2,
                "signal_date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-01", "2026-01-02"],
                "return_10d_pct": [-1.0, -2.0, 3.0, -1.0, 1.0, 2.0],
                "overlaps_prior_10d": [False, True, True, True, False, True],
            }
        )
        validation = pd.DataFrame(columns=["signal_type", "sample", "avg_return_pct", "count"])
        stability = build_stability_report(events, validation)
        ordinary = stability.loc[stability.signal_type == ORDINARY].iloc[0]
        self.assertEqual(2, ordinary.max_consecutive_losses)
        self.assertEqual("样本不足", ordinary.assessment)

        walk_events = pd.DataFrame(
            {
                "signal_type": [ORDINARY] * 20 + [ROCKET] * 20,
                "signal_date": [f"2026-01-{index + 1:02d}" for index in range(20)] * 2,
                "return_10d_pct": [1.0, -0.5] * 20,
            }
        )
        walkforward = build_walkforward_validation(walk_events)
        self.assertEqual(8, len(walkforward))
        self.assertEqual(4, walkforward.fold.max())

    def test_bundle_files_are_written(self) -> None:
        empty = pd.DataFrame()
        result = SignalAnalyticsResult(
            events=pd.DataFrame([{"signal_date": "2026-01-01", "symbol": "000001"}]),
            summary=pd.DataFrame([{"signal_type": ORDINARY, "count": 1}]),
            regime_summary=empty,
            validation=empty,
            walkforward=empty,
            stability=pd.DataFrame([{"signal_type": ORDINARY, "assessment": "样本不足"}]),
            regimes=pd.DataFrame([{"date": "2026-01-01", "regime": "震荡市"}]),
            metadata={
                "generated_at": "2026-01-01T00:00:00", "price_start": "2026-01-01", "price_end": "2026-01-01",
                "symbols": 1, "ordinary_events": 1, "rocket_events": 0, "analysis_trade_value": 10000,
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = save_analytics_bundle(result, Path(directory))
            self.assertTrue(paths["report"].exists())
            self.assertTrue(paths["events"].exists())


if __name__ == "__main__":
    unittest.main()
