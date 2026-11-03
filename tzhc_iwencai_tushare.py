"""Wencai-professional daily strategy backed by Tushare-style market data.

This module deliberately separates fields available from daily Tushare data
from Level-2 fields (seal amount, seal duration, auction order strength).  A
missing Level-2 field is reported as unavailable and never treated as a pass.
The output is research-only: signals are confirmed after the close and are
intended for next-session T+1 observation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class WencaiTushareRules:
    amount_min_yuan: float = 200_000_000.0
    turnover_min_pct: float = 5.0
    turnover_max_pct: float = 15.0
    volume_ratio_min: float = 1.20
    volume_ratio_max: float = 4.50
    breakout_window: int = 20
    max_return5_pct: float = 20.0
    max_return20_pct: float = 80.0
    max_extension_pct: float = 18.0
    max_board_height: int = 6
    close_from_high_pct: float = 2.0
    min_close_position: float = 0.85
    initial_stop_pct: float = 3.0
    trailing_stop_pct: float = 5.0


def _limit_pct(symbol: str) -> float:
    code = str(symbol).zfill(6)
    return 19.5 if code.startswith(("30", "68")) else 9.5


def _optional_numeric(frame: pd.DataFrame, column: str) -> tuple[pd.Series, bool]:
    if column not in frame.columns:
        return pd.Series(float("nan"), index=frame.index), False
    return pd.to_numeric(frame[column], errors="coerce"), True


def enrich_wencai_tushare_daily(
    data: pd.DataFrame,
    rules: WencaiTushareRules | None = None,
) -> pd.DataFrame:
    """Add Wencai/Tushare daily features to one symbol's price history."""
    rules = rules or WencaiTushareRules()
    result = data.copy().sort_values("date").reset_index(drop=True)
    result["symbol"] = result["symbol"].astype(str).str.zfill(6)
    for column in ("open", "high", "low", "close", "volume"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["pct_chg"] = result["close"].pct_change() * 100.0
    result["ma5"] = result["close"].rolling(5).mean()
    result["ma20"] = result["close"].rolling(20).mean()
    result["ma60"] = result["close"].rolling(60).mean()
    result["ma120"] = result["close"].rolling(120).mean()
    result["volume_ma20"] = result["volume"].rolling(20).mean()
    result["volume_ratio"] = result["volume"] / result["volume_ma20"].replace(0, pd.NA)
    result["return5_pct"] = result["close"].pct_change(5) * 100.0
    result["return20_pct"] = result["close"].pct_change(20) * 100.0
    result["prior_high"] = result["high"].rolling(rules.breakout_window).max().shift(1)
    result["breakout20"] = result["close"] > result["prior_high"]
    result["trend_ok"] = (
        (result["close"] > result["ma20"])
        & (result["ma20"] > result["ma60"])
        & (result["ma60"] > result["ma120"])
        & (result["ma20"] > result["ma20"].shift(5))
    )
    result["limit_pct"] = result["symbol"].map(_limit_pct)
    result["limit_up"] = result["pct_chg"] >= result["limit_pct"]
    prior_limit = result["limit_up"].rolling(5).max().shift(1).fillna(False).astype(bool)
    result["first_board"] = result["limit_up"] & ~prior_limit
    board_run = result["limit_up"].astype(int).groupby((~result["limit_up"]).cumsum()).cumsum()
    result["board_height"] = board_run.astype(int)
    candle_range = (result["high"] - result["low"]).replace(0, pd.NA)
    result["close_position"] = (result["close"] - result["low"]) / candle_range
    result["close_from_high_pct"] = (result["high"] - result["close"]) / result["high"].replace(0, pd.NA) * 100.0
    result["upper_shadow_ratio"] = (
        result["high"] - result[["open", "close"]].max(axis=1)
    ) / candle_range
    result["extension_pct"] = (result["close"] / result["ma20"] - 1.0) * 100.0
    result["close_strong"] = (
        (result["close_position"] >= rules.min_close_position)
        & (result["upper_shadow_ratio"] <= 0.20)
        & (result["close_from_high_pct"] <= rules.close_from_high_pct)
    )
    result["liquidity_amount_yuan"], amount_available = _optional_numeric(result, "amount_yuan")
    result["turnover_rate_pct"], turnover_available = _optional_numeric(result, "turnover_rate")
    result["is_st_or_risk"] = result["is_st_or_risk"] if "is_st_or_risk" in result else pd.Series(False, index=result.index)
    result["is_st_or_risk"] = result["is_st_or_risk"].fillna(False).astype(bool)
    result["supported_exchange"] = result["supported_exchange"] if "supported_exchange" in result else pd.Series(True, index=result.index)
    result["supported_exchange"] = result["supported_exchange"].fillna(True).astype(bool)
    result["amount_ok"] = result["liquidity_amount_yuan"] >= rules.amount_min_yuan
    result["turnover_ok"] = result["turnover_rate_pct"].between(
        rules.turnover_min_pct, rules.turnover_max_pct, inclusive="both"
    )
    # When Tushare market_daily is not available, keep the daily strategy
    # runnable but expose the missing fields explicitly for the UI.
    result["amount_field_available"] = amount_available & result["liquidity_amount_yuan"].notna()
    result["turnover_field_available"] = turnover_available & result["turnover_rate_pct"].notna()
    result["liquidity_ok"] = (
        (~result["amount_field_available"] | result["amount_ok"])
        & (~result["turnover_field_available"] | result["turnover_ok"])
    )
    result["volume_ok"] = result["volume_ratio"].between(
        rules.volume_ratio_min, rules.volume_ratio_max, inclusive="both"
    )
    result["extension_ok"] = result["extension_pct"].between(0, rules.max_extension_pct, inclusive="both")
    result["risk_ok"] = result["is_st_or_risk"].eq(False) & result["supported_exchange"]
    result["daily_candidate"] = (
        result["risk_ok"]
        & result["liquidity_ok"]
        & result["trend_ok"]
        & result["volume_ok"]
        & result["close_strong"]
        & result["extension_ok"]
        & (result["return5_pct"] <= rules.max_return5_pct)
        & (result["return20_pct"] <= rules.max_return20_pct)
        & (result["board_height"] <= rules.max_board_height)
    )
    result["first_board_candidate"] = result["daily_candidate"] & result["first_board"]
    result["breakout_candidate"] = result["daily_candidate"] & result["breakout20"]
    result["strategy_stage"] = ""
    result.loc[result["first_board_candidate"], "strategy_stage"] = "日线首板候选"
    result.loc[result["breakout_candidate"], "strategy_stage"] = "日线突破候选"
    result["signal_score"] = (
        result["trend_ok"].fillna(False).astype(int) * 25
        + result["first_board"].fillna(False).astype(int) * 20
        + result["breakout20"].fillna(False).astype(int) * 20
        + result["volume_ok"].fillna(False).astype(int) * 15
        + result["close_strong"].fillna(False).astype(int) * 10
        + result["liquidity_ok"].fillna(False).astype(int) * 10
    )
    missing = []
    if not result["amount_field_available"].any():
        missing.append("成交额")
    if not result["turnover_field_available"].any():
        missing.append("换手率")
    result["level2_status"] = "待接入封单额/封板时长/竞价强度"
    result["data_note"] = "日线Tushare字段" + (f"；未提供{','.join(missing)}" if missing else "")
    return result


def scan_wencai_tushare(
    prices: pd.DataFrame,
    watchlist: pd.DataFrame | None = None,
    recent_trading_days: int = 60,
    benchmark: pd.DataFrame | None = None,
    rules: WencaiTushareRules | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Scan recent Tushare daily data and return Wencai-style candidates."""
    rules = rules or WencaiTushareRules()
    names: dict[str, str] = {}
    if watchlist is not None and not watchlist.empty:
        names = watchlist.set_index("code")["name"].astype(str).to_dict()
    benchmark_ok: dict[pd.Timestamp, bool] = {}
    if benchmark is not None and not benchmark.empty and "date" in benchmark:
        bench = benchmark.copy()
        bench["date"] = pd.to_datetime(bench["date"], errors="coerce")
        bench["close"] = pd.to_numeric(bench["close"], errors="coerce")
        bench = bench.dropna(subset=["date", "close"]).sort_values("date")
        if not bench.empty:
            bench["ma20"] = bench["close"].rolling(20).mean()
            bench["ma60"] = bench["close"].rolling(60).mean()
            benchmark_ok = dict(zip(bench["date"], (bench["close"] > bench["ma20"]) & (bench["close"] > bench["ma60"])))
    recent = set(sorted(pd.to_datetime(prices["date"]).dropna().unique())[-max(1, recent_trading_days):])
    rows: list[dict] = []
    for symbol, stock in prices.groupby("symbol", sort=False):
        code = str(symbol).zfill(6)
        name = names.get(code, code)
        if "ST" in name.upper() or "退" in name:
            continue
        calc = enrich_wencai_tushare_daily(stock, rules)
        if benchmark_ok:
            calc["benchmark_trend_ok"] = calc["date"].map(benchmark_ok).fillna(False)
        else:
            calc["benchmark_trend_ok"] = True
        selected = calc.loc[
            calc["date"].isin(recent)
            & (calc["strategy_stage"] != "")
            & calc["benchmark_trend_ok"]
        ]
        for row in selected.itertuples():
            rows.append(
                {
                    "date": row.date,
                    "symbol": code,
                    "name": name,
                    "strategy_stage": row.strategy_stage,
                    "signal_score": int(row.signal_score),
                    "pct_chg": round(float(row.pct_chg), 2) if pd.notna(row.pct_chg) else None,
                    "volume_ratio": round(float(row.volume_ratio), 2) if pd.notna(row.volume_ratio) else None,
                    "amount_yuan": round(float(row.liquidity_amount_yuan), 0) if pd.notna(row.liquidity_amount_yuan) else None,
                    "turnover_rate_pct": round(float(row.turnover_rate_pct), 2) if pd.notna(row.turnover_rate_pct) else None,
                    "trend_ok": bool(row.trend_ok),
                    "first_board": bool(row.first_board),
                    "breakout20": bool(row.breakout20),
                    "benchmark_trend_ok": bool(row.benchmark_trend_ok),
                    "level2_status": row.level2_status,
                    "data_note": row.data_note,
                    "reason": "趋势多头 + 量能确认 + 收盘强；" + row.strategy_stage + "；次日T+1观察",
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(["date", "signal_score"], ascending=[False, False]).reset_index(drop=True)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save = result.copy()
        if not save.empty:
            save["date"] = pd.to_datetime(save["date"]).dt.strftime("%Y-%m-%d")
        save.to_csv(output_path, index=False, encoding="utf-8-sig")
    return result


def forward_returns_wencai(prices: pd.DataFrame, rules: WencaiTushareRules | None = None) -> pd.DataFrame:
    """Next-open entry and close exits for all daily candidates."""
    rules = rules or WencaiTushareRules()
    rows: list[dict] = []
    for symbol, stock in prices.groupby("symbol", sort=False):
        calc = enrich_wencai_tushare_daily(stock, rules)
        selected = calc["first_board_candidate"] | calc["breakout_candidate"]
        for idx in calc.index[selected.fillna(False)]:
            if idx + 1 >= len(calc):
                continue
            entry = float(calc.iloc[idx + 1]["open"])
            if not pd.notna(entry) or entry <= 0:
                continue
            record = {"symbol": str(symbol).zfill(6), "signal_date": calc.iloc[idx]["date"], "entry": entry}
            for horizon in (1, 3, 5, 10):
                exit_idx = idx + 1 + horizon
                record[f"ret_{horizon}d_pct"] = (
                    (float(calc.iloc[exit_idx]["close"]) / entry - 1.0) * 100.0
                    if exit_idx < len(calc) else None
                )
            rows.append(record)
    return pd.DataFrame(rows)
