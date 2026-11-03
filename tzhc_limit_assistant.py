from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class LimitAssistantRules:
    ma_fast: int = 5
    ma_trend: int = 20
    ma_long: int = 60
    volume_window: int = 20
    breakout_window: int = 20
    min_volume_ratio: float = 1.10
    max_volume_ratio: float = 4.00
    min_close_position: float = 0.85
    max_upper_shadow_ratio: float = 0.20
    max_entry_extension_pct: float = 18.0
    initial_stop_pct: float = 3.0
    trailing_stop_pct: float = 5.0


def _limit_pct(symbol: str) -> float:
    code = str(symbol).zfill(6)
    return 19.5 if code.startswith(("30", "68")) else 9.5


def enrich_limit_assistant(data: pd.DataFrame, rules: LimitAssistantRules | None = None) -> pd.DataFrame:
    """Calculate daily-data equivalents of the visible limit-assistant signals."""
    rules = rules or LimitAssistantRules()
    result = data.copy().sort_values("date").reset_index(drop=True)
    result["symbol"] = result["symbol"].astype(str).str.zfill(6)
    result["prev_close"] = result["close"].shift(1)
    result["change_pct"] = result["close"].pct_change() * 100.0
    result["limit_pct"] = result["symbol"].map(_limit_pct)
    result["ma5"] = result["close"].rolling(rules.ma_fast).mean()
    result["ma20"] = result["close"].rolling(rules.ma_trend).mean()
    result["ma60"] = result["close"].rolling(rules.ma_long).mean()
    result["vol_ma20"] = result["volume"].rolling(rules.volume_window).mean()
    result["volume_ratio"] = result["volume"] / result["vol_ma20"]
    candle_range = (result["high"] - result["low"]).replace(0, pd.NA)
    result["close_position"] = (result["close"] - result["low"]) / candle_range
    result["upper_shadow_ratio"] = (
        result["high"] - result[["open", "close"]].max(axis=1)
    ) / candle_range
    result["extension_pct"] = (result["close"] / result["ma20"] - 1.0) * 100.0
    result["limit_up"] = result["change_pct"] >= result["limit_pct"]
    result["near_limit_up"] = result["change_pct"] >= result["limit_pct"] - 1.5
    result["previous_limit_up"] = result["limit_up"].shift(1, fill_value=False)
    recent_limit = result["limit_up"].rolling(5).max().shift(1).fillna(False).astype(bool)
    result["first_board"] = result["limit_up"] & ~recent_limit
    result["breakout20"] = result["close"] >= result["close"].rolling(rules.breakout_window).max().shift(1)
    result["trend_ok"] = (
        (result["close"] > result["ma20"])
        & (result["ma20"] > result["ma60"])
        & (result["ma20"] > result["ma20"].shift(5))
    )
    result["volume_ok"] = result["volume_ratio"].between(
        rules.min_volume_ratio, rules.max_volume_ratio, inclusive="both"
    )
    result["close_strong"] = (
        (result["close_position"] >= rules.min_close_position)
        & (result["upper_shadow_ratio"] <= rules.max_upper_shadow_ratio)
    )
    result["extension_ok"] = result["extension_pct"].between(0, rules.max_entry_extension_pct, inclusive="both")
    result["assistant_buy"] = (
        result["first_board"]
        & result["trend_ok"]
        & result["breakout20"]
        & result["volume_ok"]
        & result["close_strong"]
        & result["extension_ok"]
    )
    result["signal_score"] = (
        result["first_board"].fillna(False).astype(int) * 25
        + result["trend_ok"].fillna(False).astype(int) * 20
        + result["breakout20"].fillna(False).astype(int) * 20
        + result["volume_ok"].fillna(False).astype(int) * 15
        + result["close_strong"].fillna(False).astype(int) * 10
        + result["extension_ok"].fillna(False).astype(int) * 10
    )
    result["event_type"] = ""
    result.loc[result["limit_up"], "event_type"] = "涨停"
    result.loc[result["first_board"], "event_type"] = "首板"
    result.loc[result["limit_up"] & result["previous_limit_up"], "event_type"] = "连板"
    result.loc[result["near_limit_up"] & ~result["limit_up"], "event_type"] = "冲板未封"
    result["assistant_reason"] = ""
    for idx, row in result.loc[result["assistant_buy"]].iterrows():
        result.at[idx, "assistant_reason"] = (
            f"首板+20日突破；趋势多头；量比{float(row['volume_ratio']):.2f}；"
            f"收盘强度{float(row['close_position']):.0%}"
        )
    return result


def scan_limit_assistant(
    prices: pd.DataFrame,
    watchlist: pd.DataFrame | None = None,
    recent_trading_days: int = 10,
    output_path: Path | None = None,
) -> pd.DataFrame:
    names = {}
    if watchlist is not None and not watchlist.empty:
        names = watchlist.set_index("code")["name"].to_dict()
    recent = sorted(prices["date"].dropna().unique())[-max(1, recent_trading_days):]
    rows: list[dict] = []
    for symbol, stock in prices.groupby("symbol", sort=False):
        name = str(names.get(str(symbol).zfill(6), ""))
        if "ST" in name.upper() or "退" in name:
            continue
        calc = enrich_limit_assistant(stock)
        for row in calc.loc[calc["date"].isin(recent) & (calc["event_type"] != "")].itertuples():
            rows.append(
                {
                    "date": row.date,
                    "symbol": str(symbol).zfill(6),
                    "name": name or str(symbol).zfill(6),
                    "event_type": row.event_type,
                    "change_pct": round(float(row.change_pct), 2),
                    "volume_ratio": round(float(row.volume_ratio), 2) if pd.notna(row.volume_ratio) else None,
                    "signal_score": int(row.signal_score),
                    "buy_candidate": bool(row.assistant_buy),
                    "reason": row.assistant_reason or "尚未满足趋势、突破、量价和收盘强度全部条件",
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(["date", "buy_candidate", "signal_score"], ascending=[False, False, False])
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save = result.copy()
        if not save.empty:
            save["date"] = pd.to_datetime(save["date"]).dt.strftime("%Y-%m-%d")
        save.to_csv(output_path, index=False, encoding="utf-8-sig")
    return result


def event_forward_returns(prices: pd.DataFrame, rules: LimitAssistantRules | None = None) -> pd.DataFrame:
    rows: list[dict] = []
    for symbol, stock in prices.groupby("symbol", sort=False):
        calc = enrich_limit_assistant(stock, rules)
        for idx in calc.index[calc["assistant_buy"].fillna(False)]:
            if idx + 1 >= len(calc):
                continue
            entry = float(calc.iloc[idx + 1]["open"])
            record = {"symbol": str(symbol).zfill(6), "signal_date": calc.iloc[idx]["date"], "entry": entry}
            for horizon in (1, 3, 5, 10):
                exit_idx = idx + 1 + horizon
                record[f"ret_{horizon}d_pct"] = (
                    (float(calc.iloc[exit_idx]["close"]) / entry - 1.0) * 100.0
                    if exit_idx < len(calc) else None
                )
            rows.append(record)
    return pd.DataFrame(rows)
