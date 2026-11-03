from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PULLBACK_STRATEGY = "pullback_ignition"
PULLBACK_LABEL = "回调起爆"
GROWTH_PREFIXES = ("300", "301", "688", "689")


@dataclass(frozen=True)
class PullbackIgnitionRules:
    pullback_days: int = 10
    rise60_threshold_pct: float = 40.0
    pullback_min_pct: float = 10.0
    pullback_max_pct: float = 20.0
    main_gain_pct: float = 3.0
    growth_gain_pct: float = 5.0
    main_max_gain_pct: float = 9.5
    growth_max_gain_pct: float = 15.0
    pullback_volume_ratio: float = 0.70
    box_range_pct: float = 12.0
    platform_break_pct: float = 1.0
    min_volume_ratio: float = 1.8
    min_average_amount: float = 30_000_000.0
    body_ratio: float = 0.60
    max_upper_shadow_ratio: float = 0.30
    cooldown_days: int = 20
    main_stop_pct: float = 4.0
    growth_stop_pct: float = 5.0
    trailing_stop_pct: float = 5.0


def rules_from_config(config: dict) -> PullbackIgnitionRules:
    defaults = PullbackIgnitionRules()
    values = {}
    for field in defaults.__dataclass_fields__:
        key = f"pullback_ignition_{field}"
        values[field] = config.get(key, getattr(defaults, field))
    values["pullback_days"] = int(values["pullback_days"])
    values["cooldown_days"] = int(values["cooldown_days"])
    return PullbackIgnitionRules(**values)


def is_growth_board(symbol: str) -> bool:
    return str(symbol).zfill(6).startswith(GROWTH_PREFIXES)


def calculate_pullback_signals(
    data: pd.DataFrame,
    growth_board: bool,
    rules: PullbackIgnitionRules | None = None,
) -> pd.DataFrame:
    rule = rules or PullbackIgnitionRules()
    rows = data.copy().sort_values("date").reset_index(drop=True)
    if rows.empty:
        rows["pullback_ignition"] = pd.Series(dtype=bool)
        return rows

    rows["ma20"] = rows["close"].rolling(20, min_periods=20).mean()
    rows["ma60"] = rows["close"].rolling(60, min_periods=60).mean()
    rows["pullback_trend_ok"] = (
        (rows["close"] > rows["ma20"])
        & (rows["ma20"] > rows["ma60"])
        & (rows["ma20"] > rows["ma20"].shift(5))
        & (rows["ma60"] >= rows["ma60"].shift(5))
    ).fillna(False)

    high60 = rows["high"].shift(1).rolling(60, min_periods=60).max()
    low60 = rows["low"].shift(1).rolling(60, min_periods=60).min()
    rows["rise60_pct"] = (high60 / low60 - 1.0) * 100.0

    days = max(3, int(rule.pullback_days))
    stage_high = rows["high"].shift(1).rolling(20, min_periods=20).max()
    pullback_low = rows["low"].shift(1).rolling(days, min_periods=days).min()
    rows["pullback_pct"] = (stage_high / pullback_low - 1.0) * 100.0
    above_ma60 = (
        (rows["low"].shift(1) >= rows["ma60"].shift(1))
        .rolling(days, min_periods=days)
        .sum()
    )
    rows["pullback_holds_ma60"] = (above_ma60 == days).fillna(False)

    pullback_volume = rows["volume"].shift(1).rolling(days, min_periods=days).mean()
    prior_up_volume = rows["volume"].shift(days + 1).rolling(20, min_periods=20).mean()
    rows["pullback_volume_ratio"] = pullback_volume / prior_up_volume

    box_high = rows["high"].shift(1).rolling(days, min_periods=days).max()
    box_low = rows["low"].shift(1).rolling(days, min_periods=days).min()
    rows["box_range_pct"] = (box_high / box_low - 1.0) * 100.0
    rows["platform_break_pct"] = (rows["close"] / box_high - 1.0) * 100.0
    previous_close_high20 = rows["close"].shift(1).rolling(20, min_periods=20).max()

    rows["daily_gain_pct"] = rows["close"].pct_change(fill_method=None) * 100.0
    candle_range = (rows["high"] - rows["low"]).replace(0, pd.NA)
    rows["body_ratio"] = (rows["close"] - rows["open"]) / candle_range
    rows["upper_shadow_ratio"] = (
        rows["high"] - rows[["open", "close"]].max(axis=1)
    ) / candle_range

    previous_volume20 = rows["volume"].shift(1).rolling(20, min_periods=20).mean()
    rows["volume_ratio20"] = rows["volume"] / previous_volume20
    rows["amount_proxy"] = rows["close"] * rows["volume"] * 100.0
    rows["amount_top20_threshold"] = (
        rows["amount_proxy"].shift(1).rolling(30, min_periods=30).quantile(0.80)
    )
    rows["avg_amount20"] = rows["amount_proxy"].shift(1).rolling(20, min_periods=20).mean()

    minimum_gain = rule.growth_gain_pct if growth_board else rule.main_gain_pct
    maximum_gain = rule.growth_max_gain_pct if growth_board else rule.main_max_gain_pct
    raw_signal = (
        rows["pullback_trend_ok"]
        & (rows["rise60_pct"] >= rule.rise60_threshold_pct)
        & rows["pullback_pct"].between(rule.pullback_min_pct, rule.pullback_max_pct, inclusive="both")
        & rows["pullback_holds_ma60"]
        & (rows["pullback_volume_ratio"] <= rule.pullback_volume_ratio)
        & (rows["box_range_pct"] <= rule.box_range_pct)
        & (rows["close"] > previous_close_high20)
        & (rows["platform_break_pct"] >= rule.platform_break_pct)
        & rows["daily_gain_pct"].between(minimum_gain, maximum_gain, inclusive="both")
        & (rows["volume_ratio20"] >= rule.min_volume_ratio)
        & (rows["amount_proxy"] >= rows["amount_top20_threshold"])
        & (rows["avg_amount20"] >= rule.min_average_amount)
        & (rows["low"] >= rows["ma20"])
        & (rows["body_ratio"] >= rule.body_ratio)
        & (rows["upper_shadow_ratio"] <= rule.max_upper_shadow_ratio)
    ).fillna(False)
    rows["pullback_ignition_raw"] = raw_signal

    accepted = pd.Series(False, index=rows.index, dtype=bool)
    last_index = -10_000
    for index in rows.index[raw_signal]:
        if int(index) - last_index > int(rule.cooldown_days):
            accepted.loc[index] = True
            last_index = int(index)
    rows["pullback_ignition"] = accepted
    return rows


def latest_signal_is_valid(
    data: pd.DataFrame,
    symbol: str,
    config: dict,
    signal_date: str | None = None,
) -> bool:
    source = data
    if signal_date:
        source = data.loc[pd.to_datetime(data["date"], errors="coerce") <= pd.Timestamp(signal_date)]
    calculated = calculate_pullback_signals(
        source,
        is_growth_board(symbol),
        rules_from_config(config),
    )
    return bool(not calculated.empty and calculated.iloc[-1]["pullback_ignition"])


def scan_pullback_signals(
    prices: pd.DataFrame,
    watchlist: pd.DataFrame,
    output_path: Path,
    config: dict,
    recent_days: int = 10,
) -> pd.DataFrame:
    names = dict(zip(watchlist["code"].astype(str).str.zfill(6), watchlist["name"].astype(str)))
    all_dates = sorted(pd.to_datetime(prices["date"], errors="coerce").dropna().unique())
    recent = all_dates[-max(1, int(recent_days)):]
    age_by_date = {pd.Timestamp(date): len(recent) - 1 - index for index, date in enumerate(recent)}
    records: list[dict] = []
    rules = rules_from_config(config)
    for symbol, stock in prices.groupby("symbol", sort=False):
        code = str(symbol).zfill(6)
        name = names.get(code, code)
        upper_name = name.upper()
        if "ST" in upper_name or "退" in upper_name:
            continue
        calculated = calculate_pullback_signals(stock, is_growth_board(code), rules)
        hits = calculated.loc[calculated["pullback_ignition"] & calculated["date"].isin(recent)]
        for row in hits.itertuples():
            signal_date = pd.Timestamp(row.date)
            records.append(
                {
                    "date": signal_date.strftime("%Y-%m-%d"),
                    "age_trade_days": age_by_date.get(signal_date, 99),
                    "symbol": code,
                    "name": name,
                    "board": "创业/科创" if is_growth_board(code) else "沪深主板",
                    "signal": PULLBACK_LABEL,
                    "close": round(float(row.close), 4),
                    "daily_gain_pct": round(float(row.daily_gain_pct), 2),
                    "volume_ratio20": round(float(row.volume_ratio20), 2),
                    "rise60_pct": round(float(row.rise60_pct), 2),
                    "pullback_pct": round(float(row.pullback_pct), 2),
                    "pullback_volume_ratio": round(float(row.pullback_volume_ratio), 3),
                    "box_range_pct": round(float(row.box_range_pct), 2),
                    "platform_break_pct": round(float(row.platform_break_pct), 2),
                    "avg_amount20": round(float(row.avg_amount20), 2),
                    "initial_stop_pct": rules.growth_stop_pct if is_growth_board(code) else rules.main_stop_pct,
                    "trailing_stop_pct": rules.trailing_stop_pct,
                    "reason": "强趋势缩量回调结束，放量大阳突破平台；收盘确认、次日开盘模拟",
                }
            )
    columns = [
        "date", "age_trade_days", "symbol", "name", "board", "signal", "close",
        "daily_gain_pct", "volume_ratio20", "rise60_pct", "pullback_pct",
        "pullback_volume_ratio", "box_range_pct", "platform_break_pct", "avg_amount20",
        "initial_stop_pct", "trailing_stop_pct", "reason",
    ]
    result = pd.DataFrame(records, columns=columns)
    if not result.empty:
        result = result.sort_values(["date", "daily_gain_pct", "volume_ratio20"], ascending=[False, False, False])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    return result
