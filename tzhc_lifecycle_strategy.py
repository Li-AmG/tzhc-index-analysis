"""女鬼 -> 女人 -> 女神生命周期策略。

This module is intentionally independent from ordinary double-high and
pullback-ignition signals.  It only creates close-confirmed candidates; the
paper account executes them on the next tradable open.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


LIFECYCLE_STRATEGY = "ghost_lifecycle"
LIFECYCLE_LABEL = "女鬼生命周期95%"


@dataclass(frozen=True)
class LifecycleRules:
    price_proximity_pct: float = 5.0
    min_volume_ratio: float = 1.8
    min_amount: float = 30_000_000.0
    recent_days: int = 800


def rules_from_config(config: dict) -> LifecycleRules:
    defaults = LifecycleRules()
    return LifecycleRules(
        price_proximity_pct=float(config.get("ghost_lifecycle_price_proximity_pct", defaults.price_proximity_pct)),
        min_volume_ratio=float(config.get("ghost_lifecycle_min_volume_ratio", defaults.min_volume_ratio)),
        min_amount=float(config.get("ghost_lifecycle_min_amount", defaults.min_amount)),
        recent_days=int(config.get("ghost_lifecycle_recent_days", defaults.recent_days)),
    )


def calculate_lifecycle_signals(
    data: pd.DataFrame,
    rules: LifecycleRules | None = None,
) -> pd.DataFrame:
    rule = rules or LifecycleRules()
    rows = data.copy()
    if rows.empty:
        for column in ("woman_day1", "lifecycle_buy", "goddess", "upper_break", "hot_stock"):
            rows[column] = pd.Series(dtype=bool)
        return rows
    rows["date"] = pd.to_datetime(rows["date"], errors="coerce")
    rows = rows.sort_values("date").reset_index(drop=True)
    for column in ("open", "high", "low", "close", "volume"):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")

    close = rows["close"]
    high = rows["high"]
    low = rows["low"]
    volume = rows["volume"]
    rows["ma5"] = close.rolling(5, min_periods=5).mean()
    rows["ma10"] = close.rolling(10, min_periods=10).mean()
    rows["ma20"] = close.rolling(20, min_periods=20).mean()
    rows["ma30"] = close.rolling(30, min_periods=30).mean()
    rows["ma60"] = close.rolling(60, min_periods=60).mean()
    rows["ma120"] = close.rolling(120, min_periods=120).mean()
    rows["boll_mid"] = rows["ma20"]
    sd20 = close.rolling(20, min_periods=20).std()
    rows["boll_upper"] = rows["ma20"] + 2.0 * sd20
    rows["boll_lower"] = rows["ma20"] - 2.0 * sd20
    rows["volume_ratio"] = volume / volume.rolling(20, min_periods=20).mean().shift(0)
    rows["previous_high60"] = close.shift(1).rolling(60, min_periods=60).max()
    rows["pct_chg"] = close.pct_change(fill_method=None) * 100.0

    prev_close = close.shift(1)
    prev_mid = rows["boll_mid"].shift(1)
    prev_upper = rows["boll_upper"].shift(1)
    mid_cross = (close > rows["boll_mid"]) & (prev_close <= prev_mid)
    recent_mid_cross = mid_cross.shift(1).rolling(30, min_periods=1).max().fillna(0).astype(bool)
    ghost_setup = (
        (rows["ma5"].shift(1) < rows["ma10"].shift(1))
        & (rows["ma10"].shift(1) < rows["ma20"].shift(1))
        & (rows["ma20"].shift(1) < rows["ma30"].shift(1))
        & (rows["ma30"].shift(1) < rows["ma60"].shift(1))
    )
    band_open = (rows["boll_upper"] > rows["boll_upper"].shift(5)).fillna(False)
    mid_rising = (rows["boll_mid"] > rows["boll_mid"].shift(1)).fillna(False)
    short_rising = (
        (rows["ma5"] > rows["ma5"].shift(1))
        & (rows["ma10"] > rows["ma10"].shift(1))
    ).fillna(False)
    ma5_launch = (
        (low <= rows["ma5"] * 1.02)
        & (close > rows["ma5"])
        & (close > rows["open"])
    ).fillna(False)
    daily_pct = rows["pct_chg"]
    rows["woman_day1"] = (
        mid_cross
        & ~recent_mid_cross
        & ghost_setup
        & (close > rows["ma30"])
        & (rows["volume_ratio"] >= 1.2)
        & band_open
        & mid_rising
        & short_rising
        & ma5_launch
        & daily_pct.between(0.5, 7.0, inclusive="both")
    ).fillna(False)
    rows["raw_double_high"] = (
        (close >= rows["previous_high60"] * (1.0 - rule.price_proximity_pct / 100.0))
        & (rows["volume_ratio"] >= rule.min_volume_ratio)
        & (close >= rows["boll_mid"])
    ).fillna(False)
    rows["lifecycle_buy"] = rows["woman_day1"].shift(1).fillna(False) & rows["raw_double_high"]
    rows["goddess"] = (close >= rows["ma120"]).fillna(False)
    rows["upper_break"] = (
        (close > rows["boll_upper"]) & (prev_close <= prev_upper)
    ).fillna(False)
    max_gain10 = daily_pct.rolling(10, min_periods=1).max()
    rows["hot_stock"] = (
        (max_gain10 >= 9.5)
        | (close / close.shift(20) - 1.0 >= 0.30)
    ).fillna(False)
    rows["amount_proxy"] = close * volume * 100.0
    rows["eligible"] = (
        rows["lifecycle_buy"]
        & (rows["amount_proxy"] >= rule.min_amount)
        & (rows["ma120"].notna())
    ).fillna(False)
    return rows


def latest_signal_is_valid(
    data: pd.DataFrame,
    config: dict,
    signal_date: str | None = None,
) -> bool:
    source = data
    if signal_date:
        dates = pd.to_datetime(data["date"], errors="coerce")
        source = data.loc[dates <= pd.Timestamp(signal_date)]
    result = calculate_lifecycle_signals(source, rules_from_config(config))
    return bool(not result.empty and bool(result.iloc[-1]["eligible"]))


def _is_excluded(name: str) -> bool:
    upper = str(name).upper()
    return "ST" in upper or "退" in str(name)


def scan_lifecycle_signals(
    prices: pd.DataFrame,
    watchlist: pd.DataFrame,
    output_path: Path,
    config: dict,
    recent_days: int | None = None,
) -> pd.DataFrame:
    rules = rules_from_config(config)
    names = dict(zip(watchlist["code"].astype(str).str.zfill(6), watchlist["name"].astype(str)))
    all_dates = sorted(pd.to_datetime(prices["date"], errors="coerce").dropna().unique())
    recent = set(all_dates[-max(1, int(recent_days or rules.recent_days)):])
    age_by_date = {pd.Timestamp(date): len(all_dates) - 1 - index for index, date in enumerate(all_dates)}
    records: list[dict] = []
    for symbol, stock in prices.groupby("symbol", sort=False):
        code = str(symbol).zfill(6)
        name = names.get(code, code)
        if not code.startswith(("00", "30", "60", "68")) or _is_excluded(name):
            continue
        calculated = calculate_lifecycle_signals(stock, rules)
        hits = calculated.loc[calculated["eligible"] & calculated["date"].isin(recent)]
        for row in hits.itertuples():
            records.append({
                "date": pd.Timestamp(row.date).strftime("%Y-%m-%d"),
                "age_trade_days": age_by_date.get(pd.Timestamp(row.date), 99),
                "symbol": code,
                "name": name,
                "signal": LIFECYCLE_LABEL,
                "close": round(float(row.close), 4),
                "pct_chg": round(float(row.pct_chg), 2),
                "volume_ratio": round(float(row.volume_ratio), 2),
                "previous_high60": round(float(row.previous_high60), 4),
                "price_proximity_pct": round(float(row.close / row.previous_high60 * 100.0), 2),
                "ma120": round(float(row.ma120), 4),
                "boll_mid": round(float(row.boll_mid), 4),
                "boll_upper": round(float(row.boll_upper), 4),
                "reason": "女人第1日后的第2交易日双高；收盘确认，次日开盘模拟买入",
            })
    columns = [
        "date", "age_trade_days", "symbol", "name", "signal", "close", "pct_chg", "volume_ratio",
        "previous_high60", "price_proximity_pct", "ma120", "boll_mid", "boll_upper", "reason",
    ]
    result = pd.DataFrame(records, columns=columns)
    if not result.empty:
        result = result.sort_values(["date", "volume_ratio"], ascending=[False, False])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    return result
