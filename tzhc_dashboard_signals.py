from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class SignalRules:
    price_window: int = 60
    volume_window: int = 20
    min_volume_ratio: float = 1.8
    max_volume_ratio: float = 3.5
    max_extension_pct: float = 15.0
    min_daily_gain_pct: float = 1.5
    max_daily_gain_pct: float = 9.5
    stop_loss_pct: float = 6.0
    profit_arm_pct: float = 14.0
    trailing_stop_pct: float = 5.0


def calculate_signals(data: pd.DataFrame, rules: SignalRules | None = None) -> pd.DataFrame:
    rules = rules or SignalRules()
    result = data.copy().sort_values("date").reset_index(drop=True)
    result["ma5"] = result["close"].rolling(5).mean()
    result["ma20"] = result["close"].rolling(20).mean()
    result["ma60"] = result["close"].rolling(60).mean()
    result["boll_mid"] = result["ma20"]
    result["vol_ma20"] = result["volume"].rolling(rules.volume_window).mean()
    result["previous_high60"] = result["close"].rolling(rules.price_window).max().shift(1)
    result["volume_ratio"] = result["volume"] / result["vol_ma20"]
    result["daily_gain_pct"] = result["close"].pct_change() * 100.0
    candle_range = (result["high"] - result["low"]).replace(0, pd.NA)
    result["close_position"] = (result["close"] - result["low"]) / candle_range
    result["upper_shadow_ratio"] = (result["high"] - result[["open", "close"]].max(axis=1)) / candle_range
    result["extension_pct"] = (result["close"] / result["ma20"] - 1.0) * 100.0

    result["raw_double_high"] = (
        (result["close"] >= result["previous_high60"] * 0.98)
        & (result["volume_ratio"] >= rules.min_volume_ratio)
    )
    breakout = result["close"] >= result["previous_high60"]
    result["first_breakout"] = breakout & ~breakout.shift(1, fill_value=False)
    result["trend_ok"] = (
        (result["close"] > result["ma20"])
        & (result["ma20"] > result["ma60"])
        & (result["ma20"] > result["ma20"].shift(5))
        & (result["ma60"] >= result["ma60"].shift(5))
    )
    result["volume_ok"] = result["volume_ratio"].between(rules.min_volume_ratio, rules.max_volume_ratio, inclusive="both")
    result["candle_ok"] = (
        (result["close"] > result["open"])
        & (result["close_position"] >= 0.65)
        & (result["upper_shadow_ratio"] <= 0.35)
        & result["daily_gain_pct"].between(rules.min_daily_gain_pct, rules.max_daily_gain_pct, inclusive="both")
    )
    result["extension_ok"] = result["extension_pct"].between(0, rules.max_extension_pct, inclusive="both")

    result["signal_score"] = (
        result["trend_ok"].fillna(False).astype(int) * 25
        + result["first_breakout"].fillna(False).astype(int) * 25
        + result["volume_ok"].fillna(False).astype(int) * 20
        + result["candle_ok"].fillna(False).astype(int) * 20
        + result["extension_ok"].fillna(False).astype(int) * 10
    )
    result["buy_candidate"] = (
        result["raw_double_high"]
        & result["trend_ok"]
        & result["first_breakout"]
        & result["volume_ok"]
        & result["candle_ok"]
        & result["extension_ok"]
        & (result["signal_score"] >= 80)
    )

    result["trade_action"] = ""
    result["trade_reason"] = ""
    result["holding_state"] = False
    in_position = False
    entry_price = 0.0
    peak_price = 0.0
    entry_index = -1

    for index, row in result.iterrows():
        if not in_position and bool(row["buy_candidate"]):
            in_position = True
            entry_price = float(row["close"])
            peak_price = entry_price
            entry_index = index
            result.at[index, "trade_action"] = "BUY"
            result.at[index, "trade_reason"] = (
                f"趋势向上；首次突破60日高点；量比{row['volume_ratio']:.2f}；"
                f"收盘位于K线上部；距MA20为{row['extension_pct']:.1f}%"
            )
        elif in_position:
            close = float(row["close"])
            peak_price = max(peak_price, close)
            pnl_pct = (close / entry_price - 1.0) * 100.0
            peak_gain_pct = (peak_price / entry_price - 1.0) * 100.0
            peak_drawdown_pct = (peak_price - close) / peak_price * 100.0 if peak_price else 0.0
            stop_loss = pnl_pct <= -rules.stop_loss_pct
            trailing_stop = peak_gain_pct >= rules.profit_arm_pct and peak_drawdown_pct >= rules.trailing_stop_pct
            volume_break = (
                pd.notna(row["ma20"])
                and close < float(row["ma20"])
                and pd.notna(row["volume_ratio"])
                and float(row["volume_ratio"]) >= 1.2
            )
            cross_down = (
                index > 0
                and pd.notna(row["ma5"])
                and pd.notna(row["ma20"])
                and pd.notna(result.at[index - 1, "ma5"])
                and pd.notna(result.at[index - 1, "ma20"])
                and float(row["ma5"]) < float(row["ma20"])
                and float(result.at[index - 1, "ma5"]) >= float(result.at[index - 1, "ma20"])
            )
            reason = ""
            if index > entry_index and stop_loss:
                reason = f"收盘亏损{pnl_pct:.1f}%，触发{rules.stop_loss_pct:.0f}%止损"
            elif index > entry_index and trailing_stop:
                reason = f"盈利曾达{peak_gain_pct:.1f}%，从高点回撤{peak_drawdown_pct:.1f}%"
            elif index > entry_index and volume_break:
                reason = f"放量跌破BOLL20中轨，量比{row['volume_ratio']:.2f}"
            elif index > entry_index and cross_down:
                reason = "MA5下穿BOLL20中轨，趋势转弱"
            if reason:
                result.at[index, "trade_action"] = "SELL"
                result.at[index, "trade_reason"] = reason
                in_position = False
                entry_price = 0.0
                peak_price = 0.0
                entry_index = -1
        result.at[index, "holding_state"] = in_position

    return result


def scan_watchlist_signals(
    prices: pd.DataFrame,
    watchlist: pd.DataFrame,
    output_path: Path,
    recent_trading_days: int = 10,
) -> pd.DataFrame:
    trading_dates = sorted(prices["date"].dropna().unique())
    recent_dates = trading_dates[-max(1, recent_trading_days) :]
    age_by_date = {pd.Timestamp(date): len(recent_dates) - 1 - index for index, date in enumerate(recent_dates)}
    names = watchlist.set_index("code")["name"].to_dict()
    tags = watchlist.set_index("code")["tags"].to_dict()
    rows = []
    for code, stock in prices.groupby("symbol", sort=False):
        calculated = calculate_signals(stock)
        events = calculated.loc[calculated["date"].isin(recent_dates) & calculated["raw_double_high"]]
        for event in events.itertuples():
            is_rocket = bool(event.buy_candidate)
            missing = []
            if not event.trend_ok:
                missing.append("趋势未通过")
            if not event.first_breakout:
                missing.append("不是首次突破")
            if not event.volume_ok:
                missing.append("量比过低或过热")
            if not event.candle_ok:
                missing.append("K线质量不足")
            if not event.extension_ok:
                missing.append("距MA20过远")
            reason = (
                f"趋势、首次突破、量价、K线和乖离全部通过"
                if is_rocket
                else "；".join(missing) or "未通过高质量过滤"
            )
            event_date = pd.Timestamp(event.date)
            rows.append(
                {
                    "date": event_date,
                    "age_trade_days": age_by_date.get(event_date, recent_trading_days),
                    "symbol": str(code).zfill(6),
                    "name": names.get(str(code).zfill(6), "") or str(code).zfill(6),
                    "signal": "🚀 高质量买入候选" if is_rocket else "◆ 普通双高",
                    "score": int(event.signal_score),
                    "close": round(float(event.close), 2),
                    "volume_ratio": round(float(event.volume_ratio), 2),
                    "extension_pct": round(float(event.extension_pct), 2),
                    "reason": reason,
                    "tags": tags.get(str(code).zfill(6), ""),
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(["date", "signal", "score", "volume_ratio"], ascending=[False, False, False, False])
        save = result.copy()
        save["date"] = save["date"].dt.strftime("%Y-%m-%d")
    else:
        save = pd.DataFrame(columns=["date", "age_trade_days", "symbol", "name", "signal", "score", "close", "volume_ratio", "extension_pct", "reason", "tags"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save.to_csv(output_path, index=False, encoding="utf-8-sig")
    return result
