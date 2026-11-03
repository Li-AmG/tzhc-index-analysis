from __future__ import annotations

import csv
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from tzhc_engine import Candle, DEFAULT_CONFIG, sma


SCHEMA_VERSION = 1
RISK_NORMAL = "NORMAL"
RISK_SOFT_LOCK = "SOFT_LOCK"
RISK_HARD_LOCK = "HARD_LOCK"


@dataclass(frozen=True)
class PaperPaths:
    state: Path
    equity: Path
    trades: Path
    events: Path
    reports: Path

    @classmethod
    def for_root(cls, root: Path) -> "PaperPaths":
        return cls(
            state=root / "data" / "paper_account.json",
            equity=root / "data" / "paper_equity.csv",
            trades=root / "data" / "paper_trades.csv",
            events=root / "data" / "paper_events.csv",
            reports=root / "reports" / "paper_daily",
        )


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _save_state(paths: PaperPaths, state: dict) -> None:
    state["updated_at"] = _now()
    _atomic_text(paths.state, json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def load_paper_account(paths: PaperPaths) -> Optional[dict]:
    if not paths.state.exists():
        return None
    state = json.loads(paths.state.read_text(encoding="utf-8"))
    if int(state.get("schema_version", 0)) != SCHEMA_VERSION:
        raise ValueError("模拟账户数据版本不兼容，请先备份后重新初始化")
    state.setdefault("paper_watchlist", [])
    state.setdefault("pending_actions", {})
    return state


def _csv_text(rows: List[dict], fields: Iterable[str]) -> str:
    from io import StringIO

    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(fields), extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return "\ufeff" + output.getvalue()


def _materialize_csv(paths: PaperPaths, state: dict) -> None:
    equity_fields = [
        "date", "equity", "cash", "market_value", "positions", "daily_pnl",
        "daily_return_pct", "drawdown_pct", "risk_status",
    ]
    trade_fields = [
        "trade_id", "symbol", "strategy", "entry_date", "exit_date", "entry_price", "exit_price",
        "shares", "pnl", "pnl_pct", "fees", "holding_days", "reason",
    ]
    event_fields = ["event_id", "date", "symbol", "action", "price", "drawdown_pct", "reason", "created_at"]
    _atomic_text(paths.equity, _csv_text(state["equity_history"], equity_fields))
    _atomic_text(paths.trades, _csv_text(state["trades"], trade_fields))
    _atomic_text(paths.events, _csv_text(state["events"], event_fields))


def _append_event(
    state: dict,
    date: str,
    symbol: str,
    action: str,
    reason: str,
    price: float = 0.0,
    drawdown_pct: float = 0.0,
) -> None:
    state["event_counter"] += 1
    state["events"].append(
        {
            "event_id": f"E{state['event_counter']:08d}",
            "date": date,
            "symbol": symbol,
            "action": action,
            "price": round(float(price), 4),
            "drawdown_pct": round(float(drawdown_pct), 2),
            "reason": reason,
            "created_at": _now(),
        }
    )


def _report_text(state: dict, row: dict) -> str:
    date = row["date"]
    day_events = [item for item in state["events"] if item["date"] == date]
    positions = list(state["positions"].values())
    status_labels = {
        RISK_NORMAL: "正常，可按规则开仓",
        RISK_SOFT_LOCK: "5%风险锁：暂停开仓，等待人工复核",
        RISK_HARD_LOCK: "8%硬熔断：禁止开仓，等待清仓或重置账户",
    }
    lines = [
        f"# 模拟账户日报 {date}",
        "",
        "> 本报告为本地模拟研究记录，不连接券商，不构成投资建议。",
        "",
        "## 账户概况",
        "",
        f"- 账户权益：{row['equity']:,.2f}",
        f"- 现金：{row['cash']:,.2f}",
        f"- 持仓市值：{row['market_value']:,.2f}",
        f"- 当日盈亏：{row['daily_pnl']:+,.2f}（{row['daily_return_pct']:+.2f}%）",
        f"- 当前回撤：{row['drawdown_pct']:.2f}%",
        f"- 风控状态：{status_labels.get(row['risk_status'], row['risk_status'])}",
        "",
        "## 当前持仓",
        "",
    ]
    if positions:
        lines.extend(["| 代码 | 策略 | 数量 | 成本价 | 建仓日 | 最新价 |", "|---|---|---:|---:|---|---:|"])
        last_prices = state.get("last_prices", {})
        for position in sorted(positions, key=lambda item: item["symbol"]):
            latest = float(last_prices.get(position["symbol"], position["entry_price"]))
            lines.append(
                f"| {position['symbol']} | {position.get('strategy', 'quality_double_high')} | "
                f"{position['shares']} | {position['entry_price']:.4f} | "
                f"{position['entry_date']} | {latest:.4f} |"
            )
    else:
        lines.append("当前无持仓。")
    lines.extend(["", "## 当日事件", ""])
    if day_events:
        lines.extend(["| 事件 | 代码 | 价格 | 回撤 | 说明 |", "|---|---|---:|---:|---|"])
        for event in day_events:
            reason = str(event["reason"]).replace("|", "/")
            lines.append(
                f"| {event['action']} | {event['symbol']} | {event['price']:.4f} | "
                f"{event['drawdown_pct']:.2f}% | {reason} |"
            )
    else:
        lines.append("当日无成交或风控事件。")
    lines.extend(
        [
            "",
            "## 风控说明",
            "",
            f"- 软回撤阈值：{state['config']['soft_drawdown_pct']}%，触发后持久暂停开仓，必须人工填写原因解锁。",
            f"- 硬回撤阈值：{state['config']['hard_drawdown_pct']}%，触发后持久熔断，按下一可成交开盘价模拟清仓。",
            "- 同一交易日重复运行不会重复成交、重复记净值或重复生成事件。",
            "",
        ]
    )
    return "\n".join(lines)


def _write_report(paths: PaperPaths, state: dict, row: dict) -> Path:
    paths.reports.mkdir(parents=True, exist_ok=True)
    content = _report_text(state, row)
    dated = paths.reports / f"paper_daily_{row['date']}.md"
    latest = paths.reports / "latest.md"
    _atomic_text(dated, content)
    _atomic_text(latest, content)
    return dated


def initialize_paper_account(
    paths: PaperPaths,
    config: dict,
    opening_date: str,
    overwrite: bool = False,
) -> dict:
    if paths.state.exists() and not overwrite:
        state = load_paper_account(paths)
        if state is None:
            raise RuntimeError("模拟账户读取失败")
        return state
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(config)
    initial = round(float(cfg["initial_capital"]), 2)
    created = _now()
    state = {
        "schema_version": SCHEMA_VERSION,
        "account_id": f"PAPER-{datetime.now():%Y%m%d-%H%M%S}",
        "created_at": created,
        "updated_at": created,
        "opening_date": opening_date,
        "last_processed_date": opening_date,
        "initial_capital": initial,
        "cash": initial,
        "peak_equity": initial,
        "max_drawdown_pct": 0.0,
        "positions": {},
        "pending_entries": {},
        "pending_exits": {},
        "pending_actions": {},
        "last_prices": {},
        "paper_watchlist": [],
        "risk_lock": {
            "status": RISK_NORMAL,
            "triggered_at": "",
            "reason": "",
            "history": [],
        },
        "config": cfg,
        "event_counter": 0,
        "trade_counter": 0,
        "equity_history": [],
        "trades": [],
        "events": [],
    }
    opening_row = {
        "date": opening_date,
        "equity": initial,
        "cash": initial,
        "market_value": 0.0,
        "positions": 0,
        "daily_pnl": 0.0,
        "daily_return_pct": 0.0,
        "drawdown_pct": 0.0,
        "risk_status": RISK_NORMAL,
    }
    state["equity_history"].append(opening_row)
    _append_event(state, opening_date, "SYSTEM", "ACCOUNT_OPENED", "模拟账户已初始化")
    _save_state(paths, state)
    _materialize_csv(paths, state)
    _write_report(paths, state, opening_row)
    return state


def _commission(cfg: dict, amount: float) -> float:
    return max(float(cfg["min_commission"]), amount * float(cfg["commission_rate"]))


def _fill_price(cfg: dict, candle: Candle, action: str) -> float:
    slip = float(cfg["slippage_pct"]) / 100.0
    return candle.open * (1.0 + slip if action == "BUY" else 1.0 - slip)


def _can_fill(cfg: dict, rows: List[Candle], index: int, action: str) -> bool:
    if index <= 0:
        return True
    previous = rows[index - 1].close
    opened = rows[index].open
    limit = float(cfg["price_limit_pct"]) / 100.0
    if action == "BUY":
        return opened < previous * (1.0 + limit - 1e-9)
    return opened > previous * (1.0 - limit + 1e-9)


def _position_equity(state: dict) -> tuple[float, float]:
    market_value = 0.0
    for symbol, position in state["positions"].items():
        market_value += int(position["shares"]) * float(state["last_prices"].get(symbol, position["entry_price"]))
    return float(state["cash"]) + market_value, market_value


def _quality_entry(cfg: dict, rows: List[Candle], index: int) -> Optional[float]:
    closes = [row.close for row in rows]
    volumes = [row.volume for row in rows]
    row = rows[index]
    vol_avg = sma(volumes, index, int(cfg["volume_window"]))
    strategy = str(cfg.get("entry_strategy", "quality_double_high"))
    if strategy == "ma_cross":
        ma_fast_now = sma(closes, index, int(cfg["ma_fast"]))
        ma_fast_prev = sma(closes, index - 1, int(cfg["ma_fast"]))
        ma_trend = sma(closes, index, int(cfg["ma_trend"]))
        if not ma_fast_now or not ma_fast_prev or not ma_trend or not vol_avg or index <= 0:
            return None
        crossed = closes[index - 1] <= ma_fast_prev and row.close > ma_fast_now
        trend_ok = row.close > ma_trend
        vol_ok = row.volume >= vol_avg * float(cfg["min_volume_ratio"])
        return (row.close / ma_trend - 1.0) * 100.0 + row.volume / vol_avg if crossed and trend_ok and vol_ok else None

    long_window = int(cfg["ma_long"])
    trend_window = int(cfg["ma_trend"])
    if index < max(long_window + 1, int(cfg["volume_window"]), trend_window + 5):
        return None
    ma20_now = sma(closes, index, trend_window)
    ma20_prev5 = sma(closes, index - 5, trend_window)
    ma60_now = sma(closes, index, long_window)
    ma60_prev5 = sma(closes, index - 5, long_window)
    if not ma20_now or not ma20_prev5 or not ma60_now or not ma60_prev5 or not vol_avg:
        return None
    previous_high = max(closes[index - long_window:index])
    previous_high_before = max(closes[index - long_window - 1:index - 1])
    volume_ratio = row.volume / vol_avg
    daily_gain_pct = (row.close / closes[index - 1] - 1.0) * 100.0
    candle_range = row.high - row.low
    close_position = (row.close - row.low) / candle_range if candle_range > 0 else 0.0
    upper_shadow_ratio = (row.high - max(row.open, row.close)) / candle_range if candle_range > 0 else 1.0
    extension_pct = (row.close / ma20_now - 1.0) * 100.0
    trend_ok = row.close > ma20_now > ma60_now and ma20_now > ma20_prev5 and ma60_now >= ma60_prev5
    first_breakout = row.close >= previous_high and closes[index - 1] < previous_high_before
    volume_ok = float(cfg["min_volume_ratio"]) <= volume_ratio <= float(cfg["max_volume_ratio"])
    candle_ok = (
        row.close > row.open
        and close_position >= 0.65
        and upper_shadow_ratio <= 0.35
        and float(cfg["min_daily_gain_pct"]) <= daily_gain_pct <= float(cfg["max_daily_gain_pct"])
    )
    extension_ok = 0.0 <= extension_pct <= float(cfg["max_extension_pct"])
    return 100.0 + volume_ratio - extension_pct / 100.0 if trend_ok and first_breakout and volume_ok and candle_ok and extension_ok else None


def _exit_reason(cfg: dict, rows: List[Candle], index: int, position: dict) -> str:
    row = rows[index]
    closes = [item.close for item in rows]
    volumes = [item.volume for item in rows]
    position["peak_price"] = max(float(position["peak_price"]), row.close)
    pnl_pct = (row.close / float(position["entry_price"]) - 1.0) * 100.0
    strategy = str(position.get("strategy") or cfg.get("entry_strategy", "quality_double_high"))
    if strategy == "pullback_ignition":
        initial_stop = float(position.get("initial_stop_pct", cfg.get("pullback_ignition_main_stop_pct", 4.0)))
        trailing_stop = float(position.get("trailing_stop_pct", cfg.get("pullback_ignition_trailing_stop_pct", 5.0)))
        peak_drawdown = (float(position["peak_price"]) - row.close) / float(position["peak_price"]) * 100.0
        if pnl_pct <= -initial_stop:
            return f"回调起爆初始止损 {initial_stop:g}%"
        if peak_drawdown >= trailing_stop:
            return f"回调起爆从最高收盘回撤 {peak_drawdown:.1f}%（阈值{trailing_stop:g}%）"
        return ""
    if pnl_pct <= -float(cfg["stop_loss_pct"]):
        return f"止损 {cfg['stop_loss_pct']}%"
    if strategy == "ma_cross":
        if pnl_pct >= float(cfg["take_profit_pct"]):
            return f"止盈 {cfg['take_profit_pct']}%"
        ma_exit = sma(closes, index, int(cfg["ma_exit"]))
        return f"跌破MA{cfg['ma_exit']}退出" if ma_exit and row.close < ma_exit else ""
    ma5_now = sma(closes, index, 5)
    ma5_prev = sma(closes, index - 1, 5)
    ma20_now = sma(closes, index, 20)
    ma20_prev = sma(closes, index - 1, 20)
    vol_avg = sma(volumes, index, int(cfg["volume_window"]))
    volume_ratio = row.volume / vol_avg if vol_avg else 0.0
    peak_gain = (float(position["peak_price"]) / float(position["entry_price"]) - 1.0) * 100.0
    peak_drawdown = (float(position["peak_price"]) - row.close) / float(position["peak_price"]) * 100.0
    if peak_gain >= float(cfg["take_profit_pct"]) and peak_drawdown >= float(cfg["trailing_stop_pct"]):
        return f"盈利曾达 {peak_gain:.1f}%，从高点回撤 {peak_drawdown:.1f}%"
    if ma20_now and row.close < ma20_now and volume_ratio >= 1.2:
        return f"放量跌破BOLL20中轨，量比 {volume_ratio:.2f}"
    if ma5_now and ma5_prev and ma20_now and ma20_prev and ma5_now < ma20_now and ma5_prev >= ma20_prev:
        return "MA5下穿BOLL20中轨"
    return ""


def _lifecycle_action(cfg: dict, rows: List[Candle], index: int, position: dict) -> Optional[dict]:
    """Return one close-confirmed lifecycle action for an open position.

    The position stores lifecycle state so MA120/goddess transitions are not
    inferred from future data.  Action execution remains next-open and is
    handled by ``advance_paper_account``.
    """
    if index < 20:
        return None
    import pandas as pd

    from tzhc_lifecycle_strategy import calculate_lifecycle_signals, rules_from_config

    frame = pd.DataFrame(
        [
            {
                "date": pd.Timestamp(row.date), "symbol": row.symbol,
                "open": row.open, "high": row.high, "low": row.low,
                "close": row.close, "volume": row.volume,
            }
            for row in rows[: index + 1]
        ]
    )
    lifecycle_rules = rules_from_config({
        "ghost_lifecycle_price_proximity_pct": cfg.get("ghost_lifecycle_price_proximity_pct", 5.0),
        "ghost_lifecycle_min_volume_ratio": cfg.get("ghost_lifecycle_min_volume_ratio", 1.8),
        "ghost_lifecycle_min_amount": cfg.get("ghost_lifecycle_min_amount", 30_000_000.0),
    })
    calculated = calculate_lifecycle_signals(frame, lifecycle_rules)
    row = calculated.iloc[-1]
    close = float(row["close"])
    position["peak_price"] = max(float(position.get("peak_price", close)), close)
    previous_close = float(position.get("previous_close", position.get("entry_price", close)))
    position["down_days"] = int(position.get("down_days", 0)) + 1 if close < previous_close else 0
    position["previous_close"] = close
    if close >= float(row["ma120"]):
        position["goddess_seen"] = True
    gain_pct = (close / float(position["entry_price"]) - 1.0) * 100.0
    drawdown_pct = (float(position["peak_price"]) - close) / float(position["peak_price"]) * 100.0
    woman_again = bool(position.get("goddess_seen")) and float(row["boll_mid"]) <= close < float(row["ma120"])
    upper_break = bool(row["upper_break"])
    hot_stock = bool(row["hot_stock"])
    if hot_stock and close < float(row["ma5"]):
        return {"kind": "sell", "fraction": 1.0, "reason": "妖股跌破MA5，清仓"}
    if close < float(row["boll_mid"]):
        return {"kind": "sell", "fraction": 1.0, "reason": "跌破BOLL中轨，重新进入女鬼区"}
    if drawdown_pct >= float(cfg.get("ghost_lifecycle_drawdown_exit_pct", 5.0)):
        return {"kind": "sell", "fraction": 1.0, "reason": "从持仓高点回撤5%，清仓"}
    if woman_again:
        return {"kind": "sell", "fraction": 1.0, "reason": "女神退回女人区，清仓"}
    if drawdown_pct >= float(cfg.get("ghost_lifecycle_drawdown_reduce_pct", 3.0)) and not position.get("drawdown_reduced"):
        position["drawdown_reduced"] = True
        return {"kind": "sell", "fraction": 0.5, "reason": "从持仓高点回撤3%，减仓一半"}
    if gain_pct >= float(cfg.get("ghost_lifecycle_profit_reduce_pct", 10.0)) and not position.get("profit_reduced"):
        position["profit_reduced"] = True
        return {"kind": "sell", "fraction": 0.5, "reason": "盈利达到10%，减仓一半"}
    if bool(position.get("goddess_seen")) and upper_break and not position.get("lifecycle_added"):
        return {
            "kind": "add",
            "fraction": float(cfg.get("ghost_lifecycle_add_pct", 5.0)),
            "reason": "女神阶段首次突破BOLL上轨，加仓5%",
        }
    return None


def _lock_risk(state: dict, date: str, status: str, reason: str, drawdown_pct: float) -> None:
    previous = state["risk_lock"]["status"]
    state["risk_lock"]["status"] = status
    state["risk_lock"]["triggered_at"] = date
    state["risk_lock"]["reason"] = reason
    state["risk_lock"]["history"].append(
        {"at": _now(), "date": date, "from": previous, "to": status, "reason": reason}
    )


def _execute_paper_sell(
    state: dict,
    cfg: dict,
    global_date_index: dict,
    date: str,
    symbol: str,
    pending: dict,
    row: Candle,
) -> None:
    position = state["positions"][symbol]
    old_shares = int(position["shares"])
    fraction = min(1.0, max(0.0, float(pending.get("fraction", 1.0))))
    lot = max(1, int(cfg["lot_size"]))
    shares = old_shares if fraction >= 0.999 else int(old_shares * fraction // lot) * lot
    if shares <= 0:
        shares = old_shares
    price = _fill_price(cfg, row, "SELL")
    proceeds = shares * price
    sell_fee = _commission(cfg, proceeds) + proceeds * float(cfg["stamp_tax_rate"])
    allocated_entry_fee = float(position.get("entry_fee", 0.0)) * shares / max(old_shares, 1)
    invested = float(position["entry_price"]) * shares + allocated_entry_fee
    state["cash"] = float(state["cash"]) + proceeds - sell_fee
    position["shares"] = old_shares - shares
    position["entry_fee"] = max(0.0, float(position.get("entry_fee", 0.0)) - allocated_entry_fee)
    state["trade_counter"] += 1
    state["trades"].append(
        {
            "trade_id": f"T{state['trade_counter']:08d}",
            "symbol": symbol,
            "strategy": position.get("strategy", "quality_double_high"),
            "entry_date": position["entry_date"],
            "exit_date": date,
            "entry_price": round(float(position["entry_price"]), 4),
            "exit_price": round(price, 4),
            "shares": shares,
            "pnl": round(proceeds - sell_fee - invested, 2),
            "pnl_pct": round((proceeds - sell_fee - invested) / invested * 100.0 if invested else 0.0, 2),
            "fees": round(allocated_entry_fee + sell_fee, 2),
            "holding_days": max(1, global_date_index[date] - global_date_index.get(position["entry_date"], 0)),
            "reason": pending["reason"],
        }
    )
    if position["shares"] <= 0:
        state["positions"].pop(symbol, None)
    state["pending_exits"].pop(symbol, None)
    _append_event(
        state,
        date,
        symbol,
        "SELL_FILLED" if shares >= old_shares else "SELL_PARTIAL_FILLED",
        pending["reason"],
        price,
    )


def _execute_lifecycle_add(
    state: dict,
    cfg: dict,
    date: str,
    symbol: str,
    pending: dict,
    row: Candle,
) -> bool:
    if symbol not in state["positions"]:
        state["pending_actions"].pop(symbol, None)
        return False
    position = state["positions"][symbol]
    equity, _ = _position_equity(state)
    budget = min(float(state["cash"]), equity * float(pending.get("fraction", 5.0)) / 100.0)
    price = _fill_price(cfg, row, "BUY")
    lot = max(1, int(cfg["lot_size"]))
    shares = int((budget - float(cfg["min_commission"])) // price // lot) * lot
    amount = shares * price
    fee = _commission(cfg, amount) if shares else 0.0
    if shares <= 0 or amount + fee > float(state["cash"]):
        return False
    old_shares = int(position["shares"])
    old_price = float(position["entry_price"])
    state["cash"] = float(state["cash"]) - amount - fee
    position["entry_price"] = (old_shares * old_price + amount) / (old_shares + shares)
    position["shares"] = old_shares + shares
    position["entry_fee"] = float(position.get("entry_fee", 0.0)) + fee
    position["lifecycle_added"] = True
    state["pending_actions"].pop(symbol, None)
    _append_event(state, date, symbol, "ADD_FILLED", pending["reason"], price)
    return True
    _append_event(
        state,
        date,
        "SYSTEM",
        "HARD_LOCK" if status == RISK_HARD_LOCK else "SOFT_LOCK",
        reason,
        drawdown_pct=drawdown_pct,
    )


def advance_paper_account(
    paths: PaperPaths,
    price_map: Dict[str, List[Candle]],
    config: dict,
    until_date: Optional[str] = None,
) -> dict:
    state = load_paper_account(paths)
    if state is None:
        raise FileNotFoundError("模拟账户尚未初始化")
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(config)
    state["config"] = cfg
    all_dates = sorted({row.date for rows in price_map.values() for row in rows})
    dates = [
        date for date in all_dates
        if date > state["last_processed_date"] and (until_date is None or date <= until_date)
    ]
    indexes = {symbol: {row.date: index for index, row in enumerate(rows)} for symbol, rows in price_map.items()}
    global_date_index = {date: index for index, date in enumerate(all_dates)}

    for date in dates:
        for symbol, rows in price_map.items():
            index = indexes[symbol].get(date)
            if index is not None:
                state["last_prices"][symbol] = rows[index].close

        for symbol, pending in list(state["pending_exits"].items()):
            rows = price_map.get(symbol, [])
            index = indexes.get(symbol, {}).get(date)
            if index is None or date <= pending["signal_date"] or symbol not in state["positions"]:
                continue
            row = rows[index]
            if not _can_fill(cfg, rows, index, "SELL"):
                _append_event(state, date, symbol, "SELL_BLOCKED", "跌停价附近，卖出顺延", row.open)
                continue
            _execute_paper_sell(state, cfg, global_date_index, date, symbol, pending, row)

        for symbol, pending in list(state["pending_actions"].items()):
            rows = price_map.get(symbol, [])
            index = indexes.get(symbol, {}).get(date)
            if index is None or date <= pending["signal_date"] or symbol not in state["positions"]:
                continue
            row = rows[index]
            if pending.get("kind") != "add":
                state["pending_actions"].pop(symbol, None)
                continue
            if state["risk_lock"]["status"] != RISK_NORMAL:
                state["pending_actions"].pop(symbol, None)
                _append_event(state, date, symbol, "ADD_CANCELLED", "账户风险锁生效，加仓取消", row.open)
                continue
            if not _can_fill(cfg, rows, index, "BUY"):
                _append_event(state, date, symbol, "ADD_BLOCKED", "涨停价附近，加仓顺延", row.open)
                continue
            if not _execute_lifecycle_add(state, cfg, date, symbol, pending, row):
                state["pending_actions"].pop(symbol, None)
                _append_event(state, date, symbol, "ADD_CANCELLED", "可用资金不足一手，加仓取消", row.open)

        for symbol, pending in list(state["pending_entries"].items()):
            rows = price_map.get(symbol, [])
            index = indexes.get(symbol, {}).get(date)
            if index is None or date <= pending["signal_date"]:
                continue
            del state["pending_entries"][symbol]
            row = rows[index]
            if state["risk_lock"]["status"] != RISK_NORMAL or symbol in state["positions"]:
                _append_event(state, date, symbol, "BUY_CANCELLED", "风险锁或已有持仓，取消买入", row.open)
                continue
            if len(state["positions"]) >= int(cfg["max_positions"]):
                _append_event(state, date, symbol, "BUY_CANCELLED", "达到持仓数量上限", row.open)
                continue
            if not _can_fill(cfg, rows, index, "BUY"):
                _append_event(state, date, symbol, "BUY_BLOCKED", "涨停价附近，取消买入", row.open)
                continue
            price = _fill_price(cfg, row, "BUY")
            equity_before, _ = _position_equity(state)
            budget = min(float(state["cash"]), equity_before * float(cfg["max_position_pct"]) / 100.0)
            if pending.get("target_value") is not None:
                budget = min(budget, float(pending["target_value"]))
            lot = max(1, int(cfg["lot_size"]))
            shares = int((budget - float(cfg["min_commission"])) // price // lot) * lot
            amount = shares * price
            fee = _commission(cfg, amount) if shares else 0.0
            while shares > 0 and amount + fee > float(state["cash"]):
                shares -= lot
                amount = shares * price
                fee = _commission(cfg, amount) if shares else 0.0
            if shares <= 0:
                _append_event(state, date, symbol, "BUY_CANCELLED", "可用资金不足一手", row.open)
                continue
            state["cash"] = float(state["cash"]) - amount - fee
            state["positions"][symbol] = {
                "symbol": symbol,
                "shares": shares,
                "entry_price": price,
                "entry_date": date,
                "entry_fee": fee,
                "peak_price": price,
                "strategy": pending.get("strategy", cfg.get("entry_strategy", "quality_double_high")),
                "initial_stop_pct": pending.get("initial_stop_pct"),
                "trailing_stop_pct": pending.get("trailing_stop_pct"),
                "previous_close": price,
                "down_days": 0,
                "goddess_seen": False,
                "lifecycle_added": False,
                "drawdown_reduced": False,
                "profit_reduced": False,
            }
            _append_event(state, date, symbol, "BUY_FILLED", pending["reason"], price)

        equity, market_value = _position_equity(state)
        state["peak_equity"] = max(float(state["peak_equity"]), equity)
        drawdown = (float(state["peak_equity"]) - equity) / float(state["peak_equity"]) * 100.0 if state["peak_equity"] else 0.0
        state["max_drawdown_pct"] = max(float(state["max_drawdown_pct"]), drawdown)
        risk_status = state["risk_lock"]["status"]
        if risk_status != RISK_HARD_LOCK and drawdown >= float(cfg["hard_drawdown_pct"]):
            state["pending_entries"].clear()
            state["pending_actions"].clear()
            reason = f"回撤达到 {cfg['hard_drawdown_pct']}%，持久熔断并在下一可成交开盘清仓"
            _lock_risk(state, date, RISK_HARD_LOCK, reason, drawdown)
            for symbol in state["positions"]:
                state["pending_exits"][symbol] = {"signal_date": date, "reason": reason}
        elif risk_status == RISK_NORMAL and drawdown >= float(cfg["soft_drawdown_pct"]):
            state["pending_entries"].clear()
            reason = f"回撤达到 {cfg['soft_drawdown_pct']}%，持久暂停开仓，等待人工复核"
            _lock_risk(state, date, RISK_SOFT_LOCK, reason, drawdown)

        if state["risk_lock"]["status"] != RISK_HARD_LOCK:
            for symbol, position in list(state["positions"].items()):
                if symbol in state["pending_exits"] or symbol in state["pending_actions"]:
                    continue
                rows = price_map.get(symbol, [])
                index = indexes.get(symbol, {}).get(date)
                if index is None:
                    continue
                if str(position.get("strategy")) == "ghost_lifecycle":
                    action = _lifecycle_action(cfg, rows, index, position)
                    if action:
                        if action["kind"] == "sell":
                            state["pending_exits"][symbol] = {
                                "signal_date": date,
                                "reason": action["reason"],
                                "fraction": float(action.get("fraction", 1.0)),
                            }
                            _append_event(
                                state, date, symbol,
                                "SELL_SIGNAL" if float(action.get("fraction", 1.0)) >= 0.999 else "SELL_PARTIAL_SIGNAL",
                                "收盘确认，下一交易日开盘执行：" + action["reason"],
                                rows[index].close,
                                drawdown,
                            )
                        elif action["kind"] == "add" and state["risk_lock"]["status"] == RISK_NORMAL:
                            state["pending_actions"][symbol] = {
                                "signal_date": date,
                                "kind": "add",
                                "fraction": float(action.get("fraction", 5.0)),
                                "reason": action["reason"],
                            }
                            _append_event(
                                state, date, symbol, "ADD_SIGNAL",
                                "收盘确认，下一交易日开盘执行：" + action["reason"],
                                rows[index].close,
                                drawdown,
                            )
                    continue
                reason = _exit_reason(cfg, rows, index, position)
                if reason:
                    state["pending_exits"][symbol] = {"signal_date": date, "reason": reason, "fraction": 1.0}
                    _append_event(state, date, symbol, "SELL_SIGNAL", "收盘确认，下一交易日开盘执行：" + reason, rows[index].close, drawdown)

        if state["risk_lock"]["status"] == RISK_NORMAL:
            ranked = []
            for symbol, rows in price_map.items():
                if symbol in state["positions"] or symbol in state["pending_entries"]:
                    continue
                index = indexes[symbol].get(date)
                if index is None:
                    continue
                score = _quality_entry(cfg, rows, index)
                if score is not None:
                    ranked.append((score, symbol, rows[index]))
            ranked.sort(reverse=True)
            for _, symbol, row in ranked:
                if len(state["positions"]) + len(state["pending_entries"]) >= int(cfg["max_positions"]):
                    break
                reason = (
                    f"首次上穿MA{cfg['ma_fast']}且放量；信号日 {date}"
                    if str(cfg.get("entry_strategy")) == "ma_cross"
                    else f"高质量双高：趋势、首次突破、量价和K线质量合格；信号日 {date}"
                )
                state["pending_entries"][symbol] = {
                    "signal_date": date,
                    "reason": reason,
                    "strategy": cfg.get("entry_strategy", "quality_double_high"),
                }
                _append_event(state, date, symbol, "BUY_SIGNAL", "收盘确认，下一交易日开盘执行", row.close, drawdown)

        equity, market_value = _position_equity(state)
        previous_equity = float(state["equity_history"][-1]["equity"])
        daily_pnl = equity - previous_equity
        daily_return = daily_pnl / previous_equity * 100.0 if previous_equity else 0.0
        state["peak_equity"] = max(float(state["peak_equity"]), equity)
        drawdown = (float(state["peak_equity"]) - equity) / float(state["peak_equity"]) * 100.0 if state["peak_equity"] else 0.0
        row = {
            "date": date,
            "equity": round(equity, 2),
            "cash": round(float(state["cash"]), 2),
            "market_value": round(market_value, 2),
            "positions": len(state["positions"]),
            "daily_pnl": round(daily_pnl, 2),
            "daily_return_pct": round(daily_return, 4),
            "drawdown_pct": round(drawdown, 2),
            "risk_status": state["risk_lock"]["status"],
        }
        state["equity_history"].append(row)
        state["last_processed_date"] = date
        _save_state(paths, state)
        _materialize_csv(paths, state)
        _write_report(paths, state, row)

    _materialize_csv(paths, state)
    if state["equity_history"]:
        _write_report(paths, state, state["equity_history"][-1])
    return state


def add_paper_watch_symbol(paths: PaperPaths, symbol: str, name: str = "") -> dict:
    state = load_paper_account(paths)
    if state is None:
        raise FileNotFoundError("模拟账户尚未初始化")
    clean_symbol = symbol.strip().zfill(6)
    if len(clean_symbol) != 6 or not clean_symbol.isdigit():
        raise ValueError("股票代码必须是6位数字")
    if any(item["symbol"] == clean_symbol for item in state["paper_watchlist"]):
        return state
    state["paper_watchlist"].append({"symbol": clean_symbol, "name": name.strip() or clean_symbol})
    _append_event(
        state,
        state["last_processed_date"],
        clean_symbol,
        "WATCH_ADDED",
        f"加入模拟自选：{name.strip() or clean_symbol}",
    )
    _save_state(paths, state)
    _materialize_csv(paths, state)
    _write_report(paths, state, state["equity_history"][-1])
    return state


def remove_paper_watch_symbol(paths: PaperPaths, symbol: str) -> dict:
    state = load_paper_account(paths)
    if state is None:
        raise FileNotFoundError("模拟账户尚未初始化")
    clean_symbol = symbol.strip().zfill(6)
    before = len(state["paper_watchlist"])
    state["paper_watchlist"] = [item for item in state["paper_watchlist"] if item["symbol"] != clean_symbol]
    if len(state["paper_watchlist"]) != before:
        _append_event(state, state["last_processed_date"], clean_symbol, "WATCH_REMOVED", "移出模拟自选")
        _save_state(paths, state)
        _materialize_csv(paths, state)
        _write_report(paths, state, state["equity_history"][-1])
    return state


def queue_paper_buy_signal(
    paths: PaperPaths,
    price_map: Dict[str, List[Candle]],
    config: dict,
    symbol: str,
    target_value: float,
    confirmed: bool,
    strategy: str = "quality_double_high",
) -> dict:
    state = load_paper_account(paths)
    if state is None:
        raise FileNotFoundError("模拟账户尚未初始化")
    clean_symbol = symbol.strip().zfill(6)
    if not confirmed:
        raise ValueError("必须确认这是模拟委托，不会连接券商")
    if state["risk_lock"]["status"] != RISK_NORMAL:
        raise ValueError("当前风险锁禁止新开仓")
    if not any(item["symbol"] == clean_symbol for item in state["paper_watchlist"]):
        raise ValueError("请先把股票加入模拟自选")
    if clean_symbol in state["positions"]:
        raise ValueError("该股票已经持仓")
    if clean_symbol in state["pending_entries"]:
        raise ValueError("该股票已有待成交模拟买单")
    rows = price_map.get(clean_symbol, [])
    index_by_date = {row.date: index for index, row in enumerate(rows)}
    signal_date = state["last_processed_date"]
    index = index_by_date.get(signal_date)
    if index is None:
        raise ValueError("模拟账户日期与该股票最新行情不一致，请先推进账户或更新行情")
    cfg = {**DEFAULT_CONFIG, **config}
    clean_strategy = str(strategy).strip() or "quality_double_high"
    if clean_strategy == "pullback_ignition":
        import pandas as pd

        from tzhc_pullback_strategy import is_growth_board, latest_signal_is_valid, rules_from_config

        frame = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp(row.date),
                    "symbol": clean_symbol,
                    "open": row.open,
                    "high": row.high,
                    "low": row.low,
                    "close": row.close,
                    "volume": row.volume,
                }
                for row in rows
            ]
        )
        if not latest_signal_is_valid(frame, clean_symbol, cfg, signal_date):
            raise ValueError("当前交易日不是回调起爆信号，不能提交该策略模拟买入")
        pullback_rules = rules_from_config(cfg)
        initial_stop_pct = (
            pullback_rules.growth_stop_pct if is_growth_board(clean_symbol) else pullback_rules.main_stop_pct
        )
        trailing_stop_pct = pullback_rules.trailing_stop_pct
        strategy_label = "回调起爆"
    elif clean_strategy == "ghost_lifecycle":
        import pandas as pd

        from tzhc_lifecycle_strategy import latest_signal_is_valid

        frame = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp(row.date),
                    "symbol": clean_symbol,
                    "open": row.open,
                    "high": row.high,
                    "low": row.low,
                    "close": row.close,
                    "volume": row.volume,
                }
                for row in rows
            ]
        )
        if not latest_signal_is_valid(frame, cfg, signal_date):
            raise ValueError("当前交易日不是女鬼生命周期95%买入信号，不能提交模拟买入")
        initial_stop_pct = float(cfg.get("ghost_lifecycle_drawdown_exit_pct", 5.0))
        trailing_stop_pct = float(cfg.get("ghost_lifecycle_drawdown_exit_pct", 5.0))
        strategy_label = "女鬼生命周期95%"
    else:
        score = _quality_entry(cfg, rows, index)
        if score is None:
            raise ValueError("当前交易日不是高质量买入信号，普通双高不能模拟买入")
        initial_stop_pct = float(cfg["stop_loss_pct"])
        trailing_stop_pct = float(cfg["trailing_stop_pct"])
        strategy_label = "高质量双高"
    equity, _ = _position_equity(state)
    initial_pct = 5.0 if clean_strategy == "ghost_lifecycle" else float(config.get("max_position_pct", DEFAULT_CONFIG["max_position_pct"]))
    maximum = min(float(state["cash"]), equity * initial_pct / 100.0)
    requested = float(target_value)
    if requested <= 0:
        raise ValueError("计划买入市值必须大于0")
    if requested > maximum + 0.01:
        raise ValueError(f"计划买入市值超过单股风控上限 {maximum:,.2f} 元")
    reason = f"人工确认{strategy_label}买入信号；信号日 {signal_date}；计划市值 {requested:,.2f} 元"
    state["pending_entries"][clean_symbol] = {
        "signal_date": signal_date,
        "reason": reason,
        "target_value": round(requested, 2),
        "source": "paper_watchlist",
        "strategy": clean_strategy,
        "initial_stop_pct": initial_stop_pct,
        "trailing_stop_pct": trailing_stop_pct,
    }
    _append_event(
        state,
        signal_date,
        clean_symbol,
        "BUY_QUEUED",
        reason + "，下一交易日开盘模拟成交",
        rows[index].close,
        state["equity_history"][-1]["drawdown_pct"],
    )
    _save_state(paths, state)
    _materialize_csv(paths, state)
    _write_report(paths, state, state["equity_history"][-1])
    return state


def unlock_soft_risk(paths: PaperPaths, reason: str, confirmed: bool) -> dict:
    state = load_paper_account(paths)
    if state is None:
        raise FileNotFoundError("模拟账户尚未初始化")
    if not confirmed:
        raise ValueError("必须确认已完成人工复核")
    clean_reason = reason.strip()
    if len(clean_reason) < 4:
        raise ValueError("请填写至少4个字的复核说明")
    if state["risk_lock"]["status"] != RISK_SOFT_LOCK:
        raise ValueError("当前不是5%暂停状态，不能执行人工解锁")
    date = state["last_processed_date"]
    previous_reason = state["risk_lock"].get("reason", "")
    state["risk_lock"]["history"].append(
        {"at": _now(), "date": date, "from": RISK_SOFT_LOCK, "to": RISK_NORMAL, "reason": clean_reason}
    )
    state["risk_lock"].update({"status": RISK_NORMAL, "triggered_at": "", "reason": ""})
    _append_event(state, date, "SYSTEM", "SOFT_UNLOCK", f"人工复核解锁：{clean_reason}；原锁定原因：{previous_reason}")
    if state["equity_history"]:
        state["equity_history"][-1]["risk_status"] = RISK_NORMAL
    _save_state(paths, state)
    _materialize_csv(paths, state)
    _write_report(paths, state, state["equity_history"][-1])
    return state
