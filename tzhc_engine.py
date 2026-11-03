from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


DEFAULT_CONFIG = {
    "initial_capital": 100000.0,
    "soft_drawdown_pct": 5.0,
    "hard_drawdown_pct": 8.0,
    "max_positions": 5,
    "max_position_pct": 18.0,
    "stop_loss_pct": 6.0,
    "take_profit_pct": 14.0,
    "entry_strategy": "quality_double_high",
    "min_volume_ratio": 1.8,
    "max_volume_ratio": 3.5,
    "volume_window": 20,
    "ma_fast": 5,
    "ma_trend": 20,
    "ma_long": 60,
    "ma_exit": 10,
    "max_extension_pct": 15.0,
    "min_daily_gain_pct": 1.5,
    "max_daily_gain_pct": 9.5,
    "trailing_stop_pct": 5.0,
    "lot_size": 100,
    "commission_rate": 0.00025,
    "min_commission": 5.0,
    "stamp_tax_rate": 0.0005,
    "slippage_pct": 0.05,
    "price_limit_pct": 10.0,
    "turtle_index_drop_pct": -2.0,
    "turtle_shanghai_drop_pct": -1.0,
    "turtle_ordinary_gain_pct": 5.0,
    "turtle_growth_gain_pct": 10.0,
    "turtle_min_volume_ratio": 1.2,
    "turtle_max_volume_ratio": 2.0,
    "ma_consolidation_days": 5,
    "ma_consolidation_range_pct": 8.0,
    "ma_entry_variant": "combined",
    "trade_start_date": "",
    "breakout_window": 60,
    "breakout_consolidation_days": 10,
    "breakout_consolidation_range_pct": 8.0,
    "breakout_volume_contraction_ratio": 0.85,
    "breakout_min_volume_ratio": 1.5,
    "breakout_max_volume_ratio": 4.0,
    "breakout_main_min_gain_pct": 4.0,
    "breakout_main_max_gain_pct": 9.5,
    "breakout_growth_min_gain_pct": 6.0,
    "breakout_growth_max_gain_pct": 15.0,
    "breakout_min_turnover_amount": 50000000.0,
    "breakout_profit_arm_pct": 10.0,
    "breakout_main_stop_pct": 6.0,
    "breakout_growth_stop_pct": 8.0,
    "breakout_main_trailing_pct": 6.0,
    "breakout_growth_trailing_pct": 8.0,
    "pullback_ignition_enabled": True,
    "pullback_ignition_pullback_days": 10,
    "pullback_ignition_rise60_threshold_pct": 40.0,
    "pullback_ignition_pullback_min_pct": 10.0,
    "pullback_ignition_pullback_max_pct": 20.0,
    "pullback_ignition_main_gain_pct": 3.0,
    "pullback_ignition_growth_gain_pct": 5.0,
    "pullback_ignition_main_max_gain_pct": 9.5,
    "pullback_ignition_growth_max_gain_pct": 15.0,
    "pullback_ignition_pullback_volume_ratio": 0.70,
    "pullback_ignition_box_range_pct": 12.0,
    "pullback_ignition_platform_break_pct": 1.0,
    "pullback_ignition_min_volume_ratio": 1.8,
    "pullback_ignition_min_average_amount": 30000000.0,
    "pullback_ignition_body_ratio": 0.60,
    "pullback_ignition_max_upper_shadow_ratio": 0.30,
    "pullback_ignition_cooldown_days": 20,
    "pullback_ignition_main_stop_pct": 4.0,
    "pullback_ignition_growth_stop_pct": 5.0,
    "pullback_ignition_trailing_stop_pct": 5.0,
    "ghost_lifecycle_enabled": True,
    "ghost_lifecycle_price_proximity_pct": 5.0,
    "ghost_lifecycle_min_volume_ratio": 1.8,
    "ghost_lifecycle_min_amount": 30000000.0,
    "ghost_lifecycle_recent_days": 800,
    "ghost_lifecycle_add_pct": 5.0,
    "ghost_lifecycle_profit_reduce_pct": 10.0,
    "ghost_lifecycle_drawdown_reduce_pct": 3.0,
    "ghost_lifecycle_drawdown_exit_pct": 5.0,
}


@dataclass
class Candle:
    date: str
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Position:
    symbol: str
    shares: int
    entry_price: float
    entry_date: str
    entry_fee: float
    peak_price: float


@dataclass
class Trade:
    symbol: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    shares: int
    pnl: float
    pnl_pct: float
    fees: float
    holding_days: int
    reason: str


@dataclass
class Signal:
    date: str
    symbol: str
    action: str
    reason: str
    close: float
    drawdown_pct: float


@dataclass
class BacktestResult:
    config: dict
    initial_capital: float
    final_equity: float
    total_return_pct: float
    max_drawdown_pct: float
    trade_count: int
    win_rate_pct: float
    annual_return_pct: float
    sharpe_ratio: float
    profit_factor: float
    total_fees: float
    equity_curve: List[dict]
    trades: List[Trade]
    signals: List[Signal]
    halted: bool
    halt_date: Optional[str]


def ensure_dirs(root: Path) -> None:
    for name in ("config", "data", "reports", "logs", "exports", "docs"):
        (root / name).mkdir(parents=True, exist_ok=True)


def load_config(path: Path) -> dict:
    if not path.exists():
        save_config(path, DEFAULT_CONFIG)
        return dict(DEFAULT_CONFIG)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    config = dict(DEFAULT_CONFIG)
    config.update(data)
    return config


def save_config(path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def _to_float(value: str, field: str, row_no: int) -> float:
    try:
        return float(str(value).strip())
    except Exception as exc:
        raise ValueError(f"第 {row_no} 行字段 {field} 不是数字: {value!r}") from exc


def load_prices_csv(path: Path) -> Dict[str, List[Candle]]:
    if not path.exists():
        raise FileNotFoundError(f"找不到CSV文件: {path}")

    price_map: Dict[str, List[Candle]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"date", "open", "high", "low", "close", "volume"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError("CSV表头至少需要: date,symbol,open,high,low,close,volume；symbol可为空但列建议保留。")
        for row_no, row in enumerate(reader, start=2):
            symbol = (row.get("symbol") or "SAMPLE").strip() or "SAMPLE"
            candle = Candle(
                date=str(row["date"]).strip(),
                symbol=symbol,
                open=_to_float(row["open"], "open", row_no),
                high=_to_float(row["high"], "high", row_no),
                low=_to_float(row["low"], "low", row_no),
                close=_to_float(row["close"], "close", row_no),
                volume=_to_float(row["volume"], "volume", row_no),
            )
            if min(candle.open, candle.high, candle.low, candle.close) <= 0 or candle.volume < 0:
                raise ValueError(f"第 {row_no} 行价格必须大于0，成交量不能为负。")
            if candle.high < max(candle.open, candle.close) or candle.low > min(candle.open, candle.close):
                raise ValueError(f"第 {row_no} 行 OHLC 关系不正确。")
            price_map.setdefault(symbol, []).append(candle)

    for rows in price_map.values():
        rows.sort(key=lambda x: x.date)
    return price_map


def write_sample_prices(path: Path, symbols: Iterable[str] = ("SAMPLE_A", "SAMPLE_B", "SAMPLE_C", "SAMPLE_D"), days: int = 260) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20260709)
    start = datetime.now().date() - timedelta(days=days * 2)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "symbol", "open", "high", "low", "close", "volume"])
        for symbol_index, symbol in enumerate(symbols):
            price = 18.0 + symbol_index * 5.0
            written = 0
            day = 0
            while written < days:
                current = start + timedelta(days=day)
                day += 1
                if current.weekday() >= 5:
                    continue
                drift = 0.0008 + math.sin(written / 18.0 + symbol_index) * 0.002
                shock = rng.gauss(0, 0.018)
                if written in (95 + symbol_index * 5, 160 + symbol_index * 4):
                    shock -= 0.055
                open_price = price * (1 + rng.gauss(0, 0.006))
                close_price = max(1.0, price * (1 + drift + shock))
                high = max(open_price, close_price) * (1 + abs(rng.gauss(0.01, 0.006)))
                low = min(open_price, close_price) * (1 - abs(rng.gauss(0.01, 0.006)))
                volume = int(900000 + rng.random() * 1200000)
                if close_price > open_price and written % 17 == 0:
                    volume = int(volume * 1.8)
                writer.writerow([
                    current.isoformat(),
                    symbol,
                    round(open_price, 3),
                    round(high, 3),
                    round(low, 3),
                    round(close_price, 3),
                    volume,
                ])
                price = close_price
                written += 1


def sma(values: List[float], idx: int, window: int) -> Optional[float]:
    if idx + 1 < window:
        return None
    chunk = values[idx + 1 - window : idx + 1]
    return sum(chunk) / window


def _price_on_or_before(rows: List[Candle], date: str, cursor: int) -> Tuple[Optional[Candle], int]:
    while cursor + 1 < len(rows) and rows[cursor + 1].date <= date:
        cursor += 1
    if cursor >= 0 and rows[cursor].date <= date:
        return rows[cursor], cursor
    return None, cursor


def run_backtest(price_map: Dict[str, List[Candle]], config: dict) -> BacktestResult:
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(config)

    initial_capital = float(cfg["initial_capital"])
    cash = initial_capital
    positions: Dict[str, Position] = {}
    pending_entries: Dict[str, Tuple[str, str]] = {}
    pending_exits: Dict[str, Tuple[str, str]] = {}
    trades: List[Trade] = []
    signals: List[Signal] = []
    equity_curve: List[dict] = []
    peak_equity = initial_capital
    max_drawdown = 0.0
    halted = False
    halt_date: Optional[str] = None
    soft_paused = False

    dates = sorted({c.date for rows in price_map.values() for c in rows})
    cursors = {symbol: -1 for symbol in price_map}
    close_history = {symbol: [c.close for c in rows] for symbol, rows in price_map.items()}
    volume_history = {symbol: [c.volume for c in rows] for symbol, rows in price_map.items()}
    row_by_date = {symbol: {row.date: row for row in rows} for symbol, rows in price_map.items()}
    date_index = {date: idx for idx, date in enumerate(dates)}

    def commission(amount: float) -> float:
        return max(float(cfg["min_commission"]), amount * float(cfg["commission_rate"]))

    def buy_fee(amount: float) -> float:
        return commission(amount)

    def sell_fee(amount: float) -> float:
        return commission(amount) + amount * float(cfg["stamp_tax_rate"])

    def fill_price(candle: Candle, action: str) -> float:
        slip = float(cfg["slippage_pct"]) / 100.0
        return candle.open * (1.0 + slip if action == "BUY" else 1.0 - slip)

    def previous_close(symbol: str, candle: Candle) -> Optional[float]:
        idx = next((i for i, row in enumerate(price_map[symbol]) if row.date == candle.date), -1)
        return price_map[symbol][idx - 1].close if idx > 0 else None

    def can_fill(symbol: str, candle: Candle, action: str) -> bool:
        prev = previous_close(symbol, candle)
        if not prev:
            return True
        limit = float(cfg["price_limit_pct"]) / 100.0
        if action == "BUY":
            return candle.open < prev * (1.0 + limit - 1e-9)
        return candle.open > prev * (1.0 - limit + 1e-9)

    def current_equity(current_date: str) -> float:
        total = cash
        for symbol, pos in positions.items():
            row, cursors[symbol] = _price_on_or_before(price_map[symbol], current_date, cursors[symbol])
            if row:
                total += pos.shares * row.close
        return total

    def exit_position(symbol: str, candle: Candle, reason: str, drawdown_pct: float, at_open: bool = True) -> None:
        nonlocal cash
        pos = positions.pop(symbol)
        price = fill_price(candle, "SELL") if at_open else candle.close * (1.0 - float(cfg["slippage_pct"]) / 100.0)
        proceeds = pos.shares * price
        fee = sell_fee(proceeds)
        cash += proceeds - fee
        invested = pos.entry_price * pos.shares + pos.entry_fee
        pnl = proceeds - fee - invested
        pnl_pct = pnl / invested * 100.0 if invested else 0.0
        trades.append(
            Trade(
                symbol=symbol,
                entry_date=pos.entry_date,
                exit_date=candle.date,
                entry_price=round(pos.entry_price, 4),
                exit_price=round(price, 4),
                shares=pos.shares,
                pnl=round(pnl, 2),
                pnl_pct=round(pnl_pct, 2),
                fees=round(pos.entry_fee + fee, 2),
                holding_days=max(1, date_index.get(candle.date, 0) - date_index.get(pos.entry_date, 0)),
                reason=reason,
            )
        )
        signals.append(Signal(candle.date, symbol, "SELL_FILLED", reason, round(price, 4), round(drawdown_pct, 2)))

    for date in dates:
        # Orders confirmed after a close are filled at the next available open.
        for symbol, (signal_date, reason) in list(pending_exits.items()):
            row = row_by_date[symbol].get(date)
            if not row or date <= signal_date:
                continue
            if can_fill(symbol, row, "SELL"):
                equity_before = current_equity(date)
                dd_before = (peak_equity - equity_before) / peak_equity * 100.0 if peak_equity else 0.0
                exit_position(symbol, row, reason, dd_before)
                pending_exits.pop(symbol, None)
            else:
                signals.append(Signal(date, symbol, "SELL_BLOCKED", "跌停价附近，卖出顺延", row.open, 0.0))

        for symbol, (signal_date, reason) in list(pending_entries.items()):
            row = row_by_date[symbol].get(date)
            if not row or date <= signal_date:
                continue
            pending_entries.pop(symbol, None)
            if halted or soft_paused or symbol in positions or len(positions) >= int(cfg["max_positions"]):
                signals.append(Signal(date, symbol, "BUY_CANCELLED", "风控状态或持仓上限取消买入", row.open, 0.0))
                continue
            if not can_fill(symbol, row, "BUY"):
                signals.append(Signal(date, symbol, "BUY_BLOCKED", "涨停价附近，取消买入", row.open, 0.0))
                continue
            price = fill_price(row, "BUY")
            equity_before = current_equity(date)
            budget = min(cash, equity_before * float(cfg["max_position_pct"]) / 100.0)
            lot = max(1, int(cfg["lot_size"]))
            shares = int((budget - float(cfg["min_commission"])) // price // lot) * lot
            amount = shares * price
            fee = buy_fee(amount) if shares else 0.0
            while shares > 0 and amount + fee > cash:
                shares -= lot
                amount = shares * price
                fee = buy_fee(amount) if shares else 0.0
            if shares <= 0:
                signals.append(Signal(date, symbol, "BUY_CANCELLED", "可用资金不足一手", row.open, 0.0))
                continue
            cash -= amount + fee
            positions[symbol] = Position(symbol, shares, price, row.date, fee, price)
            signals.append(Signal(date, symbol, "BUY_FILLED", reason, round(price, 4), 0.0))

        equity = current_equity(date)
        peak_equity = max(peak_equity, equity)
        drawdown_pct = (peak_equity - equity) / peak_equity * 100.0 if peak_equity else 0.0
        max_drawdown = max(max_drawdown, drawdown_pct)

        if not halted and drawdown_pct >= float(cfg["hard_drawdown_pct"]):
            halted = True
            halt_date = date
            pending_entries.clear()
            for symbol in positions:
                pending_exits[symbol] = (date, f"硬回撤达到 {cfg['hard_drawdown_pct']}%，次日开盘清仓停机")
            signals.append(Signal(date, "SYSTEM", "HALT", "触发硬回撤，停止后续开仓", 0.0, round(drawdown_pct, 2)))

        if not halted:
            for symbol, rows in price_map.items():
                row, cursors[symbol] = _price_on_or_before(rows, date, cursors[symbol])
                if row is None:
                    continue
                idx = cursors[symbol]
                closes = close_history[symbol]
                ma_exit = sma(closes, idx, int(cfg["ma_exit"]))
                if symbol in positions:
                    pos = positions[symbol]
                    pos.peak_price = max(pos.peak_price, row.close)
                    pnl_pct = (row.close / pos.entry_price - 1.0) * 100.0
                    if symbol in pending_exits:
                        continue
                    strategy = str(cfg.get("entry_strategy", "quality_double_high"))
                    growth_board = symbol.startswith(("300", "301", "688", "689"))
                    stop_pct = (
                        float(cfg["breakout_growth_stop_pct"] if growth_board else cfg["breakout_main_stop_pct"])
                        if strategy == "a_share_breakout" else float(cfg["stop_loss_pct"])
                    )
                    if pnl_pct <= -stop_pct:
                        pending_exits[symbol] = (date, f"止损 {stop_pct}%")
                        signals.append(Signal(date, symbol, "SELL_SIGNAL", "收盘触发止损，次日开盘执行", row.close, round(drawdown_pct, 2)))
                    elif strategy == "ma_cross" and pnl_pct >= float(cfg["take_profit_pct"]):
                        pending_exits[symbol] = (date, f"止盈 {cfg['take_profit_pct']}%")
                        signals.append(Signal(date, symbol, "SELL_SIGNAL", "收盘触发止盈，次日开盘执行", row.close, round(drawdown_pct, 2)))
                    elif strategy == "ma_cross" and ma_exit and row.close < ma_exit:
                        pending_exits[symbol] = (date, f"跌破MA{cfg['ma_exit']}退出")
                        signals.append(Signal(date, symbol, "SELL_SIGNAL", f"收盘跌破MA{cfg['ma_exit']}，次日开盘执行", row.close, round(drawdown_pct, 2)))
                    elif strategy != "ma_cross":
                        ma5_now = sma(closes, idx, 5)
                        ma5_prev = sma(closes, idx - 1, 5)
                        ma20_now = sma(closes, idx, 20)
                        ma20_prev = sma(closes, idx - 1, 20)
                        vol_avg = sma(volume_history[symbol], idx, int(cfg["volume_window"]))
                        volume_ratio = row.volume / vol_avg if vol_avg else 0.0
                        peak_gain_pct = (pos.peak_price / pos.entry_price - 1.0) * 100.0
                        peak_drawdown_pct = (pos.peak_price - row.close) / pos.peak_price * 100.0 if pos.peak_price else 0.0
                        profit_arm_pct = (
                            float(cfg["breakout_profit_arm_pct"])
                            if strategy == "a_share_breakout" else float(cfg["take_profit_pct"])
                        )
                        trailing_pct = (
                            float(cfg["breakout_growth_trailing_pct"] if growth_board else cfg["breakout_main_trailing_pct"])
                            if strategy == "a_share_breakout" else float(cfg["trailing_stop_pct"])
                        )
                        if peak_gain_pct >= profit_arm_pct and peak_drawdown_pct >= trailing_pct:
                            reason = f"盈利曾达 {peak_gain_pct:.1f}%，从高点回撤 {peak_drawdown_pct:.1f}%"
                            pending_exits[symbol] = (date, reason)
                            signals.append(Signal(date, symbol, "SELL_SIGNAL", "移动止盈触发，次日开盘执行", row.close, round(drawdown_pct, 2)))
                        elif ma20_now and row.close < ma20_now and volume_ratio >= 1.2:
                            reason = f"放量跌破BOLL20中轨，量比 {volume_ratio:.2f}"
                            pending_exits[symbol] = (date, reason)
                            signals.append(Signal(date, symbol, "SELL_SIGNAL", "趋势破坏，次日开盘执行", row.close, round(drawdown_pct, 2)))
                        elif ma5_now and ma5_prev and ma20_now and ma20_prev and ma5_now < ma20_now and ma5_prev >= ma20_prev:
                            pending_exits[symbol] = (date, "MA5下穿BOLL20中轨")
                            signals.append(Signal(date, symbol, "SELL_SIGNAL", "趋势转弱，次日开盘执行", row.close, round(drawdown_pct, 2)))

            open_allowed = drawdown_pct < float(cfg["soft_drawdown_pct"])
            if not open_allowed and not soft_paused:
                signals.append(Signal(date, "SYSTEM", "PAUSE_BUY", f"软回撤达到 {cfg['soft_drawdown_pct']}%，暂停开新仓", 0.0, round(drawdown_pct, 2)))
            if open_allowed and soft_paused:
                signals.append(Signal(date, "SYSTEM", "RESUME_BUY", "回撤恢复到软阈值以内，恢复开仓", 0.0, round(drawdown_pct, 2)))
            soft_paused = not open_allowed

            if open_allowed:
                ranked: List[Tuple[float, str, Candle]] = []
                for symbol, rows in price_map.items():
                    if symbol in positions or symbol in pending_entries:
                        continue
                    idx = cursors[symbol]
                    if idx <= 0:
                        continue
                    row = rows[idx]
                    closes = close_history[symbol]
                    volumes = volume_history[symbol]
                    vol_avg = sma(volumes, idx, int(cfg["volume_window"]))
                    strategy = str(cfg.get("entry_strategy", "quality_double_high"))
                    if strategy == "ma_cross":
                        ma_fast_now = sma(closes, idx, int(cfg["ma_fast"]))
                        ma_fast_prev = sma(closes, idx - 1, int(cfg["ma_fast"]))
                        ma_trend = sma(closes, idx, int(cfg["ma_trend"]))
                        if not ma_fast_now or not ma_fast_prev or not ma_trend or not vol_avg:
                            continue
                        crossed = closes[idx - 1] <= ma_fast_prev and row.close > ma_fast_now
                        trend_ok = row.close > ma_trend
                        vol_ok = row.volume >= vol_avg * float(cfg["min_volume_ratio"])
                        if crossed and trend_ok and vol_ok:
                            score = (row.close / ma_trend - 1.0) * 100.0 + row.volume / vol_avg
                            ranked.append((score, symbol, row))
                    elif strategy == "ma_pullback":
                        ma5_now = sma(closes, idx, 5)
                        ma5_prev = sma(closes, idx - 1, 5)
                        ma20_now = sma(closes, idx, 20)
                        ma20_prev = sma(closes, idx - 1, 20)
                        consolidation_days = max(3, int(cfg.get("ma_consolidation_days", 5)))
                        ma20_past = sma(closes, idx - consolidation_days, 20)
                        if not ma5_now or not ma5_prev or not ma20_now or not ma20_prev:
                            continue
                        ma5_cross_ma20 = ma5_prev <= ma20_prev and ma5_now > ma20_now
                        close_cross_ma5 = closes[idx - 1] <= ma5_prev and row.close > ma5_now
                        start = max(0, idx - consolidation_days)
                        consolidation_closes = closes[start:idx]
                        held_above_ma20 = len(consolidation_closes) == consolidation_days and all(
                            value >= (sma(closes, position, 20) or float("inf"))
                            for position, value in zip(range(start, idx), consolidation_closes)
                        )
                        consolidation_range_pct = (
                            (max(consolidation_closes) - min(consolidation_closes)) / ma20_now * 100.0
                            if consolidation_closes else 999.0
                        )
                        pullback_reclaim = (
                            close_cross_ma5
                            and held_above_ma20
                            and ma20_past is not None
                            and ma20_now >= ma20_past
                            and consolidation_range_pct <= float(cfg.get("ma_consolidation_range_pct", 8.0))
                        )
                        variant = str(cfg.get("ma_entry_variant", "combined"))
                        entry_ok = {
                            "ma5_cross_ma20": ma5_cross_ma20,
                            "close_cross_ma5": close_cross_ma5,
                            "pullback_reclaim": pullback_reclaim,
                            "combined": ma5_cross_ma20 or close_cross_ma5 or pullback_reclaim,
                        }.get(variant, ma5_cross_ma20 or close_cross_ma5 or pullback_reclaim)
                        if entry_ok:
                            score = (
                                (60.0 if ma5_cross_ma20 else 0.0)
                                + (40.0 if pullback_reclaim else 0.0)
                                + (20.0 if close_cross_ma5 else 0.0)
                                + max(-10.0, min(10.0, (row.close / ma20_now - 1.0) * 100.0))
                            )
                            ranked.append((score, symbol, row))
                    elif strategy == "a_share_breakout":
                        excluded_symbols = set(cfg.get("excluded_symbols", []))
                        if symbol in excluded_symbols:
                            continue
                        breakout_window = int(cfg.get("breakout_window", 60))
                        consolidation_days = int(cfg.get("breakout_consolidation_days", 10))
                        minimum_history = max(65, breakout_window + 2, consolidation_days + 21)
                        if idx < minimum_history:
                            continue
                        ma20_now = sma(closes, idx, 20)
                        ma20_prev5 = sma(closes, idx - 5, 20)
                        ma60_now = sma(closes, idx, 60)
                        ma60_prev5 = sma(closes, idx - 5, 60)
                        prior_volume20 = sma(volumes, idx - 1, 20)
                        if not ma20_now or not ma20_prev5 or not ma60_now or not ma60_prev5 or not prior_volume20:
                            continue
                        previous_high = max(closes[idx - breakout_window:idx])
                        first_breakout = row.close > previous_high
                        trend_ok = row.close > ma20_now > ma60_now and ma20_now > ma20_prev5 and ma60_now >= ma60_prev5
                        box_rows = rows[idx - consolidation_days:idx]
                        box_high = max(item.high for item in box_rows)
                        box_low = min(item.low for item in box_rows)
                        box_range_pct = (box_high / box_low - 1.0) * 100.0 if box_low else 999.0
                        prior_base_volumes = volumes[idx - consolidation_days - 20:idx - consolidation_days]
                        prior_base_volume = sum(prior_base_volumes) / len(prior_base_volumes) if prior_base_volumes else 0.0
                        box_volume = sum(item.volume for item in box_rows) / len(box_rows)
                        contraction_ok = (
                            prior_base_volume > 0
                            and box_volume <= prior_base_volume * float(cfg.get("breakout_volume_contraction_ratio", 0.85))
                        )
                        volume_ratio = row.volume / prior_volume20
                        volume_ok = float(cfg["breakout_min_volume_ratio"]) <= volume_ratio <= float(cfg["breakout_max_volume_ratio"])
                        daily_gain_pct = (row.close / closes[idx - 1] - 1.0) * 100.0
                        growth_board = symbol.startswith(("300", "301", "688", "689"))
                        min_gain = float(cfg["breakout_growth_min_gain_pct"] if growth_board else cfg["breakout_main_min_gain_pct"])
                        max_gain = float(cfg["breakout_growth_max_gain_pct"] if growth_board else cfg["breakout_main_max_gain_pct"])
                        candle_range = row.high - row.low
                        close_position = (row.close - row.low) / candle_range if candle_range > 0 else 0.0
                        upper_shadow_ratio = (row.high - max(row.open, row.close)) / candle_range if candle_range > 0 else 1.0
                        candle_ok = row.close > row.open and close_position >= 0.65 and upper_shadow_ratio <= 0.35 and min_gain <= daily_gain_pct <= max_gain
                        liquidity_amount = sum(closes[position] * volumes[position] * 100.0 for position in range(idx - 20, idx)) / 20.0
                        liquidity_ok = liquidity_amount >= float(cfg.get("breakout_min_turnover_amount", 50000000.0))
                        if trend_ok and first_breakout and box_range_pct <= float(cfg["breakout_consolidation_range_pct"]) and contraction_ok and volume_ok and candle_ok and liquidity_ok:
                            score = 100.0 + daily_gain_pct + volume_ratio - box_range_pct / 10.0
                            ranked.append((score, symbol, row))
                    else:
                        long_window = int(cfg["ma_long"])
                        trend_window = int(cfg["ma_trend"])
                        if idx < max(long_window + 1, int(cfg["volume_window"]), trend_window + 5):
                            continue
                        ma20_now = sma(closes, idx, trend_window)
                        ma20_prev5 = sma(closes, idx - 5, trend_window)
                        ma60_now = sma(closes, idx, long_window)
                        ma60_prev5 = sma(closes, idx - 5, long_window)
                        if not ma20_now or not ma20_prev5 or not ma60_now or not ma60_prev5 or not vol_avg:
                            continue
                        previous_high = max(closes[idx - long_window : idx])
                        previous_high_before = max(closes[idx - long_window - 1 : idx - 1])
                        volume_ratio = row.volume / vol_avg
                        daily_gain_pct = (row.close / closes[idx - 1] - 1.0) * 100.0
                        candle_range = row.high - row.low
                        close_position = (row.close - row.low) / candle_range if candle_range > 0 else 0.0
                        upper_shadow_ratio = (row.high - max(row.open, row.close)) / candle_range if candle_range > 0 else 1.0
                        extension_pct = (row.close / ma20_now - 1.0) * 100.0
                        trend_ok = row.close > ma20_now > ma60_now and ma20_now > ma20_prev5 and ma60_now >= ma60_prev5
                        first_breakout = row.close >= previous_high and closes[idx - 1] < previous_high_before
                        volume_ok = float(cfg["min_volume_ratio"]) <= volume_ratio <= float(cfg["max_volume_ratio"])
                        candle_ok = (
                            row.close > row.open
                            and close_position >= 0.65
                            and upper_shadow_ratio <= 0.35
                            and float(cfg["min_daily_gain_pct"]) <= daily_gain_pct <= float(cfg["max_daily_gain_pct"])
                        )
                        extension_ok = 0.0 <= extension_pct <= float(cfg["max_extension_pct"])
                        if trend_ok and first_breakout and volume_ok and candle_ok and extension_ok:
                            score = 100.0 + volume_ratio - extension_pct / 100.0
                            ranked.append((score, symbol, row))
                ranked.sort(reverse=True)

                trade_start_date = str(cfg.get("trade_start_date", "") or "")
                for _, symbol, row in ranked:
                    if trade_start_date and date < trade_start_date:
                        continue
                    if len(positions) + len(pending_entries) >= int(cfg["max_positions"]):
                        break
                    if str(cfg.get("entry_strategy", "quality_double_high")) == "ma_cross":
                        reason = f"首次上穿MA{cfg['ma_fast']}且放量；信号日 {date}"
                    elif str(cfg.get("entry_strategy", "quality_double_high")) == "ma_pullback":
                        reason = f"均线买点 {cfg.get('ma_entry_variant', 'combined')}；信号日 {date}"
                    elif str(cfg.get("entry_strategy", "quality_double_high")) == "a_share_breakout":
                        reason = f"A股强趋势缩量箱体放量突破{cfg.get('breakout_window', 60)}日高点；信号日 {date}"
                    else:
                        reason = f"高质量双高：趋势、首次突破、量价和K线质量合格；信号日 {date}"
                    pending_entries[symbol] = (date, reason)
                    signals.append(Signal(row.date, symbol, "BUY_SIGNAL", "收盘确认，下一交易日开盘执行", row.close, round(drawdown_pct, 2)))

        equity = current_equity(date)
        peak_equity = max(peak_equity, equity)
        drawdown_pct = (peak_equity - equity) / peak_equity * 100.0 if peak_equity else 0.0
        max_drawdown = max(max_drawdown, drawdown_pct)
        equity_curve.append(
            {
                "date": date,
                "equity": round(equity, 2),
                "cash": round(cash, 2),
                "positions": len(positions),
                "drawdown_pct": round(drawdown_pct, 2),
            }
        )

    if dates:
        last_date = dates[-1]
        for symbol in list(positions.keys()):
            row, cursors[symbol] = _price_on_or_before(price_map[symbol], last_date, cursors[symbol])
            if row:
                equity = current_equity(last_date)
                drawdown_pct = (peak_equity - equity) / peak_equity * 100.0 if peak_equity else 0.0
                exit_position(symbol, row, "回测结束按收盘价清算", drawdown_pct, at_open=False)

    final_equity = current_equity(dates[-1]) if dates else initial_capital
    total_return_pct = (final_equity / initial_capital - 1.0) * 100.0
    wins = sum(1 for t in trades if t.pnl > 0)
    win_rate = wins / len(trades) * 100.0 if trades else 0.0
    total_fees = sum(t.fees for t in trades)
    returns = []
    for previous, current in zip(equity_curve, equity_curve[1:]):
        if previous["equity"]:
            returns.append(current["equity"] / previous["equity"] - 1.0)
    if len(returns) > 1:
        mean = sum(returns) / len(returns)
        variance = sum((value - mean) ** 2 for value in returns) / (len(returns) - 1)
        sharpe = mean / math.sqrt(variance) * math.sqrt(252) if variance > 0 else 0.0
    else:
        sharpe = 0.0
    years = max(1.0 / 252.0, max(1, len(equity_curve) - 1) / 252.0)
    annual_return = ((final_equity / initial_capital) ** (1.0 / years) - 1.0) * 100.0 if final_equity > 0 else -100.0
    profits = sum(t.pnl for t in trades if t.pnl > 0)
    losses = abs(sum(t.pnl for t in trades if t.pnl < 0))
    profit_factor = profits / losses if losses else (999.0 if profits else 0.0)

    return BacktestResult(
        config=cfg,
        initial_capital=round(initial_capital, 2),
        final_equity=round(final_equity, 2),
        total_return_pct=round(total_return_pct, 2),
        max_drawdown_pct=round(max_drawdown, 2),
        trade_count=len(trades),
        win_rate_pct=round(win_rate, 2),
        annual_return_pct=round(annual_return, 2),
        sharpe_ratio=round(sharpe, 2),
        profit_factor=round(profit_factor, 2),
        total_fees=round(total_fees, 2),
        equity_curve=equity_curve,
        trades=trades,
        signals=signals,
        halted=halted,
        halt_date=halt_date,
    )


def save_result_bundle(result: BacktestResult, out_dir: Path, run_label: str = "") -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_label = "".join(char for char in str(run_label) if char.isalnum() or char in ("-", "_"))
    suffix = f"{safe_label}_{stamp}" if safe_label else stamp
    report_path = out_dir / f"回测报告_{suffix}.txt"
    equity_path = out_dir / f"equity_{suffix}.csv"
    trades_path = out_dir / f"trades_{suffix}.csv"
    signals_path = out_dir / f"signals_{suffix}.csv"

    with report_path.open("w", encoding="utf-8") as f:
        f.write("量化回撤控制系统 5%-8% 模拟回测报告\n")
        f.write("=" * 52 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"初始资金: {result.initial_capital:.2f}\n")
        f.write(f"期末权益: {result.final_equity:.2f}\n")
        f.write(f"总收益率: {result.total_return_pct:.2f}%\n")
        f.write(f"最大回撤: {result.max_drawdown_pct:.2f}%\n")
        f.write(f"软回撤阈值: {result.config['soft_drawdown_pct']}%（暂停开新仓）\n")
        f.write(f"硬回撤阈值: {result.config['hard_drawdown_pct']}%（模拟清仓停机）\n")
        f.write(f"交易次数: {result.trade_count}\n")
        f.write(f"胜率: {result.win_rate_pct:.2f}%\n")
        f.write(f"年化收益率: {result.annual_return_pct:.2f}%\n")
        f.write(f"夏普比率: {result.sharpe_ratio:.2f}\n")
        f.write(f"盈亏比因子: {result.profit_factor:.2f}\n")
        f.write(f"交易费用合计: {result.total_fees:.2f}\n")
        f.write(f"是否触发停机: {'是，日期 ' + result.halt_date if result.halted else '否'}\n\n")
        f.write("重要说明：本系统只用于历史回测和模拟演示，不构成投资建议，不保证收益，不接实盘账户。\n\n")
        f.write("最近交易:\n")
        for trade in result.trades[-20:]:
            f.write(
                f"{trade.symbol} {trade.entry_date}->{trade.exit_date} "
                f"{trade.entry_price}->{trade.exit_price} shares={trade.shares} "
                f"pnl={trade.pnl} {trade.pnl_pct}% fees={trade.fees} days={trade.holding_days} reason={trade.reason}\n"
            )

    with equity_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "equity", "cash", "positions", "drawdown_pct"])
        writer.writeheader()
        writer.writerows(result.equity_curve)

    with trades_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["symbol", "entry_date", "exit_date", "entry_price", "exit_price", "shares", "pnl", "pnl_pct", "fees", "holding_days", "reason"],
        )
        writer.writeheader()
        for trade in result.trades:
            writer.writerow(asdict(trade))

    with signals_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "symbol", "action", "reason", "close", "drawdown_pct"])
        writer.writeheader()
        for signal in result.signals:
            writer.writerow(asdict(signal))

    return {
        "report": str(report_path),
        "equity": str(equity_path),
        "trades": str(trades_path),
        "signals": str(signals_path),
    }
