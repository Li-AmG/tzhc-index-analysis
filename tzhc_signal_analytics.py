from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import pandas as pd

from tzhc_dashboard_signals import SignalRules, calculate_signals
from tzhc_engine import DEFAULT_CONFIG


HORIZONS = (1, 3, 5, 10)
ORDINARY = "普通双高"
ROCKET = "火箭"


@dataclass
class SignalAnalyticsResult:
    events: pd.DataFrame
    summary: pd.DataFrame
    regime_summary: pd.DataFrame
    validation: pd.DataFrame
    walkforward: pd.DataFrame
    stability: pd.DataFrame
    regimes: pd.DataFrame
    metadata: dict


def _rules_from_config(config: dict) -> SignalRules:
    return SignalRules(
        price_window=int(config.get("ma_long", 60)),
        volume_window=int(config.get("volume_window", 20)),
        min_volume_ratio=float(config.get("min_volume_ratio", 1.8)),
        max_volume_ratio=float(config.get("max_volume_ratio", 3.5)),
        max_extension_pct=float(config.get("max_extension_pct", 15.0)),
        min_daily_gain_pct=float(config.get("min_daily_gain_pct", 1.5)),
        max_daily_gain_pct=float(config.get("max_daily_gain_pct", 9.5)),
        stop_loss_pct=float(config.get("stop_loss_pct", 6.0)),
        profit_arm_pct=float(config.get("take_profit_pct", 14.0)),
        trailing_stop_pct=float(config.get("trailing_stop_pct", 5.0)),
    )


def classify_market_regimes(prices: pd.DataFrame) -> pd.DataFrame:
    data = prices[["date", "symbol", "close"]].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    data = data.dropna().sort_values(["date", "symbol"])
    pivot = data.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
    return_60 = (pivot / pivot.shift(60) - 1.0) * 100.0
    median_60 = return_60.median(axis=1, skipna=True)
    ma20 = pivot.rolling(20, min_periods=20).mean()
    valid = ma20.notna()
    breadth = ((pivot > ma20) & valid).sum(axis=1) / valid.sum(axis=1).replace(0, pd.NA) * 100.0
    result = pd.DataFrame(
        {
            "date": pivot.index,
            "median_60d_return_pct": median_60,
            "above_ma20_pct": breadth,
        }
    ).reset_index(drop=True)
    result["regime"] = "震荡市"
    result.loc[
        (result["median_60d_return_pct"] >= 5.0) & (result["above_ma20_pct"] >= 55.0),
        "regime",
    ] = "牛市"
    result.loc[
        (result["median_60d_return_pct"] <= -5.0) | (result["above_ma20_pct"] <= 40.0),
        "regime",
    ] = "下跌市"
    return result


def _commission(config: dict, amount: float) -> float:
    return max(float(config["min_commission"]), amount * float(config["commission_rate"]))


def _event_outcome(
    stock: pd.DataFrame,
    signal_index: int,
    config: dict,
    trade_value: float,
    horizons: Sequence[int],
) -> dict:
    outcome = {f"return_{horizon}d_pct": math.nan for horizon in horizons}
    outcome.update(
        {
            "entry_date": "",
            "entry_price": math.nan,
            "shares": 0,
            "analysis_value": round(float(trade_value), 2),
            "fillable": False,
            "unfilled_reason": "未来交易日不足",
            "mfe_10d_pct": math.nan,
            "mae_10d_pct": math.nan,
        }
    )
    entry_index = signal_index + 1
    if entry_index >= len(stock):
        return outcome
    signal_row = stock.iloc[signal_index]
    entry_row = stock.iloc[entry_index]
    price_limit = float(config["price_limit_pct"]) / 100.0
    if float(entry_row["open"]) >= float(signal_row["close"]) * (1.0 + price_limit - 1e-9):
        outcome["unfilled_reason"] = "次日开盘接近涨停，无法模拟买入"
        return outcome
    slip = float(config["slippage_pct"]) / 100.0
    buy_price = float(entry_row["open"]) * (1.0 + slip)
    lot = max(1, int(config["lot_size"]))
    shares = int((float(trade_value) - float(config["min_commission"])) // buy_price // lot) * lot
    if shares <= 0:
        outcome["unfilled_reason"] = "分析市值不足一手"
        return outcome
    buy_amount = shares * buy_price
    buy_fee = _commission(config, buy_amount)
    entry_cost = buy_amount + buy_fee
    outcome.update(
        {
            "entry_date": pd.Timestamp(entry_row["date"]).strftime("%Y-%m-%d"),
            "entry_price": round(buy_price, 4),
            "shares": shares,
            "fillable": True,
            "unfilled_reason": "",
        }
    )
    for horizon in horizons:
        exit_index = entry_index + horizon - 1
        if exit_index >= len(stock):
            continue
        exit_close = float(stock.iloc[exit_index]["close"]) * (1.0 - slip)
        proceeds = shares * exit_close
        sell_fee = _commission(config, proceeds) + proceeds * float(config["stamp_tax_rate"])
        outcome[f"return_{horizon}d_pct"] = round((proceeds - sell_fee - entry_cost) / entry_cost * 100.0, 4)
    max_horizon = max(horizons)
    if entry_index + max_horizon <= len(stock):
        window = stock.iloc[entry_index:entry_index + max_horizon]
        outcome["mfe_10d_pct"] = round((float(window["high"].max()) / buy_price - 1.0) * 100.0, 4)
        outcome["mae_10d_pct"] = round((float(window["low"].min()) / buy_price - 1.0) * 100.0, 4)
    return outcome


def calculate_event_outcome(
    stock: pd.DataFrame,
    signal_index: int,
    config: dict,
    trade_value: float,
    horizons: Sequence[int] = HORIZONS,
) -> dict:
    """Public cost-aware outcome calculator shared by research signals."""
    return _event_outcome(stock, signal_index, config, trade_value, horizons)


def evaluate_signal_frame(
    stock: pd.DataFrame,
    symbol: str,
    regime_by_date: dict[pd.Timestamp, str],
    config: dict,
    trade_value: float,
    horizons: Sequence[int] = HORIZONS,
) -> list[dict]:
    rows = stock.sort_values("date").reset_index(drop=True)
    events = []
    last_signal_index = -999
    for index, row in rows.iterrows():
        if not bool(row.get("raw_double_high", False)):
            continue
        event_type = ROCKET if bool(row.get("buy_candidate", False)) else ORDINARY
        outcome = _event_outcome(rows, index, config, trade_value, horizons)
        event_date = pd.Timestamp(row["date"])
        events.append(
            {
                "signal_date": event_date.strftime("%Y-%m-%d"),
                "symbol": str(symbol).zfill(6),
                "signal_type": event_type,
                "signal_score": int(row.get("signal_score", 0) or 0),
                "signal_close": round(float(row["close"]), 4),
                "volume_ratio": round(float(row["volume_ratio"]), 4) if pd.notna(row.get("volume_ratio")) else math.nan,
                "extension_pct": round(float(row["extension_pct"]), 4) if pd.notna(row.get("extension_pct")) else math.nan,
                "regime": regime_by_date.get(event_date, "震荡市"),
                "overlaps_prior_10d": index - last_signal_index <= 10,
                **outcome,
            }
        )
        last_signal_index = index
    return events


def _summary_row(values: pd.Series) -> dict:
    valid = pd.to_numeric(values, errors="coerce").dropna()
    gains = valid[valid > 0]
    losses = valid[valid <= 0]
    average_gain = float(gains.mean()) if not gains.empty else 0.0
    average_loss = float(losses.mean()) if not losses.empty else 0.0
    payoff = average_gain / abs(average_loss) if average_loss < 0 else (999.0 if average_gain > 0 else 0.0)
    return {
        "count": int(len(valid)),
        "win_rate_pct": round(float((valid > 0).mean() * 100.0), 2) if len(valid) else 0.0,
        "avg_return_pct": round(float(valid.mean()), 4) if len(valid) else 0.0,
        "median_return_pct": round(float(valid.median()), 4) if len(valid) else 0.0,
        "avg_gain_pct": round(average_gain, 4),
        "avg_loss_pct": round(average_loss, 4),
        "payoff_ratio": round(payoff, 3),
        "best_return_pct": round(float(valid.max()), 4) if len(valid) else 0.0,
        "worst_return_pct": round(float(valid.min()), 4) if len(valid) else 0.0,
    }


def summarize_returns(values: pd.Series) -> dict:
    """Return the common win-rate and payoff statistics for a return series."""
    return _summary_row(values)


def summarize_events(events: pd.DataFrame, horizons: Sequence[int] = HORIZONS) -> pd.DataFrame:
    rows = []
    for signal_type in (ORDINARY, ROCKET):
        group = events.loc[events["signal_type"] == signal_type]
        for horizon in horizons:
            row = {"signal_type": signal_type, "horizon_days": horizon, **_summary_row(group[f"return_{horizon}d_pct"])}
            if horizon == max(horizons):
                row["avg_mfe_pct"] = round(float(pd.to_numeric(group["mfe_10d_pct"], errors="coerce").mean()), 4) if not group.empty else 0.0
                row["avg_mae_pct"] = round(float(pd.to_numeric(group["mae_10d_pct"], errors="coerce").mean()), 4) if not group.empty else 0.0
            else:
                row["avg_mfe_pct"] = math.nan
                row["avg_mae_pct"] = math.nan
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_regimes(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (signal_type, regime), group in events.groupby(["signal_type", "regime"], dropna=False):
        rows.append(
            {
                "signal_type": signal_type,
                "regime": regime,
                **_summary_row(group["return_10d_pct"]),
                "avg_mfe_pct": round(float(pd.to_numeric(group["mfe_10d_pct"], errors="coerce").mean()), 4),
                "avg_mae_pct": round(float(pd.to_numeric(group["mae_10d_pct"], errors="coerce").mean()), 4),
            }
        )
    return pd.DataFrame(rows)


def _validation_summary(events: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    usable = events.dropna(subset=["return_10d_pct"]).copy()
    dates = sorted(usable["signal_date"].unique())
    if len(dates) < 2:
        return pd.DataFrame(), ""
    split_index = max(0, min(len(dates) - 2, int(len(dates) * 0.60) - 1))
    split_date = dates[split_index]
    usable["sample"] = usable["signal_date"].map(lambda value: "样本内60%" if value <= split_date else "样本外40%")
    rows = []
    for (signal_type, sample), group in usable.groupby(["signal_type", "sample"]):
        rows.append({"signal_type": signal_type, "sample": sample, "split_date": split_date, **_summary_row(group["return_10d_pct"])})
    return pd.DataFrame(rows), split_date


def build_walkforward_validation(events: pd.DataFrame, segments: int = 5) -> pd.DataFrame:
    usable = events.dropna(subset=["return_10d_pct"]).copy()
    dates = sorted(usable["signal_date"].unique())
    if len(dates) < segments:
        return pd.DataFrame()
    windows = [dates[index * len(dates) // segments:(index + 1) * len(dates) // segments] for index in range(segments)]
    rows = []
    for fold in range(1, segments):
        if not windows[fold] or not windows[fold - 1]:
            continue
        train_end = windows[fold - 1][-1]
        test_start = windows[fold][0]
        test_end = windows[fold][-1]
        for signal_type in (ORDINARY, ROCKET):
            train = usable.loc[(usable["signal_type"] == signal_type) & (usable["signal_date"] <= train_end)]
            test = usable.loc[
                (usable["signal_type"] == signal_type)
                & (usable["signal_date"] >= test_start)
                & (usable["signal_date"] <= test_end)
            ]
            train_stats = _summary_row(train["return_10d_pct"])
            test_stats = _summary_row(test["return_10d_pct"])
            rows.append(
                {
                    "signal_type": signal_type,
                    "fold": fold,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end,
                    "train_count": train_stats["count"],
                    "test_count": test_stats["count"],
                    "train_avg_return_pct": train_stats["avg_return_pct"],
                    "test_win_rate_pct": test_stats["win_rate_pct"],
                    "test_avg_return_pct": test_stats["avg_return_pct"],
                    "test_median_return_pct": test_stats["median_return_pct"],
                    "test_payoff_ratio": test_stats["payoff_ratio"],
                    "direction_consistent": train_stats["avg_return_pct"] > 0 and test_stats["avg_return_pct"] > 0,
                }
            )
    return pd.DataFrame(rows)


def _max_consecutive_losses(values: Iterable[float]) -> int:
    maximum = 0
    current = 0
    for value in values:
        if pd.notna(value) and float(value) <= 0:
            current += 1
            maximum = max(maximum, current)
        elif pd.notna(value):
            current = 0
    return maximum


def build_stability_report(
    events: pd.DataFrame,
    validation: pd.DataFrame,
    walkforward: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows = []
    for signal_type in (ORDINARY, ROCKET):
        group = events.loc[events["signal_type"] == signal_type].dropna(subset=["return_10d_pct"]).sort_values("signal_date")
        monthly = group.assign(month=group["signal_date"].str[:7]).groupby("month")["return_10d_pct"].mean()
        train = validation.loc[(validation["signal_type"] == signal_type) & (validation["sample"] == "样本内60%")]
        test = validation.loc[(validation["signal_type"] == signal_type) & (validation["sample"] == "样本外40%")]
        train_avg = float(train.iloc[0]["avg_return_pct"]) if not train.empty else math.nan
        test_avg = float(test.iloc[0]["avg_return_pct"]) if not test.empty else math.nan
        test_count = int(test.iloc[0]["count"]) if not test.empty else 0
        positive_month_ratio = float((monthly > 0).mean() * 100.0) if len(monthly) else 0.0
        folds = (
            walkforward.loc[walkforward["signal_type"] == signal_type]
            if walkforward is not None and not walkforward.empty
            else pd.DataFrame()
        )
        positive_fold_ratio = float((folds["test_avg_return_pct"] > 0).mean() * 100.0) if len(folds) else 0.0
        if len(group) < 50 or test_count < 20:
            assessment = "样本不足"
        elif train_avg > 0 and test_avg > 0 and positive_month_ratio >= 50.0 and positive_fold_ratio >= 50.0:
            assessment = "样本内外方向一致"
        else:
            assessment = "稳定性不足"
        rows.append(
            {
                "signal_type": signal_type,
                "usable_10d_samples": len(group),
                "train_avg_return_pct": round(train_avg, 4) if pd.notna(train_avg) else math.nan,
                "test_avg_return_pct": round(test_avg, 4) if pd.notna(test_avg) else math.nan,
                "return_std_pct": round(float(group["return_10d_pct"].std()), 4) if len(group) > 1 else 0.0,
                "positive_month_ratio_pct": round(positive_month_ratio, 2),
                "months": len(monthly),
                "positive_walkforward_ratio_pct": round(positive_fold_ratio, 2),
                "walkforward_folds": len(folds),
                "max_consecutive_losses": _max_consecutive_losses(group["return_10d_pct"]),
                "overlap_ratio_pct": round(float(group["overlaps_prior_10d"].mean() * 100.0), 2) if len(group) else 0.0,
                "assessment": assessment,
            }
        )
    return pd.DataFrame(rows)


def analyze_signals(
    prices: pd.DataFrame,
    config: dict,
    names: dict[str, str] | None = None,
    horizons: Sequence[int] = HORIZONS,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> SignalAnalyticsResult:
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(config)
    data = prices.copy()
    data["symbol"] = data["symbol"].astype(str).str.zfill(6)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    regimes = classify_market_regimes(data)
    regime_by_date = dict(zip(pd.to_datetime(regimes["date"]), regimes["regime"]))
    trade_value = min(float(cfg["initial_capital"]), float(cfg["initial_capital"]) * float(cfg["max_position_pct"]) / 100.0)
    rules = _rules_from_config(cfg)
    event_rows = []
    groups = data.groupby("symbol", sort=False)
    total_groups = groups.ngroups
    for index, (symbol, stock) in enumerate(groups, 1):
        if progress:
            progress(index - 1, total_groups, f"正在分析 {symbol}（{index}/{total_groups}）")
        calculated = calculate_signals(stock, rules)
        event_rows.extend(evaluate_signal_frame(calculated, symbol, regime_by_date, cfg, trade_value, horizons))
        if progress:
            progress(index, total_groups, f"已完成 {index}/{total_groups}")
    events = pd.DataFrame(event_rows)
    if events.empty:
        events = pd.DataFrame(
            columns=[
                "signal_date", "symbol", "signal_type", "signal_score", "signal_close", "volume_ratio",
                "extension_pct", "regime", "overlaps_prior_10d", "entry_date", "entry_price", "shares",
                "analysis_value", "fillable", "unfilled_reason", *[f"return_{h}d_pct" for h in horizons],
                "mfe_10d_pct", "mae_10d_pct",
            ]
        )
    events["name"] = events["symbol"].map(names or {}).fillna(events["symbol"])
    events = events.sort_values(["signal_date", "signal_type", "symbol"]).reset_index(drop=True)
    summary = summarize_events(events, horizons)
    regime_summary = summarize_regimes(events)
    validation, split_date = _validation_summary(events)
    walkforward = build_walkforward_validation(events)
    stability = build_stability_report(events, validation, walkforward)
    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "price_start": data["date"].min().strftime("%Y-%m-%d") if not data.empty else "",
        "price_end": data["date"].max().strftime("%Y-%m-%d") if not data.empty else "",
        "symbols": int(data["symbol"].nunique()),
        "events": len(events),
        "ordinary_events": int((events["signal_type"] == ORDINARY).sum()),
        "rocket_events": int((events["signal_type"] == ROCKET).sum()),
        "unfillable_events": int((~events["fillable"].astype(bool)).sum()) if not events.empty else 0,
        "analysis_trade_value": round(trade_value, 2),
        "validation_split_date": split_date,
        "regime_method": "股票池60日收益中位数 + 站上MA20比例",
        "config": cfg,
    }
    return SignalAnalyticsResult(events, summary, regime_summary, validation, walkforward, stability, regimes, metadata)


def _markdown_report(result: SignalAnalyticsResult) -> str:
    def table(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "无可用样本。"
        columns = [str(column) for column in frame.columns]
        lines = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
        ]
        for values in frame.itertuples(index=False, name=None):
            cells = []
            for value in values:
                text = "" if pd.isna(value) else str(value)
                cells.append(text.replace("|", "/").replace("\n", " "))
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    lines = [
        "# 普通双高与火箭信号统计报告",
        "",
        f"生成时间：{result.metadata['generated_at']}",
        f"行情范围：{result.metadata['price_start']} 至 {result.metadata['price_end']}",
        f"股票数量：{result.metadata['symbols']}",
        f"事件数量：普通双高 {result.metadata['ordinary_events']}，火箭 {result.metadata['rocket_events']}",
        f"统一分析市值：{result.metadata['analysis_trade_value']:,.2f}元",
        "",
        "> 收盘确认信号，下一交易日开盘模拟买入，按100股整数手并计入佣金、印花税和双边滑点。回测和统计不能保证未来收益。",
        "",
        "## 1/3/5/10日结果",
        "",
        table(result.summary),
        "",
        "## 市场状态分组（10日）",
        "",
        table(result.regime_summary),
        "",
        "## 样本内外验证（10日）",
        "",
        table(result.validation),
        "",
        "## 稳定性",
        "",
        table(result.stability),
        "",
        "## 扩展窗口滚动验证（10日）",
        "",
        table(result.walkforward),
        "",
        "## 口径限制",
        "",
        "- 牛市/震荡市/下跌市使用当前股票池代理，不等同于官方市场指数。",
        "- 同一股票10个交易日内重复出现信号会标记为重叠事件，不能视为完全独立样本。",
        "- 最近尚无足够未来交易日的事件不会伪造对应期限收益。",
        "- 所有结果均为历史模拟研究，不构成投资建议。",
        "",
    ]
    return "\n".join(lines)


def save_analytics_bundle(result: SignalAnalyticsResult, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "events": output_dir / "events_latest.csv",
        "summary": output_dir / "summary_latest.csv",
        "regime": output_dir / "regime_summary_latest.csv",
        "validation": output_dir / "validation_latest.csv",
        "walkforward": output_dir / "walkforward_latest.csv",
        "stability": output_dir / "stability_latest.csv",
        "regimes": output_dir / "market_regimes_latest.csv",
        "metadata": output_dir / "metadata_latest.json",
        "report": output_dir / "report_latest.md",
    }
    result.events.to_csv(paths["events"], index=False, encoding="utf-8-sig")
    result.summary.to_csv(paths["summary"], index=False, encoding="utf-8-sig")
    result.regime_summary.to_csv(paths["regime"], index=False, encoding="utf-8-sig")
    result.validation.to_csv(paths["validation"], index=False, encoding="utf-8-sig")
    result.walkforward.to_csv(paths["walkforward"], index=False, encoding="utf-8-sig")
    result.stability.to_csv(paths["stability"], index=False, encoding="utf-8-sig")
    result.regimes.to_csv(paths["regimes"], index=False, encoding="utf-8-sig")
    paths["metadata"].write_text(json.dumps(result.metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["report"].write_text(_markdown_report(result), encoding="utf-8")
    return paths
