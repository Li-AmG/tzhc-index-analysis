from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import pandas as pd

from tzhc_engine import DEFAULT_CONFIG, BacktestResult, Candle, run_backtest, save_result_bundle
from tzhc_signal_analytics import HORIZONS, calculate_event_outcome, summarize_returns


BREAKOUT_20 = "20日强趋势突破"
BREAKOUT_60 = "60日强趋势突破"
STRATEGIES = (BREAKOUT_20, BREAKOUT_60)


@dataclass(frozen=True)
class BreakoutRules:
    consolidation_days: int = 10
    consolidation_range_pct: float = 8.0
    volume_contraction_ratio: float = 0.85
    min_volume_ratio: float = 1.5
    max_volume_ratio: float = 4.0
    main_min_gain_pct: float = 4.0
    main_max_gain_pct: float = 9.5
    growth_min_gain_pct: float = 6.0
    growth_max_gain_pct: float = 15.0
    min_turnover_amount: float = 50_000_000.0
    start_date: str = "2024-01-01"


@dataclass
class BreakoutResult:
    events: pd.DataFrame
    summary: pd.DataFrame
    validation: pd.DataFrame
    walkforward: pd.DataFrame
    stability: pd.DataFrame
    metadata: dict


def calculate_breakout_signals(data: pd.DataFrame, growth_board: bool, rules: BreakoutRules) -> pd.DataFrame:
    rows = data.copy().sort_values("date").reset_index(drop=True)
    rows["ma20"] = rows["close"].rolling(20, min_periods=20).mean()
    rows["ma60"] = rows["close"].rolling(60, min_periods=60).mean()
    rows["trend_ok"] = (
        (rows["close"] > rows["ma20"])
        & (rows["ma20"] > rows["ma60"])
        & (rows["ma20"] > rows["ma20"].shift(5))
        & (rows["ma60"] >= rows["ma60"].shift(5))
    ).fillna(False)
    days = max(3, int(rules.consolidation_days))
    box_high = rows["high"].shift(1).rolling(days, min_periods=days).max()
    box_low = rows["low"].shift(1).rolling(days, min_periods=days).min()
    rows["box_range_pct"] = (box_high / box_low - 1.0) * 100.0
    box_volume = rows["volume"].shift(1).rolling(days, min_periods=days).mean()
    prior_base_volume = rows["volume"].shift(days + 1).rolling(20, min_periods=20).mean()
    rows["volume_contraction_ratio"] = box_volume / prior_base_volume
    rows["consolidation_ok"] = (
        (rows["box_range_pct"] <= rules.consolidation_range_pct)
        & (rows["volume_contraction_ratio"] <= rules.volume_contraction_ratio)
    ).fillna(False)
    prior_volume20 = rows["volume"].shift(1).rolling(20, min_periods=20).mean()
    rows["volume_ratio20"] = rows["volume"] / prior_volume20
    rows["volume_ok"] = rows["volume_ratio20"].between(
        rules.min_volume_ratio, rules.max_volume_ratio, inclusive="both"
    ).fillna(False)
    rows["daily_gain_pct"] = rows["close"].pct_change(fill_method=None) * 100.0
    min_gain = rules.growth_min_gain_pct if growth_board else rules.main_min_gain_pct
    max_gain = rules.growth_max_gain_pct if growth_board else rules.main_max_gain_pct
    candle_range = (rows["high"] - rows["low"]).replace(0, pd.NA)
    rows["close_position"] = (rows["close"] - rows["low"]) / candle_range
    rows["upper_shadow_ratio"] = (rows["high"] - rows[["open", "close"]].max(axis=1)) / candle_range
    rows["candle_ok"] = (
        (rows["close"] > rows["open"])
        & (rows["close_position"] >= 0.65)
        & (rows["upper_shadow_ratio"] <= 0.35)
        & rows["daily_gain_pct"].between(min_gain, max_gain, inclusive="both")
    ).fillna(False)
    amount_proxy = rows["close"] * rows["volume"] * 100.0
    rows["avg_turnover_amount20"] = amount_proxy.shift(1).rolling(20, min_periods=20).mean()
    rows["liquidity_ok"] = (rows["avg_turnover_amount20"] >= rules.min_turnover_amount).fillna(False)
    common = rows["trend_ok"] & rows["consolidation_ok"] & rows["volume_ok"] & rows["candle_ok"] & rows["liquidity_ok"]
    for window in (20, 60):
        previous_high = rows["close"].shift(1).rolling(window, min_periods=window).max()
        first_breakout = rows["close"] > previous_high
        rows[f"breakout_{window}"] = (common & first_breakout).fillna(False)
    return rows


def _summaries(events: pd.DataFrame, horizons: Sequence[int]) -> pd.DataFrame:
    records = []
    for strategy in STRATEGIES:
        group = events.loc[events["strategy"] == strategy]
        for horizon in horizons:
            records.append(
                {
                    "strategy": strategy,
                    "horizon_days": horizon,
                    **summarize_returns(group[f"return_{horizon}d_pct"]),
                    "avg_mfe_pct": round(float(pd.to_numeric(group["mfe_10d_pct"], errors="coerce").mean()), 4)
                    if horizon == max(horizons) and not group.empty else math.nan,
                    "avg_mae_pct": round(float(pd.to_numeric(group["mae_10d_pct"], errors="coerce").mean()), 4)
                    if horizon == max(horizons) and not group.empty else math.nan,
                }
            )
    return pd.DataFrame(records)


def _validation(events: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    usable = events.dropna(subset=["return_10d_pct"]).copy()
    dates = sorted(usable["signal_date"].unique())
    if len(dates) < 2:
        return pd.DataFrame(), ""
    split_index = max(0, min(len(dates) - 2, int(len(dates) * 0.60) - 1))
    split_date = dates[split_index]
    usable["sample"] = usable["signal_date"].map(lambda value: "样本内60%" if value <= split_date else "样本外40%")
    records = []
    for (strategy, sample), group in usable.groupby(["strategy", "sample"]):
        records.append({"strategy": strategy, "sample": sample, "split_date": split_date, **summarize_returns(group["return_10d_pct"])})
    return pd.DataFrame(records), str(split_date)


def _walkforward(events: pd.DataFrame, segments: int = 5) -> pd.DataFrame:
    usable = events.dropna(subset=["return_10d_pct"]).copy()
    dates = sorted(usable["signal_date"].unique())
    if len(dates) < segments:
        return pd.DataFrame()
    windows = [dates[index * len(dates) // segments:(index + 1) * len(dates) // segments] for index in range(segments)]
    records = []
    for fold in range(1, segments):
        if not windows[fold] or not windows[fold - 1]:
            continue
        train_end = windows[fold - 1][-1]
        test_start, test_end = windows[fold][0], windows[fold][-1]
        for strategy in STRATEGIES:
            train = usable.loc[(usable["strategy"] == strategy) & (usable["signal_date"] <= train_end)]
            test = usable.loc[
                (usable["strategy"] == strategy)
                & (usable["signal_date"] >= test_start)
                & (usable["signal_date"] <= test_end)
            ]
            train_stats, test_stats = summarize_returns(train["return_10d_pct"]), summarize_returns(test["return_10d_pct"])
            records.append(
                {
                    "strategy": strategy,
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
                    "direction_consistent": train_stats["avg_return_pct"] > 0 and test_stats["avg_return_pct"] > 0,
                }
            )
    return pd.DataFrame(records)


def _max_consecutive_losses(values: Iterable[float]) -> int:
    maximum = current = 0
    for value in values:
        if pd.notna(value) and float(value) <= 0:
            current += 1
            maximum = max(maximum, current)
        elif pd.notna(value):
            current = 0
    return maximum


def _stability(events: pd.DataFrame, validation: pd.DataFrame, walkforward: pd.DataFrame) -> pd.DataFrame:
    records = []
    for strategy in STRATEGIES:
        group = events.loc[events["strategy"] == strategy].dropna(subset=["return_10d_pct"]).sort_values("signal_date")
        monthly = group.assign(month=group["signal_date"].str[:7]).groupby("month")["return_10d_pct"].mean()
        train = validation.loc[(validation["strategy"] == strategy) & (validation["sample"] == "样本内60%")] if not validation.empty else pd.DataFrame()
        test = validation.loc[(validation["strategy"] == strategy) & (validation["sample"] == "样本外40%")] if not validation.empty else pd.DataFrame()
        folds = walkforward.loc[walkforward["strategy"] == strategy] if not walkforward.empty else pd.DataFrame()
        train_avg = float(train.iloc[0]["avg_return_pct"]) if not train.empty else math.nan
        test_avg = float(test.iloc[0]["avg_return_pct"]) if not test.empty else math.nan
        test_count = int(test.iloc[0]["count"]) if not test.empty else 0
        positive_month_ratio = float((monthly > 0).mean() * 100.0) if len(monthly) else 0.0
        positive_fold_ratio = float((folds["test_avg_return_pct"] > 0).mean() * 100.0) if len(folds) else 0.0
        if len(group) < 100 or test_count < 40:
            assessment = "样本不足"
        elif train_avg > 0 and test_avg > 0 and positive_month_ratio >= 50 and positive_fold_ratio >= 50:
            assessment = "事件层面方向一致"
        else:
            assessment = "稳定性不足"
        records.append(
            {
                "strategy": strategy,
                "usable_10d_samples": len(group),
                "train_avg_return_pct": round(train_avg, 4) if pd.notna(train_avg) else math.nan,
                "test_avg_return_pct": round(test_avg, 4) if pd.notna(test_avg) else math.nan,
                "positive_month_ratio_pct": round(positive_month_ratio, 2),
                "months": len(monthly),
                "positive_walkforward_ratio_pct": round(positive_fold_ratio, 2),
                "walkforward_folds": len(folds),
                "max_consecutive_losses": _max_consecutive_losses(group["return_10d_pct"]),
                "overlap_ratio_pct": round(float(group["overlaps_prior_10d"].mean() * 100.0), 2) if len(group) else 0.0,
                "assessment": assessment,
            }
        )
    return pd.DataFrame(records)


def analyze_breakout_strategy(
    prices: pd.DataFrame,
    config: dict,
    names: dict[str, str] | None = None,
    rules: BreakoutRules | None = None,
    horizons: Sequence[int] = HORIZONS,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> BreakoutResult:
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(config)
    rule = rules or BreakoutRules(
        consolidation_days=int(cfg["breakout_consolidation_days"]),
        consolidation_range_pct=float(cfg["breakout_consolidation_range_pct"]),
        volume_contraction_ratio=float(cfg["breakout_volume_contraction_ratio"]),
        min_volume_ratio=float(cfg["breakout_min_volume_ratio"]),
        max_volume_ratio=float(cfg["breakout_max_volume_ratio"]),
        main_min_gain_pct=float(cfg["breakout_main_min_gain_pct"]),
        main_max_gain_pct=float(cfg["breakout_main_max_gain_pct"]),
        growth_min_gain_pct=float(cfg["breakout_growth_min_gain_pct"]),
        growth_max_gain_pct=float(cfg["breakout_growth_max_gain_pct"]),
        min_turnover_amount=float(cfg["breakout_min_turnover_amount"]),
    )
    data = prices.copy()
    data["symbol"] = data["symbol"].astype(str).str.zfill(6)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    numeric = ["open", "high", "low", "close", "volume"]
    data[numeric] = data[numeric].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=["date", *numeric]).sort_values(["symbol", "date"])
    trade_value = min(float(cfg["initial_capital"]), float(cfg["initial_capital"]) * float(cfg["max_position_pct"]) / 100.0)
    columns = [
        "signal_date", "symbol", "name", "strategy", "signal_close", "daily_gain_pct", "volume_ratio20",
        "box_range_pct", "volume_contraction_ratio", "avg_turnover_amount20", "ma20", "ma60",
        "overlaps_prior_10d", "entry_date", "entry_price", "shares", "analysis_value", "fillable",
        "unfilled_reason", *[f"return_{h}d_pct" for h in horizons], "mfe_10d_pct", "mae_10d_pct",
    ]
    event_rows = []
    groups = data.groupby("symbol", sort=False)
    total = groups.ngroups
    for group_index, (symbol, stock) in enumerate(groups, 1):
        name = (names or {}).get(str(symbol).zfill(6), str(symbol).zfill(6))
        upper_name = str(name).upper()
        if "ST" in upper_name or "退" in upper_name:
            continue
        growth_board = str(symbol).startswith(("300", "301", "688", "689"))
        calculated = calculate_breakout_signals(stock, growth_board, rule)
        for strategy, signal_column in ((BREAKOUT_20, "breakout_20"), (BREAKOUT_60, "breakout_60")):
            last_index = -999
            for signal_index in calculated.index[
                calculated[signal_column] & (calculated["date"] >= pd.Timestamp(rule.start_date))
            ].tolist():
                row = calculated.iloc[signal_index]
                event_cfg = dict(cfg)
                event_cfg["price_limit_pct"] = 20.0 if growth_board else 10.0
                outcome = calculate_event_outcome(calculated, signal_index, event_cfg, trade_value, horizons)
                event_rows.append(
                    {
                        "signal_date": pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
                        "symbol": str(symbol).zfill(6),
                        "name": name,
                        "strategy": strategy,
                        "signal_close": round(float(row["close"]), 4),
                        "daily_gain_pct": round(float(row["daily_gain_pct"]), 4),
                        "volume_ratio20": round(float(row["volume_ratio20"]), 4),
                        "box_range_pct": round(float(row["box_range_pct"]), 4),
                        "volume_contraction_ratio": round(float(row["volume_contraction_ratio"]), 4),
                        "avg_turnover_amount20": round(float(row["avg_turnover_amount20"]), 2),
                        "ma20": round(float(row["ma20"]), 4),
                        "ma60": round(float(row["ma60"]), 4),
                        "overlaps_prior_10d": signal_index - last_index <= 10,
                        **outcome,
                    }
                )
                last_index = signal_index
        if progress:
            progress(group_index, total, f"已分析 {group_index}/{total}")
    events = pd.DataFrame(event_rows, columns=columns)
    if not events.empty:
        events = events.sort_values(["signal_date", "strategy", "symbol"]).reset_index(drop=True)
    summary = _summaries(events, horizons)
    validation, split_date = _validation(events)
    walkforward = _walkforward(events)
    stability = _stability(events, validation, walkforward)
    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "price_start": data["date"].min().strftime("%Y-%m-%d") if not data.empty else "",
        "price_end": data["date"].max().strftime("%Y-%m-%d") if not data.empty else "",
        "research_start": rule.start_date,
        "symbols": int(data["symbol"].nunique()),
        "events": len(events),
        "analysis_trade_value": round(trade_value, 2),
        "validation_split_date": split_date,
        "rules": rule.__dict__,
        "limitations": ["当前行情不含可靠总市值字段，未执行50-80亿元市值过滤。"],
    }
    return BreakoutResult(events, summary, validation, walkforward, stability, metadata)


def _table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "无可用样本。"
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for values in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join("" if pd.isna(value) else str(value).replace("|", "/") for value in values) + " |")
    return "\n".join(lines)


def _report(result: BreakoutResult) -> str:
    rules = result.metadata["rules"]
    return "\n".join(
        [
            "# A股强趋势突破稳定性研究",
            "",
            f"生成时间：{result.metadata['generated_at']}",
            f"研究区间：{result.metadata['research_start']} 至 {result.metadata['price_end']}",
            f"股票数量：{result.metadata['symbols']}，事件数量：{result.metadata['events']}",
            "",
            "> 顺势多头 + 缩量箱体 + 放量中大阳首次突破20/60日高点；收盘确认，下一交易日开盘模拟成交并计入成本。",
            "",
            "## 程序化条件",
            "",
            f"- 前{rules['consolidation_days']}日箱体振幅不超过{rules['consolidation_range_pct']}%，均量不超过此前20日的{rules['volume_contraction_ratio'] * 100:.0f}%。",
            f"- 突破日量比{rules['min_volume_ratio']}-{rules['max_volume_ratio']}；主板涨幅{rules['main_min_gain_pct']}-{rules['main_max_gain_pct']}%，创业/科创{rules['growth_min_gain_pct']}-{rules['growth_max_gain_pct']}%。",
            f"- 前20日日均成交额代理不少于{rules['min_turnover_amount'] / 10000:.0f}万元；排除名称含ST或退市标记的股票。",
            "- 当前数据没有可靠总市值字段，未伪造50-80亿元市值过滤。",
            "",
            "## 1/3/5/10日事件统计",
            "",
            _table(result.summary),
            "",
            "## 60/40样本外",
            "",
            _table(result.validation),
            "",
            "## 扩展窗口",
            "",
            _table(result.walkforward),
            "",
            "## 稳定性",
            "",
            _table(result.stability),
            "",
        ]
    )


def save_breakout_bundle(result: BreakoutResult, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "events": output_dir / "events_latest.csv",
        "summary": output_dir / "summary_latest.csv",
        "validation": output_dir / "validation_latest.csv",
        "walkforward": output_dir / "walkforward_latest.csv",
        "stability": output_dir / "stability_latest.csv",
        "metadata": output_dir / "metadata_latest.json",
        "report": output_dir / "report_latest.md",
    }
    result.events.to_csv(paths["events"], index=False, encoding="utf-8-sig")
    result.summary.to_csv(paths["summary"], index=False, encoding="utf-8-sig")
    result.validation.to_csv(paths["validation"], index=False, encoding="utf-8-sig")
    result.walkforward.to_csv(paths["walkforward"], index=False, encoding="utf-8-sig")
    result.stability.to_csv(paths["stability"], index=False, encoding="utf-8-sig")
    paths["metadata"].write_text(json.dumps(result.metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["report"].write_text(_report(result), encoding="utf-8")
    return paths


def run_breakout_portfolios(
    price_map: dict[str, list[Candle]],
    config: dict,
    output_dir: Path,
    excluded_symbols: Sequence[str] = (),
) -> tuple[pd.DataFrame, dict[int, BacktestResult]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records, results = [], {}
    for window, label in ((20, BREAKOUT_20), (60, BREAKOUT_60)):
        cfg = dict(config)
        cfg.update(
            {
                "entry_strategy": "a_share_breakout",
                "breakout_window": window,
                "trade_start_date": "2024-01-01",
                "excluded_symbols": list(excluded_symbols),
            }
        )
        result = run_backtest(price_map, cfg)
        results[window] = result
        paths = save_result_bundle(result, output_dir / "portfolio", f"breakout_{window}")
        records.append(
            {
                "window": window,
                "strategy": label,
                "initial_capital": result.initial_capital,
                "final_equity": result.final_equity,
                "total_return_pct": result.total_return_pct,
                "annual_return_pct": result.annual_return_pct,
                "max_drawdown_pct": result.max_drawdown_pct,
                "trade_count": result.trade_count,
                "win_rate_pct": result.win_rate_pct,
                "sharpe_ratio": result.sharpe_ratio,
                "profit_factor": result.profit_factor,
                "total_fees": result.total_fees,
                "halted": result.halted,
                "halt_date": result.halt_date or "",
                "report_path": str(paths["report"]),
            }
        )
    summary = pd.DataFrame(records)
    summary.to_csv(output_dir / "portfolio_summary_latest.csv", index=False, encoding="utf-8-sig")
    report_path = output_dir / "report_latest.md"
    if report_path.exists():
        text = report_path.read_text(encoding="utf-8")
        marker = "## 5%-8%真实组合回测"
        if marker in text:
            text = text.split(marker, 1)[0].rstrip() + "\n\n"
        text += "\n".join(
            [
                marker,
                "",
                "> 最多5仓、单仓18%，5%暂停开仓、8%清仓停机；主板初始/移动熔断6%，创业/科创8%。",
                "",
                _table(summary[["strategy", "total_return_pct", "annual_return_pct", "max_drawdown_pct", "trade_count", "win_rate_pct", "sharpe_ratio", "profit_factor", "halted", "halt_date"]]),
                "",
            ]
        )
        report_path.write_text(text, encoding="utf-8")
    return summary, results
