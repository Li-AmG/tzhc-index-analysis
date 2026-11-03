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


MA5_CROSS_MA20 = "MA5首上MA20"
CLOSE_CROSS_MA5 = "K线首上MA5"
PULLBACK_RECLAIM = "震荡不破MA20再上MA5"
COMBINED = "三类买点合并"
STRATEGIES = (MA5_CROSS_MA20, CLOSE_CROSS_MA5, PULLBACK_RECLAIM, COMBINED)


@dataclass(frozen=True)
class MaStrategyRules:
    consolidation_days: int = 5
    consolidation_range_pct: float = 8.0
    start_date: str = "2024-01-01"


@dataclass
class MaStrategyResult:
    events: pd.DataFrame
    summary: pd.DataFrame
    validation: pd.DataFrame
    walkforward: pd.DataFrame
    stability: pd.DataFrame
    metadata: dict


def calculate_ma_entry_signals(data: pd.DataFrame, rules: MaStrategyRules | None = None) -> pd.DataFrame:
    rule = rules or MaStrategyRules()
    rows = data.copy().sort_values("date").reset_index(drop=True)
    rows["ma5"] = rows["close"].rolling(5, min_periods=5).mean()
    rows["ma20"] = rows["close"].rolling(20, min_periods=20).mean()
    rows["volume_ma20"] = rows["volume"].rolling(20, min_periods=20).mean()
    rows["volume_ratio20"] = rows["volume"] / rows["volume_ma20"]
    rows["ma5_cross_ma20"] = (
        (rows["ma5"].shift(1) <= rows["ma20"].shift(1))
        & (rows["ma5"] > rows["ma20"])
    ).fillna(False)
    rows["close_cross_ma5"] = (
        (rows["close"].shift(1) <= rows["ma5"].shift(1))
        & (rows["close"] > rows["ma5"])
    ).fillna(False)

    days = max(3, int(rule.consolidation_days))
    previous_above_ma20 = (rows["close"].shift(1) >= rows["ma20"].shift(1)).rolling(days, min_periods=days).sum() == days
    previous_high = rows["close"].shift(1).rolling(days, min_periods=days).max()
    previous_low = rows["close"].shift(1).rolling(days, min_periods=days).min()
    previous_range_pct = (previous_high - previous_low) / rows["ma20"] * 100.0
    ma20_not_falling = rows["ma20"] >= rows["ma20"].shift(days)
    rows["pullback_reclaim"] = (
        rows["close_cross_ma5"]
        & previous_above_ma20
        & ma20_not_falling
        & (previous_range_pct <= float(rule.consolidation_range_pct))
    ).fillna(False)
    rows["combined_entry"] = (
        rows["ma5_cross_ma20"] | rows["close_cross_ma5"] | rows["pullback_reclaim"]
    )
    rows["combined_reason"] = ""
    rows.loc[rows["close_cross_ma5"], "combined_reason"] = CLOSE_CROSS_MA5
    rows.loc[rows["pullback_reclaim"], "combined_reason"] = PULLBACK_RECLAIM
    rows.loc[rows["ma5_cross_ma20"], "combined_reason"] = MA5_CROSS_MA20
    rows["consolidation_range_pct"] = previous_range_pct
    return rows


def _event_columns(horizons: Sequence[int]) -> list[str]:
    return [
        "signal_date", "symbol", "name", "strategy", "combined_reason", "signal_close", "ma5", "ma20",
        "volume_ratio20", "consolidation_range_pct", "overlaps_prior_10d", "entry_date", "entry_price",
        "shares", "analysis_value", "fillable", "unfilled_reason", *[f"return_{h}d_pct" for h in horizons],
        "mfe_10d_pct", "mae_10d_pct",
    ]


def _summary(events: pd.DataFrame, horizons: Sequence[int]) -> pd.DataFrame:
    rows = []
    for strategy in STRATEGIES:
        group = events.loc[events["strategy"] == strategy]
        for horizon in horizons:
            rows.append(
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
    return pd.DataFrame(rows)


def _validation(events: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    usable = events.dropna(subset=["return_10d_pct"]).copy()
    dates = sorted(usable["signal_date"].unique())
    if len(dates) < 2:
        return pd.DataFrame(), ""
    split_index = max(0, min(len(dates) - 2, int(len(dates) * 0.60) - 1))
    split_date = dates[split_index]
    usable["sample"] = usable["signal_date"].map(lambda value: "样本内60%" if value <= split_date else "样本外40%")
    rows = []
    for (strategy, sample), group in usable.groupby(["strategy", "sample"]):
        rows.append({"strategy": strategy, "sample": sample, "split_date": split_date, **summarize_returns(group["return_10d_pct"])})
    return pd.DataFrame(rows), str(split_date)


def _walkforward(events: pd.DataFrame, segments: int = 5) -> pd.DataFrame:
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
        test_start, test_end = windows[fold][0], windows[fold][-1]
        for strategy in STRATEGIES:
            train = usable.loc[(usable["strategy"] == strategy) & (usable["signal_date"] <= train_end)]
            test = usable.loc[
                (usable["strategy"] == strategy)
                & (usable["signal_date"] >= test_start)
                & (usable["signal_date"] <= test_end)
            ]
            train_stats = summarize_returns(train["return_10d_pct"])
            test_stats = summarize_returns(test["return_10d_pct"])
            rows.append(
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
    return pd.DataFrame(rows)


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
    rows = []
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
            assessment = "样本内外方向一致"
        else:
            assessment = "稳定性不足"
        rows.append(
            {
                "strategy": strategy,
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


def analyze_ma_strategy(
    prices: pd.DataFrame,
    config: dict,
    names: dict[str, str] | None = None,
    rules: MaStrategyRules | None = None,
    horizons: Sequence[int] = HORIZONS,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> MaStrategyResult:
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(config)
    rule = rules or MaStrategyRules(
        consolidation_days=int(cfg.get("ma_consolidation_days", 5)),
        consolidation_range_pct=float(cfg.get("ma_consolidation_range_pct", 8.0)),
    )
    data = prices.copy()
    data["symbol"] = data["symbol"].astype(str).str.zfill(6)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    numeric = ["open", "high", "low", "close", "volume"]
    data[numeric] = data[numeric].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=["date", *numeric]).sort_values(["symbol", "date"])
    trade_value = min(float(cfg["initial_capital"]), float(cfg["initial_capital"]) * float(cfg["max_position_pct"]) / 100.0)
    event_rows = []
    strategy_columns = {
        MA5_CROSS_MA20: "ma5_cross_ma20",
        CLOSE_CROSS_MA5: "close_cross_ma5",
        PULLBACK_RECLAIM: "pullback_reclaim",
        COMBINED: "combined_entry",
    }
    groups = data.groupby("symbol", sort=False)
    total = groups.ngroups
    for group_index, (symbol, stock) in enumerate(groups, 1):
        if progress:
            progress(group_index - 1, total, f"正在分析均线买点 {symbol}（{group_index}/{total}）")
        calculated = calculate_ma_entry_signals(stock, rule)
        for strategy, column in strategy_columns.items():
            last_signal_index = -999
            hits = calculated.index[
                calculated[column]
                & (calculated["date"] >= pd.Timestamp(rule.start_date))
            ].tolist()
            for signal_index in hits:
                row = calculated.iloc[signal_index]
                event_cfg = dict(cfg)
                event_cfg["price_limit_pct"] = 20.0 if str(symbol).startswith(("300", "301", "688", "689")) else 10.0
                outcome = calculate_event_outcome(calculated, signal_index, event_cfg, trade_value, horizons)
                event_rows.append(
                    {
                        "signal_date": pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
                        "symbol": str(symbol).zfill(6),
                        "name": (names or {}).get(str(symbol).zfill(6), str(symbol).zfill(6)),
                        "strategy": strategy,
                        "combined_reason": str(row["combined_reason"]) if strategy == COMBINED else strategy,
                        "signal_close": round(float(row["close"]), 4),
                        "ma5": round(float(row["ma5"]), 4) if pd.notna(row["ma5"]) else math.nan,
                        "ma20": round(float(row["ma20"]), 4) if pd.notna(row["ma20"]) else math.nan,
                        "volume_ratio20": round(float(row["volume_ratio20"]), 4) if pd.notna(row["volume_ratio20"]) else math.nan,
                        "consolidation_range_pct": round(float(row["consolidation_range_pct"]), 4)
                        if pd.notna(row["consolidation_range_pct"]) else math.nan,
                        "overlaps_prior_10d": signal_index - last_signal_index <= 10,
                        **outcome,
                    }
                )
                last_signal_index = signal_index
        if progress:
            progress(group_index, total, f"已完成 {group_index}/{total}")

    events = pd.DataFrame(event_rows, columns=_event_columns(horizons))
    if not events.empty:
        events = events.sort_values(["signal_date", "strategy", "symbol"]).reset_index(drop=True)
    summary = _summary(events, horizons)
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
        "rules": {
            "consolidation_days": rule.consolidation_days,
            "consolidation_range_pct": rule.consolidation_range_pct,
            "stop_loss_pct": float(cfg["stop_loss_pct"]),
            "profit_arm_pct": float(cfg["take_profit_pct"]),
            "trailing_stop_pct": float(cfg["trailing_stop_pct"]),
            "soft_drawdown_pct": float(cfg["soft_drawdown_pct"]),
            "hard_drawdown_pct": float(cfg["hard_drawdown_pct"]),
        },
    }
    return MaStrategyResult(events, summary, validation, walkforward, stability, metadata)


def _table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "无可用样本。"
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for values in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join("" if pd.isna(value) else str(value).replace("|", "/") for value in values) + " |")
    return "\n".join(lines)


def _report(result: MaStrategyResult) -> str:
    rules = result.metadata["rules"]
    return "\n".join(
        [
            "# 均线买点稳定性研究报告",
            "",
            f"生成时间：{result.metadata['generated_at']}",
            f"研究区间：{result.metadata['research_start']} 至 {result.metadata['price_end']}",
            f"股票数量：{result.metadata['symbols']}，事件总数：{result.metadata['events']}",
            "",
            "> 三类买点独立统计，并额外统计三类合并。所有信号收盘确认、下一交易日开盘模拟买入，计入佣金、印花税和双边滑点。",
            "",
            "## 规则",
            "",
            "- MA5首上MA20：前一日MA5不高于MA20，当日MA5首次上穿MA20。",
            "- K线首上MA5：前一日收盘不高于MA5，当日收盘首次上穿MA5。",
            f"- 震荡不破再上MA5：此前{rules['consolidation_days']}日收盘不破各自MA20、区间不超过{rules['consolidation_range_pct']}%、MA20不下行，当日收盘重新上穿MA5。",
            f"- 卖出与组合风控保持：{rules['stop_loss_pct']}%止损；盈利{rules['profit_arm_pct']}%后从高点回撤{rules['trailing_stop_pct']}%；放量跌破MA20或MA5下穿MA20；组合{rules['soft_drawdown_pct']}%暂停、{rules['hard_drawdown_pct']}%熔断。",
            "",
            "## 1/3/5/10日净收益",
            "",
            _table(result.summary),
            "",
            "## 60/40样本外",
            "",
            _table(result.validation),
            "",
            "## 扩展窗口验证",
            "",
            _table(result.walkforward),
            "",
            "## 稳定性",
            "",
            _table(result.stability),
            "",
            "- 结果用于筛选规则，不构成收益保证或投资建议。",
            "",
        ]
    )


def save_ma_strategy_bundle(result: MaStrategyResult, output_dir: Path) -> dict[str, Path]:
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


def run_ma_portfolio_variants(
    price_map: dict[str, list[Candle]],
    config: dict,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, BacktestResult]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = (
        ("ma5_cross_ma20", MA5_CROSS_MA20),
        ("close_cross_ma5", CLOSE_CROSS_MA5),
        ("pullback_reclaim", PULLBACK_RECLAIM),
        ("combined", COMBINED),
    )
    rows = []
    results = {}
    for key, label in variants:
        cfg = dict(config)
        cfg.update(
            {
                "entry_strategy": "ma_pullback",
                "ma_entry_variant": key,
                "trade_start_date": "2024-01-01",
            }
        )
        result = run_backtest(price_map, cfg)
        results[key] = result
        paths = save_result_bundle(result, output_dir / "portfolio", f"ma_{key}")
        rows.append(
            {
                "variant": key,
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
                "trades_path": str(paths["trades"]),
                "equity_path": str(paths["equity"]),
                "signals_path": str(paths["signals"]),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "portfolio_summary_latest.csv", index=False, encoding="utf-8-sig")
    (output_dir / "portfolio_summary_latest.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "trade_start_date": "2024-01-01",
                "soft_drawdown_pct": float(config.get("soft_drawdown_pct", 5.0)),
                "hard_drawdown_pct": float(config.get("hard_drawdown_pct", 8.0)),
                "rows": summary.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    report_path = output_dir / "report_latest.md"
    if report_path.exists():
        report_text = report_path.read_text(encoding="utf-8")
        marker = "## 5%-8%真实组合回测"
        if marker in report_text:
            report_text = report_text.split(marker, 1)[0].rstrip() + "\n\n"
        report_text += "\n".join(
            [
                marker,
                "",
                "> 使用最多5个持仓、单仓18%、5%暂停开仓、8%清仓停机及原卖出规则。该结果优先于重叠事件的固定10日均值。",
                "",
                _table(
                    summary[
                        [
                            "strategy", "total_return_pct", "annual_return_pct", "max_drawdown_pct",
                            "trade_count", "win_rate_pct", "sharpe_ratio", "profit_factor", "halted", "halt_date",
                        ]
                    ]
                ),
                "",
            ]
        )
        report_path.write_text(report_text, encoding="utf-8")
    return summary, results
