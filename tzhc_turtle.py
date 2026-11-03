from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Sequence

import pandas as pd

from tzhc_engine import DEFAULT_CONFIG
from tzhc_signal_analytics import HORIZONS, calculate_event_outcome, summarize_returns


TURTLE_SIGNAL = "🐢 海龟"


@dataclass(frozen=True)
class TurtleRules:
    index_drop_pct: float = -2.0
    shanghai_drop_pct: float = -1.0
    ordinary_gain_pct: float = 5.0
    growth_gain_pct: float = 10.0
    min_volume_ratio: float = 1.2
    max_volume_ratio: float = 2.0
    start_date: str = "2024-01-01"


@dataclass(frozen=True)
class BoardBenchmark:
    board: str
    benchmark_code: str
    benchmark_name: str
    gain_threshold_pct: float
    price_limit_pct: float


@dataclass
class TurtleAnalyticsResult:
    events: pd.DataFrame
    summary: pd.DataFrame
    board_summary: pd.DataFrame
    validation: pd.DataFrame
    walkforward: pd.DataFrame
    stability: pd.DataFrame
    metadata: dict


def board_benchmark(symbol: str, rules: TurtleRules) -> BoardBenchmark | None:
    code = str(symbol).zfill(6)
    if code.startswith(("300", "301")):
        return BoardBenchmark("创业板", "399006", "创业板指", rules.growth_gain_pct, 20.0)
    if code.startswith(("688", "689")):
        return BoardBenchmark("科创板", "000688", "科创50", rules.growth_gain_pct, 20.0)
    if code.startswith(("000", "001", "002", "003")):
        return BoardBenchmark("深市主板", "399001", "深证成指", rules.ordinary_gain_pct, 10.0)
    if code.startswith(("600", "601", "603", "605")):
        return BoardBenchmark("沪市主板", "000001", "上证指数", rules.ordinary_gain_pct, 10.0)
    return None


def _empty_events(horizons: Sequence[int]) -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "signal_date", "symbol", "name", "signal_type", "board", "benchmark_code", "benchmark_name",
            "index_drop_threshold_pct", "shanghai_drop_threshold_pct", "market_trigger",
            "stock_gain_threshold_pct", "min_volume_ratio", "max_volume_ratio",
            "volume_ratio20", "stock_return_pct", "benchmark_return_pct", "shanghai_return_pct",
            "relative_strength_pct", "signal_close", "overlaps_prior_10d", "entry_date", "entry_price", "shares",
            "analysis_value", "fillable", "unfilled_reason", *[f"return_{h}d_pct" for h in horizons],
            *[f"benchmark_forward_{h}d_pct" for h in horizons], *[f"excess_return_{h}d_pct" for h in horizons],
            "mfe_10d_pct", "mae_10d_pct",
        ]
    )


def _benchmark_forward_outcome(index_rows: pd.DataFrame, signal_date: pd.Timestamp, horizons: Sequence[int]) -> dict:
    result = {f"benchmark_forward_{h}d_pct": math.nan for h in horizons}
    matches = index_rows.index[index_rows["date"] == signal_date].tolist()
    if not matches:
        return result
    signal_index = int(matches[-1])
    entry_index = signal_index + 1
    if entry_index >= len(index_rows):
        return result
    entry_open = float(index_rows.iloc[entry_index]["open"])
    if entry_open <= 0:
        return result
    for horizon in horizons:
        exit_index = entry_index + horizon - 1
        if exit_index < len(index_rows):
            exit_close = float(index_rows.iloc[exit_index]["close"])
            result[f"benchmark_forward_{horizon}d_pct"] = round((exit_close / entry_open - 1.0) * 100.0, 4)
    return result


def _summaries(events: pd.DataFrame, horizons: Sequence[int]) -> pd.DataFrame:
    rows = []
    for horizon in horizons:
        stats = summarize_returns(events[f"return_{horizon}d_pct"])
        excess = pd.to_numeric(events[f"excess_return_{horizon}d_pct"], errors="coerce").dropna()
        rows.append(
            {
                "signal_type": TURTLE_SIGNAL,
                "horizon_days": horizon,
                **stats,
                "avg_excess_return_pct": round(float(excess.mean()), 4) if len(excess) else 0.0,
                "excess_win_rate_pct": round(float((excess > 0).mean() * 100.0), 2) if len(excess) else 0.0,
                "avg_mfe_pct": round(float(pd.to_numeric(events["mfe_10d_pct"], errors="coerce").mean()), 4)
                if horizon == max(horizons) and not events.empty else math.nan,
                "avg_mae_pct": round(float(pd.to_numeric(events["mae_10d_pct"], errors="coerce").mean()), 4)
                if horizon == max(horizons) and not events.empty else math.nan,
            }
        )
    return pd.DataFrame(rows)


def _board_summaries(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for board, group in events.groupby("board", dropna=False):
        rows.append(
            {
                "board": board,
                "benchmark_name": group.iloc[0]["benchmark_name"],
                "event_count": len(group),
                "avg_volume_ratio20": round(float(group["volume_ratio20"].mean()), 4),
                "avg_event_stock_gain_pct": round(float(group["stock_return_pct"].mean()), 4),
                "avg_event_index_return_pct": round(float(group["benchmark_return_pct"].mean()), 4),
                **summarize_returns(group["return_10d_pct"]),
                "avg_excess_return_pct": round(float(pd.to_numeric(group["excess_return_10d_pct"], errors="coerce").mean()), 4),
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
    for sample, group in usable.groupby("sample"):
        excess = pd.to_numeric(group["excess_return_10d_pct"], errors="coerce").dropna()
        rows.append(
            {
                "sample": sample,
                "split_date": split_date,
                **summarize_returns(group["return_10d_pct"]),
                "avg_excess_return_pct": round(float(excess.mean()), 4) if len(excess) else 0.0,
            }
        )
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
        test_start = windows[fold][0]
        test_end = windows[fold][-1]
        train = usable.loc[usable["signal_date"] <= train_end]
        test = usable.loc[(usable["signal_date"] >= test_start) & (usable["signal_date"] <= test_end)]
        train_stats = summarize_returns(train["return_10d_pct"])
        test_stats = summarize_returns(test["return_10d_pct"])
        rows.append(
            {
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


def _max_consecutive_losses(values: pd.Series) -> int:
    maximum = current = 0
    for value in values:
        if pd.notna(value) and float(value) <= 0:
            current += 1
            maximum = max(maximum, current)
        elif pd.notna(value):
            current = 0
    return maximum


def _stability(events: pd.DataFrame, validation: pd.DataFrame, walkforward: pd.DataFrame) -> pd.DataFrame:
    usable = events.dropna(subset=["return_10d_pct"]).sort_values("signal_date")
    monthly = usable.assign(month=usable["signal_date"].str[:7]).groupby("month")["return_10d_pct"].mean()
    train = validation.loc[validation["sample"] == "样本内60%"] if not validation.empty else pd.DataFrame()
    test = validation.loc[validation["sample"] == "样本外40%"] if not validation.empty else pd.DataFrame()
    train_avg = float(train.iloc[0]["avg_return_pct"]) if not train.empty else math.nan
    test_avg = float(test.iloc[0]["avg_return_pct"]) if not test.empty else math.nan
    test_count = int(test.iloc[0]["count"]) if not test.empty else 0
    positive_month_ratio = float((monthly > 0).mean() * 100.0) if len(monthly) else 0.0
    positive_fold_ratio = float((walkforward["test_avg_return_pct"] > 0).mean() * 100.0) if not walkforward.empty else 0.0
    if len(usable) < 50 or test_count < 20:
        assessment = "样本不足"
    elif train_avg > 0 and test_avg > 0 and positive_month_ratio >= 50 and positive_fold_ratio >= 50:
        assessment = "样本内外方向一致"
    else:
        assessment = "稳定性不足"
    return pd.DataFrame(
        [
            {
                "signal_type": TURTLE_SIGNAL,
                "usable_10d_samples": len(usable),
                "train_avg_return_pct": round(train_avg, 4) if pd.notna(train_avg) else math.nan,
                "test_avg_return_pct": round(test_avg, 4) if pd.notna(test_avg) else math.nan,
                "return_std_pct": round(float(usable["return_10d_pct"].std()), 4) if len(usable) > 1 else 0.0,
                "positive_month_ratio_pct": round(positive_month_ratio, 2),
                "months": len(monthly),
                "positive_walkforward_ratio_pct": round(positive_fold_ratio, 2),
                "walkforward_folds": len(walkforward),
                "max_consecutive_losses": _max_consecutive_losses(usable["return_10d_pct"]),
                "overlap_ratio_pct": round(float(usable["overlaps_prior_10d"].mean() * 100.0), 2) if len(usable) else 0.0,
                "assessment": assessment,
            }
        ]
    )


def analyze_turtle_events(
    prices: pd.DataFrame,
    benchmarks: pd.DataFrame,
    config: dict,
    names: dict[str, str] | None = None,
    rules: TurtleRules | None = None,
    horizons: Sequence[int] = HORIZONS,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> TurtleAnalyticsResult:
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(config)
    rule = rules or TurtleRules(
        index_drop_pct=float(cfg["turtle_index_drop_pct"]),
        shanghai_drop_pct=float(cfg.get("turtle_shanghai_drop_pct", -1.0)),
        ordinary_gain_pct=float(cfg["turtle_ordinary_gain_pct"]),
        growth_gain_pct=float(cfg["turtle_growth_gain_pct"]),
        min_volume_ratio=float(cfg["turtle_min_volume_ratio"]),
        max_volume_ratio=float(cfg["turtle_max_volume_ratio"]),
    )
    if rule.min_volume_ratio <= 0 or rule.max_volume_ratio < rule.min_volume_ratio:
        raise ValueError("海龟量比范围无效：上限必须大于或等于下限，且下限必须大于0。")
    data = prices.copy()
    data["symbol"] = data["symbol"].astype(str).str.zfill(6)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    numeric = ["open", "high", "low", "close", "volume"]
    data[numeric] = data[numeric].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=["date", *numeric]).sort_values(["symbol", "date"])

    indices = benchmarks.copy()
    indices["benchmark_code"] = indices["benchmark_code"].astype(str).str.zfill(6)
    indices["date"] = pd.to_datetime(indices["date"], errors="coerce")
    indices[["open", "close"]] = indices[["open", "close"]].apply(pd.to_numeric, errors="coerce")
    indices = indices.dropna(subset=["date", "open", "close"]).sort_values(["benchmark_code", "date"])
    index_groups = {}
    for code, rows in indices.groupby("benchmark_code", sort=False):
        frame = rows.sort_values("date").reset_index(drop=True).copy()
        frame["benchmark_return_pct"] = frame["close"].pct_change(fill_method=None) * 100.0
        index_groups[str(code)] = frame
    shanghai_rows = index_groups.get("000001")
    shanghai_daily = (
        shanghai_rows.set_index("date")["benchmark_return_pct"]
        if shanghai_rows is not None
        else pd.Series(dtype=float)
    )

    trade_value = min(float(cfg["initial_capital"]), float(cfg["initial_capital"]) * float(cfg["max_position_pct"]) / 100.0)
    event_rows = []
    groups = data.groupby("symbol", sort=False)
    total = groups.ngroups
    for group_index, (symbol, stock) in enumerate(groups, 1):
        if progress:
            progress(group_index - 1, total, f"正在检验海龟 {symbol}（{group_index}/{total}）")
        board = board_benchmark(symbol, rule)
        if board is None or board.benchmark_code not in index_groups:
            continue
        stock_name = str((names or {}).get(str(symbol).zfill(6), str(symbol).zfill(6)))
        if "ST" in stock_name.upper() or "退" in stock_name:
            continue
        rows = stock.sort_values("date").reset_index(drop=True).copy()
        rows["stock_return_pct"] = rows["close"].pct_change(fill_method=None) * 100.0
        rows["volume_ratio20"] = rows["volume"] / rows["volume"].shift(1).rolling(20, min_periods=20).mean()
        index_rows = index_groups[board.benchmark_code]
        index_daily = index_rows.set_index("date")["benchmark_return_pct"]
        rows["benchmark_return_pct"] = rows["date"].map(index_daily)
        rows["shanghai_return_pct"] = rows["date"].map(shanghai_daily)
        rows["board_crash"] = rows["benchmark_return_pct"] <= rule.index_drop_pct
        rows["shanghai_crash"] = rows["shanghai_return_pct"] <= rule.shanghai_drop_pct
        hits = rows.index[
            (rows["date"] >= pd.Timestamp(rule.start_date))
            & (rows["board_crash"] | rows["shanghai_crash"])
            & (rows["stock_return_pct"] >= board.gain_threshold_pct)
            & (rows["volume_ratio20"] >= rule.min_volume_ratio)
            & (rows["volume_ratio20"] <= rule.max_volume_ratio)
        ].tolist()
        last_hit = -999
        for signal_index in hits:
            row = rows.iloc[signal_index]
            event_cfg = dict(cfg)
            event_cfg["price_limit_pct"] = board.price_limit_pct
            outcome = calculate_event_outcome(rows, signal_index, event_cfg, trade_value, horizons)
            benchmark_outcome = _benchmark_forward_outcome(index_rows, pd.Timestamp(row["date"]), horizons)
            excess = {}
            for horizon in horizons:
                stock_return = outcome[f"return_{horizon}d_pct"]
                index_return = benchmark_outcome[f"benchmark_forward_{horizon}d_pct"]
                excess[f"excess_return_{horizon}d_pct"] = (
                    round(float(stock_return) - float(index_return), 4)
                    if pd.notna(stock_return) and pd.notna(index_return) else math.nan
                )
            event_rows.append(
                {
                    "signal_date": pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
                    "symbol": str(symbol).zfill(6),
                    "name": stock_name,
                    "signal_type": TURTLE_SIGNAL,
                    "board": board.board,
                    "benchmark_code": board.benchmark_code,
                    "benchmark_name": board.benchmark_name,
                    "index_drop_threshold_pct": rule.index_drop_pct,
                    "shanghai_drop_threshold_pct": rule.shanghai_drop_pct,
                    "market_trigger": (
                        "对应指数+上证"
                        if bool(row["board_crash"]) and bool(row["shanghai_crash"])
                        else ("对应指数" if bool(row["board_crash"]) else "上证指数")
                    ),
                    "stock_gain_threshold_pct": board.gain_threshold_pct,
                    "min_volume_ratio": rule.min_volume_ratio,
                    "max_volume_ratio": rule.max_volume_ratio,
                    "volume_ratio20": round(float(row["volume_ratio20"]), 4),
                    "stock_return_pct": round(float(row["stock_return_pct"]), 4),
                    "benchmark_return_pct": round(float(row["benchmark_return_pct"]), 4),
                    "shanghai_return_pct": round(float(row["shanghai_return_pct"]), 4),
                    "relative_strength_pct": round(float(row["stock_return_pct"] - row["benchmark_return_pct"]), 4),
                    "signal_close": round(float(row["close"]), 4),
                    "overlaps_prior_10d": signal_index - last_hit <= 10,
                    **outcome,
                    **benchmark_outcome,
                    **excess,
                }
            )
            last_hit = signal_index
        if progress:
            progress(group_index, total, f"已完成 {group_index}/{total}")

    events = pd.DataFrame(event_rows) if event_rows else _empty_events(horizons)
    if not events.empty:
        events = events.sort_values(["signal_date", "relative_strength_pct", "symbol"], ascending=[True, False, True]).reset_index(drop=True)
    summary = _summaries(events, horizons)
    board_summary = _board_summaries(events)
    validation, split_date = _validation(events)
    walkforward = _walkforward(events)
    stability = _stability(events, validation, walkforward)
    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "price_start": data["date"].min().strftime("%Y-%m-%d") if not data.empty else "",
        "price_end": data["date"].max().strftime("%Y-%m-%d") if not data.empty else "",
        "benchmark_start": indices["date"].min().strftime("%Y-%m-%d") if not indices.empty else "",
        "benchmark_end": indices["date"].max().strftime("%Y-%m-%d") if not indices.empty else "",
        "research_start": rule.start_date,
        "symbols": int(data["symbol"].nunique()),
        "events": len(events),
        "usable_10d_samples": int(events["return_10d_pct"].notna().sum()) if not events.empty else 0,
        "unfillable_events": int((~events["fillable"].astype(bool)).sum()) if not events.empty else 0,
        "analysis_trade_value": round(trade_value, 2),
        "validation_split_date": split_date,
        "rules": {
            "index_drop_pct": rule.index_drop_pct,
            "shanghai_drop_pct": rule.shanghai_drop_pct,
            "ordinary_gain_pct": rule.ordinary_gain_pct,
            "growth_gain_pct": rule.growth_gain_pct,
            "min_volume_ratio": rule.min_volume_ratio,
            "max_volume_ratio": rule.max_volume_ratio,
        },
    }
    return TurtleAnalyticsResult(events, summary, board_summary, validation, walkforward, stability, metadata)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "无可用样本。"
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for values in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join("" if pd.isna(value) else str(value).replace("|", "/") for value in values) + " |")
    return "\n".join(lines)


def _markdown_report(result: TurtleAnalyticsResult) -> str:
    rules = result.metadata["rules"]
    return "\n".join(
        [
            "# 逆势海龟研究报告",
            "",
            f"生成时间：{result.metadata['generated_at']}",
            f"研究范围：{result.metadata['research_start']} 至 {result.metadata['price_end']}",
            f"事件数量：{result.metadata['events']}，可用10日样本：{result.metadata['usable_10d_samples']}",
            f"规则：对应指数单日跌幅 <= {rules['index_drop_pct']}%，或上证指数单日跌幅 <= {rules['shanghai_drop_pct']}%；沪深主板个股涨幅 >= {rules['ordinary_gain_pct']}%，创业板/科创板个股涨幅 >= {rules['growth_gain_pct']}%，当日成交量为前20日均量的 {rules['min_volume_ratio']}-{rules['max_volume_ratio']} 倍。",
            "",
            "> 本报告把用户提出的“海龟”定义为逆势强势研究信号，不等同于经典海龟交易法。收盘确认，下一交易日开盘模拟买入，并计入佣金、印花税和双边滑点。",
            "",
            "## 1/3/5/10日统计",
            "",
            _markdown_table(result.summary),
            "",
            "## 板块分组（10日）",
            "",
            _markdown_table(result.board_summary),
            "",
            "## 样本内外验证（10日）",
            "",
            _markdown_table(result.validation),
            "",
            "## 扩展窗口验证",
            "",
            _markdown_table(result.walkforward),
            "",
            "## 稳定性",
            "",
            _markdown_table(result.stability),
            "",
            "## 限制",
            "",
            "- 事件只覆盖当前自选股和可取得的历史日线，不代表全市场。",
            "- 北交所股票暂不纳入，避免用不匹配的沪深指数代替北证基准。",
            "- 同一股票10个交易日内重复信号会标记为重叠，不能视为完全独立样本。",
            "- 统计结论只用于检验历史规律，不构成收益保证或投资建议。",
            "",
        ]
    )


def save_turtle_bundle(result: TurtleAnalyticsResult, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "events": output_dir / "events_latest.csv",
        "summary": output_dir / "summary_latest.csv",
        "board": output_dir / "board_summary_latest.csv",
        "validation": output_dir / "validation_latest.csv",
        "walkforward": output_dir / "walkforward_latest.csv",
        "stability": output_dir / "stability_latest.csv",
        "metadata": output_dir / "metadata_latest.json",
        "report": output_dir / "report_latest.md",
    }
    result.events.to_csv(paths["events"], index=False, encoding="utf-8-sig")
    result.summary.to_csv(paths["summary"], index=False, encoding="utf-8-sig")
    result.board_summary.to_csv(paths["board"], index=False, encoding="utf-8-sig")
    result.validation.to_csv(paths["validation"], index=False, encoding="utf-8-sig")
    result.walkforward.to_csv(paths["walkforward"], index=False, encoding="utf-8-sig")
    result.stability.to_csv(paths["stability"], index=False, encoding="utf-8-sig")
    paths["metadata"].write_text(json.dumps(result.metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["report"].write_text(_markdown_report(result), encoding="utf-8")
    return paths
