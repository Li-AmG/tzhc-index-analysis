from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


GRADE_A = "A类·低位反转海龟"
GRADE_B = "B类·待观察"


@dataclass
class TurtleReversalResult:
    events: pd.DataFrame
    summary: pd.DataFrame
    metadata: dict


def _commission(config: dict, amount: float) -> float:
    return max(float(config["min_commission"]), amount * float(config["commission_rate"]))


def _simulate_trailing_exit(
    stock: pd.DataFrame,
    entry_date: pd.Timestamp,
    entry_price: float,
    shares: int,
    config: dict,
    trailing_pct: float,
    max_holding_days: int = 10,
) -> dict:
    result = {
        f"trailing_{int(trailing_pct)}_return_pct": math.nan,
        f"trailing_{int(trailing_pct)}_exit_date": "",
        f"trailing_{int(trailing_pct)}_exit_reason": "未来交易日不足",
    }
    rows = stock.sort_values("date").reset_index(drop=True)
    matches = rows.index[rows["date"] == entry_date].tolist()
    if not matches or shares <= 0 or entry_price <= 0:
        return result

    entry_index = int(matches[-1])
    last_index = min(len(rows) - 1, entry_index + max_holding_days - 1)
    if last_index <= entry_index:
        return result

    slip = float(config["slippage_pct"]) / 100.0
    buy_amount = shares * entry_price
    entry_cost = buy_amount + _commission(config, buy_amount)
    peak_close = max(entry_price, float(rows.iloc[entry_index]["close"]))
    stop_price = peak_close * (1.0 - trailing_pct / 100.0)
    exit_index = last_index
    exit_raw = float(rows.iloc[last_index]["close"])
    reason = f"持有满{max_holding_days}日"

    # 买入当天不能卖。从下一交易日开始，只使用前一收盘后已经确定的止损价。
    for index in range(entry_index + 1, last_index + 1):
        row = rows.iloc[index]
        if float(row["open"]) <= stop_price:
            exit_index = index
            exit_raw = float(row["open"])
            reason = f"开盘低于{trailing_pct:g}%移动保护价"
            break
        if float(row["low"]) <= stop_price:
            exit_index = index
            exit_raw = stop_price
            reason = f"盘中触及{trailing_pct:g}%移动保护价"
            break
        peak_close = max(peak_close, float(row["close"]))
        stop_price = max(stop_price, peak_close * (1.0 - trailing_pct / 100.0))

    exit_price = exit_raw * (1.0 - slip)
    proceeds = shares * exit_price
    sell_fee = _commission(config, proceeds) + proceeds * float(config["stamp_tax_rate"])
    key = f"trailing_{int(trailing_pct)}"
    result.update(
        {
            f"{key}_return_pct": round((proceeds - sell_fee - entry_cost) / entry_cost * 100.0, 4),
            f"{key}_exit_date": pd.Timestamp(rows.iloc[exit_index]["date"]).strftime("%Y-%m-%d"),
            f"{key}_exit_reason": reason,
        }
    )
    return result


def _feature_frame(prices: pd.DataFrame) -> pd.DataFrame:
    rows = prices.copy()
    rows["symbol"] = rows["symbol"].astype(str).str.zfill(6)
    rows["date"] = pd.to_datetime(rows["date"], errors="coerce")
    numeric = ["open", "high", "low", "close", "volume"]
    rows[numeric] = rows[numeric].apply(pd.to_numeric, errors="coerce")
    rows = rows.dropna(subset=["date", *numeric]).sort_values(["symbol", "date"]).reset_index(drop=True)
    grouped = rows.groupby("symbol", sort=False)
    for days in (5, 10, 20, 60):
        rows[f"ma{days}"] = grouped["close"].transform(
            lambda values, window=days: values.rolling(window, min_periods=window).mean()
        )
    rows["previous_close"] = grouped["close"].shift(1)
    for days in (5, 10, 20):
        rows[f"previous_ma{days}"] = grouped[f"ma{days}"].shift(1)
    rows["ma60_5d_ago"] = grouped["ma60"].shift(5)
    rows["prior_10d_high"] = grouped["high"].transform(
        lambda values: values.shift(1).rolling(10, min_periods=10).max()
    )
    spread = (rows["high"] - rows["low"]).replace(0, pd.NA)
    rows["close_location"] = (rows["close"] - rows["low"]) / spread
    return rows


def score_turtle_reversal_events(prices: pd.DataFrame, turtle_events: pd.DataFrame) -> pd.DataFrame:
    events = turtle_events.copy()
    if events.empty:
        return events
    events["symbol"] = events["symbol"].astype(str).str.zfill(6)
    events["signal_date"] = pd.to_datetime(events["signal_date"], errors="coerce")
    features = _feature_frame(prices)
    feature_columns = [
        "date", "symbol", "ma5", "ma10", "ma20", "ma60", "ma60_5d_ago",
        "previous_close", "previous_ma5", "previous_ma10", "previous_ma20",
        "prior_10d_high", "close_location",
    ]
    result = events.merge(
        features[feature_columns],
        left_on=["signal_date", "symbol"],
        right_on=["date", "symbol"],
        how="left",
    )
    result["温和确认量"] = pd.to_numeric(result["volume_ratio20"], errors="coerce") >= 1.5
    result["一阳收复三线"] = (
        (result["previous_close"] <= result["previous_ma5"])
        & (result["previous_close"] <= result["previous_ma10"])
        & (result["previous_close"] <= result["previous_ma20"])
        & (result["signal_close"] > result["ma5"])
        & (result["signal_close"] > result["ma10"])
        & (result["signal_close"] > result["ma20"])
    )
    result["突破十日平台"] = result["signal_close"] > result["prior_10d_high"]
    result["低位下降趋势"] = (
        (result["signal_close"] < result["ma60"])
        & (result["ma60"] < result["ma60_5d_ago"])
    )
    result["收盘强势"] = result["close_location"] >= 0.70
    score_columns = ["温和确认量", "一阳收复三线", "突破十日平台", "低位下降趋势", "收盘强势"]
    result["反转评分"] = result[score_columns].fillna(False).astype(int).sum(axis=1)
    result["研究等级"] = ""
    result.loc[result["反转评分"] == 4, "研究等级"] = GRADE_B
    result.loc[result["反转评分"] == 5, "研究等级"] = GRADE_A
    result["signal_date"] = result["signal_date"].dt.strftime("%Y-%m-%d")
    return result.drop(columns=["date"])


def _summarize(values: pd.Series) -> dict:
    data = pd.to_numeric(values, errors="coerce").dropna()
    if data.empty:
        return {"count": 0, "win_rate_pct": 0.0, "avg_return_pct": 0.0, "median_return_pct": 0.0,
                "worst_return_pct": 0.0, "best_return_pct": 0.0}
    return {
        "count": len(data),
        "win_rate_pct": round(float((data > 0).mean() * 100.0), 2),
        "avg_return_pct": round(float(data.mean()), 4),
        "median_return_pct": round(float(data.median()), 4),
        "worst_return_pct": round(float(data.min()), 4),
        "best_return_pct": round(float(data.max()), 4),
    }


def analyze_turtle_reversal(
    prices: pd.DataFrame,
    turtle_events: pd.DataFrame,
    config: dict,
) -> TurtleReversalResult:
    scored = score_turtle_reversal_events(prices, turtle_events)
    price_groups = {
        str(symbol).zfill(6): rows.sort_values("date").reset_index(drop=True)
        for symbol, rows in _feature_frame(prices).groupby("symbol", sort=False)
    }
    simulation_rows = []
    for row in scored.itertuples(index=False):
        base = {}
        if bool(getattr(row, "fillable", False)) and str(row.symbol) in price_groups:
            entry_date = pd.to_datetime(getattr(row, "entry_date", ""), errors="coerce")
            if pd.notna(entry_date):
                for trailing in (5.0, 8.0):
                    base.update(
                        _simulate_trailing_exit(
                            price_groups[str(row.symbol)], entry_date, float(row.entry_price), int(row.shares),
                            config, trailing,
                        )
                    )
        simulation_rows.append(base)
    simulated = pd.DataFrame(simulation_rows, index=scored.index)
    events = pd.concat([scored, simulated], axis=1)
    events["sample"] = events["signal_date"].map(
        lambda value: "研究期2024-2025" if str(value) < "2026-01-01" else "样本外2026"
    )

    summary_rows = []
    cohorts = {
        "评分至少3": events["反转评分"] >= 3,
        "B级及以上": events["反转评分"] >= 4,
        "A级五项全满足": events["反转评分"] == 5,
        "A级沪深主板": (events["反转评分"] == 5) & ~events["symbol"].str.startswith(("300", "301", "688", "689")),
    }
    methods = {
        "固定持有10日": "return_10d_pct",
        "5%移动保护": "trailing_5_return_pct",
        "8%移动保护": "trailing_8_return_pct",
    }
    for cohort_name, cohort_mask in cohorts.items():
        for sample in ("研究期2024-2025", "样本外2026", "全部"):
            sample_mask = pd.Series(True, index=events.index) if sample == "全部" else events["sample"].eq(sample)
            group = events.loc[cohort_mask & sample_mask]
            for method_name, column in methods.items():
                summary_rows.append(
                    {"cohort": cohort_name, "sample": sample, "exit_method": method_name,
                     **_summarize(group[column] if column in group else pd.Series(dtype=float))}
                )
    summary = pd.DataFrame(summary_rows)
    a_events = events.loc[events["反转评分"] == 5]
    a_oos = a_events.loc[a_events["sample"] == "样本外2026"]
    a_returns = pd.to_numeric(a_oos.get("trailing_5_return_pct"), errors="coerce").dropna()
    gate_passed = (
        len(a_events) >= 50
        and len(a_returns) >= 20
        and float(a_returns.mean()) > 0
        and float(a_returns.median()) > 0
    )
    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "price_end": pd.to_datetime(prices["date"], errors="coerce").max().strftime("%Y-%m-%d"),
        "base_turtle_events": len(events),
        "grade_a_events": int((events["反转评分"] == 5).sum()),
        "grade_b_events": int((events["反转评分"] == 4).sum()),
        "grade_a_oos_usable": len(a_returns),
        "promotion_gate": "通过" if gate_passed else "未通过",
        "promotion_rule": "A级总样本>=50、样本外>=20，且样本外5%移动保护的平均值和中位数均>0",
        "automatic_paper_buy": False,
    }
    return TurtleReversalResult(events, summary, metadata)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "无可用样本。"
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for values in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join("" if pd.isna(value) else str(value).replace("|", "/") for value in values) + " |")
    return "\n".join(lines)


def save_turtle_reversal_bundle(result: TurtleReversalResult, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "events": output_dir / "events_latest.csv",
        "summary": output_dir / "summary_latest.csv",
        "metadata": output_dir / "metadata_latest.json",
        "report": output_dir / "report_latest.md",
    }
    result.events.to_csv(paths["events"], index=False, encoding="utf-8-sig")
    result.summary.to_csv(paths["summary"], index=False, encoding="utf-8-sig")
    paths["metadata"].write_text(json.dumps(result.metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    a_events = result.events.loc[result.events["反转评分"] == 5]
    report = "\n".join(
        [
            "# 片仔癀型低位反转海龟研究",
            "",
            f"生成时间：{result.metadata['generated_at']}",
            f"行情截至：{result.metadata['price_end']}",
            f"A级：{result.metadata['grade_a_events']}个；B级：{result.metadata['grade_b_events']}个。",
            f"正式准入：{result.metadata['promotion_gate']}。{result.metadata['promotion_rule']}。",
            "",
            "> 本模块与原海龟、双高相互独立，只做研究观察，不自动进入模拟账户。",
            "",
            "## 固定条件",
            "",
            "1. 海龟基础条件已经满足。",
            "2. 量比20不低于1.5（原海龟上限仍为2.0）。",
            "3. 前一日位于MA5/MA10/MA20下方，信号日一阳同时收复三线。",
            "4. 收盘突破前10日最高价。",
            "5. 股价仍在下降的MA60下方，属于低位反转而非高位追涨。",
            "6. 收盘位于当日振幅上部30%。",
            "",
            "## 分期与退出统计",
            "",
            _markdown_table(result.summary),
            "",
            "## A级事件",
            "",
            _markdown_table(a_events[[column for column in [
                "signal_date", "symbol", "name", "stock_return_pct", "volume_ratio20", "relative_strength_pct",
                "return_10d_pct", "trailing_5_return_pct", "trailing_8_return_pct", "sample",
            ] if column in a_events.columns]]),
            "",
            "## 结论边界",
            "",
            "- A级不足50个或样本外不足20个时，一律保持研究状态。",
            "- 片仔癀是成功样本，不允许以单笔结果反向修改阈值。",
            "- T+1模拟在次日开盘买入；移动保护最早从买入后的下一交易日执行，并计入费用与滑点。",
            "",
        ]
    )
    paths["report"].write_text(report, encoding="utf-8")
    return paths
