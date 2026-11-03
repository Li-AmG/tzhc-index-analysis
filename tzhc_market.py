from __future__ import annotations

import csv
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Callable, Iterable, List, Optional

from tzhc_watchlist import WatchItem, save_watchlist


ROOT = Path(__file__).resolve().parent
VENDOR_DIR = ROOT / "vendor"
if VENDOR_DIR.exists():
    sys.path.insert(0, str(VENDOR_DIR))

try:
    from pytdx.hq import TdxHq_API
    TDX_IMPORT_ERROR = ""
except Exception as exc:
    TdxHq_API = None
    TDX_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


SERVER_LIST = [
    ("119.147.212.81", 7709),
    ("106.120.74.86", 7709),
    ("115.238.90.165", 7709),
    ("124.74.236.94", 7721),
    ("60.191.117.167", 7709),
    ("218.108.98.244", 7709),
]


@dataclass
class MarketUpdateResult:
    requested: int
    succeeded: int
    failed: int
    rows: int
    latest_date: str
    cutoff_date: str
    server: str
    output_path: str
    report_path: str
    errors: List[str]


@dataclass
class NameUpdateResult:
    requested: int
    resolved: int
    updated: int
    server: str
    output_path: str


@dataclass
class BenchmarkUpdateResult:
    requested: int
    succeeded: int
    failed: int
    rows: int
    latest_date: str
    cutoff_date: str
    server: str
    output_path: str
    report_path: str
    errors: List[str]


BENCHMARK_SPECS = (
    {"benchmark_code": "000001", "benchmark_name": "上证指数", "market": 1},
    {"benchmark_code": "399001", "benchmark_name": "深证成指", "market": 0},
    {"benchmark_code": "399006", "benchmark_name": "创业板指", "market": 0},
    {"benchmark_code": "000688", "benchmark_name": "科创50", "market": 1},
)


def market_code(stock_code: str) -> int:
    if stock_code.startswith(("60", "68", "51", "11")):
        return 1
    if stock_code.startswith(("00", "30", "15", "12")):
        return 0
    if stock_code.startswith(("43", "83", "87", "88", "92")):
        return 2
    return -1


def latest_completed_market_date(now: Optional[datetime] = None) -> str:
    moment = now or datetime.now()
    candidate = moment.date()
    if moment.weekday() >= 5 or moment.time() < time(15, 10):
        candidate -= timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate -= timedelta(days=1)
    return candidate.isoformat()


def normalize_bars(code: str, bars: Iterable[dict], max_date: str = "") -> List[dict]:
    rows = []
    for bar in bars:
        date = str(bar.get("datetime") or "")[:10]
        row = {
            "date": date,
            "symbol": code,
            "open": float(bar.get("open", 0)),
            "high": float(bar.get("high", 0)),
            "low": float(bar.get("low", 0)),
            "close": float(bar.get("close", 0)),
            "volume": float(bar.get("vol", bar.get("volume", 0))),
        }
        if not date or (max_date and date > max_date) or min(row["open"], row["high"], row["low"], row["close"]) <= 0:
            continue
        if row["high"] < max(row["open"], row["close"]) or row["low"] > min(row["open"], row["close"]):
            continue
        rows.append(row)
    rows.sort(key=lambda item: item["date"])
    return rows


def normalize_index_bars(spec: dict, bars: Iterable[dict], max_date: str = "") -> List[dict]:
    rows = normalize_bars(str(spec["benchmark_code"]), bars, max_date)
    for row in rows:
        row["benchmark_code"] = row.pop("symbol")
        row["benchmark_name"] = str(spec["benchmark_name"])
    return rows


def _connect():
    if TdxHq_API is None:
        raise RuntimeError(f"行情组件未能加载，无法联网更新：{TDX_IMPORT_ERROR}")
    errors = []
    for ip, port in SERVER_LIST:
        api = TdxHq_API(raise_exception=True)
        try:
            if api.connect(ip, port, time_out=3):
                return api, f"{ip}:{port}"
        except Exception as exc:
            errors.append(f"{ip}:{port} {type(exc).__name__}")
            try:
                api.disconnect()
            except Exception:
                pass
    raise RuntimeError("没有可用的行情节点：" + "；".join(errors))


def fetch_security_names(
    items: Iterable[WatchItem],
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> tuple[dict[str, str], str]:
    targets = {item.code for item in items if market_code(item.code) in (0, 1)}
    if not targets:
        return {}, ""
    api, server = _connect()
    names: dict[str, str] = {}
    try:
        page_jobs = []
        for market in (0, 1):
            total = int(api.get_security_count(market) or 0)
            page_jobs.extend((market, start) for start in range(0, total, 1000))
        for index, (market, start) in enumerate(page_jobs, 1):
            if progress:
                progress(index - 1, len(page_jobs), f"正在读取通达信证券名称 {index}/{len(page_jobs)}")
            for row in api.get_security_list(market, start) or []:
                code = str(row.get("code") or "").strip()
                name = str(row.get("name") or "").strip()
                if code in targets and name:
                    names[code] = name
            if progress:
                progress(index, len(page_jobs), f"已读取通达信证券名称 {index}/{len(page_jobs)}")
    finally:
        try:
            api.disconnect()
        except Exception:
            pass
    return names, server


def apply_security_names(items: Iterable[WatchItem], names: dict[str, str]) -> tuple[List[WatchItem], int]:
    rows = []
    updated = 0
    for item in items:
        resolved = names.get(item.code, "").strip()
        if resolved and (not item.name or item.name == item.code):
            rows.append(WatchItem(item.code, resolved, item.tags, item.source))
            updated += 1
        else:
            rows.append(item)
    return rows, updated


def refresh_watchlist_names(
    items: Iterable[WatchItem],
    output_path: Path,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> NameUpdateResult:
    rows = list(items)
    names, server = fetch_security_names(rows, progress)
    updated_rows, updated = apply_security_names(rows, names)
    if updated:
        save_watchlist(updated_rows, output_path)
    return NameUpdateResult(len(rows), len(names), updated, server, str(output_path))


def update_watchlist_prices(
    items: Iterable[WatchItem],
    output_path: Path,
    report_path: Path,
    days: int = 400,
    progress: Optional[Callable[[int, int, str], None]] = None,
    now: Optional[datetime] = None,
) -> MarketUpdateResult:
    watchlist = list(items)
    if not watchlist:
        raise ValueError("自选股为空，请先导入自选股。")
    api, server = _connect()
    all_rows = []
    errors = []
    latest_dates = []
    succeeded_symbols = set()
    cutoff_date = latest_completed_market_date(now)
    try:
        for index, item in enumerate(watchlist, 1):
            if progress:
                progress(index - 1, len(watchlist), f"正在更新 {item.code} {item.name}")
            market = market_code(item.code)
            if market < 0:
                errors.append(f"{item.code}：不支持的市场代码")
                continue
            try:
                bars = api.get_security_bars(9, market, item.code, 0, min(800, max(80, int(days)))) or []
                rows = normalize_bars(item.code, bars, cutoff_date)
                if len(rows) < 60:
                    raise ValueError(f"有效日线不足60条（{len(rows)}）")
                all_rows.extend(rows)
                latest_dates.append(rows[-1]["date"])
                succeeded_symbols.add(item.code)
            except Exception as exc:
                errors.append(f"{item.code} {item.name}：{exc}")
            if progress:
                progress(index, len(watchlist), f"已完成 {index}/{len(watchlist)}")
    finally:
        try:
            api.disconnect()
        except Exception:
            pass

    succeeded = len(watchlist) - len(errors)
    if not all_rows or succeeded <= 0:
        raise RuntimeError("行情更新全部失败，原行情文件未被覆盖。")
    if output_path.exists():
        with output_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                symbol = (row.get("symbol") or "").strip()
                date = (row.get("date") or "").strip()
                if symbol and symbol not in succeeded_symbols and date and date <= cutoff_date:
                    all_rows.append(
                        {
                            "date": date,
                            "symbol": symbol,
                            "open": row.get("open", ""),
                            "high": row.get("high", ""),
                            "low": row.get("low", ""),
                            "close": row.get("close", ""),
                            "volume": row.get("volume", ""),
                        }
                    )
    all_rows.sort(key=lambda row: (str(row["symbol"]), str(row["date"])))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["date", "symbol", "open", "high", "low", "close", "volume"])
        writer.writeheader()
        writer.writerows(all_rows)
    if output_path.exists():
        backup_dir = output_path.parent / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_path = backup_dir / f"a_share_prices_{datetime.now():%Y%m%d_%H%M%S_%f}.csv"
        shutil.copy2(output_path, backup_path)
    temp_path.replace(output_path)

    latest_date = max((str(row["date"]) for row in all_rows), default="")
    result = MarketUpdateResult(
        requested=len(watchlist),
        succeeded=succeeded,
        failed=len(errors),
        rows=len(all_rows),
        latest_date=latest_date,
        cutoff_date=cutoff_date,
        server=server,
        output_path=str(output_path),
        report_path=str(report_path),
        errors=errors,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps({"updated_at": datetime.now().isoformat(timespec="seconds"), **asdict(result)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def update_benchmark_prices(
    output_path: Path,
    report_path: Path,
    days: int = 800,
    progress: Optional[Callable[[int, int, str], None]] = None,
    now: Optional[datetime] = None,
) -> BenchmarkUpdateResult:
    api, server = _connect()
    all_rows = []
    errors = []
    succeeded_codes = set()
    cutoff_date = latest_completed_market_date(now)
    try:
        for index, spec in enumerate(BENCHMARK_SPECS, 1):
            code = str(spec["benchmark_code"])
            name = str(spec["benchmark_name"])
            if progress:
                progress(index - 1, len(BENCHMARK_SPECS), f"正在更新{name}")
            try:
                bars = api.get_index_bars(9, int(spec["market"]), code, 0, min(800, max(80, int(days)))) or []
                rows = normalize_index_bars(spec, bars, cutoff_date)
                if len(rows) < 60:
                    raise ValueError(f"有效指数日线不足60条（{len(rows)}）")
                all_rows.extend(rows)
                succeeded_codes.add(code)
            except Exception as exc:
                errors.append(f"{code} {name}：{exc}")
            if progress:
                progress(index, len(BENCHMARK_SPECS), f"已完成 {index}/{len(BENCHMARK_SPECS)}")
    finally:
        try:
            api.disconnect()
        except Exception:
            pass

    if not all_rows:
        raise RuntimeError("基准指数更新全部失败，原指数文件未被覆盖。")
    if output_path.exists():
        with output_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                code = (row.get("benchmark_code") or "").strip()
                date = (row.get("date") or "").strip()
                if code and code not in succeeded_codes and date and date <= cutoff_date:
                    all_rows.append(row)
    all_rows.sort(key=lambda row: (str(row["benchmark_code"]), str(row["date"])))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    fields = ["date", "benchmark_code", "benchmark_name", "open", "high", "low", "close", "volume"]
    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    temp_path.replace(output_path)

    result = BenchmarkUpdateResult(
        requested=len(BENCHMARK_SPECS),
        succeeded=len(succeeded_codes),
        failed=len(errors),
        rows=len(all_rows),
        latest_date=max((str(row["date"]) for row in all_rows), default=""),
        cutoff_date=cutoff_date,
        server=server,
        output_path=str(output_path),
        report_path=str(report_path),
        errors=errors,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps({"updated_at": datetime.now().isoformat(timespec="seconds"), **asdict(result)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result
