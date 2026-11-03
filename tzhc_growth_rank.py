from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from tzhc_market import _connect, latest_completed_market_date, normalize_bars
from tzhc_watchlist import WatchItem, load_watchlist, save_watchlist


GROWTH_PREFIXES = ("300", "301")


@dataclass(frozen=True)
class GrowthRankResult:
    rows: pd.DataFrame
    server: str
    quote_time: str
    shanghai_return_pct: float
    growth_index_return_pct: float


def _index_return(quotes: list[dict], code: str) -> float:
    row = next((item for item in quotes if str(item.get("code")) == code), None)
    if not row:
        return float("nan")
    price = float(row.get("price") or 0)
    previous = float(row.get("last_close") or 0)
    return (price / previous - 1.0) * 100.0 if price > 0 and previous > 0 else float("nan")


def _fetch_quotes_safe(api, requests: list[tuple[int, str]]) -> list[dict]:
    if not requests:
        return []
    try:
        return list(api.get_security_quotes(requests) or [])
    except Exception:
        if len(requests) == 1:
            return []
        midpoint = len(requests) // 2
        return _fetch_quotes_safe(api, requests[:midpoint]) + _fetch_quotes_safe(api, requests[midpoint:])


def fetch_growth_top_rank(config: dict, limit: int = 15) -> GrowthRankResult:
    api, server = _connect()
    try:
        securities: list[tuple[str, str]] = []
        total = int(api.get_security_count(0) or 0)
        for start in range(0, total, 1000):
            for row in api.get_security_list(0, start) or []:
                code = str(row.get("code") or "").strip()
                name = str(row.get("name") or "").strip()
                if not code.startswith(GROWTH_PREFIXES):
                    continue
                upper_name = name.upper()
                if "ST" in upper_name or "退" in name:
                    continue
                securities.append((code, name or code))

        quote_rows: list[dict] = []
        for start in range(0, len(securities), 80):
            batch = securities[start:start + 80]
            names = dict(batch)
            for quote in _fetch_quotes_safe(api, [(0, code) for code, _ in batch]):
                code = str(quote.get("code") or "").strip()
                price = float(quote.get("price") or 0)
                previous = float(quote.get("last_close") or 0)
                if price <= 0 or previous <= 0:
                    continue
                quote_rows.append(
                    {
                        "symbol": code,
                        "name": names.get(code, code),
                        "price": price,
                        "previous_close": previous,
                        "gain_pct": (price / previous - 1.0) * 100.0,
                        "volume": float(quote.get("vol") or 0),
                        "amount": float(quote.get("amount") or 0),
                        "quote_time": str(quote.get("servertime") or ""),
                    }
                )

        index_quotes = _fetch_quotes_safe(api, [(1, "000001"), (0, "399006")])
        shanghai_return = _index_return(index_quotes, "000001")
        growth_return = _index_return(index_quotes, "399006")
        ranked = pd.DataFrame(quote_rows).sort_values(["gain_pct", "amount"], ascending=[False, False]).head(limit).copy()
        cutoff = latest_completed_market_date()
        volume_ratios: list[float] = []
        for row in ranked.itertuples():
            bars = api.get_security_bars(9, 0, str(row.symbol), 0, 35) or []
            history = normalize_bars(str(row.symbol), bars, cutoff)
            previous_volume = pd.Series([float(item["volume"]) for item in history[-20:]], dtype=float)
            average = float(previous_volume.mean()) if len(previous_volume) >= 20 else float("nan")
            volume_ratios.append(float(row.volume) / average if average > 0 else float("nan"))
        ranked["volume_ratio20_preview"] = volume_ratios

        board_drop = float(config.get("turtle_index_drop_pct", -2.0))
        shanghai_drop = float(config.get("turtle_shanghai_drop_pct", -1.0))
        growth_gain = float(config.get("turtle_growth_gain_pct", 10.0))
        minimum_volume = float(config.get("turtle_min_volume_ratio", 1.2))
        maximum_volume = float(config.get("turtle_max_volume_ratio", 2.0))
        market_ok = growth_return <= board_drop or shanghai_return <= shanghai_drop
        ranked["market_trigger"] = (
            "创业板指+上证"
            if growth_return <= board_drop and shanghai_return <= shanghai_drop
            else ("创业板指" if growth_return <= board_drop else ("上证指数" if shanghai_return <= shanghai_drop else "未触发"))
        )
        ranked["turtle_preview"] = (
            market_ok
            & (ranked["gain_pct"] >= growth_gain)
            & ranked["volume_ratio20_preview"].between(minimum_volume, maximum_volume, inclusive="both")
        )
        ranked["shanghai_return_pct"] = shanghai_return
        ranked["growth_index_return_pct"] = growth_return
        ranked["source"] = "通达信实时兜底"
        ranked["snapshot_date"] = datetime.now().strftime("%Y-%m-%d")
        ranked.insert(0, "rank", range(1, len(ranked) + 1))
        quote_time = max((str(value) for value in ranked.get("quote_time", [])), default="")
        return GrowthRankResult(ranked.reset_index(drop=True), server, quote_time, shanghai_return, growth_return)
    finally:
        try:
            api.disconnect()
        except Exception:
            pass


def add_rank_to_watchlist(rows: pd.DataFrame, watchlist_path: Path, source: str = "创业板实时涨幅榜") -> int:
    current = load_watchlist(watchlist_path)
    index = {item.code: position for position, item in enumerate(current)}
    changed = 0
    tag = f"创业板涨幅前15 {datetime.now():%Y-%m-%d}"
    for row in rows.itertuples():
        code = str(row.symbol).zfill(6)
        name = str(row.name).strip() or code
        if code in index:
            position = index[code]
            old = current[position]
            sources = [part for part in old.source.split("+") if part]
            if source not in sources:
                sources.append(source)
            updated = WatchItem(code, name if old.name == code else old.name, old.tags or tag, "+".join(sources))
            if updated != old:
                current[position] = updated
                changed += 1
            continue
        index[code] = len(current)
        current.append(WatchItem(code, name, tag, source))
        changed += 1
    if changed:
        save_watchlist(current, watchlist_path)
    return changed
