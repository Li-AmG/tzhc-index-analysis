"""Candidate03 live observation files and manually confirmed brokerage holdings."""

from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path

import pandas as pd


HOLDING_COLUMNS = [
    "symbol", "name", "entry_date", "entry_price", "shares", "notes", "created_at"
]
SUPPORTED_SYMBOL = re.compile(r"^(?:00|30|60|68)\d{4}$")


def empty_holdings() -> pd.DataFrame:
    return pd.DataFrame(columns=HOLDING_COLUMNS)


def load_candidate03_holdings(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return empty_holdings()
    try:
        frame = pd.read_csv(path, dtype={"symbol": str}, encoding="utf-8-sig")
    except pd.errors.EmptyDataError:
        return empty_holdings()
    for column in HOLDING_COLUMNS:
        if column not in frame:
            frame[column] = ""
    frame = frame[HOLDING_COLUMNS].copy()
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    frame["entry_date"] = pd.to_datetime(frame["entry_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    frame["entry_price"] = pd.to_numeric(frame["entry_price"], errors="coerce")
    frame["shares"] = pd.to_numeric(frame["shares"], errors="coerce").fillna(0).astype(int)
    return frame.dropna(subset=["entry_date", "entry_price"]).reset_index(drop=True)


def _atomic_save(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame[HOLDING_COLUMNS].to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)
    return path


def add_candidate03_holding(
    path: Path,
    symbol: str,
    name: str,
    entry_date: str | date,
    entry_price: float,
    shares: int,
    notes: str = "",
) -> pd.DataFrame:
    clean_symbol = str(symbol).strip().zfill(6)
    if not SUPPORTED_SYMBOL.fullmatch(clean_symbol):
        raise ValueError("候选03只支持沪深主板、创业板和科创板6位代码")
    parsed_date = pd.to_datetime(entry_date, errors="coerce")
    if pd.isna(parsed_date):
        raise ValueError("买入日期无效")
    if parsed_date.date() > datetime.now().date():
        raise ValueError("买入日期不能晚于今天")
    clean_price = float(entry_price)
    clean_shares = int(shares)
    if clean_price <= 0:
        raise ValueError("实际成本价必须大于0")
    if clean_shares <= 0:
        raise ValueError("实际持股数必须大于0")

    frame = load_candidate03_holdings(path)
    if clean_symbol in set(frame["symbol"]):
        raise ValueError("该股票已在候选03实盘观察持仓中，请先移除再重新登记")
    row = {
        "symbol": clean_symbol,
        "name": str(name).strip() or clean_symbol,
        "entry_date": parsed_date.strftime("%Y-%m-%d"),
        "entry_price": round(clean_price, 4),
        "shares": clean_shares,
        "notes": str(notes).strip(),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
    _atomic_save(frame, path)
    return frame


def remove_candidate03_holding(path: Path, symbol: str) -> pd.DataFrame:
    clean_symbol = str(symbol).strip().zfill(6)
    frame = load_candidate03_holdings(path)
    frame = frame.loc[frame["symbol"] != clean_symbol].reset_index(drop=True)
    _atomic_save(frame, path)
    return frame
