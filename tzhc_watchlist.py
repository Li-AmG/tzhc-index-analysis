from __future__ import annotations

import csv
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List


STOCK_CODE_PATTERN = rb"(?<!\d)(?:00|30|60|68|43|83|87|88|92)\d{4}(?!\d)"
SUPPORTED_STOCK_CODE = re.compile(r"^(?:00|30|60|68|43|83|87|88|92)\d{4}$")


@dataclass
class WatchItem:
    code: str
    name: str
    tags: str
    source: str


def extract_codes_from_sel(path: Path) -> List[str]:
    if not path.exists():
        return []
    seen = set()
    codes = []
    for match in re.finditer(STOCK_CODE_PATTERN, path.read_bytes()):
        code = match.group(0).decode("ascii")
        if code not in seen:
            seen.add(code)
            codes.append(code)
    return codes


def parse_text_watchlist(path: Path) -> List[WatchItem]:
    if not path.exists():
        return []
    text = None
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            text = path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise ValueError(f"无法识别自选股文本编码: {path}")
    items = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split(",")]
        code = parts[0] if parts else ""
        if not re.fullmatch(r"\d{6}", code):
            continue
        name = parts[1] if len(parts) > 1 and parts[1] else code
        tags = " ".join(part for part in parts[2:] if part)
        items.append(WatchItem(code, name, tags, "手工"))
    return items


def merge_watchlists(sel_path: Path, manual_path: Path) -> List[WatchItem]:
    rows = [WatchItem(code, code, "", "同花顺") for code in extract_codes_from_sel(sel_path)]
    index = {item.code: idx for idx, item in enumerate(rows)}
    for item in parse_text_watchlist(manual_path):
        if item.code in index:
            old = rows[index[item.code]]
            rows[index[item.code]] = WatchItem(
                item.code,
                item.name if item.name != item.code else old.name,
                item.tags or old.tags,
                "同花顺+手工",
            )
        else:
            index[item.code] = len(rows)
            rows.append(item)
    return rows


def save_watchlist(items: Iterable[WatchItem], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        backup_dir = output_path.parent / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup = backup_dir / f"watchlist_{datetime.now():%Y%m%d_%H%M%S_%f}.csv"
        backup.write_bytes(output_path.read_bytes())
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["code", "name", "tags", "source"])
        writer.writeheader()
        for item in items:
            writer.writerow(asdict(item))
    return output_path


def load_watchlist(path: Path) -> List[WatchItem]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            WatchItem(
                (row.get("code") or "").strip(),
                (row.get("name") or "").strip(),
                (row.get("tags") or "").strip(),
                (row.get("source") or "").strip(),
            )
            for row in reader
            if (row.get("code") or "").strip()
        ]


def add_watch_item(
    path: Path,
    code: str,
    name: str = "",
    tags: str = "手工新增 待更新行情",
    source: str = "系统直接添加",
) -> tuple[WatchItem, bool]:
    clean_code = code.strip()
    if not SUPPORTED_STOCK_CODE.fullmatch(clean_code):
        raise ValueError("请输入受支持的6位A股代码")
    rows = load_watchlist(path)
    for index, item in enumerate(rows):
        if item.code != clean_code:
            continue
        clean_name = name.strip()
        clean_tags = tags.strip()
        sources = [part for part in item.source.split("+") if part]
        if source and source not in sources:
            sources.append(source)
        updated = WatchItem(
            clean_code,
            clean_name if clean_name and item.name == clean_code else (item.name or clean_name or clean_code),
            item.tags or clean_tags,
            "+".join(sources),
        )
        if updated != item:
            rows[index] = updated
            save_watchlist(rows, path)
        return updated, False
    item = WatchItem(clean_code, name.strip() or clean_code, tags.strip(), source)
    rows.append(item)
    save_watchlist(rows, path)
    return item, True


def import_legacy_watchlist(legacy_root: Path, output_path: Path) -> List[WatchItem]:
    sel_path = legacy_root / "自选股.sel"
    manual_path = legacy_root / "手工追加自选.txt"
    if not sel_path.exists() and not manual_path.exists():
        raise FileNotFoundError(f"旧系统没有找到自选股文件: {legacy_root}")
    items = merge_watchlists(sel_path, manual_path)
    if not items:
        raise ValueError("旧系统自选股为空，未导入任何代码。")
    save_watchlist(items, output_path)
    return items
