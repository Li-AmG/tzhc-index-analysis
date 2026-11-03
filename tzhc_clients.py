from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

from tzhc_watchlist import WatchItem, load_watchlist, save_watchlist


A_SHARE_CODE = re.compile(r"^(?:00|30|60|68|43|83|87|88|92)\d{4}$")


@dataclass(frozen=True)
class ClientSources:
    tdx_root: Optional[Path]
    tdx_watchlist: Optional[Path]
    ths_root: Optional[Path]
    ths_watchlist: Optional[Path]


@dataclass(frozen=True)
class SyncResult:
    source: str
    source_count: int
    before_count: int
    added_count: int
    after_count: int
    output_path: str


def _first_existing(paths: Iterable[Path]) -> Optional[Path]:
    return next((path for path in paths if path.exists()), None)


def discover_client_sources() -> ClientSources:
    tdx_root = _first_existing(
        Path(path)
        for path in (r"C:\new_tdx", r"D:\new_tdx", r"E:\new_tdx", r"F:\new_tdx")
    )
    ths_root = _first_existing(
        Path(path)
        for path in (r"C:\thsrj\ths", r"C:\同花顺软件\同花顺", r"D:\同花顺决策先锋")
    )
    tdx_watchlist = tdx_root / "T0002" / "blocknew" / "zxg.blk" if tdx_root else None
    if tdx_watchlist and not tdx_watchlist.exists():
        tdx_watchlist = None
    ths_watchlist = None
    if ths_root:
        candidates = list(ths_root.glob("mx_*/SelfStockInfo.json"))
        candidates.extend(ths_root.glob("thsguest*/SelfStockInfo.json"))
        valid = [path for path in candidates if path.exists() and path.stat().st_size > 4]
        ths_watchlist = max(valid, key=lambda path: path.stat().st_mtime) if valid else None
    return ClientSources(tdx_root, tdx_watchlist, ths_root, ths_watchlist)


def _valid_codes(values: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        code = str(value).strip()
        if A_SHARE_CODE.fullmatch(code) and code not in seen:
            seen.add(code)
            result.append(code)
    return result


def parse_tdx_watchlist(path: Path) -> List[str]:
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
        raise ValueError(f"无法识别通达信自选文件编码：{path}")
    codes = []
    for raw in text.splitlines():
        token = raw.strip()
        if re.fullmatch(r"\d{7}", token):
            codes.append(token[-6:])
        elif re.fullmatch(r"\d{6}", token):
            codes.append(token)
    return _valid_codes(codes)


def _extract_ths_codes(data) -> List[str]:
    values = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                values.append(str(item.get("C") or item.get("code") or ""))
            elif isinstance(item, str):
                values.append(item)
    elif isinstance(data, dict):
        if isinstance(data.get("data"), dict) and isinstance(data["data"].get("selfstock"), str):
            values.extend(data["data"]["selfstock"].split(",", 1)[0].split("|"))
        elif isinstance(data.get("selfstock"), str):
            values.extend(data["selfstock"].split(",", 1)[0].split("|"))
        else:
            for value in data.values():
                values.extend(_extract_ths_codes(value))
    return values


def parse_ths_watchlist(path: Path) -> List[str]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取同花顺自选文件：{path}") from exc
    return _valid_codes(_extract_ths_codes(data))


def merge_client_watchlist(existing: Iterable[WatchItem], codes: Iterable[str], source: str) -> List[WatchItem]:
    rows = list(existing)
    indexes = {item.code: index for index, item in enumerate(rows)}
    for code in _valid_codes(codes):
        if code in indexes:
            item = rows[indexes[code]]
            sources = [part for part in item.source.split("+") if part]
            if source not in sources:
                sources.append(source)
            rows[indexes[code]] = WatchItem(item.code, item.name or code, item.tags, "+".join(sources))
        else:
            indexes[code] = len(rows)
            rows.append(WatchItem(code, code, "客户端同步", source))
    return rows


def import_client_watchlist(source: str, client_path: Path, system_path: Path) -> SyncResult:
    if source == "通达信客户端":
        codes = parse_tdx_watchlist(client_path)
    elif source == "同花顺客户端":
        codes = parse_ths_watchlist(client_path)
    else:
        raise ValueError(f"不支持的客户端来源：{source}")
    existing = load_watchlist(system_path)
    merged = merge_client_watchlist(existing, codes, source)
    save_watchlist(merged, system_path)
    return SyncResult(source, len(codes), len(existing), len(merged) - len(existing), len(merged), str(system_path))


def client_process_running(executable_name: str) -> bool:
    if os.name != "nt":
        return False
    result = subprocess.run(
        ["tasklist", "/FI", f"IMAGENAME eq {executable_name}", "/FO", "CSV", "/NH"],
        capture_output=True,
        text=True,
        check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    return executable_name.lower() in result.stdout.lower()


def _tdx_line(code: str) -> str:
    if code.startswith(("60", "68")):
        return "1" + code
    if code.startswith(("00", "30")):
        return "0" + code
    if code.startswith(("43", "83", "87", "88", "92")):
        return "2" + code
    raise ValueError(f"暂不支持导出该证券代码：{code}")


def export_tdx_watchlist(
    codes: Iterable[str],
    tdx_path: Path,
    backup_dir: Path,
    client_closed_confirmed: bool,
) -> dict:
    if not client_closed_confirmed:
        raise ValueError("必须确认通达信客户端已关闭")
    if client_process_running("TdxW.exe"):
        raise RuntimeError("检测到通达信仍在运行，请先退出客户端再导出")
    clean_codes = _valid_codes(codes)
    if not clean_codes:
        raise ValueError("没有可导出的A股代码")
    existing_raw = []
    if tdx_path.exists():
        existing_raw = [line.strip() for line in tdx_path.read_text(encoding="gb18030").splitlines() if line.strip()]
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup = backup_dir / f"tdx_zxg_{datetime.now():%Y%m%d_%H%M%S_%f}.blk"
        shutil.copy2(tdx_path, backup)
    existing_codes = parse_tdx_watchlist(tdx_path) if tdx_path.exists() else []
    merged_codes = _valid_codes([*existing_codes, *clean_codes])
    preserved = [line for line in existing_raw if not (re.fullmatch(r"\d{7}", line) and A_SHARE_CODE.fullmatch(line[-6:]))]
    content = "\r\n".join([*preserved, *[_tdx_line(code) for code in merged_codes]]) + "\r\n"
    tdx_path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".zxg.", suffix=".tmp", dir=str(tdx_path.parent))
    try:
        with os.fdopen(handle, "w", encoding="gb18030", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, tdx_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {
        "exported": len(clean_codes),
        "before": len(existing_codes),
        "after": len(merged_codes),
        "path": str(tdx_path),
    }
