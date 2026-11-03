from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(data: dict) -> bytes:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def append_backtest_audit(result, input_path: Path, output_paths: Iterable[str], audit_path: Path) -> dict:
    previous_hash = "0" * 64
    if audit_path.exists():
        lines = [line for line in audit_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if lines:
            previous_hash = json.loads(lines[-1])["record_hash"]
    record = {
        "schema": 1,
        "run_id": datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input": {"path": str(input_path), "sha256": file_sha256(input_path)},
        "config": result.config,
        "result": {
            "final_equity": result.final_equity,
            "total_return_pct": result.total_return_pct,
            "max_drawdown_pct": result.max_drawdown_pct,
            "trade_count": result.trade_count,
            "sharpe_ratio": result.sharpe_ratio,
            "halted": result.halted,
        },
        "outputs": [
            {"path": str(path), "sha256": file_sha256(Path(path))}
            for path in output_paths
            if Path(path).exists()
        ],
        "previous_hash": previous_hash,
    }
    record["record_hash"] = hashlib.sha256(_canonical(record)).hexdigest()
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return record


def verify_audit_chain(audit_path: Path) -> tuple[bool, int, str]:
    previous_hash = "0" * 64
    count = 0
    if not audit_path.exists():
        return True, 0, "审计链尚无记录"
    for count, line in enumerate(audit_path.read_text(encoding="utf-8").splitlines(), 1):
        record = json.loads(line)
        outputs = list(record.get("outputs", []))
        stored_hash = record.pop("record_hash", "")
        expected = hashlib.sha256(_canonical(record)).hexdigest()
        if record.get("previous_hash") != previous_hash or stored_hash != expected:
            return False, count, f"第 {count} 条记录校验失败"
        for output in outputs:
            path = Path(output.get("path", ""))
            if not path.exists() or file_sha256(path) != output.get("sha256"):
                return False, count, f"第 {count} 条记录的结果文件缺失或被修改：{path}"
        previous_hash = stored_hash
    return True, count, f"审计链完整，共 {count} 条记录"
