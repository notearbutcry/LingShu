
"""本地 JSON 运行记录（不记录 API 密钥）。默认保存在项目目录 .lingshu_runs/。"""
from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from privacy import redact

DATA_DIR = Path(__file__).resolve().parent / ".lingshu_runs"
CHECKPOINT_DIR = Path(__file__).resolve().parent / ".lingshu_checkpoints"


def _path_for(run_id: str, folder: Path | None = None) -> Path:
    if not re.fullmatch(r"[0-9a-f]{12}", str(run_id)):
        raise ValueError("无效的运行编号")
    return (folder or DATA_DIR) / f"{run_id}.json"


def save_record(result: dict, events: list[dict], approval: dict | None = None,
                folder: Path | None = None) -> Path:
    """原子写入：中断时不破坏已有文件；不接收或保存 api_key 字段。"""
    path = _path_for(result.get("run_id", ""), folder)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = redact({
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "result": dict(result), "events": list(events), "approval_record": approval,
    })
    # 双重排除：若调用方误传密钥键，避免它进入本地记录。
    for key in ("api_key", "DEEPSEEK_API_KEY"):
        payload["result"].pop(key, None)
    handle = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         suffix=".tmp", delete=False) as tmp:
            handle = Path(tmp.name)
            json.dump(payload, tmp, ensure_ascii=False, indent=2)
        os.replace(handle, path)
    finally:
        if handle is not None and handle.exists():
            handle.unlink()
    return path


def load_record(run_id: str, folder: Path | None = None) -> dict:
    path = _path_for(run_id, folder)
    if path.stat().st_size > 20_000_000:
        raise ValueError("历史记录过大")
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or not isinstance(record.get("result"), dict):
        raise ValueError("历史记录格式错误")
    if record["result"].get("run_id") != run_id or not isinstance(record.get("events"), list):
        raise ValueError("历史记录编号或事件不匹配")
    return redact(record)


def list_records(limit: int = 20, folder: Path | None = None) -> list[dict]:
    directory = folder or DATA_DIR
    if not directory.exists():
        return []
    records = []
    for path in sorted(directory.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        if len(records) >= limit:
            break
        try:
            r = load_record(path.stem, directory)
            result = r["result"]
            records.append({"run_id": result["run_id"], "status": result.get("status", ""),
                            "task": result.get("task", "")[:70], "saved_at": r.get("saved_at", "")})
        except (KeyError, ValueError, OSError, TypeError):
            continue
    return records


def save_checkpoint(run_id, snapshot, folder=None):
    """Persist checkpoints for explicit compatible recovery; never restore approval."""
    path = _path_for(run_id, folder or CHECKPOINT_DIR)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         suffix=".tmp", delete=False) as tmp:
            handle = Path(tmp.name)
            json.dump(redact(snapshot), tmp, ensure_ascii=False, indent=2)
        os.replace(handle, path)
    finally:
        if handle is not None and handle.exists():
            handle.unlink()
    return path
