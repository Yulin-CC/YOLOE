"""
# @Author: 算法组 蔡雨霖
# @Date: 2026-08-26
# @Description: 复核任务落盘：task.json 原子写、路径白名单、历史列表
"""
from __future__ import annotations

import json
import os
import re
import shutil
import threading
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path


TASK_ID_RE = re.compile(r"^[0-9]{8}_[0-9]{6}_[a-f0-9]{8}$")
FILE_NAME_RE = re.compile(r"^[0-9]{4}_[A-Za-z0-9._-]+$")
SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
TZ = timezone(timedelta(hours=8))


def now_iso() -> str:
    return datetime.now(TZ).isoformat(timespec="seconds")


def new_task_id() -> str:
    stamp = datetime.now(TZ).strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{uuid.uuid4().hex[:8]}"


def safe_filename(name: str) -> str:
    base = Path(str(name or "image")).name
    base = SAFE_NAME_RE.sub("_", base).strip("._") or "image.jpg"
    return base[:120]


class TaskStore:
    def __init__(self, data_root: Path):
        self.data_root = data_root.resolve()
        self.data_root.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, threading.Lock] = {}
        self._meta = threading.Lock()

    def _lock(self, task_id: str) -> threading.Lock:
        with self._meta:
            lock = self._locks.get(task_id)
            if lock is None:
                lock = threading.Lock()
                self._locks[task_id] = lock
            return lock

    def task_dir(self, task_id: str) -> Path:
        if not TASK_ID_RE.match(task_id):
            raise ValueError("非法任务 ID")
        raw = self.data_root / task_id
        if raw.is_symlink():
            raise ValueError("拒绝符号链接")
        resolved = raw.resolve()
        if resolved.parent != self.data_root:
            raise ValueError("任务路径越界")
        return resolved

    def files_dir(self, task_id: str) -> Path:
        path = self.task_dir(task_id) / "files"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def resolve_file(self, task_id: str, filename: str) -> Path:
        if not FILE_NAME_RE.match(filename):
            raise ValueError("非法文件名")
        root = self.files_dir(task_id).resolve()
        path = (root / filename).resolve()
        if path.parent != root or path.is_symlink():
            raise ValueError("文件路径越界")
        return path

    def _write_json(self, path: Path, payload: dict):
        tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
        data = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        tmp.write_text(data, encoding="utf-8")
        os.replace(tmp, path)

    def save_task(self, task: dict):
        task_id = task["id"]
        task["updatedAt"] = now_iso()
        folder = self.task_dir(task_id)
        folder.mkdir(parents=True, exist_ok=True)
        with self._lock(task_id):
            self._write_json(folder / "task.json", task)

    def load_task(self, task_id: str) -> dict:
        path = self.task_dir(task_id) / "task.json"
        if not path.is_file():
            raise FileNotFoundError(task_id)
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"任务记录损坏: {exc}") from exc

    def summarize(self, task: dict) -> dict:
        items = task.get("items") or []
        return {
            "id": task.get("id"),
            "createdAt": task.get("createdAt"),
            "updatedAt": task.get("updatedAt"),
            "status": task.get("status"),
            "model": task.get("model"),
            "params": task.get("params"),
            "itemCount": task.get("itemCount", len(items)),
            "doneCount": sum(1 for it in items if it.get("status") in ("success", "failed")),
            "okCount": sum(1 for it in items if it.get("status") == "success"),
        }

    def list_tasks(self) -> list[dict]:
        rows = []
        for child in sorted(self.data_root.iterdir(), reverse=True):
            if not child.is_dir() or child.is_symlink() or not TASK_ID_RE.match(child.name):
                continue
            path = child / "task.json"
            if not path.is_file():
                continue
            try:
                task = json.loads(path.read_text(encoding="utf-8"))
                rows.append(self.summarize(task))
            except (json.JSONDecodeError, OSError):
                print(f"[store] skip damaged task {child.name}", flush=True)
        return rows

    def create_task(self, model: dict, params: dict, items: list[dict]) -> dict:
        task_id = new_task_id()
        stamp = now_iso()
        task = {
            "id": task_id,
            "createdAt": stamp,
            "updatedAt": stamp,
            "status": "running",
            "model": model,
            "params": params,
            "itemCount": len(items),
            "items": items,
        }
        self.save_task(task)
        return task

    def update_item(self, task_id: str, index: int, patch: dict) -> dict:
        with self._lock(task_id):
            task = self.load_task(task_id)
            items = task.get("items") or []
            if index < 0 or index >= len(items):
                raise IndexError(index)
            items[index].update(patch)
            task["items"] = items
            task["updatedAt"] = now_iso()
            folder = self.task_dir(task_id)
            self._write_json(folder / "task.json", task)
            return task

    def finish_task(self, task_id: str, status: str | None = None) -> dict:
        with self._lock(task_id):
            task = self.load_task(task_id)
            items = task.get("items") or []
            if status is None:
                ok = sum(1 for it in items if it.get("status") == "success")
                fail = sum(1 for it in items if it.get("status") == "failed")
                if ok and fail:
                    status = "partial_failed"
                elif ok:
                    status = "completed"
                else:
                    status = "failed"
            task["status"] = status
            task["updatedAt"] = now_iso()
            self._write_json(self.task_dir(task_id) / "task.json", task)
            return task

    def mark_interrupted(self) -> int:
        n = 0
        for row in self.list_tasks():
            if row.get("status") != "running":
                continue
            try:
                self.finish_task(row["id"], "interrupted")
                n += 1
            except (ValueError, FileNotFoundError, OSError):
                continue
        return n

    def delete_task(self, task_id: str) -> dict:
        folder = self.task_dir(task_id)
        if not folder.is_dir():
            raise FileNotFoundError(task_id)
        summary = None
        path = folder / "task.json"
        if path.is_file():
            try:
                summary = self.summarize(json.loads(path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                summary = {"id": task_id}
        shutil.rmtree(folder)
        return summary or {"id": task_id}


def clip01(value: float) -> float:
    if value < 0:
        return 0.0
    if value > 1:
        return 1.0
    return float(value)


def public_item(item: dict) -> dict:
    out = dict(item)
    out.pop("rawResponse", None)
    return out


def public_task(task: dict) -> dict:
    out = dict(task)
    out["items"] = [public_item(it) for it in (task.get("items") or [])]
    return out
