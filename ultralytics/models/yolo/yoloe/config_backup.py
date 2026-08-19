# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Persist train snapshots into runs/.../{data,config}/ (DDP-safe via env payload)."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from ultralytics.utils import LOGGER, RANK

ENV_KEY = "YOLOE_TRAIN_CONFIG_BACKUP"


def set_config_backup_plan(copy_items: list[dict]):
    """Store absolute copy plan for DDP children (fresh interpreter keeps os.environ)."""
    os.environ[ENV_KEY] = json.dumps({"items": copy_items})


def apply_config_backup(save_dir) -> bool:
    """Copy planned files into ``<save_dir>/{data,config}/`` (rel paths from repo root).

    Returns True if a plan existed and rank-0 applied it.
    """
    raw = os.environ.get(ENV_KEY)
    if not raw:
        return False
    if RANK not in {-1, 0}:
        return False
    if not save_dir:
        return False

    try:
        plan = json.loads(raw)
    except json.JSONDecodeError:
        LOGGER.warning(f"⚠️  invalid {ENV_KEY}, skip config backup")
        return False

    items = plan.get("items") or []
    if not items:
        return False

    root = Path(save_dir)
    copied = 0
    for item in items:
        src = Path(item["src"])
        dst = root / item["rel"]
        if not src.is_file():
            LOGGER.warning(f"⚠️  snapshot missing: {src}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    if copied:
        LOGGER.info(f"📋 snapshot {copied} files → {root}/{{data,config}}/")
    return copied > 0


def register_trainer_config_backup(trainer) -> None:
    """Attach on_train_start hook (safe to call from Trainer.__init__, including DDP)."""

    def _on_train_start(t):
        apply_config_backup(t.save_dir)

    trainer.add_callback("on_train_start", _on_train_start)
