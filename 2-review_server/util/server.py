"""
# @Author: 算法组 蔡雨霖
# @Date: 2026-08-26
# @Description: YOLOE 复核推理服务入口（静态页 + JSON API + 后台推理队列）
# @Command: python 2-review_server/util/server.py --port 8088 --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


UTIL_DIR = Path(__file__).resolve().parent
SERVER_DIR = UTIL_DIR.parent
if str(UTIL_DIR) not in sys.path:
    sys.path.insert(0, str(UTIL_DIR))

from adapter import find_project_root
from app import make_server


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOE review server")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--config", default=str(UTIL_DIR / "config.json"))
    parser.add_argument("--data-dir", default=str(SERVER_DIR / "data"))
    return parser.parse_args()


def load_config(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"找不到配置: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = Path.cwd() / cfg_path
    cfg = load_config(cfg_path)
    project_root = find_project_root(UTIL_DIR)
    host = args.host or str(cfg.get("host") or "0.0.0.0")
    port = int(args.port or cfg.get("port") or 8088)
    device = args.device or str(cfg.get("device") or "cuda:0")
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = Path.cwd() / data_dir
    static_dir = UTIL_DIR / "static"

    mobileclip = cfg.get("mobileclip") or "weights/mobileclip_blt.pt"
    clip_path = Path(mobileclip)
    if not clip_path.is_absolute():
        clip_path = project_root / clip_path
    os.environ.setdefault("MOBILECLIP_PATH", str(clip_path))

    print(f"project {project_root}", flush=True)
    print(f"listen  http://{host}:{port}", flush=True)
    print(f"device  {device}", flush=True)
    server = make_server(cfg, project_root, data_dir, static_dir, host, port, device)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped", flush=True)
        server.shutdown()
