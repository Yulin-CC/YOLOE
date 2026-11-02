"""
# @Author: 算法组 蔡雨霖
# @Date: 2026-08-26
# @Description: 复核服务 HTTP：静态页、任务 API、本地图片；推理在后台队列执行
"""
from __future__ import annotations

import json
import mimetypes
import queue
import re
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from adapter import YoloeAdapter
from store import FILE_NAME_RE, TaskStore, public_task, safe_filename


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _ok(data):
    return {"ok": True, "data": data}


def _err(message: str):
    return {"ok": False, "error": {"message": message}}


def looks_like_image(name: str, blob: bytes) -> str | None:
    ext = Path(name).suffix.lower()
    if ext not in IMAGE_EXTS:
        ext = ""
    if blob.startswith(b"\xff\xd8\xff"):
        return ext if ext in {".jpg", ".jpeg"} else ".jpg"
    if blob.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if blob.startswith(b"BM"):
        return ".bmp"
    if blob.startswith(b"RIFF") and blob[8:12] == b"WEBP":
        return ".webp"
    if blob[:4] in (b"II*\x00", b"MM\x00*"):
        return ext if ext in {".tif", ".tiff"} else ".tif"
    return ext or None


def parse_multipart(content_type: str, body: bytes) -> tuple[dict[str, str], list[tuple[str, bytes]]]:
    match = re.search(r"boundary=([^;]+)", content_type or "", re.I)
    if not match:
        raise ValueError("缺少 multipart boundary")
    boundary = match.group(1).strip().strip('"')
    delim = b"--" + boundary.encode("ascii", "ignore")
    fields: dict[str, str] = {}
    files: list[tuple[str, bytes]] = []
    for raw in body.split(delim):
        chunk = raw
        if chunk.startswith(b"\r\n"):
            chunk = chunk[2:]
        if not chunk or chunk.startswith(b"--"):
            continue
        header_blob, sep, data = chunk.partition(b"\r\n\r\n")
        if not sep:
            continue
        if data.endswith(b"\r\n"):
            data = data[:-2]
        header_text = header_blob.decode("utf-8", "replace")
        name_m = re.search(r'name="([^"]*)"', header_text, re.I)
        if not name_m:
            continue
        name = name_m.group(1)
        star = re.search(r"filename\*=(?:UTF-8|utf-8)''([^;\r\n]+)", header_text)
        plain = re.search(r'filename="([^"]*)"', header_text)
        if star or plain is not None:
            filename = unquote(star.group(1)) if star else (plain.group(1) if plain else "")
            files.append((filename or "image.jpg", data))
        else:
            fields[name] = data.decode("utf-8", "replace").strip()
    return fields, files


class ReviewApp:
    def __init__(self, cfg: dict, project_root: Path, data_dir: Path, static_dir: Path, device: str):
        self.cfg = cfg
        self.project_root = project_root
        self.static_dir = static_dir.resolve()
        self.store = TaskStore(data_dir)
        self.device = device
        self.max_bytes = int(cfg.get("max_request_bytes") or 64 * 1024 * 1024)
        self.max_files = int(cfg.get("max_files_per_task") or 50)
        self.models = self._load_models(cfg.get("models") or [])
        predict_cfg = self._load_predict_cfg(cfg.get("predict_config") or "")
        self.adapter = YoloeAdapter(project_root, device, predict_cfg)
        self.jobs: queue.Queue[str] = queue.Queue()
        self.worker = threading.Thread(target=self._worker_loop, name="yoloe-review-worker", daemon=True)
        n = self.store.mark_interrupted()
        if n:
            print(f"[review] marked {n} leftover running task(s) interrupted", flush=True)
        self.worker.start()

    def _load_predict_cfg(self, rel: str) -> dict:
        if not rel:
            return {}
        path = Path(rel)
        if not path.is_absolute():
            path = self.project_root / path
        if not path.is_file():
            return {}
        try:
            import yaml
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            return data.get("predict") or {}
        except Exception:
            return {}

    def _load_models(self, rows: list) -> list[dict]:
        models = []
        for row in rows:
            mid = str(row.get("id") or "").strip()
            weights = str(row.get("weights") or "").strip()
            if not mid or not weights:
                continue
            path = Path(weights)
            if not path.is_absolute():
                path = (self.project_root / path).resolve()
            else:
                path = path.resolve()
            item = {
                "id": mid,
                "name": str(row.get("name") or mid),
                "weights": str(path),
                "available": path.is_file(),
            }
            models.append(item)
            if not item["available"]:
                print(f"[review] model missing: {mid} -> {path}", flush=True)
        return models

    def public_config(self) -> dict:
        defaults = dict(self.cfg.get("defaults") or {})
        return {
            "device": self.device,
            "maxUploadMB": round(self.max_bytes / (1024 * 1024), 1),
            "maxFilesPerTask": self.max_files,
            "defaults": {
                "conf": float(defaults.get("conf", 0.25)),
                "names": str(defaults.get("names") or "person, car"),
            },
            "models": [
                {"id": m["id"], "name": m["name"], "available": m["available"]}
                for m in self.models
            ],
        }

    def model_by_id(self, model_id: str) -> dict:
        for row in self.models:
            if row["id"] == model_id:
                return row
        raise ValueError(f"未知模型: {model_id}")

    def enqueue(self, task_id: str):
        self.jobs.put(task_id)

    def _worker_loop(self):
        while True:
            task_id = self.jobs.get()
            try:
                self._run_task(task_id)
            except Exception as exc:
                print(f"[review] task {task_id} worker error: {exc}", flush=True)
                traceback.print_exc()
                try:
                    self.store.finish_task(task_id, "failed")
                except Exception:
                    pass
            finally:
                self.jobs.task_done()

    def _run_task(self, task_id: str):
        task = self.store.load_task(task_id)
        model = self.model_by_id((task.get("model") or {}).get("id", ""))
        weights = Path(model["weights"])
        params = task.get("params") or {}
        items = task.get("items") or []
        for item in items:
            if item.get("status") not in ("queued", "running"):
                continue
            index = int(item["index"])
            self.store.update_item(task_id, index, {"status": "running", "error": None})
            try:
                filename = Path(item["inputs"][0]["url"]).name
                image_path = self.store.resolve_file(task_id, filename)
                t0 = time.monotonic()
                out = self.adapter.infer(image_path, weights, params)
                self.store.update_item(task_id, index, {
                    "status": "success",
                    "result": out["result"],
                    "error": None,
                    "costMs": out["costMs"],
                    "elapsedMs": int((time.monotonic() - t0) * 1000),
                })
            except Exception as exc:
                print(f"[review] item {task_id}#{index} failed: {exc}", flush=True)
                self.store.update_item(task_id, index, {
                    "status": "failed",
                    "result": None,
                    "error": {"message": str(exc)},
                })
        self.store.finish_task(task_id)


def make_handler(app: ReviewApp):
    class ReviewHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            print(f"[http] {self.address_string()} {fmt % args}", flush=True)

        def _send(self, code: int, payload: dict | bytes, content_type: str = "application/json; charset=utf-8"):
            if isinstance(payload, dict):
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            else:
                body = payload
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, payload: dict):
            self._send(code, payload)

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                return {}
            if length > app.max_bytes:
                got = length / (1024 * 1024)
                lim = app.max_bytes / (1024 * 1024)
                raise ValueError(f"请求过大：收到 {got:.1f} MB，上限 {lim:.0f} MB，请少选几张")
            raw = self.rfile.read(length)
            if not raw:
                return {}
            return json.loads(raw.decode("utf-8"))

        def do_GET(self):
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            if path in ("/", "/index.html"):
                return self._static("index.html")
            if path.startswith("/static/"):
                return self._static(path[len("/static/"):])
            if path == "/api/config":
                return self._json(200, _ok(app.public_config()))
            if path == "/api/health":
                return self._json(200, _ok({"status": "ok"}))
            if path == "/api/tasks":
                return self._json(200, _ok(app.store.list_tasks()))
            if path.startswith("/api/tasks/"):
                task_id = path[len("/api/tasks/"):].strip("/")
                try:
                    task = app.store.load_task(task_id)
                    return self._json(200, _ok(public_task(task)))
                except FileNotFoundError:
                    return self._json(404, _err("任务不存在"))
                except ValueError as exc:
                    return self._json(400, _err(str(exc)))
            if path.startswith("/data/"):
                return self._data_file(path[len("/data/"):])
            return self._json(404, _err("未找到"))

        def do_DELETE(self):
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            if not path.startswith("/api/tasks/"):
                return self._json(404, _err("未找到"))
            task_id = path[len("/api/tasks/"):].strip("/")
            try:
                summary = app.store.delete_task(task_id)
                return self._json(200, _ok(summary))
            except FileNotFoundError:
                return self._json(404, _err("任务不存在"))
            except ValueError as exc:
                return self._json(400, _err(str(exc)))

        def do_POST(self):
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            if path != "/api/tasks":
                return self._json(404, _err("未找到"))
            try:
                task = self._create_task()
                app.enqueue(task["id"])
                return self._json(202, _ok(app.store.summarize(task)))
            except ValueError as exc:
                print(f"[http] POST /api/tasks 400: {exc}", flush=True)
                return self._json(400, _err(str(exc)))
            except Exception as exc:
                traceback.print_exc()
                return self._json(500, _err(str(exc)))

        def _create_task(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                raise ValueError("空请求")
            if length > app.max_bytes:
                got = length / (1024 * 1024)
                lim = app.max_bytes / (1024 * 1024)
                raise ValueError(f"请求过大：收到 {got:.1f} MB，上限 {lim:.0f} MB，请少选几张")
            ctype = self.headers.get("Content-Type") or ""
            if "multipart/form-data" not in ctype.lower():
                raise ValueError("请使用 multipart 上传图片")
            body = self.rfile.read(length)
            fields, files = parse_multipart(ctype, body)
            model_id = (fields.get("model") or "").strip()
            names = (fields.get("names") or "").strip()
            conf_raw = (fields.get("conf") or "").strip()
            if not names:
                names = str((app.cfg.get("defaults") or {}).get("names") or "person, car")
            model = app.model_by_id(model_id)
            if not model["available"]:
                raise ValueError(f"模型权重不存在: {model['id']}，请换 official 或 demo-v0~v5")
            params = app.adapter.validate_params({
                "names": names,
                "conf": conf_raw if conf_raw else app.cfg.get("defaults", {}).get("conf", 0.25),
            })
            if not files:
                raise ValueError("请至少上传一张图片")
            if len(files) > app.max_files:
                raise ValueError(f"单次最多 {app.max_files} 张")

            from PIL import Image
            import io

            items = []
            blobs = []
            for i, (orig, blob) in enumerate(files):
                if not blob:
                    raise ValueError(f"空文件: {orig}")
                ext = looks_like_image(orig, blob)
                try:
                    im = Image.open(io.BytesIO(blob))
                    fmt = (im.format or "").upper()
                    im.verify()
                    fmt_ext = {
                        "JPEG": ".jpg", "PNG": ".png", "BMP": ".bmp",
                        "WEBP": ".webp", "TIFF": ".tif",
                    }.get(fmt)
                    ext = ext or fmt_ext or Path(orig).suffix.lower() or ".jpg"
                except Exception as exc:
                    raise ValueError(f"无法解码图片: {orig}") from exc
                if ext not in IMAGE_EXTS:
                    raise ValueError(f"不支持的文件: {orig}")
                filename = f"{i:04d}_{safe_filename(orig)}"
                if not FILE_NAME_RE.match(filename):
                    filename = f"{i:04d}_image{ext}"
                blobs.append((filename, blob))
                items.append({
                    "index": i,
                    "name": Path(orig).name,
                    "status": "queued",
                    "inputs": [{"role": "image", "url": f"/data/PLACEHOLDER/files/{filename}"}],
                    "result": None,
                    "error": None,
                    "costMs": None,
                })
            task = app.store.create_task(
                model={"id": model["id"], "name": model["name"]},
                params={"names": params["names"], "conf": params["conf"]},
                items=items,
            )
            files_dir = app.store.files_dir(task["id"])
            for filename, blob in blobs:
                (files_dir / filename).write_bytes(blob)
            for item in task["items"]:
                filename = Path(item["inputs"][0]["url"]).name
                item["inputs"][0]["url"] = f"/data/{task['id']}/files/{filename}"
            app.store.save_task(task)
            return task

        def _static(self, rel: str):
            rel = rel.lstrip("/")
            if not rel or ".." in rel.split("/"):
                return self._json(400, _err("非法路径"))
            path = (app.static_dir / rel).resolve()
            if app.static_dir not in path.parents and path != app.static_dir:
                return self._json(400, _err("路径越界"))
            if not path.is_file():
                return self._json(404, _err("静态文件不存在"))
            mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
            self._send(200, path.read_bytes(), mime)

        def _data_file(self, rel: str):
            parts = [p for p in rel.split("/") if p]
            if len(parts) != 3 or parts[1] != "files":
                return self._json(400, _err("非法数据路径"))
            task_id, _, filename = parts
            try:
                path = app.store.resolve_file(task_id, filename)
            except ValueError as exc:
                return self._json(400, _err(str(exc)))
            if not path.is_file():
                return self._json(404, _err("文件不存在"))
            mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
            self._send(200, path.read_bytes(), mime)

    return ReviewHandler


def make_server(cfg: dict, project_root: Path, data_dir: Path, static_dir: Path,
                host: str, port: int, device: str) -> ThreadingHTTPServer:
    app = ReviewApp(cfg, project_root, data_dir, static_dir, device)
    handler = make_handler(app)
    server = ThreadingHTTPServer((host, port), handler)
    print(f"[review] models: {', '.join(m['id'] + ('*' if m['available'] else '') for m in app.models) or '(none)'}", flush=True)
    print("[review] * = weight file found", flush=True)
    return server
