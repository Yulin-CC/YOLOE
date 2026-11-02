"""
# @Author: 算法组 蔡雨霖
# @Date: 2026-08-26
# @Description: YOLOE 开集文本推理 adapter：校验参数、本地 predict、规范化框/多边形
"""
from __future__ import annotations

import math
import re
import sys
import time
from pathlib import Path

from store import clip01


def find_project_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "0-QuickStart").is_dir() and (parent / "ultralytics").is_dir():
            return parent
        if (parent / "ultralytics").is_dir() and (parent / "config").is_dir():
            return parent
    return start.parents[2]


def parse_names(raw) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        parts = [str(x) for x in raw]
    else:
        parts = re.split(r"[,;，、/]+", str(raw))
    return [x.strip() for x in parts if x.strip()]


def _finite(value) -> bool:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v)


class YoloeAdapter:
    def __init__(self, project_root: Path, device: str, predict_cfg: dict | None = None):
        self.project_root = project_root
        self.device = device
        self.predict_cfg = dict(predict_cfg or {})
        self._model = None
        self._weights = None
        self._names: tuple[str, ...] | None = None

    def validate_params(self, params: dict) -> dict:
        names = parse_names(params.get("names"))
        if not names:
            raise ValueError("提示词不能为空")
        if len(names) > 64:
            raise ValueError("提示词过多（最多 64 个）")
        try:
            conf = float(params.get("conf", 0.25))
        except (TypeError, ValueError) as exc:
            raise ValueError("conf 必须是数字") from exc
        if not 0 < conf <= 1:
            raise ValueError("conf 需在 (0, 1] 内")
        return {"names": names, "conf": conf}

    def _discard(self):
        if self._model is None:
            return
        try:
            del self._model
        except Exception:
            pass
        self._model = None
        self._weights = None
        self._names = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _load_weights(self, weights: Path):
        if str(self.project_root) not in sys.path:
            sys.path.insert(0, str(self.project_root))
        from ultralytics import YOLOE

        if not weights.is_file():
            raise FileNotFoundError(f"权重不存在: {weights}")
        model = YOLOE(str(weights))
        model.to(self.device)
        model.predictor = None
        self._model = model
        self._weights = str(weights.resolve())
        self._names = None
        print(f"[review] load {weights.name}", flush=True)

    def _bind_prompt(self, weights: Path, names: list[str]):
        #-------------#
        # 换词必须重载权重：第一次 predict 会 fuse TPE，旧 head 不能改类别数
        #-------------#
        name_key = tuple(names)
        same = (
            self._model is not None
            and self._weights == str(weights.resolve())
            and self._names == name_key
        )
        if not same:
            self._discard()
            self._load_weights(weights)
            self._model.set_classes(names, self._model.get_text_pe(names))
            self._names = name_key
            print(f"[review] TPE  names={list(names)}  n={len(names)}", flush=True)
        self._model.predictor = None

    def infer(self, image_path: Path, weights: Path, params: dict) -> dict:
        checked = self.validate_params(params)
        self._bind_prompt(weights, checked["names"])

        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        if width < 1 or height < 1:
            raise ValueError("图片尺寸无效")

        scripts = self.project_root / "0-QuickStart" / "scripts"
        if str(scripts) not in sys.path:
            sys.path.insert(0, str(scripts))
        import predict as pred

        cfg = dict(self.predict_cfg)
        cfg["conf"] = checked["conf"]
        kwargs = pred._build_predict_kwargs(
            cfg, source=image, verbose=False,
            save=False, save_crop=False, save_txt=False, save_frames=False,
            conf=checked["conf"], device=self.device,
        )

        t0 = time.monotonic()
        result = self._model.predict(**kwargs)[0]
        cost_ms = int((time.monotonic() - t0) * 1000)
        n = 0 if result.boxes is None else len(result.boxes)
        print(
            f"[review] {image_path.name}  names={checked['names']}  conf={checked['conf']}  dets={n}  {cost_ms}ms",
            flush=True,
        )
        return {
            "result": self.normalize(result, width, height),
            "costMs": cost_ms,
            "width": width,
            "height": height,
        }

    def normalize(self, result, width: int, height: int) -> dict:
        #-------------#
        # 像素 → [0,1]
        #-------------#
        boxes = result.boxes
        regions = []
        if boxes is None or len(boxes) == 0:
            return {"type": "boxes", "width": width, "height": height, "regions": []}

        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy()
        class_ids = boxes.cls.int().detach().cpu().tolist()
        names = result.names
        mask_xy = None
        if getattr(result, "masks", None) is not None and getattr(result.masks, "xy", None) is not None:
            mask_xy = result.masks.xy

        for i, (box, score, cid) in enumerate(zip(xyxy, confs, class_ids)):
            if not all(_finite(v) for v in list(box) + [score]):
                continue
            x1, y1, x2, y2 = (float(v) for v in box)
            label = names[cid] if isinstance(names, dict) else names[int(cid)]
            region = {
                "label": str(label),
                "score": round(float(score), 4),
                "box": {
                    "x": clip01(x1 / width),
                    "y": clip01(y1 / height),
                    "w": clip01((x2 - x1) / width),
                    "h": clip01((y2 - y1) / height),
                },
            }
            if mask_xy is not None and i < len(mask_xy):
                pts = mask_xy[i]
                if pts is not None and len(pts) >= 3:
                    points = []
                    ok = True
                    for pt in pts:
                        if len(pt) < 2 or not _finite(pt[0]) or not _finite(pt[1]):
                            ok = False
                            break
                        points.append({"x": clip01(float(pt[0]) / width), "y": clip01(float(pt[1]) / height)})
                    if ok:
                        region["points"] = points
            regions.append(region)
        return {"type": "boxes", "width": width, "height": height, "regions": regions}
