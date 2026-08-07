"""
# @Author: 算法组 蔡雨霖
# @Date: 2026-06-15
# @Description: YOLOE 统一推理脚本，支持文本提示词 / 视觉提示词 / Prompt-Free 三种模式
#   超参默认值从 config/default_notrain.yaml predict 段读取，CLI 可覆盖
# @Command: python 0-QuickStart/scripts/predict.py --config config/default_notrain.yaml
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import re
import numpy as np
from PIL import Image
import supervision as sv
from ultralytics import YOLOE
from ultralytics.utils import yaml_load

_DEFAULT_CFG = Path("config/default_notrain.yaml")

# predict yaml 专用字段，不传给 model.predict()
_PREDICT_CFG_SKIP = frozenset({"weights", "mode", "names", "output", "show_mask"})


#----------------------------#
# 配置读取
#----------------------------#
def load_predict_cfg(config_path: str | None = None) -> dict:
    cfg_path = Path(config_path or _DEFAULT_CFG)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    if not cfg_path.is_file():
        return {}
    return yaml_load(str(cfg_path)).get("predict", {}) or {}


def _pick(cli_val, cfg: dict, key: str, default=None):
    if cli_val is not None and cli_val != "":
        return cli_val
    if key in cfg and cfg[key] is not None:
        return cfg[key]
    return default


def _build_predict_kwargs(cfg: dict, **overrides) -> dict:
    kwargs = {k: v for k, v in cfg.items() if k not in _PREDICT_CFG_SKIP and v is not None}
    kwargs.update(overrides)
    return kwargs


#----------------------------#
# 收集输入图片路径（单文件 / 目录）
#----------------------------#
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _list_images(source: str) -> list[Path]:
    src = Path(source)
    if src.is_file():
        return [src]
    if src.is_dir():
        images = sorted(p for p in src.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
        if not images:
            raise FileNotFoundError(f"目录中未找到图片：{src}")
        return images
    raise FileNotFoundError(f"输入不存在：{src}")


#----------------------------#
# 文本提示词推理
#----------------------------#
def predict_text(model, source, names, output, device, predict_cfg: dict):
    model.to(device)
    model.set_classes(names, model.get_text_pe(names))

    images = _list_images(source)
    print(f"📷 待推理图片：{len(images)} 张")
    for i, img_path in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}] {img_path.name}")
        image = Image.open(img_path).convert("RGB")
        kwargs = _build_predict_kwargs(predict_cfg, source=image, verbose=True, save=False)
        results = model.predict(**kwargs)
        _print_detections(results[0])
        _save_annotated(image, results[0], output, str(img_path), predict_cfg)


#----------------------------#
# 视觉提示词推理（示例：用 bus.jpg 内框作为提示，对同图推理）
#----------------------------#
def predict_visual(model, source, predict_cfg: dict):
    from ultralytics.models.yolo.yoloe.predict_vp import YOLOEVPSegPredictor

    visuals = dict(
        bboxes=[np.array([[221.52, 405.8, 344.98, 857.54]])],
        cls=[np.array([0])],
    )
    kwargs = _build_predict_kwargs(
        predict_cfg,
        source=source,
        prompts=visuals,
        predictor=YOLOEVPSegPredictor,
        verbose=True,
    )
    results = model.predict(**kwargs)
    for r in results:
        _print_detections(r)
    print(f"\n✅ 视觉提示词推理完成，结果已保存至 runs/")


#----------------------------#
# 从 .pt 权重路径解析架构 yaml（ultralytics/cfg/models 下）
#----------------------------#
def _resolve_arch_yaml(weights_path: str) -> str:
    """返回带 scale 字母的 yaml 名（如 yoloe-11s-seg.yaml），供 yaml_model_load 识别 n/s/m/l/x。"""
    from ultralytics.utils.checks import check_yaml

    stem = Path(weights_path).stem
    unified = re.sub(r"(\d+)([nslmx])(.+)?$", r"\1\3", stem)
    if not check_yaml(f"{unified}.yaml", hard=False) and not check_yaml(f"{stem}.yaml", hard=False):
        raise FileNotFoundError(
            f"找不到架构 yaml：{stem}.yaml / {unified}.yaml "
            f"（请确认 ultralytics/cfg/models 中存在对应配置）"
        )
    return f"{stem}.yaml"


#----------------------------#
# Prompt-Free 推理（内置大词表，无需指定类别）
#----------------------------#
def predict_promptfree(weights_base, weights_pf, source, device, predict_cfg: dict):
    arch_yaml = _resolve_arch_yaml(weights_base)
    unfused = YOLOE(arch_yaml)
    unfused.load(weights_base)
    unfused.eval()
    unfused.to(device)

    with open(ROOT / "tools/ram_tag_list.txt") as f:
        names = [x.strip() for x in f.readlines()]
    vocab = unfused.get_vocab(names)

    model = YOLOE(weights_pf).to(device)
    model.set_vocab(vocab, names=names)
    model.model.model[-1].is_fused = True
    model.model.model[-1].conf = predict_cfg.get("conf", 0.001)
    model.model.model[-1].max_det = predict_cfg.get("max_det", 1000)

    kwargs = _build_predict_kwargs(predict_cfg, source=source, verbose=True)
    results = model.predict(**kwargs)
    for r in results:
        _print_detections(r)
    print(f"\n✅ Prompt-Free 推理完成，结果已保存至 runs/")


#----------------------------#
# 打印检测结果摘要
#----------------------------#
def _print_detections(result):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        print("  ⚠️  未检测到目标（可尝试降低 conf 阈值）")
        return

    from collections import Counter
    names     = result.names
    class_ids = boxes.cls.int().tolist()
    confs     = boxes.conf.tolist()

    counts = Counter(names[c] for c in class_ids)
    print(f"\n  检测到 {len(class_ids)} 个目标：" + "  ".join(f"{k}×{v}" for k, v in counts.items()))
    for i, (cid, conf) in enumerate(zip(class_ids, confs)):
        print(f"    [{i+1}] {names[cid]:<15s}  conf={conf:.3f}")


#----------------------------#
# 可视化并保存结果
#----------------------------#
def _save_annotated(image, result, output, source, predict_cfg: dict | None = None):
    cfg = predict_cfg or {}
    show_mask   = cfg.get("show_mask", True)
    show_boxes  = cfg.get("show_boxes", True)
    show_labels = cfg.get("show_labels", True)
    show_conf   = cfg.get("show_conf", True)
    line_width  = cfg.get("line_width")

    detections = sv.Detections.from_ultralytics(result)

    resolution_wh = image.size
    thickness  = line_width or sv.calculate_optimal_line_thickness(resolution_wh=resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=resolution_wh)

    labels = []
    if show_labels:
        for cls, conf in zip(detections["class_name"], detections.confidence):
            labels.append(f"{cls} {conf:.2f}" if show_conf else str(cls))

    annotated = image.copy()
    if show_mask:
        annotated = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX, opacity=0.4).annotate(annotated, detections)
    if show_boxes:
        annotated = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX, thickness=thickness).annotate(annotated, detections)
    if labels:
        annotated = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX, text_scale=text_scale, smart_position=True).annotate(annotated, detections, labels)

    out_path = Path(output) / Path(source).name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(str(out_path))
    print(f"✅ 推理完成，结果已保存：{out_path}")


#----------------------------#
# 参数解析
#----------------------------#
def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLOE 推理（CLI > config/default_notrain.yaml predict 段）",
    )
    parser.add_argument("--config",  type=str, default="config/default_notrain.yaml",
                        help="推理配置文件路径")
    parser.add_argument("--mode",    type=str, default=None,
                        choices=["text", "visual", "promptfree"],
                        help="推理模式")
    parser.add_argument("--weights", type=str, default=None,
                        help="预训练权重路径")
    parser.add_argument("--source",  type=str, default=None,
                        help="输入图片或目录")
    parser.add_argument("--names",   type=str, default=None,
                        help="检测类别（text 模式），逗号分隔，类名内可含空格，如 'black car,person'")
    parser.add_argument("--output",  type=str, default=None,
                        help="结果保存目录（text 模式）")
    parser.add_argument("--device",  type=str, default=None,
                        help="推理设备，如 cpu / cuda:0")
    return parser.parse_args()


def _parse_names(names_arg, cfg) -> list:
    """解析类别名：CLI 逗号分隔字符串，或 yaml list/逗号字符串。"""
    if names_arg is not None:
        return [x.strip() for x in str(names_arg).split(",") if x.strip()]
    cfg_names = cfg.get("names", ["person"])
    if isinstance(cfg_names, str):
        return [x.strip() for x in cfg_names.split(",") if x.strip()]
    return list(cfg_names)


if __name__ == "__main__":
    args = parse_args()
    cfg = load_predict_cfg(args.config)

    mode = _pick(args.mode, cfg, "mode", "text")
    weights = _pick(args.weights, cfg, "weights", "weights/yoloe-11s-seg.pt")
    source = _pick(args.source, cfg, "source", "ultralytics/assets/bus.jpg")
    names = _parse_names(args.names, cfg)
    output = _pick(args.output, cfg, "output", "runs/1-predict")
    device = _pick(args.device, cfg, "device", "cuda:0")

    print(f"📄 配置：{args.config}")
    weights_path = Path(weights)
    if not weights_path.is_absolute():
        weights_path = (ROOT / weights_path).resolve()
    else:
        weights_path = weights_path.resolve()
    print(f"   weights={weights_path}")
    print(f"   mode={mode}  source={source}  device={device}")

    if mode == "text":
        model = YOLOE(weights)
        predict_text(model, source, names, output, device, cfg)

    elif mode == "visual":
        model = YOLOE(weights)
        model.to(device)
        predict_visual(model, source, cfg)

    elif mode == "promptfree":
        weights_pf = weights.replace(".pt", "-pf.pt")
        weights_pf_path = Path(weights_pf)
        if not weights_pf_path.is_absolute():
            weights_pf_path = (ROOT / weights_pf_path).resolve()
        else:
            weights_pf_path = weights_pf_path.resolve()
        print(f"   weights_pf={weights_pf_path}")
        if not weights_pf_path.is_file():
            raise FileNotFoundError(f"Prompt-Free 权重不存在: {weights_pf_path}")
        predict_promptfree(weights, str(weights_pf_path), source, device, cfg)
