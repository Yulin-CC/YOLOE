"""
# @Author: 算法组 蔡雨霖
# @Date: 2026-07-07
# @Description: 从 YOLO yaml + grounding cache 收集全部类别/短语，生成 train_label_embeddings.pt。
# @Command: python 1-data-process/tools/vocab_generate_label_embedding.py --yolo-yaml data/yolo/0-Objects365v1.yaml --grounding-yaml data/grounding/0-mixed.yaml
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["PYTHONHASHSEED"] = "0"


#----------------------------#
# 路径解析
#----------------------------#
def _resolve_path(path: str) -> Path:
    p = Path(path)
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


#----------------------------#
# 读取文本模型名
#----------------------------#
def _get_text_model_name() -> str:
    from ultralytics.utils import yaml_load

    return yaml_load(str(ROOT / "ultralytics/cfg/default.yaml"))["text_model"]


#----------------------------#
# 从 YOLO yaml 收集检测类别
#----------------------------#
def collect_detection_labels(yaml_path: Path) -> set[str]:
    from ultralytics.utils import yaml_load

    cat_names: set[str] = set()
    data = yaml_load(str(yaml_path), append_filename=True)
    for name in data["names"].values():
        for n in str(name).split("/"):
            n = n.strip()
            if n:
                cat_names.add(n)
    return cat_names


#----------------------------#
# 从 grounding cache 收集短语
#----------------------------#
def collect_grounding_labels(cache_path: Path) -> set[str]:
    cat_names: set[str] = set()
    labels = np.load(str(cache_path), allow_pickle=True)
    for label in labels:
        for text in label["texts"]:
            for t in text:
                t = t.strip()
                if t:
                    cat_names.add(t)
    return cat_names


#----------------------------#
# 从 grounding yaml 解析 cache 路径
#----------------------------#
def grounding_caches_from_yaml(grounding_yaml: Path) -> list[Path]:
    from ultralytics.utils import yaml_load

    grounding_cfg = yaml_load(str(grounding_yaml))
    grounding_data = (grounding_cfg.get("train") or {}).get("grounding_data") or []
    if not grounding_data:
        raise ValueError(f"grounding yaml 缺少 train.grounding_data: {grounding_yaml}")

    caches: list[Path] = []
    for entry in grounding_data:
        json_file = _resolve_path(entry["json_file"])
        cache_path = json_file.with_suffix(".cache")
        if not cache_path.is_file():
            raise FileNotFoundError(
                f"未找到 grounding cache: {cache_path}\n"
                f"请先运行: bash 1-data-process/2-create_grounding.sh "
                f"--json-path {json_file} --img-path {entry['img_path']}"
            )
        caches.append(cache_path)
    return caches


#----------------------------#
# 合并全部标签来源
#----------------------------#
def collect_all_labels(yolo_yaml: str, grounding_yaml: str | None = None) -> tuple[list[str], list[Path]]:
    yolo_path = _resolve_path(yolo_yaml)
    if not yolo_path.is_file():
        raise FileNotFoundError(f"YOLO 数据集 yaml 不存在: {yolo_path}")

    sources: list[Path] = [yolo_path]
    all_names = collect_detection_labels(yolo_path)
    print(f"  Objects365 / YOLO: {len(all_names)} 类 ← {yolo_path}")

    if grounding_yaml:
        grounding_path = _resolve_path(grounding_yaml)
        caches = grounding_caches_from_yaml(grounding_path)
        sources.extend(caches)
        for cache_path in caches:
            names = collect_grounding_labels(cache_path)
            print(f"  Grounding cache: {len(names)} 短语 ← {cache_path.name}")
            all_names |= names

    return sorted(all_names), sources


#----------------------------#
# 批量编码类别文本
#----------------------------#
def encode_texts(text_model_name: str, texts: list[str], batch: int = 512) -> torch.Tensor:
    from ultralytics.nn.text_model import build_text_model
    from ultralytics.utils.torch_utils import smart_inference_mode

    model = build_text_model(text_model_name, device="cuda")

    @smart_inference_mode()
    def _encode():
        tokens = model.tokenize(texts)
        feats = []
        for tok in tqdm(tokens.split(batch), desc="  编码"):
            feats.append(model.encode_text(tok))
        return torch.cat(feats, dim=0).cpu()

    return _encode()


#----------------------------#
# 判断是否需要重新生成
#----------------------------#
def _needs_regenerate(out_path: Path, sources: list[Path]) -> bool:
    if not out_path.is_file():
        return True
    out_mtime = out_path.stat().st_mtime
    return any(src.stat().st_mtime > out_mtime for src in sources if src.is_file())


#----------------------------#
# 生成并保存 train_label_embeddings.pt
#----------------------------#
def generate_label_embedding(
    yolo_yaml: str,
    grounding_yaml: str | None = None,
    output_dir: str | None = None,
    vocab_json_path: str | None = None,
    force: bool = False,
) -> str | None:
    text_model_name = _get_text_model_name()
    categories, sources = collect_all_labels(yolo_yaml, grounding_yaml)

    if output_dir is None:
        output_dir = str(ROOT / "config" / text_model_name)
    os.makedirs(output_dir, exist_ok=True)
    out_path = Path(output_dir) / "train_label_embeddings.pt"

    if not force and not _needs_regenerate(out_path, sources):
        existing = torch.load(out_path, map_location="cpu")
        if len(existing) == len(categories):
            print(f"⏭️  跳过：{out_path} 已是最新（{len(categories)} 类，加 --force 强制重建）")
            return str(out_path)
        print(f"⚠️  已有 .pt 仅 {len(existing)} 类，当前数据需 {len(categories)} 类，重新生成…")

    print(f"📚 词汇表合计：{len(categories)} 个类别/短语")
    print(f"🗺️  text_model: {text_model_name}")

    if vocab_json_path:
        json_path = _resolve_path(vocab_json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(categories, f, indent=2, ensure_ascii=False)
        print(f"📝 词汇表 JSON：{json_path}")

    feats = encode_texts(text_model_name, categories)
    cat_feat_map = {name: feat for name, feat in zip(categories, feats)}
    torch.save(cat_feat_map, out_path)
    print(f"✅ 已保存：{out_path}  ({len(categories)} 类，dim={feats.shape[-1]})")
    return str(out_path)


#----------------------------#
# 参数解析
#----------------------------#
def parse_args():
    parser = argparse.ArgumentParser(
        description="从 YOLO yaml + grounding cache 生成 train_label_embeddings.pt"
    )
    parser.add_argument("--yolo-yaml", type=str, required=True, help="YOLO 数据集 yaml")
    parser.add_argument("--grounding-yaml", type=str, default=None, help="Grounding 数据集 yaml（scratch 必需）")
    parser.add_argument("--vocab-json", type=str, default="config/vocab/train_label_embeddings.json", help="可选：同步写出词汇表 JSON 备份")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录（默认 config/{text_model}/）")
    parser.add_argument("--force", action="store_true", help="强制重建，忽略已有 .pt")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_label_embedding(
        yolo_yaml=args.yolo_yaml,
        grounding_yaml=args.grounding_yaml,
        output_dir=args.output_dir,
        vocab_json_path=args.vocab_json,
        force=args.force,
    )
