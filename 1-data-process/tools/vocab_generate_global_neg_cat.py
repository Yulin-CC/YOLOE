"""
# @Author: 算法组 蔡雨霖
# @Date: 2026-07-07
# @Description: 从 grounding cache 统计高频短语，生成负样本词表 JSON 与 global_grounding_neg_embeddings.pt。
# @Command: python 1-data-process/tools/vocab_generate_global_neg_cat.py --grounding-yaml data/grounding/0-mixed.yaml
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["PYTHONHASHSEED"] = "0"

from vocab_generate_label_embedding import (
    _get_text_model_name,
    _needs_regenerate,
    _resolve_path,
    encode_texts,
    grounding_caches_from_yaml,
)


#----------------------------#
# 统计 grounding cache 中短语频次
#----------------------------#
def collect_grounding_freq(cache_paths: list[Path]) -> dict[str, int]:
    freq: dict[str, int] = defaultdict(int)
    for cache_path in cache_paths:
        labels = np.load(str(cache_path), allow_pickle=True)
        for label in labels:
            for text in label["texts"]:
                for t in text:
                    t = t.strip()
                    if t:
                        freq[t] += 1
        print(f"  统计频次 ← {cache_path.name}（累计 {len(freq)} 个短语）")
    return freq


#----------------------------#
# 生成负样本词表与嵌入
#----------------------------#
def generate_global_neg_cat(
    grounding_yaml: str,
    min_freq: int = 100,
    neg_vocab_path: str | None = None,
    output_dir: str | None = None,
    force: bool = False,
) -> str | None:
    grounding_path = _resolve_path(grounding_yaml)
    caches = grounding_caches_from_yaml(grounding_path)
    sources = list(caches)

    text_model_name = _get_text_model_name()
    if output_dir is None:
        output_dir = str(ROOT / "config" / text_model_name)
    os.makedirs(output_dir, exist_ok=True)

    if neg_vocab_path is None:
        neg_vocab_path = str(ROOT / "config/vocab/global_grounding_neg_cat.json")
    neg_json = _resolve_path(neg_vocab_path)
    neg_json.parent.mkdir(parents=True, exist_ok=True)
    pt_path = Path(output_dir) / "global_grounding_neg_embeddings.pt"

    if not force and neg_json.is_file() and not _needs_regenerate(pt_path, sources):
        existing = torch.load(pt_path, map_location="cpu")
        with open(neg_json, encoding="utf-8") as f:
            saved_cats = json.load(f)
        if len(existing) == len(saved_cats):
            print(f"⏭️  跳过：{pt_path} 已是最新（{len(saved_cats)} 类，加 --force 强制重建）")
            return str(pt_path)
        print(f"⚠️  已有 .pt 仅 {len(existing)} 类，词表需 {len(saved_cats)} 类，重新生成…")

    freq = collect_grounding_freq(caches)
    global_neg_cat = sorted(k for k, v in freq.items() if v >= min_freq)
    print(f"📚 负样本词表：{len(global_neg_cat)} 个短语（min_freq={min_freq}）")
    print(f"🗺️  text_model: {text_model_name}")

    with open(neg_json, "w", encoding="utf-8") as f:
        json.dump(global_neg_cat, f, indent=2, ensure_ascii=False)
    print(f"📝 负样本 JSON：{neg_json}")

    feats = encode_texts(text_model_name, global_neg_cat)
    torch.save(feats, pt_path)
    print(f"✅ 已保存：{pt_path}  ({len(global_neg_cat)} 类，dim={feats.shape[-1]})")
    return str(pt_path)


#----------------------------#
# 参数解析
#----------------------------#
def parse_args():
    parser = argparse.ArgumentParser(
        description="从 grounding cache 生成 global_grounding_neg_cat.json 与 global_grounding_neg_embeddings.pt"
    )
    parser.add_argument("--grounding-yaml", type=str, required=True, help="Grounding 数据集 yaml")
    parser.add_argument("--min-freq", type=int, default=100, help="短语最小出现次数（默认 100）")
    parser.add_argument("--neg-vocab", type=str, default="config/vocab/global_grounding_neg_cat.json",
                        help="负样本词表 JSON 输出路径")
    parser.add_argument("--output-dir", type=str, default=None, help="嵌入输出目录（默认 config/{text_model}/）")
    parser.add_argument("--force", action="store_true", help="强制重建，忽略已有 .pt")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_global_neg_cat(
        grounding_yaml=args.grounding_yaml,
        min_freq=args.min_freq,
        neg_vocab_path=args.neg_vocab,
        output_dir=args.output_dir,
        force=args.force,
    )
