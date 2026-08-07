"""
# @Author: 算法组 蔡雨霖
# @Date: 2026-06-16
# @Description: YOLOE 开集训练封装：
#   词汇表 .pt 由 1-data-process/3-create_vocab_pt.sh 离线生成
#   Step 1 开集训练（与 train_pe.py 共用 linear/full/visual/scratch 分发）
#   训练启动时自动备份配置至 runs/0-train/{project}/config/{args,dataset,vocab}/
# @Command: bash 0-QuickStart/0-train_open.sh
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_SCRIPTS = Path(__file__).parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

os.environ["PYTHONHASHSEED"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from train_pe import (
    set_config_path,
    _load_train_cfg,
    _parse_device,
    train_linear,
    train_full,
    train_visual,
    train_scratch,
)


#----------------------------#
# 路径解析
#----------------------------#
def _resolve_path(path: str) -> Path:
    p = Path(path)
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


#----------------------------#
# 解析 0-YOLO.yaml 索引 / 单份数据集 yaml
#----------------------------#
def _expand_yolo_index(yolo_yaml: str) -> tuple[list[str], list[str]]:
    """展开 YOLO 入口 yaml。

    索引格式（data/0-YOLO.yaml）::
        train: [data/train-yolo_dataset/A.yaml, ...]
        val:   [data/val-yolo_dataset/B.yaml, ...]

    普通数据集 yaml（含 names/nc）则 train=[自身]，val=[]。
    """
    from ultralytics.utils import yaml_load

    index_path = _resolve_path(yolo_yaml)
    if not index_path.is_file():
        raise FileNotFoundError(f"YOLO 数据集 yaml 不存在: {index_path}")

    cfg = yaml_load(str(index_path))
    train = cfg.get("train")

    def _as_yaml_list(value) -> list[str] | None:
        if value is None:
            return []
        items = value if isinstance(value, list) else [value]
        items = [str(x).strip() for x in items if x is not None and str(x).strip()]
        if not items:
            return []
        if all(p.endswith((".yaml", ".yml")) for p in items):
            return items
        return None  # 普通数据集的 train.txt / 图片路径列表

    train_refs = _as_yaml_list(train)
    if train_refs is not None:
        train_yamls = [str(_resolve_path(p)) for p in train_refs]
        val_refs = _as_yaml_list(cfg.get("val")) or []
        val_yamls = [str(_resolve_path(p)) for p in val_refs]
        for p in train_yamls + val_yamls:
            if not Path(p).is_file():
                raise FileNotFoundError(f"索引中的数据集 yaml 不存在: {p}")
        if not train_yamls:
            raise ValueError(f"YOLO 索引缺少 train 子数据集: {index_path}")
        return train_yamls, val_yamls

    return [str(index_path)], []


#----------------------------#
# 合并 yolo 索引 + grounding yaml → scratch 训练 dict
#----------------------------#
def build_scratch_data(yolo_yaml: str, grounding_yaml: str, val_yaml: str | None = None) -> dict:
    from ultralytics.utils import yaml_load

    yolo_paths, index_val_yamls = _expand_yolo_index(yolo_yaml)

    grounding_path = _resolve_path(grounding_yaml)
    if not grounding_path.is_file():
        raise FileNotFoundError(f"Grounding 数据集 yaml 不存在: {grounding_path}")

    grounding_cfg = yaml_load(str(grounding_path))
    grounding_data = (grounding_cfg.get("train") or {}).get("grounding_data")
    if not grounding_data:
        raise ValueError(f"grounding yaml 缺少 train.grounding_data: {grounding_path}")

    # DDP 子进程通过 temp file 反序列化 data dict，路径须为绝对字符串
    grounding_data = [
        {
            "img_path": str(_resolve_path(entry["img_path"])),
            "json_file": str(_resolve_path(entry["json_file"])),
        }
        for entry in grounding_data
    ]

    # val 优先：0-YOLO.yaml 索引 > grounding.yaml > config.val_data
    val_cfg = grounding_cfg.get("val") or {}
    if index_val_yamls:
        val_yolo = index_val_yamls
    elif val_cfg.get("yolo_data"):
        val_yolo = val_cfg["yolo_data"]
    elif val_yaml:
        val_yolo = [str(_resolve_path(val_yaml))]
    else:
        raise ValueError("未找到验证集：请在 data/0-YOLO.yaml 的 val 中指定，或配置 val_data")

    if len(val_yolo) > 1:
        print(f"  ⚠️  scratch 目前仅支持 1 个 val 集，使用: {val_yolo[0]}（忽略 {val_yolo[1:]}）")
        val_yolo = val_yolo[:1]

    return {
        "train": {
            "yolo_data": yolo_paths,
            "grounding_data": grounding_data,
        },
        "val": {"yolo_data": val_yolo if isinstance(val_yolo, list) else [val_yolo]},
    }


#----------------------------#
# 参数解析
#----------------------------#
def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLOE 开集训练（超参默认值从 --config yaml 读取）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config",  type=str, default="config/train_open.yaml",
                        help="训练配置文件路径")
    parser.add_argument("--mode",    type=str, default=None,
                        choices=["linear", "full", "visual", "scratch"],
                        help="训练模式（不填则从 --config train.defaults.mode 读取）")
    parser.add_argument("--model",   type=str, default="weights/yoloe-11s-seg.pt")
    parser.add_argument("--data",    type=str, default="data/0-YOLO.yaml",
                        help="YOLO 索引 yaml（train/val 指向子数据集）或单份数据集 yaml")
    parser.add_argument("--grounding-data", type=str, default="data/grounding/0-mixed.yaml",
                        help="Grounding 数据集 yaml（scratch 模式，含 train.grounding_data）")
    parser.add_argument("--project", type=str, default="YOLOE-open-exp01")
    parser.add_argument("--epochs",  type=int, default=None)
    parser.add_argument("--batch",   type=int, default=None)
    parser.add_argument("--device",  type=str, default="0")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.device = _parse_device(args.device)

    set_config_path(args.config)
    if args.mode is None:
        args.mode = _load_train_cfg().get("mode", "linear")

    #--------------------------------------
    # 开集训练（Ultralytics 多卡 device 自动 spawn DDP）
    # 训练启动时 on_train_start 自动备份 config/{args,dataset,vocab}/
    #--------------------------------------
    print("\n════════════════════════════════════════")
    print(f" 开集训练（mode={args.mode}）")
    print("════════════════════════════════════════")
    cfg = _load_train_cfg()
    args.backup_vocab = args.mode == "scratch"
    train_yamls, val_yamls = _expand_yolo_index(args.data)
    if args.mode == "scratch":
        fallback_val = cfg.get("val_data", "ultralytics/cfg/datasets/lvis.yaml")
        args.scratch_data = build_scratch_data(args.data, args.grounding_data, fallback_val)
        args.val_data = args.scratch_data["val"]["yolo_data"][0]
        # 备份：索引 + 展开后的 train/val 子 yaml
        args.backup_dataset_yamls = [str(_resolve_path(args.data))] + \
            args.scratch_data["train"]["yolo_data"] + args.scratch_data["val"]["yolo_data"]
        n_yolo = len(args.scratch_data["train"]["yolo_data"])
        n_gd = len(args.scratch_data["train"]["grounding_data"])
        print(f"  scratch data: index={args.data} | yolo×{n_yolo} | grounding×{n_gd} | val={args.val_data}")
        for p in args.scratch_data["train"]["yolo_data"]:
            print(f"    · train {Path(p).name}")
    else:
        if len(train_yamls) > 1:
            print(f"  ⚠️  {args.mode} 模式仅使用第一份 YOLO yaml，忽略: {[Path(p).name for p in train_yamls[1:]]}")
        args.data = train_yamls[0]
        args.backup_dataset_yamls = [args.data]
    dispatch = {
        "linear":  train_linear,
        "full":    train_full,
        "visual":  train_visual,
        "scratch": train_scratch,
    }
    dispatch[args.mode](args)
    print(f"\n🎉 完成！权重与配置均已保存至 runs/0-train/{args.project}/")