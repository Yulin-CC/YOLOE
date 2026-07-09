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
# 合并 yolo yaml + grounding yaml → scratch 训练 dict
#----------------------------#
def build_scratch_data(yolo_yaml: str, grounding_yaml: str, val_yaml: str) -> dict:
    from ultralytics.utils import yaml_load

    yolo_path = _resolve_path(yolo_yaml)
    grounding_path = _resolve_path(grounding_yaml)
    if not yolo_path.is_file():
        raise FileNotFoundError(f"YOLO 数据集 yaml 不存在: {yolo_path}")
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

    val_cfg = grounding_cfg.get("val") or {}
    val_yolo = val_cfg.get("yolo_data") or [str(_resolve_path(val_yaml))]

    return {
        "train": {
            "yolo_data": [str(yolo_path)],
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
    parser.add_argument("--data",    type=str, default="data/yolo/0-Person.yaml",
                        help="YOLO 数据集 yaml（linear/full/visual 模式）")
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
    if args.mode == "scratch":
        val_yaml = cfg.get("val_data", "ultralytics/cfg/datasets/lvis.yaml")
        args.val_data = val_yaml
        args.scratch_data = build_scratch_data(args.data, args.grounding_data, val_yaml)
        n_gd = len(args.scratch_data["train"]["grounding_data"])
        print(f"  scratch data: yolo={args.data} | grounding entries={n_gd} | val={val_yaml}")
    dispatch = {
        "linear":  train_linear,
        "full":    train_full,
        "visual":  train_visual,
        "scratch": train_scratch,
    }
    dispatch[args.mode](args)
    print(f"\n🎉 完成！权重与配置均已保存至 runs/0-train/{args.project}/")
