"""
# @Author: 算法组 蔡雨霖
# @Date: 2026-06-15
# @Description: YOLOE 分割微调训练封装，固定使用 Seg 系列 Trainer（-seg.pt 权重）。
#   超参默认值从 config/train_pe.yaml train.defaults 读取，CLI 参数可覆盖。
#   --mode linear  : 线性探测，仅训练最后分类卷积（cv3.*.2 = PE 层），其余全冻结
#   --mode full    : 全参微调，不冻结任何层
#   --mode visual  : 视觉提示词，仅训练 SAVPE 模块，其余全冻结
#   --mode scratch : 从头预训练（需 .yaml + Objects365/Flickr/GQA 数据）
# @Command: python 0-QuickStart/scripts/train.py --mode full --model weights/yoloe-11s-seg.pt --data data/0-Person.yaml
"""
import argparse
import os
import shutil
import sys
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["PYTHONHASHSEED"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

_DEFAULT_CFG = Path("config/train_pe.yaml")

# 本仓库 yaml 专用字段，不传给 Ultralytics model.train()
_TRAIN_CFG_SKIP = frozenset({
    "mode", "grounding_data", "vocab_json", "neg_vocab", "val_data",
    "model", "data", "device", "project", "name",
})


#----------------------------#
# DDP 辅助
#----------------------------#
def _parse_device(device) -> str:
    """统一 device 为 '0,1,2,3' 字符串，供 Ultralytics 自动 spawn DDP。"""
    if isinstance(device, (list, tuple)):
        return ",".join(str(d).strip() for d in device)
    return ",".join(x.strip() for x in str(device).split(",") if x.strip())


def _world_size(device) -> int:
    dev = _parse_device(device)
    if dev in {"cpu", "mps", ""}:
        return 1
    return len(dev.split(","))


def _is_main_process() -> bool:
    from ultralytics.utils import LOCAL_RANK, RANK
    return LOCAL_RANK in {-1, 0} and RANK in {-1, 0}


def _ddp_batch_note(batch: int, device) -> str:
    ws = _world_size(device)
    if ws <= 1:
        return f"batch={batch}"
    return f"global batch={batch} | per-GPU batch={max(batch // ws, 1)} | {ws} GPUs (DDP)"


def set_config_path(cfg_path: str):
    """覆盖全局配置文件路径（供 train_open.py 等调用方使用）。"""
    global _DEFAULT_CFG
    _DEFAULT_CFG = Path(cfg_path)


#----------------------------#
# 从配置文件读取训练超参（train.defaults 节点）
#----------------------------#
def _load_train_cfg(_mode: str = "") -> dict:
    if not _DEFAULT_CFG.exists():
        return {}
    try:
        import yaml
        with open(_DEFAULT_CFG) as f:
            cfg = yaml.safe_load(f)
        return cfg.get("train", {}).get("defaults", {})
    except Exception:
        return {}


#----------------------------#
# 从数据集 yaml 读取类别名
#----------------------------#
def _load_names(data_yaml: str) -> list[str]:
    from ultralytics.utils import yaml_load
    cfg = yaml_load(data_yaml)
    if "names" not in cfg:
        raise ValueError(f"数据集 yaml 缺少 'names' 字段：{data_yaml}")
    names_val = cfg["names"]
    if isinstance(names_val, dict):
        return [names_val[k] for k in sorted(names_val)]
    return list(names_val)


#----------------------------#
# 读取 scale 相关超参扩展（ultralytics/cfg/{scale}_train.yaml）
#----------------------------#
def _load_extends(model_path: str, finetune: bool) -> dict:
    from ultralytics.nn.tasks import guess_model_scale
    from ultralytics.utils import yaml_load

    scale = guess_model_scale(model_path)
    cfg_dir = Path("ultralytics/cfg")
    extend_key = f"coco_{scale}_train.yaml" if finetune else f"{scale}_train.yaml"
    extend_cfg = cfg_dir / extend_key

    defaults = yaml_load(str(cfg_dir / "default.yaml"))
    if extend_cfg.exists():
        extends = yaml_load(str(extend_cfg))
        extends = {k: v for k, v in extends.items() if v is not None and k in defaults}
    else:
        extends = {}
    return extends


#----------------------------#
# 生成并保存文本提示 PE
#----------------------------#
def _prepare_pe(model, names: list[str], pe_path: str) -> str:
    tpe = model.get_text_pe(names)
    torch.save({"names": names, "pe": tpe}, pe_path)
    print(f"  PE 已生成：{pe_path}  ({len(names)} 类)")
    return pe_path



#----------------------------#
# 备份配置快照（训练启动后写入 save_dir，避免被 Ultralytics 清空）
# 目录结构：runs/0-train/<project>/config/{args,dataset,vocab}/
#----------------------------#
def _resolve_config_path(path: str) -> Path:
    p = Path(path)
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def _resolve_embed_dir(text_model: str) -> Path:
    for candidate in (ROOT / "config" / text_model, ROOT / "tools" / text_model):
        if (candidate / "train_label_embeddings.pt").is_file():
            return candidate
    return ROOT / "config" / text_model


def backup_train_config(
    run_dir: str | Path,
    data_yaml: str,
    *,
    grounding_yaml: str | None = None,
    val_yaml: str | None = None,
    include_vocab: bool = False,
):
    config_root = Path(run_dir) / "config"
    args_dir = config_root / "args"
    dataset_dir = config_root / "dataset"
    args_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = ROOT / _DEFAULT_CFG
    if train_cfg.is_file():
        shutil.copy2(train_cfg, args_dir / train_cfg.name)
        print(f"  📋 {train_cfg.name} → {args_dir}/")

    for yaml_path in (data_yaml, grounding_yaml, val_yaml):
        if not yaml_path:
            continue
        src = _resolve_config_path(yaml_path)
        if src.is_file():
            shutil.copy2(src, dataset_dir / src.name)
            print(f"  📋 {src.name} → {dataset_dir}/")

    if not include_vocab:
        return

    vocab_dir = config_root / "vocab"
    vocab_dir.mkdir(parents=True, exist_ok=True)
    cfg = _load_train_cfg()
    vocab_json = cfg.get("vocab_json", "config/vocab/train_label_embeddings.json")
    neg_vocab = cfg.get("neg_vocab", "config/vocab/global_grounding_neg_cat.json")
    text_model = cfg.get("text_model", "mobileclip:blt")
    embed_dir = _resolve_embed_dir(text_model)

    for pt_name in ("global_grounding_neg_embeddings.pt", "train_label_embeddings.pt"):
        src = embed_dir / pt_name
        if src.is_file():
            shutil.copy2(src, vocab_dir / pt_name)
            print(f"  📦 {pt_name} → {vocab_dir}/")

    for vocab_path in (vocab_json, neg_vocab):
        vocab_src = _resolve_config_path(vocab_path)
        if vocab_src.is_file():
            shutil.copy2(vocab_src, vocab_dir / vocab_src.name)
            print(f"  📋 {vocab_src.name} → {vocab_dir}/")


def _register_config_backup(
    model,
    data_yaml: str,
    *,
    grounding_yaml: str | None = None,
    val_yaml: str | None = None,
    include_vocab: bool = False,
):
    def _on_train_start(trainer):
        if _is_main_process():
            backup_train_config(
                trainer.save_dir,
                data_yaml,
                grounding_yaml=grounding_yaml,
                val_yaml=val_yaml,
                include_vocab=include_vocab,
            )

    model.add_callback("on_train_start", _on_train_start)


#----------------------------#
# 从 yaml + scale 扩展构建 Ultralytics train kwargs
# extends（scale 默认）< train.defaults < 调用方 overrides
#----------------------------#
def _build_train_kwargs(
    cfg: dict,
    extends: dict,
    *,
    data,
    device: str,
    epochs: int,
    batch: int,
    project_name: str,
    **overrides,
) -> dict:
    from ultralytics.utils import DEFAULT_CFG_KEYS

    merged = dict(extends)
    for k in DEFAULT_CFG_KEYS:
        if k in cfg and cfg[k] is not None:
            merged[k] = cfg[k]

    kwargs = {k: v for k, v in merged.items() if k not in _TRAIN_CFG_SKIP}
    kwargs.update(
        data=data,
        device=device,
        epochs=epochs,
        batch=batch,
        project="runs/0-train",
        name=project_name,
        **overrides,
    )
    return kwargs


#----------------------------#
# 构造 linear 冻结列表
# 仅保留 head 最后一层 cv3.*.2（Prompt Embedding）可训练
#----------------------------#
def _freeze_linear(model) -> list:
    head_idx = len(model.model.model) - 1
    freeze = [str(i) for i in range(head_idx)]             # 冻结整个 backbone
    for name, _ in model.model.model[-1].named_children():
        if "cv3" not in name:
            freeze.append(f"{head_idx}.{name}")             # 冻结 head 中非 cv3 部分
    # cv3 内部：只留 cv3.*.2（最后一个卷积）；冻结 .0 和 .1
    freeze += [
        f"{head_idx}.cv3.0.0", f"{head_idx}.cv3.0.1",
        f"{head_idx}.cv3.1.0", f"{head_idx}.cv3.1.1",
        f"{head_idx}.cv3.2.0", f"{head_idx}.cv3.2.1",
    ]
    return freeze


#----------------------------#
# 构造 visual 冻结列表
# 仅保留 head 中的 SAVPE 模块可训练
#----------------------------#
def _freeze_visual(model) -> list:
    head_idx = len(model.model.model) - 1
    freeze = list(range(head_idx))                          # 冻结整个 backbone（整数列表）
    for name, _ in model.model.model[-1].named_children():
        if "savpe" not in name:
            freeze.append(f"{head_idx}.{name}")             # 冻结 head 中非 savpe 部分
    return freeze


#----------------------------#
# 模式 1：线性探测（仅 PE 层可训练）
#----------------------------#
def train_linear(args):
    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe.train_pe import YOLOEPESegTrainer

    cfg     = _load_train_cfg("linear")
    extends = _load_extends(args.model, finetune=True)
    names   = _load_names(args.data)

    epochs = args.epochs or cfg.get("epochs", 10)
    batch  = args.batch  or cfg.get("batch",  16)
    lr0    = cfg.get("lr0", 1e-3)
    print(f"\n⭐ Train mode: Linear Probing（仅 PE 层可训练）")
    print(f"💾 Weights：{args.model}")
    print(f"🗺️ MobileCLIP：{os.environ.get('MOBILECLIP_PATH', 'mobileclip_blt.pt')}")
    print(f"🛤️ dataset：{args.data} |  类别：{len(names)} 个")
    device = _parse_device(args.device or cfg.get("device", "0"))
    print(f"🔧 Config:  epochs={epochs} | {_ddp_batch_note(batch, device)} | lr0={lr0} | device={device}\n")

    model   = YOLOE(args.model)
    freeze  = _freeze_linear(model)
    pe_path = f"runs/0-train/{args.project}/pe.pt"
    Path(pe_path).parent.mkdir(parents=True, exist_ok=True)
    _prepare_pe(model, names, pe_path)
    _register_config_backup(model, args.data)

    train_kwargs = _build_train_kwargs(
        cfg, extends,
        data=args.data, device=device, epochs=epochs, batch=batch,
        project_name=args.project,
        trainer=YOLOEPESegTrainer, freeze=freeze, train_pe_path=pe_path,
    )
    model.train(**train_kwargs)


#----------------------------#
# 模式 2：全参微调（无冻结）
#----------------------------#
def train_full(args):
    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe.train_pe import YOLOEPESegTrainer

    cfg     = _load_train_cfg("full")
    extends = _load_extends(args.model, finetune=True)
    names   = _load_names(args.data)

    epochs = args.epochs or cfg.get("epochs", 80)
    batch  = args.batch  or cfg.get("batch",  16)
    lr0    = cfg.get("lr0", 1e-3)
    print(f"\n⭐ Train mode: Full Tuning（全部参数可训练）")
    print(f"💾 Weights：{args.model}")
    print(f"🗺️ MobileCLIP：{os.environ.get('MOBILECLIP_PATH', 'mobileclip_blt.pt')}")
    print(f"🛤️ dataset：{args.data} |  类别：{len(names)} 个")
    device = _parse_device(args.device or cfg.get("device", "0"))
    print(f"🔧 Config:  epochs={epochs} | {_ddp_batch_note(batch, device)} | lr0={lr0} | device={device}\n")

    model   = YOLOE(args.model)
    pe_path = f"runs/0-train/{args.project}/pe.pt"
    Path(pe_path).parent.mkdir(parents=True, exist_ok=True)
    _prepare_pe(model, names, pe_path)
    _register_config_backup(model, args.data)

    train_kwargs = _build_train_kwargs(
        cfg, extends,
        data=args.data, device=device, epochs=epochs, batch=batch,
        project_name=args.project,
        trainer=YOLOEPESegTrainer, train_pe_path=pe_path,
    )
    model.train(**train_kwargs)


#----------------------------#
# 模式 3：视觉提示词（仅 SAVPE）
#----------------------------#
def train_visual(args):
    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe.train_vp import YOLOEVPTrainer

    cfg     = _load_train_cfg("visual")
    extends = _load_extends(args.model, finetune=False)

    epochs = args.epochs or cfg.get("epochs", 2)
    batch  = args.batch  or cfg.get("batch",  16)
    lr0    = cfg.get("lr0", 8e-3)
    print(f"\n⭐ Train mode: Visual Prompt（仅 SAVPE 模块可训练）")
    print(f"💾 Weights：{args.model}")
    print(f"🛤️ dataset：{args.data}")
    device = _parse_device(args.device or cfg.get("device", "0"))
    print(f"🔧 Config:  epochs={epochs} | {_ddp_batch_note(batch, device)} | lr0={lr0} | device={device}\n")

    model  = YOLOE(args.model)
    freeze = _freeze_visual(model)
    _register_config_backup(model, args.data)

    train_kwargs = _build_train_kwargs(
        cfg, extends,
        data=args.data, device=device, epochs=epochs, batch=batch,
        project_name=args.project,
        trainer=YOLOEVPTrainer, freeze=freeze, load_vp=cfg.get("load_vp", True),
    )
    model.train(**train_kwargs)


#----------------------------#
# 模式 4：从头预训练（.yaml，大规模数据）
#----------------------------#
def train_scratch(args):
    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe.train_yoloe_seg import YOLOESegTrainerFromScratch

    cfg     = _load_train_cfg("scratch")
    extends = _load_extends(args.model, finetune=False)

    data = getattr(args, "scratch_data", None)
    if data is None:
        raise ValueError(
            "scratch 模式需要 yolo + grounding 合并后的 data dict。"
            "请通过 train_open.py 启动，或手动设置 args.scratch_data。"
        )

    epochs = args.epochs or cfg.get("epochs", 30)
    batch  = args.batch  or cfg.get("batch",  128)
    lr0    = cfg.get("lr0", 2e-3)
    n_yolo = len(data["train"]["yolo_data"])
    n_gd   = len(data["train"].get("grounding_data") or [])
    print(f"\n⭐ Train mode: Scratch 预训练")
    print(f"💾 Weights：{args.model}")
    print(f"🛤️ yolo_data：{n_yolo} 个 | grounding_data：{n_gd} 个")
    print(f"🛤️ val：{data['val']['yolo_data']}")
    device = _parse_device(args.device or cfg.get("device", "0"))
    print(f"🔧 Config:  epochs={epochs} | {_ddp_batch_note(batch, device)} | lr0={lr0} | device={device}\n")

    model = YOLOE(args.model)
    _register_config_backup(
        model,
        args.data,
        grounding_yaml=getattr(args, "grounding_data", None),
        val_yaml=getattr(args, "val_data", None) or cfg.get("val_data"),
        include_vocab=getattr(args, "backup_vocab", False),
    )

    train_kwargs = _build_train_kwargs(
        cfg, extends,
        data=data, device=device, epochs=epochs, batch=batch,
        project_name=args.project,
        trainer=YOLOESegTrainerFromScratch,
    )
    model.train(**train_kwargs)


#----------------------------#
# 参数解析
#----------------------------#
def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLOE 训练封装（超参默认值从 --config yaml 读取）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
模式说明（epochs/batch/lr 均在 --config yaml 中按 mode 分区配置）：
  linear  — 线性探测：仅训练 cv3.*.2（PE 层），冻结其余全部
  full    — 全参微调：不冻结任何层
  visual  — 视觉提示词：仅训练 SAVPE 模块，冻结其余全部
  scratch — 从头预训练：需 .yaml + 大规模数据集
        """
    )
    parser.add_argument("--config",  type=str, default="config/train_pe.yaml",
                        help="训练配置文件路径")
    parser.add_argument("--mode",    type=str, default=None,
                        choices=["linear", "full", "visual", "scratch"],
                        help="训练模式（不填则从 --config train.defaults.mode 读取）")
    parser.add_argument("--model",   type=str, default="weights/yoloe-11s-seg.pt",
                        help=".pt → 微调；.yaml → scratch")
    parser.add_argument("--data",    type=str, default="data/0-Person.yaml",
                        help="数据集配置 yaml（YOLO 格式）")
    parser.add_argument("--project", type=str, default="exp01",
                        help="运行名称 → 权重保存至 runs/0-train/<project>/")
    parser.add_argument("--epochs",  type=int, default=None,
                        help="覆盖 yaml 中的 epochs")
    parser.add_argument("--batch",   type=int, default=None,
                        help="覆盖 yaml 中的 batch")
    parser.add_argument("--device",  type=str, default="0",
                        help="GPU 设备 ID，多卡如 '0,1,2,3'")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    set_config_path(args.config)
    _yaml_cfg = _load_train_cfg()
    if args.mode is None:
        args.mode = _yaml_cfg.get("mode", "linear")

    dispatch = {
        "linear":  train_linear,
        "full":    train_full,
        "visual":  train_visual,
        "scratch": train_scratch,
    }
    dispatch[args.mode](args)
