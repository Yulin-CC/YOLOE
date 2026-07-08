"""
# @Author: 算法组 蔡雨霖
# @Date: 2026-06-15
# @Description: YOLOE 评估脚本，对齐官方 z-others/val*.py；LVIS text 模式支持自动 Fixed AP
# @Command: python 0-QuickStart/scripts/eval.py --config config/default_notrain.yaml
"""
import argparse
import io
import json
import logging
import os
import re
import sys
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 避免 tools/lvis/ 目录遮蔽真正的 lvis 包
sys.path = [p for p in sys.path if not p.endswith(("tools", "tools/"))]

import ultralytics.utils
from ultralytics import YOLOE
from ultralytics.data.converter import coco80_to_coco91_class
from ultralytics.models.yolo.segment import SegmentationValidator
from ultralytics.models.yolo.yoloe.val import YOLOEDetectValidator, YOLOESegValidator
from ultralytics.utils import LOGGER, ops, yaml_load
from lvis import LVIS, LVISResults, LVISEval

os.environ.setdefault("PYTHONUNBUFFERED", "1")

ultralytics.utils.TQDM_BAR_FORMAT = "{l_bar}{bar:30}{r_bar}"


def _setup_tqdm_desc():
    """统一 val 进度条描述（LVIS detect / COCO seg 等模式）。"""
    desc_fn = lambda self: "Validating YOLOE..."
    YOLOEDetectValidator.get_desc = desc_fn
    YOLOESegValidator.get_desc = desc_fn
    SegmentationValidator.get_desc = desc_fn


_setup_tqdm_desc()


#----------------------------#
# 读取 eval 配置
#----------------------------#
def load_eval_cfg(config_path: str) -> dict:
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    return yaml_load(str(cfg_path)).get("eval", {}) or {}


# eval yaml 专用字段，不传给 model.val()
_EVAL_CFG_SKIP = frozenset({
    "weights", "mobileclip", "mode", "fixed_ap",
})


def _build_val_kwargs(cfg: dict, **overrides) -> dict:
    kwargs = {k: v for k, v in cfg.items() if k not in _EVAL_CFG_SKIP and v is not None}
    kwargs.update(overrides)
    return kwargs


#----------------------------#
# CLI > yaml > 默认值
#----------------------------#
def pick(cli_val, cfg: dict, key: str, default=None):
    if cli_val is not None and cli_val != "":
        return cli_val
    if key in cfg and cfg[key] is not None:
        return cfg[key]
    return default


def str2bool(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("true", "1", "yes")


def resolve_path(path: str) -> str:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return str(p)


#----------------------------#
# LVIS 标注路径
#----------------------------#
def get_lvis_anno_path(data_yaml: str, split: str) -> Path | None:
    try:
        data_cfg = yaml_load(data_yaml)
        data_root = Path(data_cfg.get("path", ""))
        split_val = data_cfg.get(split, data_cfg.get("val", ""))
        if isinstance(split_val, list):
            split_val = split_val[0]
        is_minival = "minival" in str(split_val)
        return data_root / "annotations" / f"lvis_v1_{'minival' if is_minival else 'val'}.json"
    except Exception as e:
        LOGGER.warning(f"读取数据集 yaml 失败：{e}")
        return None


#----------------------------#
# Fixed AP（同 tools/eval_fixed_ap.py）
#----------------------------#
def run_fixed_ap(pred_json: Path, anno_json: Path, eval_type: str = "bbox"):
    with open(pred_json) as f:
        results = json.load(f)

    topk = 10000
    by_cat = defaultdict(list)
    for ann in results:
        by_cat[ann["category_id"]].append(ann)

    capped, missing = [], set()
    for cat, cat_anns in by_cat.items():
        if len(cat_anns) < topk:
            missing.add(cat)
        capped.extend(sorted(cat_anns, key=lambda x: x["score"], reverse=True)[:topk])

    if eval_type == "segm":
        for x in capped:
            x.pop("bbox", None)

    if missing:
        LOGGER.warning(
            f"\n===\n{len(missing)} classes had less than {topk} detections!\n==="
        )

    gt = LVIS(str(anno_json))
    lvis_results = LVISResults(gt, capped, max_dets=-1)
    lvis_eval = LVISEval(gt, lvis_results, iou_type=eval_type)
    lvis_eval.params.max_dets = -1
    lvis_eval.run()
    lvis_eval.print_results()
    metrics = {k: v for k, v in lvis_eval.results.items() if k.startswith("AP")}
    LOGGER.info("copypaste: %s", ",".join(map(str, metrics.keys())))
    LOGGER.info("copypaste: %s", ",".join(f"{v * 100:.2f}" for v in metrics.values()))


#----------------------------#
# COCO pycocotools 评估（path 为 COCO2017 时 ultralytics is_coco 检测会失败）
#----------------------------#
def get_coco_anno_path(data_yaml: str) -> Path | None:
    try:
        data_cfg = yaml_load(data_yaml)
        return Path(data_cfg["path"]) / "annotations" / "instances_val2017.json"
    except Exception as e:
        LOGGER.warning(f"读取 COCO 标注路径失败：{e}")
        return None


def get_coco_img_ids(data_yaml: str) -> list[int]:
    data_cfg = yaml_load(data_yaml)
    data_root = Path(data_cfg["path"])
    val = data_cfg.get("val", "val2017.txt")
    if isinstance(val, list):
        val = val[0]
    val_path = Path(val) if Path(val).is_absolute() else data_root / val
    with open(val_path) as f:
        return [int(Path(line.strip()).stem) for line in f if line.strip()]


def run_coco_eval(pred_json: Path, anno_json: Path, img_ids: list[int] | None = None):
    from ultralytics.utils.checks import check_requirements

    check_requirements("pycocotools>=2.0.6")
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    LOGGER.info(f"\nEvaluating pycocotools mAP using {pred_json} and {anno_json}...")
    anno = COCO(str(anno_json))
    pred = anno.loadRes(str(pred_json))

    ap_stats = {}
    for iou_type, label in (("bbox", "BOX EVAL"), ("segm", "MASK EVAL")):
        print("=" * 40, label, "=" * 40)
        ev = COCOeval(anno, pred, iou_type)
        if img_ids:
            ev.params.imgIds = img_ids
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        ap_stats[iou_type] = ev.stats[:3]

    box, mask = ap_stats["bbox"], ap_stats["segm"]
    LOGGER.info("copypaste: Box_AP,Box_AP50,Box_AP75,Mask_AP,Mask_AP50,Mask_AP75")
    LOGGER.info("copypaste: %s", ",".join(f"{v * 100:.2f}" for v in list(box) + list(mask)))
    return ap_stats


#----------------------------#
# Detect validator：仅写 bbox，跳过越界 cls（open-vocab 头偶发）
#----------------------------#
class YOLOELVISDetectValidator(YOLOEDetectValidator):
    def pred_to_json(self, predn, filename):
        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        if len(predn) == 0:
            return
        box = ops.xyxy2xywh(predn[:, :4])
        box[:, :2] -= box[:, 2:] / 2
        for p, b in zip(predn.tolist(), box.tolist()):
            cls_id = int(p[5])
            if cls_id < 0 or cls_id >= len(self.class_map):
                continue
            self.jdict.append(
                {
                    "image_id": image_id,
                    "category_id": self.class_map[cls_id] + (1 if self.is_lvis else 0),
                    "bbox": [round(x, 3) for x in b],
                    "score": round(p[4], 5),
                }
            )


#----------------------------#
# COCO validator：强制 is_coco + coco91 类别映射（COCO2017 路径大小写问题）
#----------------------------#
class YOLOECOCOSegValidator(SegmentationValidator):
    def init_metrics(self, model):
        super().init_metrics(model)
        self.is_coco = True
        self.is_lvis = False
        self.class_map = coco80_to_coco91_class()
        self.args.save_json = True


#----------------------------#
# 日志捕获 + 落盘（val + Fixed AP 同一 results.txt）
#----------------------------#
LVIS_LOGGERS = ("lvis", "lvis.results", "lvis.eval", "lvis.lvis")


class _Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

    def isatty(self):
        return any(getattr(f, "isatty", lambda: False)() for f in self.files)

    def fileno(self):
        for f in self.files:
            if hasattr(f, "fileno"):
                return f.fileno()
        raise AttributeError("fileno")


def _attach_stream_logger(stream: io.TextIOBase):
    formatter = logging.Formatter("%(message)s")
    fh = logging.StreamHandler(stream)
    fh.setFormatter(formatter)
    fh.setLevel(logging.INFO)
    LOGGER.addHandler(fh)
    for logger_name in LVIS_LOGGERS:
        lg = logging.getLogger(logger_name)
        lg.addHandler(fh)
        lg.setLevel(logging.INFO)
    return fh


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_TQDM_RE = re.compile(r"Validating YOLOE.*(%|\d+/\d+)")
_METRICS_HDR_RE = re.compile(r"^\s*Class\s+Images\s+Instances")
_ALL_ROW_RE = re.compile(r"^\s*all\s+\d")
_AP_AR_RE = re.compile(r"^(Average Precision|Average Recall)")
_MAXDETS_RE = re.compile(r"maxDets=\s*(-?\d+)")


def _coco_ap_triplet(metric_lines: list[str]) -> list[float]:
    """从 pycocotools summarize 行提取 AP / AP50 / AP75（area=all 前三项）。"""
    vals = []
    for line in metric_lines:
        if not line.startswith("Average Precision") or "area=   all" not in line:
            continue
        vals.append(float(line.rsplit("=", 1)[-1].strip()))
        if len(vals) == 3:
            break
    return vals


def _fixed_ap_banner(text: str) -> list[str]:
    """从日志中提取 Fixed AP 分隔块（含预测/标注路径）。"""
    raw_lines = [_ANSI_RE.sub("", ln).rstrip("\n\r") for ln in text.splitlines()]
    norm_lines = [ln.strip() for ln in raw_lines]
    for i, line in enumerate(norm_lines):
        if "计算 Fixed AP" not in line:
            continue
        block = ["=" * 50, line]
        for j in range(i + 1, min(i + 5, len(norm_lines))):
            if norm_lines[j].startswith("预测:") or norm_lines[j].startswith("标注:"):
                block.append(raw_lines[j])
            elif norm_lines[j] == "=" * 50:
                block.append("=" * 50)
                return block
        block.append("=" * 50)
        return block
    return ["=" * 50, "Fixed AP（maxDets=-1）", "=" * 50]


def _extract_eval_summary(text: str) -> str:
    """从完整日志提取评估摘要；LVIS 分标准 AP / Fixed AP；COCO 分 Ultralytics / Box / Mask。"""
    speed = None
    std_metrics, fixed_metrics, copypaste = [], [], []
    ultralytics_rows, box_coco, mask_coco = [], [], []
    in_box, in_mask = False, False

    for raw in text.splitlines():
        line = _ANSI_RE.sub("", raw).strip()
        if not line:
            continue
        if _TQDM_RE.search(line) or (line.startswith("val:") and "Scanning" in line):
            continue
        if line.startswith("Speed:"):
            speed = line
        elif line.startswith("copypaste:"):
            copypaste.append(line)
        elif line.startswith("Ultralytics stats:"):
            ultralytics_rows.append(line)
        elif "BOX EVAL" in line:
            in_box, in_mask = True, False
        elif "MASK EVAL" in line:
            in_box, in_mask = False, True
        elif _METRICS_HDR_RE.match(line) or _ALL_ROW_RE.match(line):
            ultralytics_rows.append(line)
        elif _AP_AR_RE.match(line):
            m = _MAXDETS_RE.search(line)
            if in_box:
                box_coco.append(line)
            elif in_mask:
                mask_coco.append(line)
            elif m and m.group(1) == "-1":
                fixed_metrics.append(line)
            elif m and m.group(1) == "100":
                # pycocotools 但未捕获到 BOX/MASK 标题时，按出现顺序分配
                (box_coco if len(box_coco) <= len(mask_coco) else mask_coco).append(line)
            else:
                std_metrics.append(line)

    parts = []
    if speed:
        parts.extend([speed, ""])

    if ultralytics_rows:
        parts.extend(["=" * 50, "Ultralytics 内置指标（COCO val2017）", "=" * 50])
        parts.extend(ultralytics_rows)
        parts.append("")

    if box_coco:
        parts.extend(["=" * 50, "COCO Box AP（pycocotools bbox）", "=" * 50])
        parts.extend(box_coco)
        parts.append("")

    if mask_coco:
        parts.extend(["=" * 50, "COCO Mask AP（pycocotools segm）", "=" * 50])
        parts.extend(mask_coco)
        parts.append("")

    if std_metrics:
        parts.extend(["=" * 50, "标准 AP（maxDets=300，val 内置 LVIS 评估）", "=" * 50])
        parts.extend(std_metrics)
        parts.append("")

    if fixed_metrics or (copypaste and not box_coco):
        parts.extend(_fixed_ap_banner(text))
        parts.extend(fixed_metrics)

    if copypaste:
        parts.extend(copypaste)
    elif box_coco and mask_coco:
        box_v = _coco_ap_triplet(box_coco)
        mask_v = _coco_ap_triplet(mask_coco)
        if len(box_v) == 3 and len(mask_v) == 3:
            parts.append("copypaste: Box_AP,Box_AP50,Box_AP75,Mask_AP,Mask_AP50,Mask_AP75")
            parts.append(
                "copypaste: "
                + ",".join(f"{v * 100:.2f}" for v in box_v + mask_v)
            )

    return "\n".join(parts) + ("\n" if parts else "")


def _detach_file_logger(fh):
    LOGGER.removeHandler(fh)
    for logger_name in LVIS_LOGGERS:
        logging.getLogger(logger_name).removeHandler(fh)
    fh.close()


@contextmanager
def eval_log_session(save_dir: Path):
    """终端打印完整日志；save_dir/results.txt 仅保留 AP/AR 等评估摘要。"""
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / "results.txt"
    buf = io.StringIO()
    fh = _attach_stream_logger(buf)
    stdout, stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(stdout, buf)
    sys.stderr = _Tee(stderr, buf)
    print(f"📝 评估摘要将保存至 {log_path}")
    try:
        yield log_path
    finally:
        sys.stdout = stdout
        sys.stderr = stderr
        _detach_file_logger(fh)
        log_path.write_text(_extract_eval_summary(buf.getvalue()), encoding="utf-8")


def _run_val(model, save_dir: Path, **kwargs):
    kwargs.setdefault("verbose", True)
    return model.val(project=str(save_dir.parent), name=save_dir.name, exist_ok=True, **kwargs)


def _stats_to_dict(stats) -> dict:
    if stats is None:
        return {}
    if isinstance(stats, dict):
        return stats
    if hasattr(stats, "results_dict"):
        return stats.results_dict
    return {}


def _log_coco_copypaste(stats):
    """Ultralytics stats 摘要（pycocotools 结果见 summarize / copypaste 行）。"""
    d = _stats_to_dict(stats)
    if not d:
        return
    b_ap = d.get("metrics/mAP50-95(B)", 0) * 100
    b50 = d.get("metrics/mAP50(B)", 0) * 100
    m_ap = d.get("metrics/mAP50-95(M)", 0) * 100
    m50 = d.get("metrics/mAP50(M)", 0) * 100
    LOGGER.info(
        "Ultralytics stats: Box mAP50-95=%.2f mAP50=%.2f | Mask mAP50-95=%.2f mAP50=%.2f",
        b_ap, b50, m_ap, m50,
    )


def make_save_dir(mode: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub = "coco" if mode == "coco" else "lvis"
    return ROOT / "runs/val" / sub / f"{mode}_{ts}"


#----------------------------#
# text：对齐 z-others/val.py
# fixed_ap=True  → max_det=1000 + detect validator（bbox json）+ Fixed AP
# fixed_ap=False → max_det=300  + 默认 seg validator（标准 AP，同 val8）
#----------------------------#
def eval_text(weights, cfg, split, batch, device, fixed_ap, max_det, save_dir):
    model = YOLOE(resolve_path(weights))
    val_kwargs = _build_val_kwargs(
        cfg,
        data=resolve_path(cfg.get("data", "ultralytics/cfg/datasets/lvis.yaml")),
        batch=batch,
        split=split,
        rect=False,
        device=device,
        is_lvis=True,
    )

    if fixed_ap:
        val_kwargs["max_det"] = max_det or 1000
        val_kwargs["validator"] = YOLOELVISDetectValidator
    else:
        val_kwargs["max_det"] = max_det or 300

    with eval_log_session(save_dir):
        if fixed_ap:
            LOGGER.info(f"text + Fixed AP：max_det={val_kwargs.get('max_det', 1000)}，YOLOELVISDetectValidator（bbox json）")
        else:
            LOGGER.info(f"text 标准 AP：max_det={val_kwargs['max_det']}，默认 YOLOESegValidator")

        _run_val(model, save_dir, **val_kwargs)

        if fixed_ap:
            pred_json = save_dir / "predictions.json"
            anno_json = get_lvis_anno_path(val_kwargs["data"], split)
            if not pred_json.exists():
                LOGGER.warning(f"未找到 {pred_json}，跳过 Fixed AP")
            elif anno_json is None or not anno_json.exists():
                LOGGER.warning(f"未找到 LVIS 标注 {anno_json}，跳过 Fixed AP")
            else:
                LOGGER.info(
                    f"\n{'=' * 50}\n计算 Fixed AP（bbox）\n  预测: {pred_json}\n  标注: {anno_json}\n{'=' * 50}\n"
                )
                run_fixed_ap(pred_json, anno_json, eval_type="bbox")

    print(f"✅ 文本提示词评估完成（{split}）→ {save_dir}")


#----------------------------#
# visual：对齐 z-others/val_vp.py
#----------------------------#
def eval_visual(weights, cfg, split, batch, device, fixed_ap, max_det, save_dir):
    weights_path = Path(resolve_path(weights))
    yaml_path = weights_path.with_suffix(".yaml")
    if not yaml_path.exists():
        yaml_path = ROOT / "ultralytics/cfg/models/11/yoloe-11-seg.yaml"

    model = YOLOE(str(yaml_path))
    model.load(str(weights_path))
    model.eval()

    val_kwargs = _build_val_kwargs(
        cfg,
        data=resolve_path(cfg.get("data", "ultralytics/cfg/datasets/lvis.yaml")),
        batch=batch,
        split=split,
        rect=False,
        device=device,
        load_vp=True,
        is_lvis=True,
        max_det=max_det or (1000 if fixed_ap else 300),
    )
    if fixed_ap:
        val_kwargs["validator"] = YOLOELVISDetectValidator

    with eval_log_session(save_dir):
        _run_val(model, save_dir, **val_kwargs)
        if fixed_ap:
            pred_json = save_dir / "predictions.json"
            anno_json = get_lvis_anno_path(val_kwargs["data"], split)
            if pred_json.exists() and anno_json and anno_json.exists():
                LOGGER.info(
                    f"\n{'=' * 50}\n计算 Fixed AP（bbox）\n  预测: {pred_json}\n  标注: {anno_json}\n{'=' * 50}\n"
                )
                run_fixed_ap(pred_json, anno_json, eval_type="bbox")

    print(f"✅ 视觉提示词评估完成（{split}）→ {save_dir}")


#----------------------------#
# promptfree：对齐 z-others/val_pe_free.py
#----------------------------#
def eval_promptfree(weights, cfg, split, batch, device, fixed_ap, max_det, save_dir):
    from ultralytics.models.yolo.yoloe.val_pe_free import YOLOEPEFreeDetectValidator
    from ultralytics.nn.modules.head import YOLOEDetect

    weights_path = resolve_path(weights)
    pf_weights = weights_path.replace("-seg.pt", "-seg-pf.pt").replace(".pt", "-pf.pt")
    if not Path(pf_weights).exists():
        pf_weights = weights_path

    unfused = YOLOE(weights_path)
    with open(ROOT / "tools/ram_tag_list.txt") as f:
        names = [x.strip() for x in f.readlines()]
    vocab = unfused.get_vocab(names)

    model = YOLOE(pf_weights)
    model.set_vocab(vocab, names=names)
    model.model.model[-1].is_fused = True
    model.model.model[-1].conf = cfg.get("conf") or 0.001
    model.model.model[-1].max_det = max_det or (1000 if fixed_ap else 300)
    model.model.model[-1].__class__ = YOLOEDetect

    val_kwargs = _build_val_kwargs(
        cfg,
        data=resolve_path(cfg.get("data", "ultralytics/cfg/datasets/lvis.yaml")),
        batch=batch,
        split=split,
        rect=False,
        device=device,
        max_det=max_det or (1000 if fixed_ap else 300),
        validator=YOLOEPEFreeDetectValidator,
        is_lvis=True,
    )
    if not fixed_ap:
        val_kwargs.setdefault("plots", cfg.get("plots", True))

    with eval_log_session(save_dir):
        _run_val(model, save_dir, **val_kwargs)

    print(f"✅ Prompt-Free 评估完成（{split}）→ {save_dir}")
    print("   Open-ended AP: python tools/eval_open_ended.py --json <lvis_json> --pred <predictions.json> --fixed")


#----------------------------#
# coco：对齐 z-others/val_coco.py
#----------------------------#
def eval_coco(weights, cfg, batch, device, save_dir):
    """对齐 z-others/val_coco.py：COCO 下游迁移评估（需 coco-pe / coco 微调权重）。"""
    data_yaml = resolve_path(cfg.get("data", str(ROOT / "ultralytics/cfg/datasets/coco.yaml")))
    model = YOLOE(resolve_path(weights))
    val_kwargs = _build_val_kwargs(
        cfg,
        data=data_yaml,
        batch=batch,
        device=device,
        save_json=True,
        validator=YOLOECOCOSegValidator,
    )
    with eval_log_session(save_dir):
        stats = _run_val(model, save_dir, **val_kwargs)
        _log_coco_copypaste(stats)
    print(f"✅ COCO 评估完成 → {save_dir}")
    print(f"   数据集: {data_yaml}")
    print("   官方权重示例: yoloe-11s-seg-coco-pe.pt（LP）或 yoloe-11s-seg-coco.pt（FT）")


#----------------------------#
# 参数解析
#----------------------------#
def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLOE 评估（CLI > config yaml eval 段 > 默认值）",
    )
    parser.add_argument("--config", type=str, default="config/default_notrain.yaml")
    parser.add_argument("--mode", type=str, default=None, choices=["text", "visual", "promptfree", "coco"])
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--mobileclip", type=str, default=None)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-det", dest="max_det", type=int, default=None)
    parser.add_argument("--fixed-ap", dest="fixed_ap", type=str, default=None,
                        help="是否执行 Fixed AP（true/false）")
    parser.add_argument("--plots", type=str, default=None,
                        help="是否保存可视化图（true/false，默认 true）")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_eval_cfg(args.config)

    mode = pick(args.mode, cfg, "mode", "text")
    weights = pick(args.weights, cfg, "weights", "weights/yoloe-11s-seg.pt")
    split = pick(args.split, cfg, "split", "minival")
    batch = pick(args.batch, cfg, "batch", 1)
    device = pick(args.device, cfg, "device", "0")
    max_det = pick(args.max_det, cfg, "max_det", None)

    fixed_ap = pick(str2bool(args.fixed_ap), cfg, "fixed_ap", True)
    if isinstance(fixed_ap, str):
        fixed_ap = str2bool(fixed_ap)

    if args.data is not None:
        cfg = {**cfg, "data": args.data}

    plots = pick(str2bool(args.plots), cfg, "plots", True)
    if isinstance(plots, str):
        plots = str2bool(plots)
    cfg = {**cfg, "plots": plots}

    mobileclip = resolve_path(pick(args.mobileclip, cfg, "mobileclip", "weights/mobileclip_blt.pt"))
    os.environ["MOBILECLIP_PATH"] = mobileclip

    save_dir = make_save_dir(mode)

    print(f"📄 配置：{args.config}")
    print(f"   mode={mode}  split={split}  batch={batch}  fixed_ap={fixed_ap}  plots={plots}  max_det={max_det or ('1000' if fixed_ap else '300')}")
    print(f"   data={cfg.get('data')}  imgsz={cfg.get('imgsz', 640)}")
    print(f"   输出：{save_dir}")

    if mode == "text":
        eval_text(weights, cfg, split, batch, device, fixed_ap, max_det, save_dir)
    elif mode == "visual":
        eval_visual(weights, cfg, split, batch, device, fixed_ap, max_det, save_dir)
    elif mode == "promptfree":
        eval_promptfree(weights, cfg, split, batch, device, fixed_ap, max_det, save_dir)
    elif mode == "coco":
        eval_coco(weights, cfg, batch, device, save_dir)
    else:
        raise ValueError(f"未知评估模式: {mode}")
