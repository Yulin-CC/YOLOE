"""
# @Author: 算法组 蔡雨霖
# @Date: 2026-07-07
# @Description: 扫描数据根目录下 GEOAI-* 子文件夹，按后缀分别生成 grounding / yolo 训练 yaml
# @Command: cd data && python create_data.py
"""
import json
import os
from datetime import datetime
from pathlib import Path

import yaml

JSONS_SEGMENT_DIR = "jsons-segment"


#----------------------------#
# 数据集目录识别：前缀 GEOAI，后缀 GD / YOLO
#----------------------------#
def is_grounding_dir(name: str) -> bool:
    return name.startswith("GEOAI") and name.endswith("GD")


def is_yolo_dir(name: str) -> bool:
    return name.startswith("GEOAI") and name.endswith("YOLO")


#----------------------------#
# 在 GEOAI-*-GD 目录下查找 segm json
#----------------------------#
def find_grounding_json_files(dataset_dir: Path) -> list[Path]:
    json_files: list[Path] = []

    for p in sorted(dataset_dir.glob("*.json")):
        json_files.append(p.resolve())

    for cache in sorted(dataset_dir.glob("*.cache")):
        json_path = cache.with_suffix(".json")
        if json_path.resolve() not in json_files:
            json_files.append(json_path.resolve())

    return json_files


#----------------------------#
# 单个 GEOAI-*-GD 目录 → grounding_data 条目
#----------------------------#
def collect_grounding_entries(root: Path, dir_name: str) -> list[dict]:
    dataset_dir = root / dir_name
    entries = []

    for json_file in find_grounding_json_files(dataset_dir):
        cache_file = json_file.with_suffix(".cache")
        if not json_file.is_file() and not cache_file.is_file():
            continue

        img_path = dataset_dir / "images"
        if not img_path.is_dir():
            print(f"  ⚠️  跳过 {dir_name}：缺少 images/（json={json_file.name}）")
            continue

        entries.append({
            "img_path": str(img_path.resolve()),
            "json_file": str(json_file.resolve()),
        })
        print(f"  ✅ {dir_name}: json={json_file.name}, cache={'有' if cache_file.is_file() else '无'}")

    return entries


#----------------------------#
# 扫描 path 列表，写入 data/grounding/0-Grounding.yaml
#----------------------------#
def build_grounding_yaml(paths: list[str], out_path: Path) -> int:
    grounding_data: list[dict] = []
    used_dirs: list[str] = []

    for p in paths:
        root = Path(p).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"数据根目录不存在: {root}")

        for dir_name in sorted(d for d in os.listdir(root) if (root / d).is_dir()):
            if not is_grounding_dir(dir_name):
                continue
            entries = collect_grounding_entries(root, dir_name)
            if entries:
                grounding_data.extend(entries)
                used_dirs.append(dir_name)

    if not grounding_data:
        print("⚠️  未找到 GEOAI-*-GD 数据集，跳过 0-Grounding.yaml")
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = {"train": {"grounding_data": grounding_data}}

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("#【Grounding datasets】\n")
        f.write(f"# Dataset Used: {len(used_dirs)}\n")
        f.write(f"# Grounding entries: {len(grounding_data)}\n")
        f.write(f"# Date：{datetime.now()}\n")
        f.write("\n")
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"\n✅ 写入 {len(grounding_data)} 条 grounding_data → {out_path}")
    return len(grounding_data)


#----------------------------#
# 校验 YOLO 子数据集标准结构
#----------------------------#
def check_yolo_dataset_standard(dataset_dir: Path, dir_name: str) -> bool:
    ok = True
    if not (dataset_dir / "images").is_dir():
        print(f"  ⚠️  {dir_name} 不符合标准：缺少 images/")
        ok = False
    if not (dataset_dir / "train.txt").is_file():
        print(f"  ⚠️  {dir_name} 不符合标准：缺少 train.txt")
        ok = False
    jsons_dir = dataset_dir / JSONS_SEGMENT_DIR
    if not jsons_dir.is_dir():
        print(f"  ⚠️  {dir_name} 不符合标准：缺少 {JSONS_SEGMENT_DIR}/")
        ok = False
    elif not any(jsons_dir.glob("*.json")):
        print(f"  ⚠️  {dir_name} 不符合标准：{JSONS_SEGMENT_DIR}/ 下无 json 文件")
        ok = False
    return ok


#----------------------------#
# 从 jsons-segment 发现类别名
#----------------------------#
def discover_categories_from_jsons_segment(dataset_dir: Path, dir_name: str) -> list[str] | None:
    json_dir = dataset_dir / JSONS_SEGMENT_DIR
    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        return None

    categories: set[str] = set()
    iterator = json_files
    if len(json_files) > 500:
        from tqdm import tqdm
        iterator = tqdm(json_files, desc=f"  {dir_name} 扫描类别", leave=False)

    for json_path in iterator:
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        for shape in data.get("shapes", []):
            label = shape.get("label")
            if label and str(label).strip():
                categories.add(str(label).strip())

    if not categories:
        print(f"  ⚠️  {dir_name} 不符合标准：{JSONS_SEGMENT_DIR}/ 中未找到有效类别名（shapes[].label）")
        return None

    print(f"  📋 {dir_name}: 从 {JSONS_SEGMENT_DIR}/ 发现 {len(categories)} 个类别")
    return sorted(categories)


#----------------------------#
# 单个 GEOAI-*-YOLO 目录 → yolo 条目
#----------------------------#
def collect_yolo_entry(root: Path, dir_name: str) -> dict | None:
    dataset_dir = (root / dir_name).resolve()
    if not check_yolo_dataset_standard(dataset_dir, dir_name):
        return None

    train_txt = dataset_dir / "train.txt"
    val_txt = dataset_dir / "val.txt"
    categories = discover_categories_from_jsons_segment(dataset_dir, dir_name)
    if categories is None:
        return None

    entry = {
        "dir_name": dir_name,
        "path": str(dataset_dir),
        "train": str(train_txt.resolve()),
        "val": str(val_txt.resolve()) if val_txt.is_file() else None,
        "categories": categories,
    }
    print(f"  ✅ {dir_name}: train.txt{' + val.txt' if entry['val'] else ''}")
    return entry


#----------------------------#
# 合并多个子数据集的类别表
#----------------------------#
def merge_categories(yolo_entries: list[dict]) -> tuple[int, dict]:
    merged: list[str] = []
    seen: set[str] = set()
    for entry in yolo_entries:
        for name in entry["categories"]:
            if name not in seen:
                seen.add(name)
                merged.append(name)
    names = {i: name for i, name in enumerate(merged)}
    return len(merged), names


#----------------------------#
# 扫描 path 列表，写入 data/yolo/0-YOLO.yaml
#----------------------------#
def build_yolo_yaml(paths: list[str], out_path: Path) -> int:
    yolo_entries: list[dict] = []

    for p in paths:
        root = Path(p).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"数据根目录不存在: {root}")

        for dir_name in sorted(d for d in os.listdir(root) if (root / d).is_dir()):
            if not is_yolo_dir(dir_name):
                continue
            entry = collect_yolo_entry(root, dir_name)
            if entry:
                yolo_entries.append(entry)

    if not yolo_entries:
        print("⚠️  未找到符合标准的 GEOAI-*-YOLO 数据集，跳过 0-YOLO.yaml")
        print(f"    标准结构：images/ + train.txt + {JSONS_SEGMENT_DIR}/*.json（含 shapes[].label）")
        return 0

    nc, names = merge_categories(yolo_entries)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if len(yolo_entries) == 1:
        e = yolo_entries[0]
        data = {
            "path": e["path"],
            "train": "train.txt",
            "val": "val.txt" if e["val"] else None,
            "nc": nc,
            "names": names,
        }
        dataset_used = 1
    else:
        data = {
            "nc": nc,
            "names": names,
            "train": [e["train"] for e in yolo_entries],
            "val": [e["val"] for e in yolo_entries if e["val"]] or None,
        }
        dataset_used = len(yolo_entries)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("#【YOLO datasets】\n")
        f.write(f"# Dataset Used: {dataset_used}\n")
        f.write(f"# Categories: {nc}\n")
        f.write(f"# Date：{datetime.now()}\n")
        f.write("\n")
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"\n✅ 写入 {dataset_used} 个 YOLO 数据集（nc={nc}）→ {out_path}")
    return dataset_used


#----------------------------#
# 主流程
#----------------------------#
def main(paths: list[str], data_dir: Path):
    print("════════════════════════════════════════")
    print(" 扫描 GEOAI-*-GD → 0-Grounding.yaml")
    print("════════════════════════════════════════")
    build_grounding_yaml(paths, data_dir / "0-Grounding.yaml")

    print("\n════════════════════════════════════════")
    print(" 扫描 GEOAI-*-YOLO → 0-YOLO.yaml")
    print("════════════════════════════════════════")
    build_yolo_yaml(paths, data_dir / "0-YOLO.yaml")


if __name__ == "__main__":

# 修改 ------------------------------------------------------------------------------------------------------------------#
    path = [
        "/home/yulin/0-data/0-public/反光衣",
        "/home/yulin/0-data/0-public/自行车",

    ]
#------------------------------------------------------------------------------------------------------------------------#

    script_dir = Path(__file__).resolve().parent
    main(path, script_dir)
