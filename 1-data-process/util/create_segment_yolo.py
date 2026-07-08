"""
# @Author: 蔡雨霖 (图像算法组)
# @Date: 2026-06-16
# @Description: 将 jsons-segment（LabelMe 多边形）转为 YOLO 分割 labels/*.txt，并生成 train.txt / val.txt
# @Command: python util/create_segment_yolo.py --path ../testdir/GEOAI-person --split_ratio 0.9
"""

import argparse
import json
import random
from pathlib import Path

from PIL import Image
from tqdm import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
LABELGEAI_VERSION = "0.1.4"
JSON_DIR = "jsons-segment"


#-------------#
# 参数解析
#-------------#
def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert jsons-segment LabelMe JSON to YOLO segment labels and train/val index files."
    )
    parser.add_argument("--path", required=True, help="Dataset root (images/ + jsons-segment/)")
    parser.add_argument("--split_ratio", type=float, default=0.9, help="Train split ratio")
    parser.add_argument("--part_ratio", type=float, default=1.0, help="Fraction of images to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    parser.add_argument("--names", default="", help="Category names, comma-separated; auto if empty")
    return parser.parse_args()


#-------------#
# 数据集扫描与划分
#-------------#
def collect_images(images_dir: Path) -> list[str]:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing images directory: {images_dir}")

    images = sorted(
        p.name for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise RuntimeError(f"No images found in {images_dir}")
    return images


def split_dataset(images: list[str], split_ratio: float, part_ratio: float, seed: int):
    rng = random.Random(seed)
    shuffled = images.copy()
    rng.shuffle(shuffled)

    part_count = max(1, int(len(shuffled) * part_ratio))
    subset = shuffled[:part_count]

    split_index = max(1, int(len(subset) * split_ratio)) if len(subset) > 1 else 1
    if split_index >= len(subset):
        split_index = len(subset) - 1

    train_set = subset[:split_index]
    val_set = subset[split_index:]
    if not val_set:
        val_set = [train_set.pop()]
    return train_set, val_set


def load_label_json(json_path: Path) -> dict | None:
    if not json_path.is_file():
        return None
    with open(json_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def create_blank_label_json(image_path: Path, output_path: Path) -> dict:
    with Image.open(image_path) as img:
        width, height = img.size

    label_data = {
        "version": LABELGEAI_VERSION,
        "flags": {},
        "shapes": [],
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(label_data, fh, ensure_ascii=False, separators=(",", ":"))

    return label_data


def ensure_label_files(data_root: Path, image_names: list[str]) -> int:
    images_dir = data_root / "images"
    json_dir = data_root / JSON_DIR
    json_dir.mkdir(parents=True, exist_ok=True)

    created = 0
    for image_name in image_names:
        image_path = images_dir / image_name
        json_path = json_dir / f"{Path(image_name).stem}.json"
        if json_path.is_file():
            continue
        if not image_path.is_file():
            print(f"Warning: image missing, skip blank label: {image_path}")
            continue
        create_blank_label_json(image_path, json_path)
        created += 1

    return created


def resolve_image_size(image_path: Path, label_data: dict | None) -> tuple[int, int]:
    if label_data:
        width = int(label_data.get("imageWidth", 0))
        height = int(label_data.get("imageHeight", 0))
        if width > 0 and height > 0:
            return width, height

    with Image.open(image_path) as img:
        width, height = img.size
    return width, height


def discover_categories(data_root: Path, image_names: list[str]) -> list[str]:
    categories = set()
    json_dir = data_root / JSON_DIR

    for image_name in image_names:
        stem = Path(image_name).stem
        label_data = load_label_json(json_dir / f"{stem}.json")
        if not label_data:
            continue
        for shape in label_data.get("shapes", []):
            label = shape.get("label")
            if label:
                categories.add(label)

    if not categories:
        raise RuntimeError(f"No categories found under {json_dir}")
    return sorted(categories)


def parse_category_names(names_arg: str, discovered: list[str]) -> list[str]:
    if not names_arg.strip():
        return discovered
    names = [name.strip() for name in names_arg.split(",") if name.strip()]
    unknown = sorted(set(names) - set(discovered))
    if unknown:
        print(f"Warning: names not found in labels: {unknown}")
    return names


#-------------#
# LabelMe shape → YOLO segment 行
#-------------#
def _normalize_points(points: list, width: int, height: int) -> list[float]:
    coords = []
    for pt in points:
        x = max(0.0, min(1.0, float(pt[0]) / width))
        y = max(0.0, min(1.0, float(pt[1]) / height))
        coords.extend([x, y])
    return coords


def shape_to_yolo_line(shape: dict, name_to_id: dict, width: int, height: int) -> str | None:
    label = shape.get("label")
    if not label or label not in name_to_id:
        return None
    if shape.get("difficult"):
        return None

    shape_type = shape.get("shape_type")
    points = shape.get("points", [])

    if shape_type == "polygon" and len(points) >= 3:
        coords = _normalize_points(points, width, height)
    elif shape_type == "rectangle" and len(points) >= 4:
        coords = _normalize_points(points, width, height)
    else:
        return None

    if len(coords) < 6:
        return None

    cls_id = name_to_id[label]
    return f"{cls_id} " + " ".join(f"{v:.6g}" for v in coords)


def convert_json_to_yolo_txt(
    json_path: Path,
    label_path: Path,
    name_to_id: dict,
    image_path: Path,
) -> int:
    label_data = load_label_json(json_path)
    if label_data is None:
        width, height = resolve_image_size(image_path, None)
        shapes = []
    else:
        width, height = resolve_image_size(image_path, label_data)
        shapes = label_data.get("shapes", [])

    lines = []
    for shape in shapes:
        line = shape_to_yolo_line(shape, name_to_id, width, height)
        if line:
            lines.append(line)

    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def write_split_txt(image_names: list[str], images_dir: Path, output_path: Path):
    lines = [str((images_dir / name).resolve()) for name in image_names]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


#-------------#
# 主流程
#-------------#
def create_yolo_segment_labels(
    data_root: Path,
    split_ratio: float = 0.9,
    part_ratio: float = 1.0,
    seed: int = 42,
    names: str = "",
):
    images_dir = data_root / "images"
    json_dir = data_root / JSON_DIR
    labels_dir = data_root / "labels"

    images = collect_images(images_dir)
    created_labels = ensure_label_files(data_root, images)
    train_set, val_set = split_dataset(images, split_ratio, part_ratio, seed)

    all_used = train_set + val_set
    discovered = discover_categories(data_root, all_used)
    category_names = parse_category_names(names, discovered)
    name_to_id = {name: idx for idx, name in enumerate(category_names)}

    print(f"Dataset root : {data_root}")
    print(f"Json dir     : {JSON_DIR}/")
    print(f"Images total : {len(images)}")
    if created_labels:
        print(f"Blank labels : {created_labels} (created in {JSON_DIR}/)")
    print(f"Train / Val  : {len(train_set)} / {len(val_set)}")
    print(f"Categories   : {category_names}")

    ann_total = 0
    for image_name in tqdm(images, desc="Converting YOLO segment labels"):
        stem = Path(image_name).stem
        json_path = json_dir / f"{stem}.json"
        label_path = labels_dir / f"{stem}.txt"
        image_path = images_dir / image_name
        ann_total += convert_json_to_yolo_txt(json_path, label_path, name_to_id, image_path)

    write_split_txt(train_set, images_dir, data_root / "train.txt")
    write_split_txt(val_set, images_dir, data_root / "val.txt")

    print(f"Saved labels : {labels_dir}/ ({len(list(labels_dir.glob('*.txt')))} files, {ann_total} instances)")
    print(f"Saved train  : {data_root / 'train.txt'} ({len(train_set)} images)")
    print(f"Saved val    : {data_root / 'val.txt'} ({len(val_set)} images)")


if __name__ == "__main__":
    args = parse_args()
    create_yolo_segment_labels(
        data_root=Path(args.path).resolve(),
        split_ratio=args.split_ratio,
        part_ratio=args.part_ratio,
        seed=args.seed,
        names=args.names,
    )
