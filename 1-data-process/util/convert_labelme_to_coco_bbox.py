"""
# @Author: AI技术平台
# @Date: 2026-07-09
# @Description: 将 LabelMe 单图 json（rectangle / rotation / polygon）合并为 COCO bbox json，供 generate_sam_masks.py 使用
# @Command: python 1-data-process/tools/labelme_to_coco_bbox.py --input /path/to/dataset --output /path/to/train_bbox.json
"""
import argparse
import json
from pathlib import Path

from PIL import Image
from tqdm import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
SUPPORTED_SHAPE_TYPES = {"rectangle", "rotation", "polygon"}


#-------------#
# 参数解析
#-------------#
def parse_args():
    parser = argparse.ArgumentParser(
        description="LabelMe jsons → COCO bbox json（供 generate_sam_masks.py 使用）"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="数据集根目录（含 images/ 与 jsons/）；与 --json-dir 二选一",
    )
    parser.add_argument(
        "--json-dir",
        type=str,
        default="",
        help="LabelMe json 目录；未指定时从 --input/jsons 读取",
    )
    parser.add_argument(
        "--img-path",
        type=str,
        default="",
        help="图片目录；未指定时从 --input/images 读取",
    )
    parser.add_argument("--output", type=str, required=True, help="输出 COCO bbox json 路径")
    parser.add_argument("--images-dir", type=str, default="images", help="--input 模式下图片子目录名")
    parser.add_argument("--jsons-dir", type=str, default="jsons", help="--input 模式下标注子目录名")
    parser.add_argument("--limit", type=int, default=0, help="仅处理前 N 个 json（调试用，0=全部）")
    return parser.parse_args()


#-------------#
# 路径解析
#-------------#
def resolve_paths(args) -> tuple[Path, Path]:
    if args.input:
        root = Path(args.input).resolve()
        json_dir = Path(args.json_dir).resolve() if args.json_dir else root / args.jsons_dir
        img_path = Path(args.img_path).resolve() if args.img_path else root / args.images_dir
    else:
        if not args.json_dir or not args.img_path:
            raise ValueError("请指定 --input，或同时指定 --json-dir 与 --img-path")
        json_dir = Path(args.json_dir).resolve()
        img_path = Path(args.img_path).resolve()

    if not json_dir.is_dir():
        raise FileNotFoundError(f"缺少标注目录: {json_dir}")
    if not img_path.is_dir():
        raise FileNotFoundError(f"缺少图片目录: {img_path}")
    return json_dir, img_path


def resolve_image(images_dir: Path, stem: str, hint_name: str | None) -> Path | None:
    if hint_name:
        candidate = images_dir / Path(hint_name).name
        if candidate.is_file():
            return candidate

    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return None


#-------------#
# LabelMe shape → COCO bbox
#-------------#
def points_to_bbox(points: list) -> list[float] | None:
    if not points:
        return None

    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    width = x_max - x_min
    height = y_max - y_min
    if width <= 0 or height <= 0:
        return None
    return [x_min, y_min, width, height]


def shape_to_annotation(shape: dict, ann_id: int, image_id: int, category_id: int) -> dict | None:
    if shape.get("difficult"):
        return None

    shape_type = shape.get("shape_type", "")
    points = shape.get("points", [])
    if shape_type not in SUPPORTED_SHAPE_TYPES:
        return None
    if shape_type in {"rectangle", "rotation"} and len(points) < 4:
        return None
    if shape_type == "polygon" and len(points) < 3:
        return None

    bbox = points_to_bbox(points)
    if bbox is None:
        return None

    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": bbox,
        "area": float(bbox[2] * bbox[3]),
        "iscrowd": 0,
        "segmentation": [],
    }


def is_labelme_json(data: dict) -> bool:
    return isinstance(data, dict) and "shapes" in data and ("imagePath" in data or "imageHeight" in data)


def parse_labelme_json(
    json_path: Path,
    images_dir: Path,
    next_image_id: int,
    next_ann_id: int,
    label_to_cat_id: dict[str, int],
) -> tuple[dict | None, list[dict], int, int]:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    if not is_labelme_json(data):
        print(f"  ⚠️  跳过（非 LabelMe 格式）: {json_path.name}")
        return None, [], next_image_id, next_ann_id

    image_file = resolve_image(images_dir, json_path.stem, data.get("imagePath"))
    if image_file is None:
        print(f"  ❌  找不到图片: {json_path.stem} → 跳过 {json_path.name}")
        return None, [], next_image_id, next_ann_id

    width = int(data.get("imageWidth") or 0)
    height = int(data.get("imageHeight") or 0)
    if width <= 0 or height <= 0:
        with Image.open(image_file) as img:
            width, height = img.size

    image = {
        "id": next_image_id,
        "file_name": image_file.name,
        "width": width,
        "height": height,
    }
    next_image_id += 1

    annotations = []
    for shape in data.get("shapes", []):
        label = (shape.get("label") or "").strip()
        if not label:
            continue
        if label not in label_to_cat_id:
            label_to_cat_id[label] = len(label_to_cat_id)
        ann = shape_to_annotation(shape, next_ann_id, image["id"], label_to_cat_id[label])
        if ann is None:
            continue
        annotations.append(ann)
        next_ann_id += 1

    return image, annotations, next_image_id, next_ann_id


#-------------#
# 主流程
#-------------#
def build_coco_bbox_json(
    json_dir: Path,
    images_dir: Path,
    output_path: Path,
    limit: int = 0,
) -> tuple[int, int, int]:
    json_files = sorted(json_dir.glob("*.json"))
    if limit > 0:
        json_files = json_files[:limit]
    if not json_files:
        raise RuntimeError(f"未找到 json 文件: {json_dir}")

    images: list[dict] = []
    annotations: list[dict] = []
    label_to_cat_id: dict[str, int] = {}
    next_image_id = 0
    next_ann_id = 0
    skipped = 0

    for json_path in tqdm(json_files, desc="LabelMe → COCO bbox"):
        image, anns, next_image_id, next_ann_id = parse_labelme_json(
            json_path, images_dir, next_image_id, next_ann_id, label_to_cat_id
        )
        if image is None:
            skipped += 1
            continue
        images.append(image)
        annotations.extend(anns)

    if not images:
        raise RuntimeError(f"未生成任何有效样本: {json_dir}")

    categories = [
        {"id": cat_id, "name": name, "supercategory": "object"}
        for name, cat_id in sorted(label_to_cat_id.items(), key=lambda x: x[1])
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "info": {"description": "Converted from LabelMe jsons"},
                "licenses": [],
                "categories": categories,
                "images": images,
                "annotations": annotations,
            },
            f,
            ensure_ascii=False,
        )

    return len(images), len(annotations), skipped


if __name__ == "__main__":
    args = parse_args()
    json_dir, images_dir = resolve_paths(args)
    output_path = Path(args.output).resolve()

    print(f"Json dir : {json_dir}")
    print(f"Img path : {images_dir}")
    print(f"Output   : {output_path}")

    n_img, n_ann, n_skip = build_coco_bbox_json(json_dir, images_dir, output_path, args.limit)
    print(f"✅ 写入 {n_img} 张图、{n_ann} 条 bbox annotation → {output_path}")
    if n_skip:
        print(f"   跳过 {n_skip} 个 json")
    print(f"   下一步: python tools/generate_sam_masks.py --img-path {images_dir} --json-path {output_path}")
