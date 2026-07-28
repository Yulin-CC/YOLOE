"""
# @Author: 算法组 蔡雨霖
# @Date: 2026-07-13
# @Description: 将 grounding 源数据（images/ + jsons/*.json）合并为 COCO segm json。
#   支持两种 per-json 格式：
#   1) GEOAI data_engine LabelMe（shapes + description_en + negative_labels）
#   2) 已有 COCO Grounding（images + annotations，透传合并）
#   rectangle 仅写 bbox；polygon 写 segmentation；label / negative_labels 保留在输出中。
# @Command: python utils/convert_geoai2coco.py --input /path/to/dataset --project geoai
"""
import argparse
import json
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
CAPTION_SEP = " . "
DEFAULT_CATEGORIES = [{"id": 1, "name": "object", "supercategory": "object"}]


#-------------#
# 参数解析
#-------------#
def parse_args():
    parser = argparse.ArgumentParser(description="合并 grounding jsons → COCO segm json（GEOAI / Flickr 双格式）")
    parser.add_argument("--input", type=str, default="testdir/geoai", help="数据集根目录（含 images/ 与 jsons/）")
    parser.add_argument("--project", type=str, default="geoai", help="输出文件前缀，默认 {input}/{project}_segm.json")
    parser.add_argument("--output", type=str, default="", help="输出 segm json 路径（默认 {input}/{project}_segm.json）")
    parser.add_argument("--images-dir", type=str, default="images", help="图片子目录名")
    parser.add_argument("--jsons-dir", type=str, default="jsons", help="标注 json 子目录名")
    parser.add_argument("--dataset-name", type=str, default="geoai", help="写入 images[].dataset_name")
    return parser.parse_args()


#-------------#
# 按 stem 查找 images/ 中的图片
#-------------#
def normalize_hint_name(hint_name: str | None) -> str | None:
    if not hint_name:
        return None
    return Path(hint_name.replace("\\", "/")).name


def resolve_image(images_dir: Path, stem: str, hint_name: str | None) -> Path | None:
    if hint_name:
        candidate = images_dir / normalize_hint_name(hint_name)
        if candidate.is_file():
            return candidate

    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return None


#-------------#
# 合成 caption 与 description_en → tokens_positive 区间
#-------------#
def build_caption_and_spans(prompts: list[str]) -> tuple[str, dict[str, tuple[int, int]]]:
    parts: list[str] = []
    spans: dict[str, tuple[int, int]] = {}
    offset = 0
    for i, prompt in enumerate(prompts):
        if i > 0:
            offset += len(CAPTION_SEP)
        start = offset
        parts.append(prompt)
        offset += len(prompt)
        spans[prompt] = (start, offset)
    return CAPTION_SEP.join(parts), spans


#-------------#
# points → bbox；polygon → segmentation
#-------------#
def shape_to_bbox(points: list[list[float]]) -> list[float] | None:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    bw, bh = xmax - xmin, ymax - ymin
    if bw <= 0 or bh <= 0:
        return None
    return [float(xmin), float(ymin), float(bw), float(bh)]


def shape_to_segmentation(shape: dict) -> list[list[float]] | None:
    shape_type = (shape.get("shape_type") or "").lower()
    points = shape.get("points") or []
    if shape_type != "polygon" or len(points) < 3:
        return None
    return [[float(c) for pt in points for c in pt]]


INVALID_DESCRIPTION_EN = {"ai generated, pending review"}


def get_caption_text(shape: dict) -> str | None:
    text = (shape.get("description_en") or "").strip()
    if not text or text.lower() in INVALID_DESCRIPTION_EN:
        return None
    return text


def is_labelme_engine_json(data: dict) -> bool:
    return bool(data.get("shapes")) and "imageWidth" in data and "imageHeight" in data


#-------------#
# GEOAI data_engine LabelMe → (images, annotations)
#-------------#
def parse_labelme_engine_json(
    json_path: Path,
    images_dir: Path,
    dataset_name: str,
) -> tuple[list[dict], list[dict], dict] | None:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    image_file = resolve_image(images_dir, json_path.stem, data.get("imagePath"))
    if image_file is None:
        print(f"  ❌  找不到图片: {json_path.stem} → 跳过 {json_path.name}")
        return None

    width = int(data["imageWidth"])
    height = int(data["imageHeight"])
    shapes = data.get("shapes") or []

    valid_shapes: list[tuple[dict, str, list[float]]] = []
    for shape in shapes:
        caption_text = get_caption_text(shape)
        if not caption_text:
            continue
        bbox = shape_to_bbox(shape.get("points") or [])
        if bbox is None:
            continue
        valid_shapes.append((shape, caption_text, bbox))

    if not valid_shapes:
        print(f"  ⚠️  跳过（无有效 description_en 框）: {json_path.name}")
        return None

    prompts_ordered: list[str] = []
    seen: set[str] = set()
    for _, prompt, _ in valid_shapes:
        if prompt not in seen:
            seen.add(prompt)
            prompts_ordered.append(prompt)

    caption, spans = build_caption_and_spans(prompts_ordered)
    metadata = data.get("metadata") or {}
    negative_labels = data.get("negative_labels") or []

    image = {
        "id": 0,
        "file_name": image_file.name,
        "height": height,
        "width": width,
        "caption": caption,
        "tokens_negative": [],
        "negative_labels": negative_labels,
        "dataset_name": dataset_name,
        "source_stem": json_path.stem,
        "ontology_version": metadata.get("ontology_version", ""),
        "media_id": metadata.get("media_id", ""),
    }

    annotations: list[dict] = []
    for shape, prompt, bbox in valid_shapes:
        start, end = spans[prompt]
        attrs = shape.get("attributes") or {}
        ann: dict = {
            "id": len(annotations),
            "image_id": 0,
            "category_id": 1,
            "bbox": bbox,
            "tokens_positive": [[start, end]],
            "label": shape.get("label", ""),
            "description": shape.get("description", ""),
            "description_en": shape.get("description_en", ""),
            "canonical_id": attrs.get("canonical_id", ""),
            "iscrowd": 0,
            "area": float(bbox[2] * bbox[3]),
        }
        segm = shape_to_segmentation(shape)
        if segm is not None:
            ann["segmentation"] = segm
        annotations.append(ann)

    meta = {
        "info": [],
        "licenses": [],
        "categories": DEFAULT_CATEGORIES,
    }
    return [image], annotations, meta


#-------------#
# 已有 COCO Grounding json → 透传合并
#-------------#
def parse_coco_json(json_path: Path, images_dir: Path) -> tuple[list[dict], list[dict], dict] | None:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    file_images = data.get("images", [])
    if not file_images:
        print(f"  ⚠️  跳过（无 images）: {json_path.name}")
        return None

    image_file = resolve_image(images_dir, json_path.stem, file_images[0].get("file_name"))
    if image_file is None:
        print(f"  ❌  找不到图片: {json_path.stem} → 跳过 {json_path.name}")
        return None

    images = []
    for img in file_images:
        new_img = dict(img)
        if new_img.get("file_name") != image_file.name:
            print(f"  🔧 {json_path.name} file_name: {new_img.get('file_name')!r} → {image_file.name!r}")
        new_img["file_name"] = image_file.name
        images.append(new_img)

    meta = {
        "info": data.get("info", []),
        "licenses": data.get("licenses", []),
        "categories": data.get("categories", []) or DEFAULT_CATEGORIES,
    }
    return images, data.get("annotations", []), meta


#-------------#
# 单 json 路由
#-------------#
def parse_json_file(
    json_path: Path,
    images_dir: Path,
    dataset_name: str,
) -> tuple[list[dict], list[dict], dict] | None:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    if is_labelme_engine_json(data):
        return parse_labelme_engine_json(json_path, images_dir, dataset_name)
    return parse_coco_json(json_path, images_dir)


#-------------#
# 扫描 jsons/ 合并为 COCO segm json
#-------------#
def build_grounding_segm(
    input_root: Path,
    output_path: Path,
    images_dir_name: str = "images",
    jsons_dir_name: str = "jsons",
    dataset_name: str = "geoai",
) -> tuple[int, int, int, int]:
    images_dir = input_root / images_dir_name
    jsons_dir = input_root / jsons_dir_name

    if not images_dir.is_dir():
        raise FileNotFoundError(f"缺少图片目录: {images_dir}")
    if not jsons_dir.is_dir():
        raise FileNotFoundError(f"缺少标注目录: {jsons_dir}")

    json_files = sorted(jsons_dir.glob("*.json"))
    if not json_files:
        raise RuntimeError(f"未找到 json 文件: {jsons_dir}")

    images, annotations = [], []
    meta = {"info": [], "licenses": [], "categories": DEFAULT_CATEGORIES}
    next_image_id = 0
    next_ann_id = 0
    skipped = 0
    labelme_count = 0

    for json_path in json_files:
        with open(json_path, encoding="utf-8") as f:
            raw = json.load(f)
        is_labelme = is_labelme_engine_json(raw)

        parsed = parse_json_file(json_path, images_dir, dataset_name)
        if parsed is None:
            skipped += 1
            continue
        if is_labelme:
            labelme_count += 1

        file_images, file_anns, file_meta = parsed
        if not meta["categories"] or meta["categories"] == DEFAULT_CATEGORIES:
            if file_meta.get("categories"):
                meta["categories"] = file_meta["categories"]

        old_to_new: dict[int, int] = {}
        for img in file_images:
            old_id = img["id"]
            new_img = dict(img)
            new_img["id"] = next_image_id
            old_to_new[old_id] = next_image_id
            images.append(new_img)
            next_image_id += 1

        for ann in file_anns:
            old_img_id = ann["image_id"]
            if old_img_id not in old_to_new:
                continue
            new_ann = dict(ann)
            new_ann["id"] = next_ann_id
            new_ann["image_id"] = old_to_new[old_img_id]
            annotations.append(new_ann)
            next_ann_id += 1

    if not images:
        raise RuntimeError(f"未生成任何有效样本: {jsons_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "info": meta.get("info", []),
                "licenses": meta.get("licenses", []),
                "categories": meta.get("categories", DEFAULT_CATEGORIES),
                "images": images,
                "annotations": annotations,
            },
            f,
            ensure_ascii=False,
        )

    return len(images), len(annotations), skipped, labelme_count


if __name__ == "__main__":
    args = parse_args()
    root = Path(args.input).resolve()
    out = Path(args.output).resolve() if args.output else root / f"{args.project}_segm.json"

    print(f"Input  : {root}")
    print(f"Output : {out}")
    n_img, n_ann, n_skip, n_labelme = build_grounding_segm(
        root, out, args.images_dir, args.jsons_dir, args.dataset_name
    )
    print(f"✅ 写入 {n_img} 条 image 记录、{n_ann} 条 annotation → {out}")
    print(f"   其中 GEOAI LabelMe 源 json: {n_labelme} 个")
    if n_skip:
        print(f"   跳过 {n_skip} 个 json")
    print(f"   训练时 img_path 指向: {root / args.images_dir}")
    print("   提示: rectangle 仅含 bbox，请用 generate_sam_masks.py 补 segmentation 后再 generate_grounding_cache")
