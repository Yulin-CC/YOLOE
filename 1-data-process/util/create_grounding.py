"""
# @Author: 算法组 蔡雨霖
# @Date: 2026-06-16
# @Description: 将 grounding 源数据（images/ + jsons/*.json）合并为 COCO segm json。
#   支持每个 json 含多条 caption（images 多条）；合并时重排全局 image_id / annotation id。
#   file_name 统一为 images/ 下的 basename，供 generate_grounding_cache 使用：img_path / file_name
# @Command: bash 1-data-process/5-create_grounding.sh
"""
import argparse
import json
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


#-------------#
# 参数解析
#-------------#
def parse_args():
    parser = argparse.ArgumentParser(description="合并 grounding jsons → COCO segm json")
    parser.add_argument("--input",  type=str, default="testdir/qga",
                        help="数据集根目录（含 images/ 与 jsons/）")
    parser.add_argument("--project", type=str, default="gqa",
                        help="输出文件前缀，默认 {input}/{project}_segm.json")
    parser.add_argument("--output", type=str, default="",
                        help="输出 segm json 路径（默认 {input}/{project}_segm.json）")
    parser.add_argument("--images-dir", type=str, default="images",
                        help="图片子目录名")
    parser.add_argument("--jsons-dir",  type=str, default="jsons",
                        help="标注 json 子目录名")
    return parser.parse_args()


#-------------#
# 按 stem 查找 images/ 中的图片
#-------------#
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
# 单个 json → (images, annotations, meta)
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
        "categories": data.get("categories", []),
    }
    return images, data.get("annotations", []), meta


#-------------#
# 扫描 jsons/ 合并为 COCO segm json
#-------------#
def build_grounding_segm(
    input_root: Path,
    output_path: Path,
    images_dir_name: str = "images",
    jsons_dir_name: str = "jsons",
) -> tuple[int, int, int]:
    images_dir = input_root / images_dir_name
    jsons_dir  = input_root / jsons_dir_name

    if not images_dir.is_dir():
        raise FileNotFoundError(f"缺少图片目录: {images_dir}")
    if not jsons_dir.is_dir():
        raise FileNotFoundError(f"缺少标注目录: {jsons_dir}")

    json_files = sorted(jsons_dir.glob("*.json"))
    if not json_files:
        raise RuntimeError(f"未找到 json 文件: {jsons_dir}")

    images, annotations = [], []
    meta = {"info": [], "licenses": [], "categories": []}
    next_image_id = 0
    next_ann_id = 0
    skipped = 0

    for json_path in json_files:
        parsed = parse_coco_json(json_path, images_dir)
        if parsed is None:
            skipped += 1
            continue

        file_images, file_anns, file_meta = parsed
        if not meta["categories"] and file_meta.get("categories"):
            meta = file_meta

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
                "categories": meta.get("categories", []),
                "images": images,
                "annotations": annotations,
            },
            f,
            ensure_ascii=False,
        )

    return len(images), len(annotations), skipped


if __name__ == "__main__":
    args = parse_args()
    root = Path(args.input).resolve()
    out  = Path(args.output).resolve() if args.output else root / f"{args.project}_segm.json"

    print(f"Input  : {root}")
    print(f"Output : {out}")
    n_img, n_ann, n_skip = build_grounding_segm(root, out, args.images_dir, args.jsons_dir)
    print(f"✅ 写入 {n_img} 条 image 记录、{n_ann} 条 annotation → {out}")
    if n_skip:
        print(f"   跳过 {n_skip} 个 json")
    print(f"   训练时 img_path 指向: {root / args.images_dir}")
