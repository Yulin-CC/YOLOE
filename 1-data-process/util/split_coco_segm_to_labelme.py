"""
# @Author: 算法组 蔡雨霖
# @Date: 2026-07-07
# @Description: 将 COCO segm json 按图像拆分为 LabelMe jsons-segment/*.json，供 create_segment_yolo.py 使用
# @Command: python 1-data-process/util/split_coco_segm_to_labelme.py --input /path/to/objects365_train_segm.json --output /path/to/jsons-segment
"""
import argparse
import json
from pathlib import Path

from pycocotools.coco import COCO
from tqdm import tqdm

LABELME_VERSION = "0.1.4"


#----------------------------#
# 参数解析
#----------------------------#
def parse_args():
    parser = argparse.ArgumentParser(
        description="拆分 COCO segm json → jsons-segment/*.json（LabelMe 格式）"
    )
    parser.add_argument("--input", type=str, required=True, help="合并后的 COCO segm json 路径")
    parser.add_argument("--output", type=str, required=True, help="输出 jsons-segment 目录")
    parser.add_argument("--skip-existing", action="store_true", help="跳过已存在的 json")
    parser.add_argument("--limit", type=int, default=0, help="仅处理前 N 张图（调试用，0=全部）")
    return parser.parse_args()


#----------------------------#
# COCO polygon → LabelMe points
#----------------------------#
def segmentation_to_polygons(segmentation) -> list[list[list[float]]]:
    if segmentation is None:
        return []
    if isinstance(segmentation, dict):
        return []
    polygons: list[list[list[float]]] = []
    for seg in segmentation:
        if not isinstance(seg, list) or len(seg) < 6:
            continue
        points = [[float(seg[i]), float(seg[i + 1])] for i in range(0, len(seg), 2)]
        if len(points) >= 3:
            polygons.append(points)
    return polygons


def ann_to_shapes(ann: dict, cat_id_to_name: dict[int, str]) -> list[dict]:
    if ann.get("iscrowd"):
        return []

    label = cat_id_to_name.get(ann["category_id"])
    if not label:
        return []

    shapes = []
    for points in segmentation_to_polygons(ann.get("segmentation")):
        shapes.append({
            "label": label,
            "points": points,
            "group_id": None,
            "description": "",
            "shape_type": "polygon",
            "flags": {},
            "mask": None,
        })
    return shapes


def build_labelme_json(image: dict, shapes: list[dict]) -> dict:
    file_name = Path(image["file_name"]).name
    return {
        "version": LABELME_VERSION,
        "flags": {},
        "shapes": shapes,
        "imagePath": file_name,
        "imageData": None,
        "imageHeight": int(image["height"]),
        "imageWidth": int(image["width"]),
    }


#----------------------------#
# 主流程
#----------------------------#
def split_coco_segm_to_labelme(
    input_path: Path,
    output_dir: Path,
    skip_existing: bool = False,
    limit: int = 0,
) -> tuple[int, int]:
    print(f"Loading COCO index: {input_path}")
    coco = COCO(str(input_path))
    cat_id_to_name = {cat["id"]: cat["name"] for cat in coco.loadCats(coco.getCatIds())}

    img_ids = coco.getImgIds()
    if limit > 0:
        img_ids = img_ids[:limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0

    for img_id in tqdm(img_ids, desc="Splitting to LabelMe"):
        image = coco.loadImgs(img_id)[0]
        stem = Path(image["file_name"]).stem
        out_path = output_dir / f"{stem}.json"
        if skip_existing and out_path.is_file():
            skipped += 1
            continue

        shapes: list[dict] = []
        seen: set[tuple] = set()
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id)):
            for shape in ann_to_shapes(ann, cat_id_to_name):
                key = (shape["label"], tuple(tuple(p) for p in shape["points"]))
                if key in seen:
                    continue
                seen.add(key)
                shapes.append(shape)

        labelme = build_labelme_json(image, shapes)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(labelme, f, ensure_ascii=False, separators=(",", ":"))

        written += 1

    return written, skipped


if __name__ == "__main__":
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    print(f"Input  : {input_path}")
    print(f"Output : {output_dir}")
    written, skipped = split_coco_segm_to_labelme(
        input_path, output_dir, skip_existing=args.skip_existing, limit=args.limit
    )
    print(f"✅ 拆分完成：写入 {written} 个 json，跳过 {skipped} 个 → {output_dir}")
