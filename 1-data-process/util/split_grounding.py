"""
# @Author: 算法组
# @Date: 2026-07-02
# @Description: 将 COCO grounding segm json 按图像拆分为 jsons/*.json，与 images/ 一一对应。
# @Command: python 1-data-process/util/split_grounding.py --input /path/to/flickr_train_segm.json --output /path/to/jsons
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


#-------------#
# 参数解析
#-------------#
def parse_args():
    parser = argparse.ArgumentParser(description="拆分 grounding segm json → jsons/*.json")
    parser.add_argument("--input", type=str, required=True, help="合并后的 segm json 路径")
    parser.add_argument("--output", type=str, required=True, help="输出 jsons 目录")
    return parser.parse_args()


#-------------#
# caption 排序：Flickr 用 sentence_id，GQA 等用全局 id
#-------------#
def image_sort_key(img: dict) -> tuple:
    if "sentence_id" in img:
        return (0, img["sentence_id"])
    return (1, img["id"])


#-------------#
# 按 file_name 分组并重映射 id
#-------------#
def split_grounding_segm(input_path: Path, output_dir: Path) -> int:
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    meta = {
        "info": data.get("info", []),
        "licenses": data.get("licenses", []),
        "categories": data.get("categories", []),
    }

    groups: dict[str, list] = defaultdict(list)
    for img in data["images"]:
        groups[img["file_name"]].append(img)

    anns_by_image_id: dict[int, list] = defaultdict(list)
    for ann in data["annotations"]:
        anns_by_image_id[ann["image_id"]].append(ann)

    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    for file_name in sorted(groups):
        imgs = sorted(groups[file_name], key=image_sort_key)
        old_to_new = {img["id"]: i for i, img in enumerate(imgs)}

        new_images = []
        for i, img in enumerate(imgs):
            new_img = dict(img)
            new_img["id"] = i
            new_images.append(new_img)

        new_annotations = []
        ann_id = 0
        for img in imgs:
            for ann in anns_by_image_id.get(img["id"], []):
                new_ann = dict(ann)
                new_ann["id"] = ann_id
                new_ann["image_id"] = old_to_new[ann["image_id"]]
                new_annotations.append(new_ann)
                ann_id += 1

        stem = Path(file_name).stem
        out_path = output_dir / f"{stem}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({**meta, "images": new_images, "annotations": new_annotations}, f, ensure_ascii=False)
        written += 1

    return written


if __name__ == "__main__":
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()

    print(f"Input  : {input_path}")
    print(f"Output : {output_dir}")
    n = split_grounding_segm(input_path, output_dir)
    print(f"✅ 拆分完成：{n} 个 json → {output_dir}")
