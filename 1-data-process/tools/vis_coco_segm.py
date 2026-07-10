"""
# @Author: AI技术平台
# @Date: 2026-07-09
# @Description: 可视化 COCO segm json：叠加 polygon 分割、bbox 与类别名
# @Command: python 1-data-process/util/vis_coco_segm.py --json sample/bbox/bbox_coco_segm.json --img-path sample/bbox/images --output sample/bbox/vis
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm

PALETTE = [
    (255, 99, 71), (50, 205, 50), (30, 144, 255), (255, 215, 0),
    (238, 130, 238), (0, 206, 209), (255, 140, 0), (147, 112, 219),
]


#-------------#
# 参数解析
#-------------#
def parse_args():
    parser = argparse.ArgumentParser(description="可视化 COCO segm json")
    parser.add_argument("--json", type=str, required=True, help="COCO segm json 路径")
    parser.add_argument("--img-path", type=str, required=True, help="图片目录")
    parser.add_argument("--output", type=str, required=True, help="可视化输出目录")
    parser.add_argument("--limit", type=int, default=0, help="仅可视化前 N 张图（0=全部）")
    parser.add_argument("--max-size", type=int, default=1920, help="输出图最长边上限（0=不缩放）")
    parser.add_argument("--alpha", type=float, default=0.45, help="mask 透明度")
    return parser.parse_args()


def resize_if_needed(image: np.ndarray, max_size: int) -> tuple[np.ndarray, float]:
    if max_size <= 0:
        return image, 1.0
    h, w = image.shape[:2]
    scale = min(1.0, max_size / max(h, w))
    if scale >= 1.0:
        return image, 1.0
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA), scale


def draw_annotation(
    canvas: np.ndarray,
    overlay: np.ndarray,
    ann: dict,
    cat_name: str,
    color: tuple[int, int, int],
    scale: float,
    alpha: float,
):
    x, y, w, h = ann["bbox"]
    x, y, w, h = x * scale, y * scale, w * scale, h * scale
    x1, y1 = int(x), int(y)
    x2, y2 = int(x + w), int(y + h)

    for seg in ann.get("segmentation") or []:
        if not isinstance(seg, list) or len(seg) < 6:
            continue
        pts = np.array(seg, dtype=np.float32).reshape(-1, 2) * scale
        pts = pts.astype(np.int32)
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(canvas, [pts], True, color, 2, cv2.LINE_AA)

    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    label = cat_name
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    ty = max(y1 - 6, th + 4)
    cv2.rectangle(canvas, (x1, ty - th - 6), (x1 + tw + 4, ty + 2), color, -1)
    cv2.putText(canvas, label, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


def visualize(json_path: Path, img_path: Path, output_dir: Path, limit: int, max_size: int, alpha: float):
    coco = COCO(json_path)
    cat_id_to_name = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
    output_dir.mkdir(parents=True, exist_ok=True)

    img_ids = sorted(coco.getImgIds())
    if limit > 0:
        img_ids = img_ids[:limit]

    for img_id in tqdm(img_ids, desc="Visualizing"):
        img_info = coco.loadImgs(img_id)[0]
        image_file = img_path / img_info["file_name"]
        if not image_file.is_file():
            print(f"  skip missing: {image_file.name}")
            continue

        image = cv2.imread(str(image_file))
        if image is None:
            print(f"  skip unreadable: {image_file.name}")
            continue

        image, scale = resize_if_needed(image, max_size)
        canvas = image.copy()
        overlay = image.copy()

        ann_ids = coco.getAnnIds(imgIds=img_id)
        for ann in coco.loadAnns(ann_ids):
            cat_name = cat_id_to_name.get(ann["category_id"], str(ann["category_id"]))
            color = PALETTE[ann["category_id"] % len(PALETTE)]
            draw_annotation(canvas, overlay, ann, cat_name, color, scale, alpha)

        result = cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0)
        out_path = output_dir / f"{Path(img_info['file_name']).stem}_segm.jpg"
        cv2.imwrite(str(out_path), result)

    print(f"Saved {len(list(output_dir.glob('*.jpg')))} images -> {output_dir}")


if __name__ == "__main__":
    args = parse_args()
    visualize(
        json_path=Path(args.json).resolve(),
        img_path=Path(args.img_path).resolve(),
        output_dir=Path(args.output).resolve(),
        limit=args.limit,
        max_size=args.max_size,
        alpha=args.alpha,
    )
