"""
# @Author: 算法组 蔡雨霖
# @Date: 2026-10-28
# @Description: jsons-GD → YOLOE val：labels/*.txt + val.txt + labels.cache
#   短语取 caption[tokens_positive]，全局去重后当 class id；一图多 caption 合并进同一 txt
# @Command: bash 1-data-process/3-create_valset.sh
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image
from tqdm import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
JSONS_CANDIDATES = ("jsons-GD", "jsons")


#-------------#
# 参数解析
#-------------#
def parse_args():
    parser = argparse.ArgumentParser(description="jsons-GD → labels/ + val.txt + labels.cache")
    parser.add_argument("--path", required=True, help="(VAL) 数据集根目录（含 images/ 与 jsons-GD/）")
    return parser.parse_args()


def slice_phrase(caption: str, tokens_positive) -> str:
    return " ".join(str(caption)[int(t[0]) : int(t[1])] for t in (tokens_positive or [])).lower().strip()


def resolve_jsons_dir(root: Path) -> Path:
    for name in JSONS_CANDIDATES:
        d = root / name
        if d.is_dir() and any(d.glob("*.json")):
            return d
    raise FileNotFoundError(f"缺少 jsons-GD/（或 jsons/）: {root}")


def collect_images(images_dir: Path) -> list[str]:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"缺少 images/: {images_dir}")
    images = sorted(
        p.name for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise RuntimeError(f"images/ 为空: {images_dir}")
    return images


def image_size(images_dir: Path, file_name: str, hint: dict) -> tuple[int, int]:
    w = int(hint.get("width") or 0)
    h = int(hint.get("height") or 0)
    if w > 0 and h > 0:
        return w, h
    with Image.open(images_dir / file_name) as im:
        return im.size


def bbox_to_poly(bbox) -> list[float]:
    x, y, w, h = (float(v) for v in bbox)
    return [x, y, x + w, y, x + w, y + h, x, y + h]


def first_polygon(ann: dict) -> list[float] | None:
    segm = ann.get("segmentation")
    if isinstance(segm, list) and segm:
        if isinstance(segm[0], list):
            poly = segm[0]
        else:
            poly = segm
        if len(poly) >= 6:
            return [float(v) for v in poly]
    bbox = ann.get("bbox")
    if bbox and len(bbox) >= 4:
        return bbox_to_poly(bbox)
    return None


def normalize_poly(poly: list[float], width: int, height: int) -> list[float]:
    coords = []
    for i in range(0, len(poly) - 1, 2):
        x = max(0.0, min(1.0, poly[i] / width))
        y = max(0.0, min(1.0, poly[i + 1] / height))
        coords.extend([x, y])
    return coords


def load_records(root: Path) -> tuple[dict[str, list[dict]], list[str]]:
    images_dir = root / "images"
    jsons_dir = resolve_jsons_dir(root)
    per_image: dict[str, list[dict]] = defaultdict(list)

    for json_path in sorted(jsons_dir.glob("*.json")):
        with open(json_path, encoding="utf-8") as fh:
            data = json.load(fh)
        id_to_img = {img["id"]: img for img in data.get("images", [])}
        if not id_to_img:
            continue
        file_name = Path(next(iter(id_to_img.values())).get("file_name") or json_path.stem).name
        if not (images_dir / file_name).is_file():
            for ext in IMAGE_EXTENSIONS:
                cand = images_dir / f"{json_path.stem}{ext}"
                if cand.is_file():
                    file_name = cand.name
                    break
        width, height = image_size(images_dir, file_name, next(iter(id_to_img.values())))
        seen = set()
        for ann in data.get("annotations", []):
            img = id_to_img.get(ann.get("image_id"))
            if not img:
                continue
            phrase = slice_phrase(img.get("caption", ""), ann.get("tokens_positive"))
            if not phrase:
                continue
            poly = first_polygon(ann)
            if not poly:
                continue
            raw_bbox = [float(v) for v in (ann.get("bbox") or [0, 0, 0, 0])[:4]]
            key = (phrase, tuple(round(v, 1) for v in raw_bbox))
            if key in seen:
                continue
            seen.add(key)
            per_image[file_name].append({
                "phrase": phrase,
                "poly": poly,
                "bbox": raw_bbox,
                "width": width,
                "height": height,
                "iscrowd": int(ann.get("iscrowd") or 0),
            })

    phrases = sorted({rec["phrase"] for recs in per_image.values() for rec in recs})
    if not phrases:
        raise RuntimeError(f"{jsons_dir} 中未解析到短语")
    return per_image, phrases


def write_labels(root: Path, image_names: list[str], per_image: dict, name_to_id: dict) -> int:
    labels_dir = root / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    n_inst = 0
    for name in image_names:
        recs = per_image.get(name, [])
        lines = []
        for rec in recs:
            cls = name_to_id[rec["phrase"]]
            coords = normalize_poly(rec["poly"], rec["width"], rec["height"])
            if len(coords) < 6:
                continue
            lines.append(f"{cls} " + " ".join(f"{v:.6g}" for v in coords))
            n_inst += 1
        (labels_dir / f"{Path(name).stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
    return n_inst


def write_val_txt(root: Path, image_names: list[str]):
    lines = [f"./images/{name}" for name in image_names]
    (root / "val.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_lvis_json(root: Path, image_names: list[str], per_image: dict, phrases: list[str]):
    """与 labels class id 对齐的评测 GT；缺了 geoai eval 读不到官方 AP。"""
    counts = defaultdict(int)
    for recs in per_image.values():
        for rec in recs:
            counts[rec["phrase"]] += 1

    def frequency(n: int) -> str:
        if n < 10:
            return "r"
        if n < 100:
            return "c"
        return "f"

    categories = [
        {"id": i + 1, "name": name, "frequency": frequency(counts[name])}
        for i, name in enumerate(phrases)
    ]
    images = []
    annotations = []
    ann_id = 1
    name_to_id = {n: i for i, n in enumerate(phrases)}
    for img_id, name in enumerate(image_names):
        recs = per_image.get(name, [])
        if recs:
            w, h = recs[0]["width"], recs[0]["height"]
        else:
            with Image.open(root / "images" / name) as im:
                w, h = im.size
        images.append({"id": img_id, "file_name": name, "width": w, "height": h})
        for rec in recs:
            bbox = rec["bbox"]
            if len(bbox) < 4:
                xs, ys = rec["poly"][0::2], rec["poly"][1::2]
                x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
                bbox = [x0, y0, x1 - x0, y1 - y0]
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": name_to_id[rec["phrase"]] + 1,
                "bbox": bbox,
                "segmentation": [rec["poly"]],
                "area": float(bbox[2] * bbox[3]),
                "iscrowd": rec["iscrowd"],
            })
            ann_id += 1

    out_dir = root / "annotations"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "lvis_v1_val.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(
            {"info": [], "licenses": [], "categories": categories, "images": images, "annotations": annotations},
            fh,
            ensure_ascii=False,
        )
    return out_path


def build_labels_cache(root: Path, phrases: list[str]):
    cache_path = root / "labels.cache"
    if cache_path.is_file():
        cache_path.unlink()
    from ultralytics.data.dataset import YOLODataset

    YOLODataset(
        img_path=str(root / "val.txt"),
        imgsz=640,
        augment=False,
        cache=False,
        data={"names": {i: n for i, n in enumerate(phrases)}, "nc": len(phrases)},
        task="segment",
    )
    if not cache_path.is_file():
        raise RuntimeError(f"未写出 {cache_path}")
    return cache_path


def main(root: Path):
    if not root.name.endswith("(YOLO)(VAL)"):
        print(f"  ⚠️  目录名建议以 (YOLO)(VAL) 结尾：{root.name}")

    image_names = collect_images(root / "images")
    per_image, phrases = load_records(root)
    name_to_id = {n: i for i, n in enumerate(phrases)}

    n_inst = write_labels(root, image_names, per_image, name_to_id)
    write_val_txt(root, image_names)
    lvis_path = write_lvis_json(root, image_names, per_image, phrases)
    cache_path = build_labels_cache(root, phrases)

    print(f"Dataset root : {root}")
    print(f"Images       : {len(image_names)}")
    print(f"Phrases / nc : {len(phrases)}")
    print(f"Instances    : {n_inst}")
    print(f"Saved labels : {root / 'labels'}")
    print(f"Saved val    : {root / 'val.txt'}")
    print(f"Saved cache  : {cache_path}")
    print(f"Saved lvis   : {lvis_path}")
    print("names:")
    for i, name in enumerate(phrases):
        print(f"  {i}: {name}")


if __name__ == "__main__":
    args = parse_args()
    main(Path(args.path).resolve())
