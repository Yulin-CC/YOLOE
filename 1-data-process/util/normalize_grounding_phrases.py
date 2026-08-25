"""
# @Author: 算法组 蔡雨霖
# @Date: 2026-08-20
# @Description: 将安防 grounding 近义短语收成短词，并回写 jsons 的 caption / tokens_positive。
# @Command: python 1-data-process/util/normalize_grounding_phrases.py --input /path/to/GD --dry-run
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


CAPTION_SEP = " . "


#-------------#
# 短语 → 短词
#-------------#
COLOR_MAP = {
    "white": "white",
    "black": "black",
    "red": "red",
    "blue": "blue",
    "yellow": "yellow",
    "green": "green",
    "orange": "orange",
    "brown": "brown",
    "pink": "pink",
    "silver": "gray",
    "grey": "gray",
    "gray": "gray",
    "beige": "gray",
    "gold": "yellow",
}

# 更具体的类型放前面
TYPE_PATTERNS = [
    (r"\bfish\b|\baquatic (animal|creature)\b", "deadfish"),
    (r"\bboat\b|\bwatercraft\b|\bvessel\b|\bship\b|\bcanoe\b|\browboat\b|\bmotorboat\b|\bspeedboat\b", "boat"),
    (r"\bfloating\b|\btrash\b|\bdebris\b|\bgarbage\b|\blitter\b|\bwaste\b|\bplastic\b|\bbottle\b|\bbuoyant\b|\bwaterborne\b", "floatingstuff"),
    (r"\bexcavator\b|\bdigger\b|\bcrane\b|\bdrilling\b|\bpiledriver\b|\bpiling\b|\bmachinery\b|\bconstruction machine\b|\bearth moving\b|\bequipment\b|\bboom\b", "excavator"),
    (r"\bbulldozer\b", "bulldozer"),
    (r"\bforklift\b", "forklift"),
    (r"\bambulance\b", "ambulance"),
    (r"\bfire\s*truck\b|\bfiretruck\b", "truck"),
    (r"\bscooter\b|\bmotorcycle\b|\bmotorbike\b|\btricycle\b", "scooter"),
    (r"\bebike\b|\be-bike\b|\bbicycle\b|\bcyclist\b|\bbiker\b", "bicycle"),
    (r"\bminivan\b|\bvan\b", "van"),
    (r"\bbus\b", "bus"),
    (r"\btruck\b|\bsemi\b|\btrailer\b", "truck"),
    (r"\bsedan\b|\bsuv\b|\bhatchback\b|\bcrossover\b|\bcoupe\b|\bwagon\b|\bautomobile\b|\bcompact car\b|\bpassenger car\b|\bcar\b|\bvehicle\b", "vehicle"),
]

PERSON_RE = re.compile(
    r"\b(person|people|man|men|woman|women|male|female|guy|individual|child|girl|boy|"
    r"figure|officer|policeman|worker|rider|pedestrian|adult|student|operator)\b"
)
CLOTHES_RE = re.compile(
    r"\b(wearing|outfit|shirt|jacket|pants|jeans|clothes|clothing|helmet|hat|shoes|t-shirt|vest|top)\b"
)
PART_RE = re.compile(r"\b(sunroof|window|windows|license plate)\b")
STAND_RE = re.compile(r"\b(standing|upright)\b")
WALK_RE = re.compile(r"\bwalk")
SIT_RE = re.compile(r"\b(sitting|squatting|bending|crouching|lying)\b")
COLOR_RE = re.compile(r"\b(" + "|".join(sorted(COLOR_MAP, key=len, reverse=True)) + r")\b")


def extract_color(text: str) -> str | None:
    m = COLOR_RE.search(text)
    if not m:
        return None
    return COLOR_MAP[m.group(1)]


def extract_type(text: str) -> str | None:
    for pattern, name in TYPE_PATTERNS:
        if re.search(pattern, text):
            return name
    return None


def normalize_phrase(raw: str, side: str = "val") -> str:
    """side=val 用 vehicle；side=train 用 car，方便开集近邻而不抄同一句。"""
    text = re.sub(r"\s+", " ", (raw or "").strip().lower())
    if not text:
        return text

    obj = extract_type(text)
    if obj in {"deadfish", "floatingstuff"}:
        return obj

    color = extract_color(text)
    if obj:
        if obj == "vehicle" and side == "train":
            obj = "car"
        return f"{color} {obj}" if color and obj in {"vehicle", "car", "truck", "van", "bus", "boat", "excavator", "bulldozer"} else obj

    if PERSON_RE.search(text) or CLOTHES_RE.search(text):
        if STAND_RE.search(text):
            return "person standing"
        if WALK_RE.search(text):
            return "person walking"
        if SIT_RE.search(text):
            return "person sitting"
        return "person"

    if STAND_RE.search(text):
        return "person standing"
    if WALK_RE.search(text):
        return "person walking"
    if SIT_RE.search(text):
        return "person sitting"
    if PART_RE.search(text) or "driving" in text:
        return "vehicle" if side == "val" else "car"
    if "three-wheeled" in text:
        return "scooter"
    if "hair" in text or "silhouette" in text:
        return "person"
    if "water" in text or "object" in text or "item" in text:
        return "floatingstuff"

    return text


def slice_phrase(caption: str, tokens_positive) -> str:
    return " ".join(caption[int(t[0]) : int(t[1])] for t in (tokens_positive or [])).strip()


def rebuild_caption(phrases: list[str]) -> tuple[str, dict[str, list[int]]]:
    ordered, seen = [], set()
    for p in phrases:
        if p and p not in seen:
            seen.add(p)
            ordered.append(p)
    parts, spans, offset = [], {}, 0
    for i, p in enumerate(ordered):
        if i:
            offset += len(CAPTION_SEP)
        start = offset
        parts.append(p)
        offset += len(p)
        spans[p] = [start, offset]
    return CAPTION_SEP.join(parts), spans


#-------------#
# 单张 json
#-------------#
def rewrite_json(data: dict, side: str) -> tuple[dict, list[tuple[str, str]]]:
    images = {img["id"]: img for img in data.get("images", [])}
    pairs = []
    by_image: dict[int, list] = defaultdict(list)
    for ann in data.get("annotations", []):
        by_image[ann["image_id"]].append(ann)

    for img in data.get("images", []):
        caption = img.get("caption") or ""
        anns = by_image.get(img["id"], [])
        new_phrases = []
        for ann in anns:
            raw = slice_phrase(caption, ann.get("tokens_positive"))
            new = normalize_phrase(raw, side=side)
            pairs.append((raw, new))
            ann["_new_phrase"] = new
            new_phrases.append(new)
        new_caption, spans = rebuild_caption(new_phrases)
        img["caption"] = new_caption
        img["tokens_negative"] = [[0, len(new_caption)]] if new_caption else []
        img["tokens_positive_eval"] = [[[s, e]] for s, e in (spans[p] for p in spans)]
        for ann in anns:
            phrase = ann.pop("_new_phrase")
            ann["tokens_positive"] = [spans[phrase]]
    return data, pairs


#-------------#
# 扫描目录
#-------------#
def iter_jsons(root: Path, jsons_dir: str):
    d = root / jsons_dir
    files = sorted(d.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"未找到 json: {d}")
    return files


def collect_pairs(files: list[Path], side: str) -> list[tuple[str, str]]:
    pairs = []
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        _, file_pairs = rewrite_json(data, side=side)
        pairs.extend(file_pairs)
    return pairs


def apply_files(files: list[Path], side: str) -> list[tuple[str, str]]:
    pairs = []
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        data, file_pairs = rewrite_json(data, side=side)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        pairs.extend(file_pairs)
    return pairs


def print_report(pairs: list[tuple[str, str]]):
    before = Counter(raw.lower().strip() for raw, _ in pairs if raw)
    after = Counter(new for _, new in pairs if new)
    changed = sum(1 for raw, new in pairs if raw.lower().strip() != new)
    print(f"框数 {len(pairs)}  |  改写 {changed}")
    print(f"独特短语  {len(before)} → {len(after)}")
    print("\n收口后 top:")
    for k, v in after.most_common(25):
        print(f"  {v:5d}  {k}")
    print("\n未收入短词表（保持原文）:")
    leftover = Counter(new for raw, new in pairs if new == raw.lower().strip() and new)
    # 原文已是短词的不算 leftover：只列出 normalize 没动、且不像短词的
    canon = set(after)
    odd = [(k, v) for k, v in leftover.most_common() if k not in {
        "person", "person standing", "person walking", "person sitting",
        "vehicle", "car", "truck", "van", "bus", "boat", "excavator",
        "bulldozer", "forklift", "ambulance", "scooter", "bicycle",
        "deadfish", "floatingstuff",
    } and " " in k or (k not in {
        "person", "vehicle", "car", "truck", "van", "bus", "boat", "excavator",
        "bulldozer", "forklift", "ambulance", "scooter", "bicycle", "deadfish", "floatingstuff",
    } and not k.startswith("person ") and not any(k.startswith(c + " ") for c in COLOR_MAP.values()))]
    for k, v in leftover.most_common(20):
        if k in {"person", "vehicle", "truck", "van", "bus", "boat", "excavator", "bulldozer",
                 "forklift", "ambulance", "scooter", "bicycle", "deadfish", "floatingstuff",
                 "person standing", "person walking", "person sitting"} or any(
            k == f"{c} {t}" for c in COLOR_MAP.values()
            for t in ("vehicle", "car", "truck", "van", "bus", "boat", "excavator", "bulldozer")
        ):
            continue
        print(f"  {v:5d}  {k}")


#-------------#
# 参数
#-------------#
def parse_args():
    parser = argparse.ArgumentParser(description="安防 grounding 短语收口")
    parser.add_argument("--input", type=str, required=True, help="数据集根目录（含 jsons/）")
    parser.add_argument("--jsons-dir", type=str, default="jsons")
    parser.add_argument("--side", type=str, default="val", choices=("val", "train"),
                        help="val→color vehicle；train→color car")
    parser.add_argument("--dry-run", action="store_true", help="只打印统计，不写文件")
    parser.add_argument("--report", type=str, default="", help="映射表输出 json 路径")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root = Path(args.input).resolve()
    files = iter_jsons(root, args.jsons_dir)
    print(f"Input : {root}")
    print(f"Jsons : {len(files)}  side={args.side}  dry_run={args.dry_run}")

    if args.dry_run:
        pairs = collect_pairs(files, args.side)
    else:
        pairs = apply_files(files, args.side)
        print(f"✅ 已回写 {len(files)} 个 json")

    print_report(pairs)
    if args.report:
        mapping = defaultdict(lambda: defaultdict(int))
        for raw, new in pairs:
            mapping[new][raw.lower().strip()] += 1
        Path(args.report).write_text(
            json.dumps({k: dict(v) for k, v in mapping.items()}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Report → {args.report}")
