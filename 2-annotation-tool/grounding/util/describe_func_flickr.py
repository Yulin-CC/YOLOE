"""
# @Author: AI产品研发组
# @Date: 2026-10-30
# @Description: Flickr describe 引擎，对应 rules/describe_rules_flickr.md
# @Command: 由 util/describe.py 在 --rules-scene flickr 时加载
"""

import re

from describe_common import (
    label_natural_reading,
    locked_axis_tokens,
    phrase_content_key,
    phrase_synonym_key,
    policy_max_phrases,
    shares_locked_axis,
)
from rules_io import get_scene_policy
from vllm_client import extract_json_object


RULES_FILE = "describe_rules_flickr.md"

SYSTEM_PROMPT = (
    "You describe ONE closed-set object for visual grounding. "
    "Write 1–3 short English noun phrases at different specificity. "
    "Articles a/an/the are allowed. "
    "Role words within the same class are allowed (person → a man / a worker). "
    "Never swap to another class. "
    "Put color, age, size, role, and visible clothing INSIDE the phrase. "
    "Do NOT output clothing or body parts as their own phrase. "
    "No action, pose, or location. No two/several/group. "
    "Output JSON only: {\"phrases\":[...]}. No markdown, no thinking."
)

_GROUP = frozenset({
    "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "several", "many", "group", "crowd", "couple", "pair",
})
_ACTION = frozenset({
    "walking", "walk", "walks", "standing", "stand", "stands", "sitting", "sit",
    "running", "run", "parked", "parking", "moving", "riding", "holding",
    "looking", "playing", "smiling",
})
_LOC = frozenset({
    "left", "right", "background", "foreground", "nearby", "outside", "inside",
})
_PART_OR_CLOTH = frozenset({
    "hair", "hand", "hands", "head", "face", "arm", "arms", "leg", "legs",
    "foot", "feet", "eye", "eyes", "finger", "fingers",
    "shirt", "shirts", "jacket", "jackets", "hat", "hats", "cap", "caps",
    "jeans", "pants", "shoes", "boots", "glasses", "helmet", "vest", "vests",
    "coat", "coats",
})
_PERSON_ROLES = frozenset({
    "person", "people", "man", "men", "woman", "women", "boy", "girl",
    "child", "children", "kid", "kids", "worker", "workers", "lady", "guy",
    "student",
})


def _tokens(phrase: str):
    return re.findall(r"[a-z0-9]+", (phrase or "").lower())


def _reject(phrase: str, hint: str) -> bool:
    toks = _tokens(phrase)
    if not toks or len(toks) > 8:
        return True
    if any(t in _GROUP or t in _ACTION or t in _LOC for t in toks):
        return True
    hint_toks = set(hint.split())
    head = toks[-1]
    if head in _PART_OR_CLOTH and not (set(toks) & _PERSON_ROLES) and not (set(toks) & hint_toks):
        return True
    return False


def sanitize_phrases(phrases, label: str, max_n: int, avoid_l, lock_axis):
    hint = label_natural_reading(label)
    kept, ban_keys, ban_syn = [], set(), set()
    for a in avoid_l:
        key = phrase_content_key(a)
        syn = phrase_synonym_key(a)
        if key:
            ban_keys.add(key)
        if syn:
            ban_syn.add(syn)
    for p in phrases or []:
        if not isinstance(p, str):
            continue
        text = re.sub(r"\s+", " ", p.strip())
        if not text or _reject(text, hint):
            continue
        if text.lower() in avoid_l:
            continue
        key = phrase_content_key(text)
        syn = phrase_synonym_key(text)
        if (key and key in ban_keys) or (syn and syn in ban_syn):
            continue
        if lock_axis and shares_locked_axis(text, lock_axis):
            continue
        if key:
            ban_keys.add(key)
        if syn:
            ban_syn.add(syn)
        kept.append(text)
        if len(kept) >= max_n:
            break
    return kept


def describe_object(client, crop_abs: str, label: str, rules: str = "",
                    avoid_phrases=None, min_phrases=None, with_zh: bool = False,
                    max_retries: int = 3, seed_phrases=None, lock_phrases=None):
    pol = get_scene_policy()
    need = max(1, int(min_phrases if min_phrases is not None else policy_max_phrases(pol)))
    avoid = [
        str(p).strip() for p in (avoid_phrases or [])
        if isinstance(p, str) and str(p).strip()
    ]
    seeds = [
        str(p).strip() for p in (seed_phrases or [])
        if isinstance(p, str) and str(p).strip()
    ]
    locked = [
        str(p).strip() for p in (lock_phrases or [])
        if isinstance(p, str) and str(p).strip()
    ]
    lock_axis = locked_axis_tokens(locked)
    hint = label_natural_reading(label)
    avoid_l = {a.lower() for a in avoid}
    example = '{"phrases":["a car","a white car","a small white car"]}'
    if hint == "person":
        example = '{"phrases":["a man","a young man","a man in a blue shirt"]}'
    extra = ""
    if locked:
        listed = "; ".join(locked[:8])
        extra += (
            f"LOCKED (already chosen; do not repeat or lightly reword): {listed}.\n"
            f"Write a different specificity still about this same object "
            f"(more generic, or extra visible color / age / size / clothing).\n"
        )
    elif seeds:
        listed = "; ".join(seeds[:8])
        extra += (
            f"Stay close to these phrases but make them more precise: {listed}.\n"
            f"Do NOT repeat them verbatim.\n"
        )
    if avoid:
        listed = "; ".join(avoid[:16])
        extra += (
            f"Do NOT reuse or lightly paraphrase: {listed}.\n"
            f"A different article only (a man / the man) counts as a duplicate.\n"
        )
    zh_line = ""
    if with_zh:
        example = example[:-1] + ',"zh":["一名男子","一名年轻男子","穿蓝衬衫的男子"]}'
        zh_line = (
            "Also provide Simplified Chinese for each phrase in parallel array \"zh\" "
            "(same count/order).\n"
        )
    task = (
        f"Class label (GROUND TRUTH): {label}\n"
        f"Natural reading: {hint}\n"
        f"Write 1–{need} short English noun phrases for THIS one object. "
        f"Fewer is OK; do not pad.\n"
        f"Rules:\n"
        f"- Same class family only. Role or near-synonym heads are OK "
        f"(person → a man / a worker). Never rename to another category.\n"
        f"- Allow a/an/the. Typical length 2–4 words, at most about 6.\n"
        f"- Allowed inside the phrase: color, age, size, role, visible clothing.\n"
        f"- Clothing stays on this object: 'a man in a blue shirt'. "
        f"Never a standalone garment or body part "
        f"(not 'a blue shirt', 'shaggy hair', 'their hands').\n"
        f"- No action or pose (walking / parked). "
        f"No location (on the road / on the left).\n"
        f"- One instance only: no two / several / a group of.\n"
        f"- Vary specificity across the list. "
        f"Do not swap only the article or a synonym.\n"
        f"{extra}"
        f"{zh_line}"
        f"Return ONLY valid JSON, example: {example}"
    )
    prompt = (str(rules).strip() + "\n\n" + task) if rules and str(rules).strip() else task
    last_err = None
    for _ in range(max(1, int(max_retries or 1))):
        try:
            raw = client.describe_image(
                crop_abs, prompt,
                system=SYSTEM_PROMPT,
                max_tokens=384,
                temperature=0.4,
            )
            data = extract_json_object(raw)
            phrases = sanitize_phrases(
                data.get("phrases") or [], label, need, avoid_l, lock_axis,
            )
            if not phrases:
                raise ValueError(f"模型未返回有效短语: {data}")
            zh = []
            if with_zh:
                zh_raw = data.get("zh") or []
                for i in range(len(phrases)):
                    if i < len(zh_raw) and isinstance(zh_raw[i], str) and zh_raw[i].strip():
                        zh.append(zh_raw[i].strip())
                    else:
                        zh.append("")
            return {
                "label": label,
                "phrases": phrases,
                "source": "vllm",
                "error": None,
                "zh": zh,
            }
        except Exception as e:
            last_err = e
    raise RuntimeError(f"描述失败 ({label}): {last_err}")
