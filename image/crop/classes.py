from __future__ import annotations

import re

from image.crop.ratios import normalize_ratio

CLASS_NAMES: dict[int, str] = {
    0: "face",
    1: "head_shot",
    2: "half_body",
    3: "cowboy_shot",
    4: "three_quarter",
    5: "full_body",
}

NAME_TO_CLASS: dict[str, int] = {name: cls_id for cls_id, name in CLASS_NAMES.items()}

_COMPOUND_RE = re.compile(r"^(?P<base>.+)_(?P<w>\d+)x(?P<h>\d+)$")


def compound_name(cls_id: int, ratio: str) -> str:
    return f"{CLASS_NAMES.get(cls_id, f'class_{cls_id}')}_{ratio}"


def split_compound_name(name: str) -> tuple[int, str] | None:
    match = _COMPOUND_RE.match(name)
    if not match:
        return None
    ratio = normalize_ratio(f"{match.group('w')}x{match.group('h')}")
    cls_id = NAME_TO_CLASS.get(match.group("base"), -1)
    return cls_id, ratio
