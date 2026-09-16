from __future__ import annotations

RATIO_ORDER = ("1x1", "1x2", "2x3", "2x1", "3x2")

RATIOS: dict[str, tuple[int, int]] = {
    "1x1": (1, 1),
    "1x2": (1, 2),
    "2x3": (2, 3),
    "2x1": (2, 1),
    "3x2": (3, 2),
}

def normalize_ratio(value) -> str:
    if value is None:
        return "1x1"
    if isinstance(value, (tuple, list)) and len(value) == 2:
        candidate = f"{int(value[0])}x{int(value[1])}"
    else:
        candidate = str(value).strip().lower().replace(":", "x").replace("×", "x")
    if candidate in RATIOS:
        return candidate
    return "1x1"


def ratio_dims(value) -> tuple[int, int]:
    return RATIOS[normalize_ratio(value)]


def ratio_sort_key(value) -> int:
    ratio = normalize_ratio(value)
    return RATIO_ORDER.index(ratio)


def pair_sort_key(pair: tuple[int, str]) -> tuple[int, int]:
    return (int(pair[0]), ratio_sort_key(pair[1]))
