QUANT_METHODS_BY_MAIN_BITS = (
    "aff-low",
    "sym-low",
    "aff-med",
    "aff-high",
    "sym-med",
    "sym-high",
)
MIXED_VARIANT_ORDER = ("nano", "mini", "")
_VARIANT_DEPTH = {"mini": 1, "nano": 2}


def split_quant_name(name: str) -> tuple[str, int]:
    """Split a method name into its base method and its rank depth.

    The depth is the number of ladder rungs the bottom rank drops: none for a
    plain method, one for `-mini`, two for `-nano`.

    Args:
        name: A method name such as `sym-med-nano`.

    Returns:
        The base method and the rank depth.

    Raises:
        ValueError: If the base method is not in the ladder.
    """
    if name == "fp16":
        return "fp16", 0

    base = name
    depth = 0
    for suffix, suffix_depth in _VARIANT_DEPTH.items():
        if name.endswith("-" + suffix):
            base = name[: -len(suffix) - 1]
            depth = suffix_depth
            break

    if base not in QUANT_METHODS_BY_MAIN_BITS:
        raise ValueError(
            f"Base quantization '{base}' not in recognized order: {QUANT_METHODS_BY_MAIN_BITS}"
        )
    return base, depth


def convert_quant_ranks(name: str, count: int = 2) -> tuple[str, ...]:
    """Return the method for each precision rank of a target set.

    Rank 1 takes the base method. The bottom rank drops by the depth of the
    variant. Ranks between rank 1 and the bottom rank drop by one rung.

    Args:
        name: A method name such as `sym-med-nano`.
        count: The number of ranks in the target set.

    Returns:
        One method per rank, most precise first.

    Raises:
        ValueError: If the base method is not in the ladder, or if a rank would
            drop below the lowest method.
    """
    if count < 1:
        raise ValueError(f"A target set needs at least one rank, got {count}.")

    base, depth = split_quant_name(name)
    if base == "fp16":
        return ("fp16",) * count

    index = QUANT_METHODS_BY_MAIN_BITS.index(base)
    methods = [base]
    for rank in range(2, count + 1):
        step = depth if rank == count else min(depth, 1)
        target = index - step
        if target < 0:
            raise ValueError(
                f"Cannot offset '{name}' to rank {rank}: "
                f"'{base}' is the lowest method."
            )
        methods.append(QUANT_METHODS_BY_MAIN_BITS[target])
    return tuple(methods)


def mixed_quant_methods() -> tuple[str, ...]:
    methods: list[str] = []
    for base_method in QUANT_METHODS_BY_MAIN_BITS:
        for variant in MIXED_VARIANT_ORDER:
            method = base_method if not variant else f"{base_method}-{variant}"
            try:
                convert_quant_ranks(method, 2)
            except ValueError:
                continue
            methods.append(method)
    return tuple(methods)


def quant_method_sort_key(method: str) -> tuple[int, int, str]:
    variant = ""
    base_method = method
    for suffix in ("-nano", "-mini"):
        if method.endswith(suffix):
            base_method = method[: -len(suffix)]
            variant = suffix[1:]
            break

    try:
        bit_order = QUANT_METHODS_BY_MAIN_BITS.index(base_method)
    except ValueError:
        bit_order = len(QUANT_METHODS_BY_MAIN_BITS)

    try:
        variant_order = MIXED_VARIANT_ORDER.index(variant)
    except ValueError:
        variant_order = len(MIXED_VARIANT_ORDER)

    return (bit_order, variant_order, method)


__all__ = [
    "MIXED_VARIANT_ORDER",
    "QUANT_METHODS_BY_MAIN_BITS",
    "convert_quant_ranks",
    "mixed_quant_methods",
    "quant_method_sort_key",
    "split_quant_name",
]
