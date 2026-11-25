QUANT_METHODS_BY_MAIN_BITS = (
    "q2",
    "q3",
    "q4",
    "q5",
    "q6",
    "q8",
)
MIXED_VARIANT_ORDER = ("mini", "max", "")
_VARIANTS = ("max", "mini")
_VARIANT_SIZE_ORDER = {"mini": 0, "": 1, "max": 2}
# The widest target set in the repo has three ranks. A method must work for
# that count to be offered as a mixed preset.
_MAX_MIXED_RANKS = 3


def split_quant_name(name: str) -> tuple[str, str]:
    """Split a method name into its base method and its variant.

    The variant is `""` for a plain method, `max` for the wide variant, and
    `mini` for the narrow variant.

    Args:
        name: A method name such as `q6-mini`.

    Returns:
        The base method and the variant.

    Raises:
        ValueError: If the base method is not in the ladder.
    """
    if name == "fp16":
        return "fp16", ""

    base = name
    variant = ""
    for suffix in _VARIANTS:
        if name.endswith("-" + suffix):
            base = name[: -len(suffix) - 1]
            variant = suffix
            break

    if base not in QUANT_METHODS_BY_MAIN_BITS:
        raise ValueError(
            f"Base quantization '{base}' not in recognized order: {QUANT_METHODS_BY_MAIN_BITS}"
        )
    return base, variant


def convert_quant_ranks(name: str, count: int = 2) -> tuple[str, ...]:
    """Return the method for each precision rank of a target set.

    Rank 1 is the most precise.

    - A plain method gives every rank the base method.
    - `-max` raises rank 1 by one rung and drops the bottom rank by one rung.
      The ranks between stay at the base method. A two-rank set raises rank 1
      and keeps rank 2 at the base method.
    - `-mini` keeps rank 1 at the base method and drops every other rank by
      one rung.

    Args:
        name: A method name such as `q6-mini`.
        count: The number of ranks in the target set.

    Returns:
        One method per rank, most precise first.

    Raises:
        ValueError: If the base method is not in the ladder, a single-rank set
            uses `-max`, or a rank would fall outside the ladder.
    """
    if count < 1:
        raise ValueError(f"A target set needs at least one rank, got {count}.")

    base, variant = split_quant_name(name)
    if base == "fp16":
        return ("fp16",) * count

    index = QUANT_METHODS_BY_MAIN_BITS.index(base)
    if not variant:
        return (base,) * count

    if variant == "max":
        if count == 1:
            raise ValueError(f"Cannot spread '{name}' across a single rank.")
        top = index + 1
        if top >= len(QUANT_METHODS_BY_MAIN_BITS):
            raise ValueError(
                f"Cannot offset '{name}' to rank 1: '{base}' is the highest method."
            )
        methods = [QUANT_METHODS_BY_MAIN_BITS[top]]
        for rank in range(2, count + 1):
            drop = 1 if (count > 2 and rank == count) else 0
            target = index - drop
            if target < 0:
                raise ValueError(
                    f"Cannot offset '{name}' to rank {rank}: "
                    f"'{base}' is the lowest method."
                )
            methods.append(QUANT_METHODS_BY_MAIN_BITS[target])
        return tuple(methods)

    methods = [base]
    for rank in range(2, count + 1):
        target = index - 1
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
                convert_quant_ranks(method, _MAX_MIXED_RANKS)
            except ValueError:
                continue
            methods.append(method)
    return tuple(methods)


def quant_method_sort_key(method: str) -> tuple[int, int, str]:
    base_method = method
    variant = ""
    for suffix in _VARIANTS:
        if method.endswith("-" + suffix):
            base_method = method[: -len(suffix) - 1]
            variant = suffix
            break

    try:
        bit_order = QUANT_METHODS_BY_MAIN_BITS.index(base_method)
    except ValueError:
        bit_order = len(QUANT_METHODS_BY_MAIN_BITS)

    variant_order = _VARIANT_SIZE_ORDER.get(variant, len(_VARIANT_SIZE_ORDER))

    return (bit_order, variant_order, method)


__all__ = [
    "MIXED_VARIANT_ORDER",
    "QUANT_METHODS_BY_MAIN_BITS",
    "convert_quant_ranks",
    "mixed_quant_methods",
    "quant_method_sort_key",
    "split_quant_name",
]
