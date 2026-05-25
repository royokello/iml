QUANT_METHODS_BY_MAIN_BITS = (
    "aff-low",
    "sym-low",
    "aff-med",
    "aff-high",
    "sym-med",
    "sym-high",
)
MIXED_VARIANT_ORDER = ("nano", "mini", "")


def convert_quant_name(name: str) -> tuple[str, str]:
    """
    Convert a quantization method name like 'sym-high-nano' into
    a (high, low) tuple based on the defined level hierarchy.

    Rules:
    - No suffix  → high = low = base
    - '-mini'    → high = base, low = base - 1 level
    - '-nano'    → high = base, low = base - 2 levels
    """
    order = QUANT_METHODS_BY_MAIN_BITS

    # Determine suffix and base name
    if name.endswith("-mini"):
        base = name[:-5]   # remove "-mini"
        suffix = "mini"
    elif name.endswith("-nano"):
        base = name[:-5]   # remove "-nano"
        suffix = "nano"
    else:
        base = name
        suffix = None

    if base not in order:
        raise ValueError(f"Base quantization '{base}' not in recognized order: {order}")

    base_idx = order.index(base)

    if suffix == "mini":
        high_idx = base_idx
        low_idx = base_idx - 1
    elif suffix == "nano":
        high_idx = base_idx
        low_idx = base_idx - 2
    else:  # no suffix
        high_idx = low_idx = base_idx

    # Ensure indices remain within bounds
    if high_idx >= len(order) or low_idx < 0:
        raise ValueError(
            f"Cannot offset '{name}': resulting high={high_idx}, low={low_idx} out of range."
        )

    return (order[high_idx], order[low_idx])


def mixed_quant_methods() -> tuple[str, ...]:
    methods: list[str] = []
    for base_method in QUANT_METHODS_BY_MAIN_BITS:
        for variant in MIXED_VARIANT_ORDER:
            method = base_method if not variant else f"{base_method}-{variant}"
            try:
                convert_quant_name(method)
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
    "convert_quant_name",
    "mixed_quant_methods",
    "quant_method_sort_key",
]
