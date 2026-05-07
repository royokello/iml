def convert_quant_name(name: str) -> tuple[str, str]:
    """
    Convert a quantization method name like 'aff-med-max' into
    a (high, low) tuple based on the defined level hierarchy.

    Rules:
    - No suffix  → high = low = base
    - '-max'     → high = base + 2 levels, low = base
    - '-mini'    → high = base, low = base - 1 level
    """
    order = ["aff-low", "sym-low", "aff-med", "aff-high", "sym-med", "sym-high"]

    # Determine suffix and base name
    if name.endswith("-max"):
        base = name[:-4]   # remove "-max"
        suffix = "max"
    elif name.endswith("-mini"):
        base = name[:-5]   # remove "-mini"
        suffix = "mini"
    else:
        base = name
        suffix = None

    if base not in order:
        raise ValueError(f"Base quantization '{base}' not in recognized order: {order}")

    base_idx = order.index(base)

    if suffix == "max":
        high_idx = base_idx + 2
        low_idx = base_idx
    elif suffix == "mini":
        high_idx = base_idx
        low_idx = base_idx - 1
    else:  # no suffix
        high_idx = low_idx = base_idx

    # Ensure indices remain within bounds
    if high_idx >= len(order) or low_idx < 0:
        raise ValueError(
            f"Cannot offset '{name}': resulting high={high_idx}, low={low_idx} out of range."
        )

    return (order[high_idx], order[low_idx])
