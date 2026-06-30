from __future__ import annotations

from pathlib import Path


def _extract_methods(directory: Path) -> list[str]:
    """Scan */*_quant.safetensors, return sorted method names (hyphenated)."""
    if not directory.is_dir():
        return []
    methods: set[str] = set()
    for fpath in directory.glob("*_quant.safetensors"):
        name = fpath.stem  # e.g. "sym_high_quant"
        if name.endswith("_quant"):
            name = name[:-6]
        methods.add(name.replace("_", "-"))
    return sorted(methods)


def scan_text_encoder(root: Path, model: str, version: str | None = None) -> list[str]:
    root = root.expanduser().resolve()
    if model == "ideogram":
        return _extract_methods(root / "ideogram" / "text_encoder")
    if "flux" in model:
        v = version or "4b"
        return _extract_methods(root / "flux2" / v / "model" / "text_encoder")
    return []


def scan_denoiser(root: Path, model: str, version: str | None = None, variant: str | None = None) -> list[str]:
    root = root.expanduser().resolve()
    if model == "ideogram":
        cond = _extract_methods(root / "ideogram" / "transformer" / "cond")
        uncond = _extract_methods(root / "ideogram" / "transformer" / "uncond")
        # A method is usable only if quant files exist in BOTH directories
        return sorted(set(cond) & set(uncond))
    if "flux" in model:
        v = version or "4b"
        var = variant or "distill"
        return _extract_methods(root / "flux2" / v / "model" / "transformer" / var)
    return []
