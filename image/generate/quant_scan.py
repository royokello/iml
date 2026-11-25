from __future__ import annotations

from pathlib import Path

from utils.quant.name import mixed_quant_methods

_PLAIN_METHODS = {"fp16", "fp32", "bf16"}


def _extract_methods(directory: Path) -> list[str]:
    """Scan *_quant.safetensors and return the loadable method names.

    Dead names from the old scheme stay on disk, but no loader can read them,
    so the list keeps only the current methods and the plain dtypes.

    Args:
        directory: The folder that holds the quantized checkpoints.

    Returns:
        The sorted, hyphenated method names.
    """
    if not directory.is_dir():
        return []
    allowed = set(mixed_quant_methods()) | _PLAIN_METHODS
    methods: set[str] = set()
    for fpath in directory.glob("*_quant.safetensors"):
        name = fpath.stem  # e.g. "q8_quant"
        if name.endswith("_quant"):
            name = name[:-6]
        method = name.replace("_", "-")
        if method in allowed:
            methods.add(method)
    return sorted(methods)


def scan_text_encoder(root: Path, model: str, version: str | None = None) -> list[str]:
    root = root.expanduser().resolve()
    if model == "ideogram":
        return _extract_methods(root / "ideogram" / "text_encoder")
    if model == "flux3":
        return _extract_methods(root / "flux3" / "model" / "text_encoder")
    if model == "flux":
        v = version or "4b"
        return _extract_methods(root / "flux2" / v / "model" / "text_encoder")
    if model == "anima":
        return _extract_methods(root / "anima" / "model" / "text_encoder")
    return []


def scan_denoiser(root: Path, model: str, version: str | None = None, variant: str | None = None) -> list[str]:
    root = root.expanduser().resolve()
    if model == "ideogram":
        cond = _extract_methods(root / "ideogram" / "transformer" / "cond")
        uncond = _extract_methods(root / "ideogram" / "transformer" / "uncond")
        # A method is usable only if quant files exist in BOTH directories
        return sorted(set(cond) & set(uncond))
    if model == "flux3":
        return _extract_methods(root / "flux3" / "model" / "transformer" / (variant or "distill"))
    if model == "flux":
        v = version or "4b"
        var = variant or "distill"
        return _extract_methods(root / "flux2" / v / "model" / "transformer" / var)
    if model == "anima":
        var = variant or "base"
        return _extract_methods(root / "anima" / "model" / "denoiser" / var)
    return []
