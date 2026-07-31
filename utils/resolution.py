RESOLUTIONS = {
    "low": 256,     "low+": 384,
    "std": 512,     "std+": 768,
    "high": 1024,   "high+": 1536,
    "ultra": 2048,  "ultra+": 3072,
    "real": 4096,   "real+": 6144,
}

def parse_resolution(value: str) -> int:
    normalized = value.strip().lower().replace("plus", "+").replace(" ", "")
    if normalized.endswith("p") and normalized[:-1].isdigit():
        return int(normalized[:-1])
    if normalized.isdigit():
        return int(normalized)
    if normalized in RESOLUTIONS:
        return RESOLUTIONS[normalized]
    raise ValueError(f"Unknown resolution: {value!r}")
