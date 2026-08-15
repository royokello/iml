SEED_MASK = (1 << 64) - 1


def image_seed(base_seed: int, sample_name: str) -> int:
    h = int.from_bytes(sample_name.encode("utf-8"), "big") & SEED_MASK
    s = (base_seed ^ h) & SEED_MASK
    s = ((s ^ (s >> 30)) * 0xBF58476D1CE4E5B9) & SEED_MASK
    s = ((s ^ (s >> 27)) * 0x94D049BB133111EB) & SEED_MASK
    return (s ^ (s >> 31)) & SEED_MASK
