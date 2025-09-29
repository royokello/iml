"""
SDXL PROFILE
py -m sdxl.profile --model "/models/sdxl.safetensors"
"""

import argparse
from pathlib import Path
import math
import re
import torch

# Optional safetensors
try:
    from safetensors.torch import load_file as _safe_load  # type: ignore
    _SAFE = True
except ImportError:
    _SAFE = False

# Section regexes
SECTIONS = {
    "text_encoder": re.compile(r"(^|\.)conditioner\.embedders\.0\."),
    "text_encoder_2": re.compile(r"(^|\.)conditioner\.embedders\.1\."),
    "unet": re.compile(r"(^|\.)model\.diffusion_model\."),
    "vae": re.compile(r"(^|\.)first_stage_model\."),
}
# Layer regexes
LAYER_REGEX = {
    "Linear": re.compile(r"\.(fc|proj|lin)[0-9]*\.weight$", re.I),
    "Conv": re.compile(r"\.conv[0-9]*\.weight$", re.I),
    "Embedding": re.compile(r"\.emb.*\.weight$", re.I),
}

# -------- helpers --------

def _load(p: Path):
    if p.suffix.lower() == ".safetensors" and _SAFE:
        return _safe_load(str(p), device="cpu")
    sd = torch.load(str(p), map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    return sd


def _sec(n: str):
    for s, pat in SECTIONS.items():
        if pat.search(n):
            return s
    return "other"


def _lay(n: str):
    if n.endswith(".bias"):
        return "Bias"
    for t, pat in LAYER_REGEX.items():
        if pat.search(n):
            return t
    return "Other"


def _human(b: int):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if b < 1024 or u == "TB":
            return f"{b:,} {u}" if u != "B" else f"{b} B"
        b //= 1024
    return f"{b} TB"

# --------- Classification Mapping ----------
# E4M3 (forward quantization): applied only to weight tensors (nn.Linear, nn.Conv, Embeddings)
# E5M2 (backward quantization): applied only to bias tensors (gradients of weights)   
# FP16 (full precision): applied to all other parameters/activations, including LayerNorm scales/shifts and non-weight/bias tensors
# E8M0 (scale metadata): one scale byte per 32 weight elements (blocks)

# ---------- profiler ----------

def profile(path: Path):
    sd = _load(path)
    data: dict[str, dict] = {}

    for k, t in sd.items():
        if not isinstance(t, torch.Tensor):
            continue
        sec = _sec(k)
        lay = _lay(k)
        s = data.setdefault(sec, {
            "layers": {},
            "w_elems": 0,  # weight elements (E4)
            "b_elems": 0,  # bias elements (E5 only)
            "blocks": 0,   # weight blocks (E8)
            "fp16": 0      # bytes
        })
        ls = s["layers"].setdefault(lay, {"w_blocks": 0})

        n = t.numel()
        if k.endswith(".weight"):
            blk = math.ceil(n/32)
            ls["w_blocks"] += blk
            s["w_elems"] += n
            s["blocks"] += blk
        elif k.endswith(".bias"):
            s["b_elems"] += n
            s["fp16"] += n*2  # kept in FP16 too
        else:
            s["fp16"] += n*2

    # build output
    out = [f"Per‑section quant summary for: {path}", ""]
    for sec, s in data.items():
        out.append(f"[{sec}]")
        for typ in ("Linear","Conv","Embedding","Other"):
            blk = s["layers"].get(typ, {}).get("w_blocks", 0)
            if blk:
                out.append(f"  E4M3({typ.lower()}): {blk:,} blocks")
        if s["b_elems"]:
            bias_blks = math.ceil(s['b_elems']/32)
            out.append(f"  E5M2(bias): {bias_blks:,} blocks")
        # totals
        e4_bytes = s['w_elems']          # 1 B per weight elem
        e5_bytes = s['b_elems']          # 1 B per bias elem
        e8_bytes = s['blocks']           # 1 B per block
        fp8_total = e4_bytes + e5_bytes + e8_bytes
        out.append(f"  Total E4M3 bytes: {_human(e4_bytes)}")
        out.append(f"  Total E5M2 bytes: {_human(e5_bytes)}")
        out.append(f"  E8M0 bytes: {_human(e8_bytes)}")
        out.append(f"  FP16 bytes: {_human(s['fp16'])}")
        out.append(f"  Section FP8 total: {_human(fp8_total)}")
        out.append(f"  Section combined: {_human(fp8_total + s['fp16'])}")
        out.append("")
    return "\n".join(out)

# ---------- CLI ----------
if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",required=True,type=Path)
    args=ap.parse_args()
    print(profile(args.model))
