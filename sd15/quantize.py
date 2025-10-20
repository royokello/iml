#!/usr/bin/env python3
import argparse, json, math, time, re
from pathlib import Path
import os

import numpy as np
from safetensors.numpy import load_file, save_file

INT8_MAX = 127
INT8_MIN = -128


def parse_args():
    p = argparse.ArgumentParser(
        description="Quantize Stable Diffusion UNet weights to symmetric int8 via direct rounding."
    )
    p.add_argument(
        "--model",
        required=True,
        help="Path to a model directory containing a unet/ subdirectory.",
    )
    p.add_argument(
        "--quant-mode",
        default="tensor",
        choices=["tensor"],
        help="Quantization granularity (tensor rounds each tensor independently).",
    )
    # Per-category overrides kept for parity, but only int8 is exposed.
    p.add_argument("--proj-precision", default="int8", choices=["int8"])
    p.add_argument("--bias-norm-precision", default="int8", choices=["int8"])
    p.add_argument("--resnet-precision", default="int8", choices=["int8"])
    p.add_argument("--sampler-precision", default="int8", choices=["int8"])
    return p.parse_args()


def find_unet_file(model_dir: Path) -> Path:
    unet = model_dir / "unet"
    if not unet.is_dir():
        raise SystemExit("missing unet/ in --model dir")
    files = list(unet.glob("*.safetensors")) + list(unet.glob("*.safetensor"))
    if len(files) != 1:
        raise SystemExit(f"expected exactly one safetensor(s) in {unet}, found {len(files)}")
    return files[0]


def out_path_for(in_path: Path) -> Path:
    stem, suffix = in_path.stem, in_path.suffix
    return in_path.with_name(f"{stem}_quantized{suffix}")


_attn_qkv = re.compile(r'^(down_blocks\.\d+|mid_block|up_blocks\.\d+)\.attentions\.\d+\.transformer_blocks\.\d+\.attn[12]\.to_[qkv]\.weight$')
_attn_out = re.compile(r'^(down_blocks\.\d+|mid_block|up_blocks\.\d+)\.attentions\.\d+\.transformer_blocks\.\d+\.attn[12]\.to_out\.0\.weight$')
_proj_io  = re.compile(r'^(down_blocks\.\d+|mid_block|up_blocks\.\d+)\.attentions\.\d+\.proj_(in|out)\.weight$')
_ff_w0    = re.compile(r'^(down_blocks\.\d+|mid_block|up_blocks\.\d+)\.attentions\.\d+\.transformer_blocks\.\d+\.ff\.net\.0\.proj\.weight$')
_ff_w2    = re.compile(r'^(down_blocks\.\d+|mid_block|up_blocks\.\d+)\.attentions\.\d+\.transformer_blocks\.\d+\.ff\.net\.2\.weight$')
_resnet   = re.compile(
    r'^(?:down_blocks\.\d+|mid_block|up_blocks\.\d+)'  # block scope
    r'\.(?:resnets\.\d+\..*|conv\d+\..*)\.(?:weight|bias)$'
)
_root_convs = re.compile(r'^(?:conv_in|conv_out)\.(?:weight|bias)$')
_time_embed = re.compile(r'^time_embedding\.linear_[12]\.(?:weight|bias)$')
_samplers_global = re.compile(r'^(?:down|up)_samplers\.\d+\..*\.(?:weight|bias)$')
_samplers_inblock = re.compile(
    r'^(?:down_blocks\.\d+\.downsamplers\.\d+|up_blocks\.\d+\.upsamplers\.\d+)'
    r'\.conv\.(?:weight|bias)$'
)
_bias     = re.compile(r'.*\.bias$')
_norm     = re.compile(r'.*norm.*\.(weight|bias)$')


def categorize_tensor(name: str):
    # Bias and normalization tensors take precedence
    if _bias.match(name) or _norm.match(name):
        return "bias_norm"
    # Projections: attention q/k/v/o + proj_in/out
    if _attn_qkv.match(name) or _attn_out.match(name) or _proj_io.match(name):
        return "proj"
    # Resnet parts inside transformer + resnets outside transformer + root convs + time-embed MLP
    if _ff_w0.match(name) or _ff_w2.match(name) or _resnet.match(name) or _root_convs.match(name) or _time_embed.match(name):
        return "resnet"
    # Samplers
    if _samplers_global.match(name) or _samplers_inblock.match(name):
        return "sampler"
    return None


def tensor_amax(x: np.ndarray) -> float:
    if not x.size:
        return 0.0
    amax = float(np.max(np.abs(x)))
    return amax if math.isfinite(amax) else 0.0


def quantize_tensor_tensor(x: np.ndarray) -> tuple[np.ndarray, float]:
    x = np.asarray(x, dtype=np.float32)
    q = np.clip(np.rint(x), INT8_MIN, INT8_MAX).astype(np.int8)
    return q, tensor_amax(x)


def quantize_int8_tensor(x: np.ndarray, quant_mode: str) -> tuple[np.ndarray, float]:
    if quant_mode != "tensor":
        raise ValueError(f"unsupported quant-mode {quant_mode}")
    return quantize_tensor_tensor(x)


def main():
    args = parse_args()
    in_path = find_unet_file(Path(args.model))
    out_path = out_path_for(in_path)
    data = load_file(str(in_path))
    out: dict[str, np.ndarray] = {}
    q_count = 0
    bytes_in_q = 0
    bytes_out_q = 0

    uncategorized: list[str] = []
    for k, v in data.items():
        cat = categorize_tensor(k)
        if cat is None or not (v.dtype.kind == "f" or str(v.dtype) == "bfloat16"):
            if cat is None:
                uncategorized.append(k)
            out[k] = v
            continue

        prec = {
            "proj": args.proj_precision,
            "resnet": args.resnet_precision,
            "sampler": args.sampler_precision,
            "bias_norm": args.bias_norm_precision,
        }[cat]

        if prec != "int8":
            out[k] = v
            uncategorized.append(k)
            continue

        orig_dtype = v.dtype
        orig_bytes = v.nbytes
        v32 = v.astype(np.float32, copy=False)

        q, amax = quantize_int8_tensor(v32, args.quant_mode)
        out[k] = q

        print(
            f"[quant] {k} cat={cat} mode={args.quant_mode} "
            f"amax={amax:.4g} in={orig_dtype}/{orig_bytes}B -> out=int8/{q.nbytes}B"
        )

        q_count += 1
        bytes_in_q += orig_bytes
        bytes_out_q += q.nbytes

    print(
        f"[summary] quantized={q_count} tensors; bytes(quantized subset): "
        f"{bytes_in_q} -> {bytes_out_q} ({(bytes_out_q/bytes_in_q - 1.0):+.1%})"
    )
    if uncategorized:
        print(f"[summary] uncategorized={len(uncategorized)} tensors:")
        for n in uncategorized[:10]:
            print(f"    {n}")
    print(f"[path] {out_path}")

    meta = {
        "quant_meta/format": "int8",
        "quant_meta/precision": "int8",
        "quant_meta/weights/targets": json.dumps([
            "attn{1,2}.to_{q,k,v}.weight",
            "attn{1,2}.to_out.0.weight",
            "proj_in.weight",
            "proj_out.weight",
            "ff.net.0.proj.weight",
            "ff.net.2.weight"
        ]),
        "quant_meta/weights/quant_mode": args.quant_mode,
        "quant_meta/weights/scales": "none",
        "quant_meta/weights/precision/proj": args.proj_precision,
        "quant_meta/weights/precision/bias_norm": args.bias_norm_precision,
        "quant_meta/weights/precision/resnet": args.resnet_precision,
        "quant_meta/weights/precision/sampler": args.sampler_precision,
        "quant_meta/bias/precision": "int8",
        "quant_meta/activations/precision": "int8",
        "quant_meta/activations/quant_mode": args.quant_mode,
        "quant_meta/activations/scales": "none",
        "quant_meta/created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    save_file(out, str(out_path), metadata=meta)

    file_bytes = os.path.getsize(out_path)
    print(f"[post-save] file_on_disk={file_bytes/1e6:.1f} MB")
    print(f"[path] {out_path}")


if __name__ == "__main__":
    main()
