#!/usr/bin/env python3
import argparse, json, math, time, re
from pathlib import Path
import os
import numpy as np
from safetensors.numpy import load_file, save_file

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--scaling-mode", required=True, choices=["off", "tensor"])
    p.add_argument("--scaling-precision", required=True, choices=["e8m0"])
    # Per-category precision overrides
    p.add_argument("--proj-precision", default="e2m1", choices=["e2m1", "e4m3"])  # q/k/v/o + proj_in/out
    p.add_argument("--bias-norm-precision", default="e4m3", choices=["e2m1", "e4m3"])  # all bias and norm tensors
    p.add_argument("--resnet-precision", default="e4m3", choices=["e2m1", "e4m3"])
    p.add_argument("--sampler-precision", default="e4m3", choices=["e2m1", "e4m3"])
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

def fp8_e4m3_max():
    return 240.0

def quantize_e4m3_tensor(x: np.ndarray, scaling_mode: str):
    x = np.asarray(x, dtype=np.float32)
    if scaling_mode == "off":
        scale = 1.0
    else:
        amax = float(np.max(np.abs(x))) if x.size else 0.0
        m = fp8_e4m3_max()
        scale = (amax / m) if amax > 0 else 1.0
    s = (x < 0).astype(np.uint8)
    ax = np.abs(x) / np.float32(scale)
    nz = ax > 0
    code = np.zeros(ax.shape, dtype=np.uint8)
    if np.any(nz):
        bias = 7
        e_bits, m_bits = 4, 3
        ax_nz = ax[nz]
        exp_uncl = np.floor(np.log2(ax_nz))
        mant_f = ax_nz / np.exp2(exp_uncl) - 1.0
        mant = np.rint(mant_f * (1 << m_bits)).astype(np.int32)
        carry = mant == (1 << m_bits)
        if np.any(carry):
            mant[carry] = 0
            exp_uncl[carry] += 1.0
        exp_biased = exp_uncl.astype(np.int32) + bias
        exp_max = (1 << e_bits) - 2
        under = exp_biased < 1
        over = exp_biased > exp_max
        exp_biased = np.clip(exp_biased, 1, exp_max)
        mant = np.clip(mant, 0, (1 << m_bits) - 1)
        c = ((exp_biased.astype(np.uint16) << m_bits) | mant.astype(np.uint16)).astype(np.uint8)
        if np.any(under):
            c[under] = 0
        if np.any(over):
            c[over] = ((exp_max << m_bits) | ((1 << m_bits) - 1)).astype(np.uint8)
        code[nz] = c
    code |= (s << 7)
    return code, float(scale)

def fp8_e2m1_max():
    # (1 + (2^m_bits - 1)/2^m_bits) * 2^(exp_max - bias) with e_bits=2, m_bits=1, bias=1
    return 3.0

def quantize_e2m1_tensor(x: np.ndarray, scaling_mode: str):
    x = np.asarray(x, dtype=np.float32)
    if scaling_mode == "off":
        scale = 1.0
    else:
        amax = float(np.max(np.abs(x))) if x.size else 0.0
        m = fp8_e2m1_max()
        scale = (amax / m) if amax > 0 else 1.0
    s = (x < 0).astype(np.uint8)
    ax = np.abs(x) / np.float32(scale)
    nz = ax > 0
    code = np.zeros(ax.shape, dtype=np.uint8)
    if np.any(nz):
        bias = 1
        e_bits, m_bits = 2, 1
        ax_nz = ax[nz]
        exp_uncl = np.floor(np.log2(ax_nz))
        mant_f = ax_nz / np.exp2(exp_uncl) - 1.0
        mant = np.rint(mant_f * (1 << m_bits)).astype(np.int32)
        carry = mant == (1 << m_bits)
        if np.any(carry):
            mant[carry] = 0
            exp_uncl[carry] += 1.0
        exp_biased = exp_uncl.astype(np.int32) + bias
        exp_max = (1 << e_bits) - 2
        under = exp_biased < 1
        over = exp_biased > exp_max
        exp_biased = np.clip(exp_biased, 1, exp_max)
        mant = np.clip(mant, 0, (1 << m_bits) - 1)
        c = ((exp_biased.astype(np.uint16) << m_bits) | mant.astype(np.uint16)).astype(np.uint8)
        if np.any(under):
            c[under] = 0
        if np.any(over):
            c[over] = ((exp_max << m_bits) | ((1 << m_bits) - 1)).astype(np.uint8)
        code[nz] = c
    code |= (s << 7)
    return code, float(scale)

def pack_scale_e8m0(scale: float) -> np.ndarray:
    if not math.isfinite(scale) or scale <= 0.0:
        return np.array([0], dtype=np.uint8)
    e = int(round(math.log2(scale) + 127.0))
    e = 0 if e < 0 else (255 if e > 255 else e)
    return np.array([e], dtype=np.uint8)

def main():
    args = parse_args()
    in_path = find_unet_file(Path(args.model))
    out_path = out_path_for(in_path)
    data = load_file(str(in_path))
    out = {}
    scales = {}
    q_count = 0
    bytes_in_q = 0
    bytes_out_q = 0

    uncategorized = []
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

        orig_dtype = v.dtype
        orig_bytes = v.nbytes
        v32 = v.astype(np.float32, copy=False)
        if prec == "e4m3":
            q, scale = quantize_e4m3_tensor(v32, args.scaling_mode)
        elif prec == "e2m1":
            q, scale = quantize_e2m1_tensor(v32, args.scaling_mode)
        else:
            # Fallback: keep original (shouldn't happen due to argparse choices)
            out[k] = v
            uncategorized.append(k)
            continue

        out[k] = q
        scale_byte = None
        if args.scaling_mode == "tensor":
            s = pack_scale_e8m0(scale)
            scales[k] = s
            scale_byte = int(s[0])
        print(
            f"[quant] {k} cat={cat} prec={prec} shape={tuple(v.shape)} "
            f"amax={float(np.max(np.abs(v32))) if v32.size else 0.0:.4g} "
            f"scale={scale:.6g} scale_byte={scale_byte if scale_byte is not None else 'n/a'} "
            f"in={orig_dtype}/{orig_bytes}B -> out=uint8/{q.nbytes}B"
        )
        q_count += 1
        bytes_in_q += orig_bytes
        bytes_out_q += q.nbytes + (1 if scale_byte is not None else 0)

    for k, s in scales.items():
        out[f"quant_scales/{k}"] = s

    print(f"[summary] quantized={q_count} tensors; bytes(quantized subset): {bytes_in_q} -> {bytes_out_q} ({(bytes_out_q/bytes_in_q - 1.0):+.1%})")
    if uncategorized:
        print(f"[summary] uncategorized={len(uncategorized)} tensors:")
        for n in uncategorized[:10]:
            print(f"    {n}")
    print(f"[path] {out_path}")

    meta = {
        "quant_meta/format": "fp8",
        "quant_meta/precision": "mixed",
        "quant_meta/weights/targets": json.dumps([
            "attn{1,2}.to_{q,k,v}.weight",
            "attn{1,2}.to_out.0.weight",
            "proj_in.weight",
            "proj_out.weight",
            "ff.net.0.proj.weight",
            "ff.net.2.weight"
        ]),
        "quant_meta/weights/scaling_mode": args.scaling_mode,
        "quant_meta/weights/scaling_precision": args.scaling_precision,
        "quant_meta/weights/precision/proj": args.proj_precision,
        "quant_meta/weights/precision/bias_norm": args.bias_norm_precision,
        "quant_meta/weights/precision/resnet": args.resnet_precision,
        "quant_meta/weights/precision/sampler": args.sampler_precision,
        "quant_meta/bias/precision": "fp16",
        "quant_meta/activations/precision": "e4m3",
        "quant_meta/activations/scaling_mode": args.scaling_mode,
        "quant_meta/activations/scaling_precision": args.scaling_precision,
        "quant_meta/created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    save_file(out, str(out_path), metadata=meta)

    file_bytes = os.path.getsize(out_path)
    print(f"[post-save] file_on_disk={file_bytes/1e6:.1f} MB")
    print(f"[path] {out_path}")

if __name__ == "__main__":
    main()
