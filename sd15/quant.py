import os
import argparse
import shutil
from typing import Dict

import torch
from safetensors.torch import load_file, save_file

# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------

FP8_DTYPE = torch.float8_e5m2  # mixed-precision MXFP8 weight format


def needs_quant(name: str, tensor: torch.Tensor) -> bool:
    """True for conv/linear weights, False for biases, norms, embeds, etc."""
    if ".weight" not in name:
        return False
    if any(tag in name for tag in ("norm", "embedding")):
        return False
    return tensor.dim() >= 2 and tensor.dtype in (torch.float32, torch.float16, torch.bfloat16)


def quantise_weights(tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Return new dict with FP8 weights + .scale companions."""
    out: Dict[str, torch.Tensor] = {}
    for k, v in tensors.items():
        if needs_quant(k, v):
            # per-tensor max-abs scale → INT8 dynamic range
            scale = v.abs().max() / 127.0
            v_fp8 = v.to(dtype=FP8_DTYPE, copy=True)

            out[k] = v_fp8
            out[f"{k}.scale"] = torch.as_tensor(scale, dtype=torch.float32)
        else:
            out[k] = v.clone()
    return out


# -------------------------------------------------------------
# Main CLI
# -------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dir", required=True)
    parser.add_argument("-o", "--output-dir", required=True)
    args = parser.parse_args()

    f_root = os.path.abspath(args.input_dir.rstrip(os.sep))
    q_root = os.path.abspath(args.output_dir.rstrip(os.sep))

    if os.path.exists(q_root):
        shutil.rmtree(q_root)
    shutil.copytree(f_root, q_root)

    f_unet_path = os.path.join(f_root, "unet", "diffusion_pytorch_model.safetensors")
    q_unet_path = os.path.join(q_root, "unet", "diffusion_pytorch_model.safetensors")

    print(f"[INFO] Loading UNet weights → {f_unet_path}")
    f_unet = load_file(f_unet_path)

    print("[INFO] Quantising weights to FP8 (E5M2)…")
    q_unet = quantise_weights(f_unet)

    print(f"[INFO] Saving quantised UNet → {q_unet_path}")
    save_file(q_unet, q_unet_path, metadata={"format": "float8_e5m2"})
    print("[DONE] Quantisation complete.")


if __name__ == "__main__":
    main()
