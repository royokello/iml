from safetensors.torch import load_file, save_file
from pathlib import Path

root = Path(r"C:\Users\roy\Documents\delos\ideogram\transformer")

for name in ["cond/sym_med_nano_quant.safetensors", "uncond/sym_med_nano_quant.safetensors"]:
    path = root / name
    tensors = load_file(path)
    before = len(tensors)
    tensors = {k: v for k, v in tensors.items() if not k.endswith("weight_scale")}
    save_file(tensors, path)
    print(f"{name}: {before} keys -> {len(tensors)} keys (removed {before - len(tensors)})")
