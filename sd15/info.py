#!/usr/bin/env python
from __future__ import annotations
import argparse, os, csv, json, glob
from typing import List, Tuple, Dict, Iterable
from safetensors.torch import load_file as load_safetensors

MODULE_PREFIXES = {
    "text": "cond_stage_model",
    "unet": "model.diffusion_model",
    "vae": "first_stage_model",
}

DIFFUSERS_SUBDIR = {
    "unet": "unet",
    "vae": "vae",
    "text": "text_encoder",
}

TOPLEVEL_PREFERRED = [
    "diffusion_pytorch_model.safetensors",
    "model.safetensors",
    "unet.safetensors",
    "final.safetensors",
]

def _derive_output_path(model_path: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.basename(model_path.rstrip(os.sep)) or "model"
    return os.path.join(out_dir, f"{base}.csv")

def _dtype_name(t) -> str:
    return str(t.dtype).split(".")[-1]

def _index_json_candidates(module_dir: str) -> List[str]:
    pats = [
        os.path.join(module_dir, "*.safetensors.index.json"),
        os.path.join(module_dir, "**", "*.safetensors.index.json"),
    ]
    out: List[str] = []
    for p in pats:
        out.extend(glob.glob(p, recursive=True))
    return out

def _files_from_index_json(idx_path: str, base_dir: str) -> List[str]:
    with open(idx_path, "r", encoding="utf-8") as f:
        idx = json.load(f)
    weight_map: Dict[str, str] = idx.get("weight_map", {})
    unique_files = sorted(set(weight_map.values()))
    return [os.path.normpath(os.path.join(base_dir, rel)) for rel in unique_files]

def _module_file_candidates(module_dir: str) -> List[str]:
    preferred = [
        "diffusion_pytorch_model.safetensors",
        "model.safetensors",
        "unet.safetensors",
        "vae.safetensors",
        "final.safetensors",
        "pytorch_model.safetensors",  # text_encoder
    ]
    for name in preferred:
        cand = os.path.join(module_dir, name)
        if os.path.exists(cand):
            return [cand]
    return sorted(glob.glob(os.path.join(module_dir, "*.safetensors")))

def _resolve_diffusers_module_files(root: str, module: str) -> List[str]:
    module_dir = os.path.join(root, DIFFUSERS_SUBDIR[module])
    if not os.path.isdir(module_dir):
        return []
    idxs = _index_json_candidates(module_dir)
    if idxs:
        files: List[str] = []
        for idx in idxs:
            files.extend(_files_from_index_json(idx, os.path.dirname(idx)))
        return sorted(set(files))
    return _module_file_candidates(module_dir)

def _resolve_from_dir_by_modules(root: str, modules: List[str] | None) -> List[Tuple[str, str]]:
    files: List[Tuple[str, str]] = []
    if modules:
        for m in modules:
            for fp in _resolve_diffusers_module_files(root, m):
                files.append((m, fp))
        if files:
            return files
    # fallback: top-level single-file checkpoint in the directory
    for name in TOPLEVEL_PREFERRED:
        cand = os.path.join(root, name)
        if os.path.exists(cand):
            return [("all", cand)]
    cands = sorted([os.path.join(root, f) for f in os.listdir(root) if f.endswith(".safetensors")])
    if len(cands) == 1:
        return [("all", cands[0])]
    return []

def _resolve_input(model_path: str, modules: List[str] | None) -> Tuple[str, List[Tuple[str, str]]]:
    if os.path.isfile(model_path):
        if not model_path.endswith(".safetensors"):
            raise ValueError(f"Unsupported file (need .safetensors): {model_path}")
        return ("file", [("all", model_path)])
    if not os.path.isdir(model_path):
        raise FileNotFoundError(model_path)
    files = _resolve_from_dir_by_modules(model_path, modules)
    if not files:
        looked = []
        if modules:
            for m in modules:
                looked.append(os.path.join(model_path, DIFFUSERS_SUBDIR[m]))
        msg = "No .safetensors found. Looked in: " + ", ".join(looked or [model_path]) + \
              " for *.safetensors or *.safetensors.index.json shards."
        raise RuntimeError(msg)
    return ("dir", files)

def _filter_keys(keys: Iterable[str], modules: List[str] | None) -> List[str]:
    if not modules:
        return sorted(keys)
    prefixes = [MODULE_PREFIXES[m] for m in modules if m in MODULE_PREFIXES]
    if not prefixes:
        return sorted(keys)
    return sorted([k for k in keys if any(k.startswith(p) for p in prefixes)])

def _load_state_dict_entries_safetensors(path: str):
    sd = load_safetensors(path, device="cpu")
    for k in sd.keys():
        yield k, sd[k]

def collect_tensor_info(model_path: str, modules: List[str] | None = None) -> List[Tuple[str, str, str]]:
    mode, files = _resolve_input(model_path, modules)
    rows: List[Tuple[str, str, str]] = []

    if mode == "file":
        path = files[0][1]
        entries = list(_load_state_dict_entries_safetensors(path))
        keys = _filter_keys((k for k, _ in entries), modules)
        tensor_map = {k: t for k, t in entries}
        for k in keys:
            t = tensor_map[k]
            rows.append((k, "[" + ",".join(str(x) for x in t.shape) + "]", _dtype_name(t)))
        return sorted(rows, key=lambda x: x[0])

    for module, path in files:
        for k, t in _load_state_dict_entries_safetensors(path):
            rows.append((k, "[" + ",".join(str(x) for x in t.shape) + "]", _dtype_name(t)))
    return sorted(rows, key=lambda x: x[0])

def main() -> None:
    ap = argparse.ArgumentParser(description="List tensor names with shape and precision (dtype) to CSV (.safetensors only)")
    ap.add_argument("--model", required=True, help="Path to model dir (Diffusers) or a single .safetensors file")
    ap.add_argument("--modules", nargs="*", choices=MODULE_PREFIXES.keys(), help="Which parts to include: text unet vae")
    ap.add_argument("--output", required=True, help="Directory to write the output .csv file")
    args = ap.parse_args()

    entries = collect_tensor_info(args.model, args.modules)
    out_path = _derive_output_path(args.model, args.output)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "shape", "precision"])
        w.writerows(entries)
    print(f"Wrote {len(entries)} tensors → {out_path}")

if __name__ == "__main__":
    main()
