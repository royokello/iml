import csv
import sys
from pathlib import Path
from safetensors import safe_open


def dump_tensors_csv(paths: list[Path], output_path: Path):
    rows = []
    for path in paths:
        with safe_open(str(path), framework="numpy") as handle:
            for name in handle.keys():
                s = handle.get_slice(name)
                dtype = str(s.get_dtype())
                shape = s.get_shape()
                shape_str = "[" + ", ".join(str(d) for d in shape) + "]"
                rows.append({"name": name, "precision": dtype, "size[y,x]": shape_str})

    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "precision", "size[y,x]"])
        w.writeheader()
        w.writerows(rows)

    return len(rows)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m utils.dump_tensors <model.safetensors|dir> [output.csv]")
        sys.exit(1)

    src = Path(sys.argv[1])
    if not src.exists():
        print(f"Error: {src} not found")
        sys.exit(1)

    if src.is_dir():
        paths = sorted(src.glob("*.safetensors"))
        if not paths:
            print(f"Error: no .safetensors files in {src}")
            sys.exit(1)
        dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_name("tensor.csv")
    else:
        paths = [src]
        dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_name("tensor.csv")

    count = dump_tensors_csv(paths, dst)
    print(f"{count} tensors -> {dst}")
