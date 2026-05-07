from __future__ import annotations

import argparse
import time
from pathlib import Path

from flux2.train.run import run_training
from flux2.train.prepare import prepare_project_dir
INITIAL_LORA_TARGET_LINEAR_NAMES = (
    "transformer_blocks.attn.to_q",
    "transformer_blocks.attn.to_k",
    "transformer_blocks.attn.to_v",
    "transformer_blocks.attn.to_out",
    "transformer_blocks.ff.linear_in",
    "transformer_blocks.ff.linear_out",
    "single_transformer_blocks.attn.to_qkv_mlp_proj",
    "single_transformer_blocks.attn.to_out",
)
INITIAL_LORA_RANK = 32
INITIAL_LORA_ALPHA = 16
MODEL_METADATA_NAMES = {
    "4b": "flux 2 klein 4b",
    "9b": "flux 2 klein 9b",
}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument(
        "--version",
        choices=("4b", "9b"),
        required=True,
        help="Model family to load.",
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--steps", type=_positive_int, default=2048)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest checkpoint.",
    )
    parser.add_argument("--text-quant-method", default="sym-high")
    parser.add_argument("--denoiser-quant-method", default="sym-med")
    parser.add_argument(
        "--trigger",
        type=str,
        help="Required for captionless datasets. Encoded once and kept on GPU for all training images.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_path = args.project

    normalized_trigger = None if args.trigger is None else args.trigger.strip()
    model_root = args.root / f"flux2_{args.version}" / "model"

    prepare_project_dir(project_path)
    print("1. Run training epochs ...")

    checkpoint_metadata = {
        "model": MODEL_METADATA_NAMES[args.version.strip().lower()],
        "rank": str(INITIAL_LORA_RANK),
        "alpha": str(INITIAL_LORA_ALPHA),
        "trigger": normalized_trigger
    }
    
    training_start = time.perf_counter()
    print(f"  * max steps: {args.steps}")

    run_training(
        transformer=None,
        project_dir=project_path,
        checkpoint_metadata=checkpoint_metadata,
        root_path=model_root,
        model_version=args.version.strip().lower(),
        text_quant_method=args.text_quant_method,
        denoiser_quant_method=args.denoiser_quant_method,
        trigger=normalized_trigger,
        max_steps=args.steps,
        resume=args.resume,
    )

    print(f"  * training done in {time.perf_counter() - training_start:.3f}s")

if __name__ == "__main__":
    main()
