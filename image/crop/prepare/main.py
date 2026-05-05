import argparse
from typing import Sequence

from image.crop.prepare import yolo
from utils.stages import find_latest_stage


def resolve_stage(project: str, stage: int | None) -> int:
    return stage if stage is not None else find_latest_stage(project)


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser("Crop dataset preparation")
    parser.add_argument(
        "--model",
        choices=("yolo",),
        default="yolo",
        help="Dataset format/model target to prepare.",
    )
    parser.add_argument("--project", required=True, help="Path to project root")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation fraction (0..1)")
    parser.add_argument("--test_split", type=float, default=0.1, help="Test fraction (0..1)")
    parser.add_argument("--stage", type=int, default=None, help="Stage number (optional)")
    parser.add_argument(
        "--balance",
        action="store_true",
        help="Downsample each class to the smallest class count before splitting",
    )
    parser.add_argument("--seed", type=int, default=19930625, help="Random seed")
    args = parser.parse_args(argv)
    stage = resolve_stage(args.project, args.stage)

    if args.model == "yolo":
        yolo.run(
            project=args.project,
            val_split=args.val_split,
            stage=stage,
            balance=args.balance,
            seed=args.seed,
        )
        return

    raise SystemExit(f"Unsupported prepare model: {args.model}")


if __name__ == "__main__":
    main()
