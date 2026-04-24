"""
Split a video into parts at comma-separated timestamps.

Example:
    py -m video.split --ffmpeg "/programs/ffmpeg/bin/ffmpeg.exe" --in video.mp4 --cuts "00:01:30,01:15:37"
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_timestamp(value: str) -> float:
    parts = value.strip().split(":")
    if not value.strip() or len(parts) > 3:
        raise argparse.ArgumentTypeError(f"Invalid timestamp: {value!r}")

    try:
        if len(parts) == 1:
            total = float(parts[0])
        elif len(parts) == 2:
            minutes = int(parts[0])
            seconds = float(parts[1])
            if seconds >= 60:
                raise ValueError
            total = (minutes * 60) + seconds
        else:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            if minutes >= 60 or seconds >= 60:
                raise ValueError
            total = (hours * 3600) + (minutes * 60) + seconds
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid timestamp: {value!r}") from exc

    if total <= 0:
        raise argparse.ArgumentTypeError(f"Timestamp must be greater than zero: {value!r}")
    return total


def parse_cuts(value: str) -> list[float]:
    cuts = [parse_timestamp(part) for part in value.split(",") if part.strip()]
    if not cuts:
        raise argparse.ArgumentTypeError("--cuts must contain at least one timestamp.")

    for prev, current in zip(cuts, cuts[1:]):
        if current <= prev:
            raise argparse.ArgumentTypeError("--cuts must be in strictly increasing order.")
    return cuts


def ffmpeg_time(seconds: float) -> str:
    return f"{seconds:.6f}".rstrip("0").rstrip(".")


def resolve_ffmpeg(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.is_file():
        return resolved
    sys.exit(f"ERROR: ffmpeg not found at {resolved}")


def output_paths(source: Path, output_dir: Path, segment_count: int) -> list[Path]:
    return [output_dir / f"{source.stem}_{index}{source.suffix}" for index in range(1, segment_count + 1)]


def split_video(ffmpeg: Path, source: Path, cuts: list[float], output_dir: Path) -> list[Path]:
    outputs = output_paths(source, output_dir, len(cuts) + 1)
    existing = [path for path in outputs if path.exists()]
    if existing:
        names = ", ".join(str(path) for path in existing)
        sys.exit(f"ERROR: output file already exists: {names}")

    boundaries: list[tuple[float | None, float | None]] = []
    starts = [None, *cuts]
    ends = [*cuts, None]
    boundaries.extend(zip(starts, ends))

    created: list[Path] = []
    for index, ((start, end), output) in enumerate(zip(boundaries, outputs), start=1):
        duration = None
        if end is not None:
            duration = end if start is None else end - start

        cmd = [
            str(ffmpeg),
            "-hide_banner",
            "-n",
            "-i",
            str(source),
            "-map",
            "0",
        ]
        if start is not None:
            cmd.extend(["-ss", ffmpeg_time(start)])
        if duration is not None:
            cmd.extend(["-t", ffmpeg_time(duration)])
        cmd.extend(["-c", "copy", "-avoid_negative_ts", "make_zero", str(output)])

        print(f"[{index}/{len(outputs)}] {output}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            sys.exit(result.stderr.strip() or f"ERROR: ffmpeg failed while creating {output}")
        created.append(output)

    return created


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split a video at comma-separated timestamps.")
    parser.add_argument("--ffmpeg", required=True, type=Path, help="Path to the ffmpeg executable.")
    parser.add_argument("--in", dest="input", required=True, type=Path, help="Path to the source video.")
    parser.add_argument("--cuts", required=True, type=parse_cuts, help='Comma-separated cut timestamps, e.g. "00:01:30,01:15:37".')
    parser.add_argument("--out", type=Path, default=None, help="Optional output folder. Defaults to the source video's parent folder.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source = args.input.expanduser().resolve()
    if not source.is_file():
        sys.exit(f"ERROR: input video not found: {source}")
    if not source.suffix:
        sys.exit(f"ERROR: input video must have a file extension: {source}")

    ffmpeg = resolve_ffmpeg(args.ffmpeg)
    output_dir = args.out.expanduser().resolve() if args.out else source.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    created = split_video(ffmpeg, source, args.cuts, output_dir)
    print("Created:")
    for path in created:
        print(f"  {path}")


if __name__ == "__main__":
    main()
