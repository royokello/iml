import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
from PIL import Image


def _run_cmd(cmd, verbose=False):
    if verbose:
        print(" ".join(cmd))
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.returncode, p.stdout, p.stderr


def _load_grid_config(image_path):
    config_path = image_path.parent / "config.json"
    if not config_path.is_file():
        raise ValueError("config.json not found next to the grid image.")
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _split_grid_cells(frame, rows, cols, cell_width, cell_height):
    height, width = frame.shape[:2]
    if rows <= 0 or cols <= 0:
        raise ValueError("Invalid grid layout.")
    if not cell_width or not cell_height or (cell_width * cols != width) or (cell_height * rows != height):
        cell_width = max(width // cols, 1)
        cell_height = max(height // rows, 1)
    cells = []
    for r in range(rows):
        for c in range(cols):
            y0 = r * cell_height
            x0 = c * cell_width
            cells.append(frame[y0 : y0 + cell_height, x0 : x0 + cell_width].copy())
    return cells, cell_width, cell_height


def create_image_animation(
    image_path,
    output_path,
    ffmpeg,
    length_seconds,
    fps=30.0,
    output_format="gif",
    codec="libx264",
    verbose=False,
):
    if length_seconds <= 0:
        raise ValueError("Animation length must be greater than 0.")
    if fps <= 0:
        raise ValueError("FPS must be greater than 0.")

    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    frame = cv2.imread(str(path))
    if frame is None:
        raise ValueError(f"Could not read image: {path}")

    config = _load_grid_config(path)
    rows = int(config.get("grid_rows", 0))
    cols = int(config.get("grid_cols", 0))
    cell_w = config.get("cell_width")
    cell_h = config.get("cell_height")
    cells, cell_width, cell_height = _split_grid_cells(frame, rows, cols, cell_w, cell_h)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_frames = max(int(round(length_seconds * fps)), 1)
    per_cell = max(total_frames // len(cells), 1)
    remainder = total_frames - (per_cell * len(cells))

    repeated_frames = []
    for idx, cell in enumerate(cells):
        count = per_cell + (1 if idx < remainder else 0)
        repeated_frames.extend([cell] * count)

    if output_format == "gif":
        duration_ms = max(int(round(1000 / fps)), 1)
        pil_frames = [
            Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            for f in repeated_frames
        ]
        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration_ms,
            loop=0,
        )
    else:
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            for i, f in enumerate(repeated_frames):
                cv2.imwrite(str(tmpdir / f"{i + 1:06d}.png"), f)
            ffmpeg_bin = str(Path(ffmpeg))
            cmd = [ffmpeg_bin, "-hide_banner",
                   "-loglevel", "info" if verbose else "error",
                   "-framerate", str(fps),
                   "-i", str(tmpdir / "%06d.png"),
                   "-c:v", codec,
                   "-pix_fmt", "yuv420p",
                   "-y",
                   str(output_path)]
            rc, _, err = _run_cmd(cmd, verbose=verbose)
            if rc != 0:
                raise RuntimeError(f"ffmpeg encoding failed: {err}")

    return {
        "output_path": str(output_path),
        "filename": output_path.name,
        "length_seconds": length_seconds,
        "fps": fps,
        "frame_count": total_frames,
        "cells": len(cells),
        "format": output_format,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Animate grid image into GIF or MP4")
    parser.add_argument("-i", "--input", required=True, help="Input grid PNG path")
    parser.add_argument("-o", "--output", required=True, help="Output file path")
    parser.add_argument("--ffmpeg", required=True, help="Path to ffmpeg executable")
    parser.add_argument("--length", type=float, required=True,
                        help="Animation length in seconds")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Frames per second (default: 30)")
    parser.add_argument("--format", default="gif", choices=["gif", "mp4"],
                        help="Output format (default: gif)")
    parser.add_argument("--codec", default="libx264",
                        choices=["libx264", "hevc_nvenc"],
                        help="Video codec for mp4 (default: libx264)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose ffmpeg output")
    args = parser.parse_args()

    try:
        result = create_image_animation(
            image_path=args.input,
            output_path=args.output,
            ffmpeg=args.ffmpeg,
            length_seconds=args.length,
            fps=args.fps,
            output_format=args.format,
            codec=args.codec,
            verbose=args.verbose,
        )
        print(f"Created animation: {result['output_path']}")
        print(f"  Format: {result['format']}")
        print(f"  Duration: {result['length_seconds']}s @ {result['fps']}fps")
        print(f"  Cells: {result['cells']}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
