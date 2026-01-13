from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
from PIL import Image


def _load_grid_config(image_path: Path) -> dict:
    config_path = image_path.parent / "config.json"
    if not config_path.is_file():
        raise ValueError("config.json not found next to the grid image.")

    import json

    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _split_grid_cells(
    frame,
    rows: int,
    cols: int,
    cell_width: int | None,
    cell_height: int | None,
) -> list:
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
    image_path: str,
    output_root: Path,
    length_seconds: float,
    fps: float = 30.0,
) -> dict[str, object]:
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
    cell_width = config.get("cell_width")
    cell_height = config.get("cell_height")
    cells, cell_width, cell_height = _split_grid_cells(
        frame, rows, cols, cell_width, cell_height
    )

    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    total_frames = max(int(round(length_seconds * fps)), 1)
    per_cell = max(total_frames // len(cells), 1)
    remainder = total_frames - (per_cell * len(cells))

    frames = []
    for idx, cell in enumerate(cells):
        frames_for_cell = per_cell + (1 if idx < remainder else 0)
        for _ in range(frames_for_cell):
            frames.append(cell)

    output_path = output_root / f"{timestamp}.gif"
    duration_ms = max(int(round(1000 / fps)), 1)
    pil_frames = [
        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames
    ]
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )

    return {
        "output_path": str(output_path),
        "filename": output_path.name,
        "length_seconds": length_seconds,
        "fps": fps,
        "frame_count": total_frames,
        "cells": len(cells),
        "ext": ".gif",
    }
