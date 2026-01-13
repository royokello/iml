from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import cv2
import numpy as np


def _center_crop_to_aspect(frame: np.ndarray, target_aspect: float) -> np.ndarray:
    if target_aspect <= 0:
        return frame

    height, width = frame.shape[:2]
    if height <= 0 or width <= 0:
        return frame

    current_aspect = width / height
    if abs(current_aspect - target_aspect) < 1e-6:
        return frame

    if current_aspect > target_aspect:
        new_width = max(int(round(height * target_aspect)), 1)
        x0 = max((width - new_width) // 2, 0)
        return frame[:, x0 : x0 + new_width]

    new_height = max(int(round(width / target_aspect)), 1)
    y0 = max((height - new_height) // 2, 0)
    return frame[y0 : y0 + new_height, :]


def _resize_frame(frame: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    if target_width <= 0 or target_height <= 0:
        return frame

    height, width = frame.shape[:2]
    if height == 0 or width == 0:
        return frame

    scale = target_width / width
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(frame, (target_width, target_height), interpolation=interpolation)


def _build_grid_image(
    frames: list[np.ndarray],
    grid_rows: int,
    grid_cols: int,
    target_aspect: float,
    cell_width: int,
    cell_height: int,
) -> np.ndarray:
    grid_height = grid_rows * cell_height
    grid_width = grid_cols * cell_width
    grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

    for idx, frame in enumerate(frames):
        cropped = _center_crop_to_aspect(frame, target_aspect)
        resized = _resize_frame(cropped, cell_width, cell_height)
        row = idx // grid_cols
        col = idx % grid_cols
        y0 = row * cell_height
        x0 = col * cell_width
        grid[y0 : y0 + cell_height, x0 : x0 + cell_width] = resized

    return grid


def create_video_grids(
    video_path: str,
    output_dir: Path,
    grid_rows: int,
    grid_cols: int,
    cell_ratio: tuple[int, int],
    cell_height: int,
    frame_interval_sec: float,
    alignment: str = "center",
) -> dict[str, object]:
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")

    if alignment != "center":
        raise ValueError("Only center alignment is supported.")

    ratio_w, ratio_h = cell_ratio
    if ratio_w <= 0 or ratio_h <= 0:
        raise ValueError("Cell ratio must be positive.")

    if grid_rows <= 0 or grid_cols <= 0:
        raise ValueError("Grid rows/cols must be positive.")

    if cell_height <= 0:
        raise ValueError("Cell height must be greater than 0.")

    if frame_interval_sec <= 0:
        raise ValueError("Frame interval must be greater than 0.")

    run_dir = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_dir = output_dir / run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            raise ValueError("Could not read FPS from the video.")

        interval_frames = max(int(round(fps * frame_interval_sec)), 1)
        frames_per_grid = grid_rows * grid_cols
        cell_width = max(int(round(cell_height * (ratio_w / ratio_h))), 1)
        target_aspect = ratio_w / ratio_h
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0:
            sampled_frames = ((total_frames - 1) // interval_frames) + 1
            grids_estimate = max(sampled_frames // frames_per_grid, 1)
        else:
            grids_estimate = 1
        pad_width = len(str(grids_estimate))

        outputs: list[dict[str, object]] = []
        buffer: list[np.ndarray] = []
        frame_index = 0
        success, frame = cap.read()
        while success:
            if frame_index % interval_frames == 0:
                buffer.append(frame.copy())
                if len(buffer) == frames_per_grid:
                    grid_image = _build_grid_image(
                        buffer,
                        grid_rows,
                        grid_cols,
                        target_aspect,
                        cell_width,
                        cell_height,
                    )
                    filename = f"{len(outputs) + 1:0{pad_width}d}.png"
                    output_path = output_dir / filename
                    if not cv2.imwrite(str(output_path), grid_image):
                        raise RuntimeError("Failed to write grid image.")
                    outputs.append(
                        {
                            "output_path": str(output_path),
                            "filename": filename,
                            "grid_index": len(outputs) + 1,
                        }
                    )
                    buffer = []
            frame_index += 1
            success, frame = cap.read()

        if not outputs:
            raise ValueError(
                f"Video too short for a {grid_rows}x{grid_cols} grid at {frame_interval_sec}s intervals."
            )

        config = {
            "video_path": str(path),
            "grid_rows": grid_rows,
            "grid_cols": grid_cols,
            "cell_ratio": {"width": ratio_w, "height": ratio_h},
            "cell_height": cell_height,
            "cell_width": cell_width,
            "frame_interval_seconds": frame_interval_sec,
            "alignment": alignment,
            "frames_per_grid": frames_per_grid,
            "interval_frames": interval_frames,
            "fps": fps,
            "total_frames": total_frames,
            "grids_made": len(outputs),
            "remainder_frames": len(buffer),
            "run_dir": run_dir,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "outputs": outputs,
        }

        config_path = output_dir / "config.json"
        with config_path.open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)

        return {
            "outputs": outputs,
            "grid_rows": grid_rows,
            "grid_cols": grid_cols,
            "cell_width": cell_width,
            "cell_height": cell_height,
            "frames_per_grid": frames_per_grid,
            "interval_seconds": frame_interval_sec,
            "grids_made": len(outputs),
            "remainder_frames": len(buffer),
            "run_dir": run_dir,
        }
    finally:
        cap.release()


def create_grid_animation(
    run_dir: Path,
    outputs: list[dict[str, object]],
    length_seconds: float,
) -> dict[str, object]:
    if length_seconds <= 0:
        raise ValueError("Animation length must be greater than 0.")

    if not outputs:
        raise ValueError("No grid images available to animate.")

    run_dir.mkdir(parents=True, exist_ok=True)

    safe_length = f"{length_seconds:.2f}".replace(".", "_")
    filename = f"animation_{safe_length}s.mp4"
    output_path = run_dir / filename

    first_item = outputs[0]
    first_path = Path(first_item.get("output_path") or (run_dir / first_item["filename"]))
    first_frame = cv2.imread(str(first_path))
    if first_frame is None:
        raise ValueError(f"Could not read grid image: {first_path}")

    height, width = first_frame.shape[:2]
    fps = max(len(outputs) / length_seconds, 1.0)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError("Failed to initialize video writer.")

    try:
        for item in outputs:
            frame_path = Path(item.get("output_path") or (run_dir / item["filename"]))
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise ValueError(f"Could not read grid image: {frame_path}")
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()

    return {
        "output_path": str(output_path),
        "filename": filename,
        "fps": fps,
        "length_seconds": length_seconds,
        "frame_count": len(outputs),
    }
