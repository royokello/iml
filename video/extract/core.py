from __future__ import annotations

import math
import os
import random
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, List, Optional

import cv2
import numpy as np


VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv")


def resize_preserve_aspect(frame, target_short_side: Optional[int]) -> "cv2.Mat":
    """Resize frame so its short side equals target_short_side."""
    h, w = frame.shape[:2]
    if target_short_side is None or target_short_side <= 0:
        return frame

    short = min(h, w)
    if short == target_short_side:
        return frame

    scale = target_short_side / short
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(frame, (new_w, new_h), interpolation=interpolation)


def read_frame_at(cap: cv2.VideoCapture, frame_index: int) -> Optional["cv2.Mat"]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    success, frame = cap.read()
    return frame if success else None


def sharpness_score(frame: "cv2.Mat") -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def exposure_penalty(frame: "cv2.Mat") -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_norm = float(gray.mean()) / 255.0
    mean_penalty = abs(mean_norm - 0.5) / 0.5
    clipped_dark = float(np.mean(gray <= 5))
    clipped_bright = float(np.mean(gray >= 250))
    clip_penalty = min((clipped_dark + clipped_bright) / 0.25, 1.0)
    return min(0.7 * mean_penalty + 0.3 * clip_penalty, 1.0)


def candidate_indices(mark_index: int, buffer: int, total_frames: int) -> List[int]:
    if buffer <= 0:
        return [mark_index]
    start = max(0, mark_index - buffer)
    end = min(total_frames - 1, mark_index + buffer)
    return list(range(start, end + 1))


def bounded_candidate_indices(
    mark_index: int,
    buffer: int,
    total_frames: int,
    min_index: Optional[int] = None,
    max_index: Optional[int] = None,
) -> List[int]:
    indices = candidate_indices(mark_index, buffer, total_frames)
    lower = 0 if min_index is None else max(0, min_index)
    upper = total_frames - 1 if max_index is None else min(total_frames - 1, max_index)
    if upper < lower:
        upper = lower
    return [idx for idx in indices if lower <= idx <= upper]


def select_best_frame(
    cap: cv2.VideoCapture,
    mark_index: int,
    total_frames: int,
    buffer: int,
    min_index: Optional[int] = None,
    max_index: Optional[int] = None,
) -> tuple[Optional["cv2.Mat"], int]:
    candidate_rows = []
    for idx in bounded_candidate_indices(mark_index, buffer, total_frames, min_index=min_index, max_index=max_index):
        frame = read_frame_at(cap, idx)
        if frame is None:
            continue
        candidate_rows.append(
            {
                "index": idx,
                "frame": frame,
                "sharpness": sharpness_score(frame),
                "exposure_penalty": exposure_penalty(frame),
            }
        )

    if not candidate_rows:
        return None, mark_index

    sharpness_values = [row["sharpness"] for row in candidate_rows]
    sharp_min = min(sharpness_values)
    sharp_max = max(sharpness_values)
    sharp_range = sharp_max - sharp_min

    def composite(row: dict[str, Any]) -> float:
        if sharp_range <= 1e-9:
            sharp_norm = 0.5
        else:
            sharp_norm = max(0.0, min((row["sharpness"] - sharp_min) / sharp_range, 1.0))
        return sharp_norm - (0.4 * row["exposure_penalty"])

    best = min(
        candidate_rows,
        key=lambda row: (-composite(row), abs(row["index"] - mark_index), row["index"]),
    )
    return best["frame"], int(best["index"])


def select_best_candidate_rows(
    candidate_rows: list[dict[str, Any]],
    mark_index: int,
) -> dict[str, Any] | None:
    if not candidate_rows:
        return None

    sharpness_values = [row["sharpness"] for row in candidate_rows]
    sharp_min = min(sharpness_values)
    sharp_max = max(sharpness_values)
    sharp_range = sharp_max - sharp_min

    def composite(row: dict[str, Any]) -> float:
        if sharp_range <= 1e-9:
            sharp_norm = 0.5
        else:
            sharp_norm = max(0.0, min((row["sharpness"] - sharp_min) / sharp_range, 1.0))
        return sharp_norm - (0.4 * row["exposure_penalty"])

    return min(
        candidate_rows,
        key=lambda row: (-composite(row), abs(row["index"] - mark_index), row["index"]),
    )


def random_mark_indices(total_frames: int, frames_per_unit: int) -> List[int]:
    n_random = frames_per_unit if frames_per_unit and frames_per_unit > 0 else total_frames
    n_random = min(n_random, total_frames)
    return sorted(random.sample(range(total_frames), n_random))


def time_grid_mark_indices(total_frames: int, fps: float, frames_per_unit: int, time_unit: str) -> List[int]:
    multiplier = 1 if time_unit == "second" else 60 if time_unit == "minute" else 3600
    frames_in_unit = fps * multiplier
    frames_per_unit = frames_per_unit if frames_per_unit and frames_per_unit > 0 else 1
    interval = max(int(round(frames_in_unit / frames_per_unit)), 1)
    return list(range(0, total_frames, interval))


def interval_mark_indices(total_frames: int, fps: float, interval_seconds: float) -> List[int]:
    if interval_seconds <= 0:
        raise ValueError("Interval seconds must be greater than 0.")
    if fps <= 0:
        raise ValueError("Cannot compute interval marks without a positive FPS.")
    interval = max(int(round(fps * interval_seconds)), 1)
    return list(range(0, total_frames, interval))


def parse_timestamp_seconds(value: str) -> float:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("Time value cannot be empty.")

    parts = cleaned.split(":")
    if len(parts) > 3:
        raise ValueError(f"Invalid time value: {value}")

    total_seconds = 0.0
    multiplier = 1.0
    for index, part in enumerate(reversed(parts)):
        chunk = part.strip()
        if not chunk:
            raise ValueError(f"Invalid time value: {value}")
        try:
            amount = float(chunk) if index == 0 else int(chunk)
        except ValueError as exc:
            raise ValueError(f"Invalid time value: {value}") from exc
        if amount < 0:
            raise ValueError(f"Time values must be positive: {value}")
        total_seconds += amount * multiplier
        multiplier *= 60.0
    return total_seconds


def parse_time_ranges(value: str) -> list[tuple[float, float]]:
    cleaned = value.strip()
    if not cleaned:
        return []

    ranges: list[tuple[float, float]] = []
    for chunk in cleaned.split(","):
        part = chunk.strip().replace("–", "-").replace("—", "-").replace("−", "-")
        if not part:
            continue
        if "-" not in part:
            raise ValueError(
                "Time ranges must look like '1:00 - 1:10, 32:40 - 40:00' or '00:15:30 - 00:18:00'."
            )
        start_text, end_text = [segment.strip() for segment in part.split("-", 1)]
        start_seconds = parse_timestamp_seconds(start_text)
        end_seconds = parse_timestamp_seconds(end_text)
        if end_seconds <= start_seconds:
            raise ValueError(f"Range end must be greater than start: {part}")
        ranges.append((start_seconds, end_seconds))

    if not ranges:
        raise ValueError("No valid time ranges were provided.")

    ranges.sort(key=lambda item: (item[0], item[1]))
    merged: list[tuple[float, float]] = []
    for start_seconds, end_seconds in ranges:
        if not merged:
            merged.append((start_seconds, end_seconds))
            continue
        prev_start, prev_end = merged[-1]
        if start_seconds <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end_seconds))
        else:
            merged.append((start_seconds, end_seconds))
    return merged


def format_timestamp(frame_index: int, fps: float) -> str:
    seconds = 0.0 if fps <= 0 else frame_index / fps
    return format_seconds_timestamp(seconds)


def format_seconds_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def build_mark_ranges(
    total_frames: int,
    fps: float,
    interval_seconds: float,
    time_ranges: Optional[list[tuple[float, float]]] = None,
) -> list[dict[str, Any]]:
    if interval_seconds <= 0:
        raise ValueError("Interval seconds must be greater than 0.")
    if fps <= 0:
        raise ValueError("Cannot compute interval marks without a positive FPS.")
    if total_frames <= 0:
        raise ValueError("Total frames must be greater than 0.")

    duration_seconds = total_frames / fps
    raw_ranges = time_ranges or [(0.0, duration_seconds)]
    interval_frames = max(int(round(fps * interval_seconds)), 1)
    normalized_ranges: list[dict[str, Any]] = []

    for start_seconds, end_seconds in raw_ranges:
        bounded_start = max(0.0, min(start_seconds, duration_seconds))
        bounded_end = max(0.0, min(end_seconds, duration_seconds))
        if bounded_end <= bounded_start:
            continue

        start_index = min(total_frames - 1, max(0, int(math.ceil(bounded_start * fps))))
        end_index = min(total_frames - 1, max(start_index, int(math.floor(bounded_end * fps))))
        if end_index < start_index:
            continue

        mark_indices = list(range(start_index, end_index + 1, interval_frames))
        if not mark_indices:
            mark_indices = [start_index]

        normalized_ranges.append(
            {
                "start_seconds": bounded_start,
                "end_seconds": bounded_end,
                "start_index": start_index,
                "end_index": end_index,
                "label": f"{format_seconds_timestamp(bounded_start)} - {format_seconds_timestamp(bounded_end)}",
                "mark_indices": mark_indices,
            }
        )

    if not normalized_ranges:
        raise ValueError("No frames fall inside the requested time ranges.")

    return normalized_ranges


def get_video_metadata(video_path: str | Path) -> dict[str, Any]:
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if total_frames <= 0:
        raise ValueError(f"Invalid frame count for video: {video_path}")
    if fps <= 0:
        raise ValueError(f"Cannot read FPS for video: {video_path}")

    return {
        "video_path": video_path,
        "total_frames": total_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_seconds": total_frames / fps,
    }


def write_image(path: str | Path, frame, params: Optional[list[int]] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), frame, params or [])
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")


def preview_frame_name(mark_index: int, filename_width: int, total_frames: int) -> str:
    width = max(filename_width, len(str(max(total_frames - 1, 0))))
    return f"{mark_index:0{width}d}.jpg"


def save_frame_name(image_count: int, filename_width: int) -> str:
    return f"{image_count:0{filename_width}d}.png"


def run_ffmpeg_preview_range(
    ffmpeg_exe: str | Path,
    video_path: str | Path,
    temp_dir: str | Path,
    start_seconds: float,
    duration_seconds: float,
    interval_frames: int,
    resolution: int,
    expected_count: int,
    source_width: int,
    source_height: int,
) -> list[Path]:
    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    output_pattern = temp_dir / "%06d.jpg"

    if source_width >= source_height:
        cuda_scale = f"scale_cuda=w=-2:h={resolution}"
        cpu_scale = f"scale=w=-2:h={resolution}"
    else:
        cuda_scale = f"scale_cuda=w={resolution}:h=-2"
        cpu_scale = f"scale=w={resolution}:h=-2"

    cuda_vf = f"{cuda_scale},hwdownload,format=nv12,select='not(mod(n\\,{interval_frames}))'"
    cpu_vf = f"{cpu_scale},select='not(mod(n\\,{interval_frames}))'"

    cmd_cuda = [
        str(ffmpeg_exe),
        "-hide_banner",
        "-loglevel",
        "error",
        "-hwaccel",
        "cuda",
        "-hwaccel_output_format",
        "cuda",
        "-y",
        "-ss",
        format_seconds_timestamp(start_seconds),
        "-i",
        str(video_path),
        "-t",
        format_seconds_timestamp(duration_seconds),
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        cuda_vf,
        "-frames:v",
        str(expected_count),
        "-vsync",
        "0",
        "-q:v",
        "2",
        str(output_pattern),
    ]

    proc = subprocess.run(cmd_cuda, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        cmd_cpu = [
            str(ffmpeg_exe),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            format_seconds_timestamp(start_seconds),
            "-i",
            str(video_path),
            "-t",
            format_seconds_timestamp(duration_seconds),
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            cpu_vf,
            "-frames:v",
            str(expected_count),
            "-vsync",
            "0",
            "-q:v",
            "2",
            str(output_pattern),
        ]
        proc = subprocess.run(cmd_cpu, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            message = proc.stderr.strip() or proc.stdout.strip() or "Unknown ffmpeg error."
            raise RuntimeError(message)

    return sorted(temp_dir.glob("*.jpg"))


def build_preview_set(
    video_path: str | Path,
    preview_dir: str | Path,
    interval_seconds: float,
    resolution: int,
    filename_width: int,
    ffmpeg_exe: str | Path,
    time_ranges: Optional[list[tuple[float, float]]] = None,
) -> dict[str, Any]:
    metadata = get_video_metadata(video_path)
    mark_ranges = build_mark_ranges(
        metadata["total_frames"],
        metadata["fps"],
        interval_seconds,
        time_ranges=time_ranges,
    )

    preview_dir = Path(preview_dir)
    if preview_dir.exists():
        shutil.rmtree(preview_dir)
    preview_dir.mkdir(parents=True, exist_ok=True)

    items: list[dict[str, Any]] = []
    interval_frames = max(int(round(metadata["fps"] * interval_seconds)), 1)

    for mark_range in mark_ranges:
        expected_count = len(mark_range["mark_indices"])
        if expected_count <= 0:
            continue

        with tempfile.TemporaryDirectory(prefix="iml_extract_preview_") as temp_dir:
            generated_files = run_ffmpeg_preview_range(
                ffmpeg_exe=ffmpeg_exe,
                video_path=video_path,
                temp_dir=temp_dir,
                start_seconds=mark_range["start_seconds"],
                duration_seconds=mark_range["end_seconds"] - mark_range["start_seconds"],
                interval_frames=interval_frames,
                resolution=resolution,
                expected_count=expected_count,
                source_width=metadata["width"],
                source_height=metadata["height"],
            )

            for mark_index, generated_path in zip(mark_range["mark_indices"], generated_files):
                preview_name = preview_frame_name(mark_index, filename_width, metadata["total_frames"])
                target_path = preview_dir / preview_name
                generated_path.replace(target_path)

                items.append(
                    {
                        "id": len(items),
                        "mark_index": int(mark_index),
                        "timestamp": format_timestamp(mark_index, metadata["fps"]),
                        "time_seconds": mark_index / metadata["fps"],
                        "preview_name": preview_name,
                        "range_label": mark_range["label"],
                        "range_start_index": mark_range["start_index"],
                        "range_end_index": mark_range["end_index"],
                    }
                )

    return {
        "video": {
            **metadata,
            "selected_ranges": [
                {
                    "label": mark_range["label"],
                    "start_seconds": mark_range["start_seconds"],
                    "end_seconds": mark_range["end_seconds"],
                }
                for mark_range in mark_ranges
            ],
        },
        "items": items,
    }


def merge_selection_windows(
    selections: list[dict[str, Any]],
    buffer: int,
    total_frames: int,
) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    sorted_selections = sorted(selections, key=lambda selection: int(selection["mark_index"]))

    for selection in sorted_selections:
        mark_index = int(selection["mark_index"])
        start_index = max(0, mark_index - buffer)
        end_index = min(total_frames - 1, mark_index + buffer)

        if windows and start_index <= windows[-1]["end_index"] + 1:
            windows[-1]["end_index"] = max(windows[-1]["end_index"], end_index)
            windows[-1]["selections"].append(selection)
        else:
            windows.append(
                {
                    "start_index": start_index,
                    "end_index": end_index,
                    "selections": [selection],
                }
            )

    return windows


def decode_ffmpeg_window(
    ffmpeg_exe: str | Path,
    video_path: str | Path,
    start_index: int,
    end_index: int,
    fps: float,
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    frame_count = max(0, end_index - start_index + 1)
    if frame_count <= 0:
        return []

    frame_bytes = width * height * 3
    if frame_bytes <= 0:
        raise ValueError("Invalid video dimensions for FFmpeg decode.")

    cmd_cuda = [
        str(ffmpeg_exe),
        "-hide_banner",
        "-loglevel",
        "error",
        "-hwaccel",
        "cuda",
        "-hwaccel_output_format",
        "cuda",
        "-ss",
        f"{start_index / fps:.6f}",
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        "hwdownload,format=nv12,format=bgr24",
        "-frames:v",
        str(frame_count),
        "-vsync",
        "0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "pipe:1",
    ]

    cmd_cpu = [
        str(ffmpeg_exe),
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_index / fps:.6f}",
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        "format=bgr24",
        "-frames:v",
        str(frame_count),
        "-vsync",
        "0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "pipe:1",
    ]

    def try_decode(cmd: list[str]) -> list[dict[str, Any]]:
        nonlocal height, width
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        local_rows: list[dict[str, Any]] = []

        try:
            if proc.stdout is None:
                raise RuntimeError("FFmpeg pipe did not open.")

            for offset in range(frame_count):
                raw = proc.stdout.read(frame_bytes)
                if len(raw) != frame_bytes:
                    break

                frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3)).copy()
                local_rows.append(
                    {
                        "index": start_index + offset,
                        "frame": frame,
                        "sharpness": sharpness_score(frame),
                        "exposure_penalty": exposure_penalty(frame),
                    }
                )
        finally:
            stderr = b""
            if proc.stdout is not None:
                proc.stdout.close()
            if proc.stderr is not None:
                stderr = proc.stderr.read()
                proc.stderr.close()
            proc.wait()

        if proc.returncode != 0:
            message = stderr.decode("utf-8", errors="replace").strip() or "Unknown ffmpeg error."
            raise RuntimeError(message)

        return local_rows

    try:
        return try_decode(cmd_cuda)
    except RuntimeError:
        return try_decode(cmd_cpu)


def save_native_frames(
    ffmpeg_exe: str | Path,
    video_path: str | Path,
    output_dir: str | Path,
    selections: list[dict[str, Any]],
    buffer: int,
    filename_width: int,
    start_count: int = 1,
) -> list[dict[str, Any]]:
    metadata = get_video_metadata(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    image_count = start_count
    windows = merge_selection_windows(selections, buffer, metadata["total_frames"])
    decoded_by_mark_index: dict[int, dict[str, Any]] = {}

    for window in windows:
        decoded_rows = decode_ffmpeg_window(
            ffmpeg_exe=ffmpeg_exe,
            video_path=video_path,
            start_index=int(window["start_index"]),
            end_index=int(window["end_index"]),
            fps=metadata["fps"],
            width=metadata["width"],
            height=metadata["height"],
        )
        rows_by_index = {int(row["index"]): row for row in decoded_rows}

        for selection in window["selections"]:
            mark_index = int(selection["mark_index"])
            candidate_rows = [
                rows_by_index[idx]
                for idx in candidate_indices(mark_index, buffer, metadata["total_frames"])
                if idx in rows_by_index
            ]
            best_row = select_best_candidate_rows(candidate_rows, mark_index)
            if best_row is None:
                continue
            decoded_by_mark_index[mark_index] = {
                "selection": selection,
                "winner_index": int(best_row["index"]),
                "frame": best_row["frame"],
            }

    for selection in selections:
        mark_index = int(selection["mark_index"])
        row = decoded_by_mark_index.get(mark_index)
        if row is None:
            continue

        filename = save_frame_name(image_count, filename_width)
        write_image(output_dir / filename, row["frame"])
        results.append(
            {
                "filename": filename,
                "mark_index": mark_index,
                "winner_index": int(row["winner_index"]),
                "mark_timestamp": format_timestamp(mark_index, metadata["fps"]),
                "winner_timestamp": format_timestamp(int(row["winner_index"]), metadata["fps"]),
                "range_label": selection.get("range_label"),
            }
        )
        image_count += 1

    return results


def extract_frames(
    video_path: str,
    output_dir: str,
    frames_per_unit: int,
    resolution: Optional[int],
    time_unit: str,
    start_count: int,
    filename_width: int,
    extract_all: bool,
    random_mode: bool,
    buffer: int,
    video_counter: int,
    video_total: int,
) -> int:
    """Extract frames from video_path and write them into output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f" ! Cannot open video: {video_path}")
        return start_count

    image_count = start_count
    filename_width = max(1, filename_width)

    if extract_all:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0:
            print(f"[{video_counter}/{video_total}] All mode: {video_path} - extracting all {total_frames} frame(s).")
        else:
            print(f"[{video_counter}/{video_total}] All mode: {video_path} - extracting all frames.")

        success, frame = cap.read()
        while success:
            frame = resize_preserve_aspect(frame, resolution)
            cv2.imwrite(os.path.join(output_dir, f"{image_count:0{filename_width}d}.png"), frame)
            image_count += 1
            success, frame = cap.read()

        cap.release()
        return image_count

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print(f" ! Skipping {video_path} (invalid frame count).")
        cap.release()
        return image_count

    if random_mode:
        mark_indices = random_mark_indices(total_frames, frames_per_unit)
        print(
            f"[{video_counter}/{video_total}] Random mode: {video_path} - extracting {len(mark_indices)} frame(s)"
            f"{'' if buffer <= 0 else f' with buffer {buffer}'}."
        )
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            print(f" ! Skipping {video_path} (cannot read FPS).")
            cap.release()
            return image_count

        mark_indices = time_grid_mark_indices(total_frames, fps, frames_per_unit, time_unit)
        interval = mark_indices[1] - mark_indices[0] if len(mark_indices) >= 2 else total_frames
        print(
            f"[{video_counter}/{video_total}] {video_path} - {frames_per_unit} frame(s) per "
            f"{time_unit} (fps={fps:.2f}) -> interval {interval}"
            f"{'' if buffer <= 0 else f', buffer {buffer}'}"
        )

    for mark_index in mark_indices:
        frame, _winner_index = select_best_frame(cap, mark_index, total_frames, buffer)
        if frame is None:
            print(f" ! Failed to read mark {mark_index} in {video_path}")
            continue

        resized = resize_preserve_aspect(frame, resolution)
        cv2.imwrite(os.path.join(output_dir, f"{image_count:0{filename_width}d}.png"), resized)
        image_count += 1

    cap.release()
    return image_count


def initial_global_count(dir_path: str | Path) -> int:
    """Return next image id for dir_path by counting existing image files."""
    existing = [
        f
        for f in os.listdir(dir_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ]
    return len(existing) + 1


def collect_video_paths(input_path: str | Path) -> List[str]:
    """Collect supported video file paths recursively from input_path."""
    video_paths: List[str] = []
    for root, _, files in os.walk(input_path):
        for file in files:
            if file.lower().endswith(VIDEO_EXTENSIONS):
                video_paths.append(os.path.join(root, file))
    video_paths.sort()
    return video_paths
