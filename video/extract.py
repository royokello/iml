import argparse
import os
import random
from typing import List, Optional

import cv2
import numpy as np


VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv")


def _resize_preserve_aspect(frame, target_short_side: Optional[int]) -> "cv2.Mat":
    """Resize frame so its short side equals target_short_side.

    If target_short_side is None or <= 0, the original frame is returned.
    """
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


def _read_frame_at(cap: cv2.VideoCapture, frame_index: int) -> Optional["cv2.Mat"]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    success, frame = cap.read()
    return frame if success else None


def _sharpness_score(frame: "cv2.Mat") -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _exposure_penalty(frame: "cv2.Mat") -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_norm = float(gray.mean()) / 255.0
    mean_penalty = abs(mean_norm - 0.5) / 0.5
    clipped_dark = float(np.mean(gray <= 5))
    clipped_bright = float(np.mean(gray >= 250))
    clip_penalty = min((clipped_dark + clipped_bright) / 0.25, 1.0)
    return min(0.7 * mean_penalty + 0.3 * clip_penalty, 1.0)


def _candidate_indices(mark_index: int, buffer: int, total_frames: int) -> List[int]:
    if buffer <= 0:
        return [mark_index]
    start = max(0, mark_index - buffer)
    end = min(total_frames - 1, mark_index + buffer)
    return list(range(start, end + 1))


def _select_best_frame(
    cap: cv2.VideoCapture,
    mark_index: int,
    total_frames: int,
    buffer: int,
) -> tuple[Optional["cv2.Mat"], int]:
    candidate_rows = []
    for idx in _candidate_indices(mark_index, buffer, total_frames):
        frame = _read_frame_at(cap, idx)
        if frame is None:
            continue
        candidate_rows.append(
            {
                "index": idx,
                "frame": frame,
                "sharpness": _sharpness_score(frame),
                "exposure_penalty": _exposure_penalty(frame),
            }
        )

    if not candidate_rows:
        return None, mark_index

    sharpness_values = [row["sharpness"] for row in candidate_rows]
    sharp_min = min(sharpness_values)
    sharp_max = max(sharpness_values)
    sharp_range = sharp_max - sharp_min

    def _composite(row) -> float:
        if sharp_range <= 1e-9:
            sharp_norm = 0.5
        else:
            sharp_norm = max(0.0, min((row["sharpness"] - sharp_min) / sharp_range, 1.0))
        return sharp_norm - (0.4 * row["exposure_penalty"])

    best = min(
        candidate_rows,
        key=lambda row: (-_composite(row), abs(row["index"] - mark_index), row["index"]),
    )
    return best["frame"], int(best["index"])


def _random_mark_indices(total_frames: int, frames_per_unit: int) -> List[int]:
    n_random = frames_per_unit if frames_per_unit and frames_per_unit > 0 else total_frames
    n_random = min(n_random, total_frames)
    return sorted(random.sample(range(total_frames), n_random))


def _time_grid_mark_indices(total_frames: int, fps: float, frames_per_unit: int, time_unit: str) -> List[int]:
    multiplier = 1 if time_unit == "second" else 60 if time_unit == "minute" else 3600
    frames_in_unit = fps * multiplier
    frames_per_unit = frames_per_unit if frames_per_unit and frames_per_unit > 0 else 1
    interval = max(int(round(frames_in_unit / frames_per_unit)), 1)
    return list(range(0, total_frames, interval))


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

    # All-frames mode: extract every frame in decode order.
    if extract_all:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0:
            print(f"[{video_counter}/{video_total}] All mode: {video_path} - extracting all {total_frames} frame(s).")
        else:
            print(f"[{video_counter}/{video_total}] All mode: {video_path} - extracting all frames.")

        success, frame = cap.read()
        while success:
            frame = _resize_preserve_aspect(frame, resolution)
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
        mark_indices = _random_mark_indices(total_frames, frames_per_unit)
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

        mark_indices = _time_grid_mark_indices(total_frames, fps, frames_per_unit, time_unit)
        interval = mark_indices[1] - mark_indices[0] if len(mark_indices) >= 2 else total_frames
        print(
            f"[{video_counter}/{video_total}] {video_path} - {frames_per_unit} frame(s) per "
            f"{time_unit} (fps={fps:.2f}) -> interval {interval}"
            f"{'' if buffer <= 0 else f', buffer {buffer}'}"
        )

    for mark_index in mark_indices:
        frame, _winner_index = _select_best_frame(cap, mark_index, total_frames, buffer)
        if frame is None:
            print(f" ! Failed to read mark {mark_index} in {video_path}")
            continue

        resized = _resize_preserve_aspect(frame, resolution)
        cv2.imwrite(os.path.join(output_dir, f"{image_count:0{filename_width}d}.png"), resized)
        image_count += 1

    cap.release()
    return image_count


def _initial_global_count(dir_path: str) -> int:
    """Return next image id for dir_path by counting existing image files."""
    existing = [f for f in os.listdir(dir_path) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
    return len(existing) + 1


def _collect_video_paths(input_path: str) -> List[str]:
    """Collect supported video file paths recursively from input_path."""
    video_paths: List[str] = []
    for root, _, files in os.walk(input_path):
        for file in files:
            if file.lower().endswith(VIDEO_EXTENSIONS):
                video_paths.append(os.path.join(root, file))
    video_paths.sort()
    return video_paths


def main(args):
    input_path = args.input
    output_root = args.output
    resolution = args.resolution

    if args.buffer < 0:
        raise SystemExit("--buffer must be >= 0.")
    if args.all and args.buffer > 0:
        print(" ! Ignoring --buffer because --all extracts exact decode-order frames.")

    if args.collate:
        os.makedirs(output_root, exist_ok=True)
        global_counter = _initial_global_count(output_root)
    else:
        global_counter = 1

    def _process_video(path: str, out_dir: str, start_idx: int, video_counter: int, video_total: int) -> int:
        return extract_frames(
            video_path=path,
            output_dir=out_dir,
            frames_per_unit=args.frames,
            resolution=resolution,
            time_unit=args.time,
            start_count=start_idx,
            filename_width=args.filename_width,
            extract_all=args.all,
            random_mode=args.random and not args.all,
            buffer=0 if args.all else args.buffer,
            video_counter=video_counter,
            video_total=video_total,
        )

    if os.path.isfile(input_path):
        if args.collate:
            _process_video(input_path, output_root, global_counter, 1, 1)
        else:
            video_name = os.path.splitext(os.path.basename(input_path))[0]
            vid_out = os.path.join(output_root, video_name)
            os.makedirs(vid_out, exist_ok=True)
            _process_video(input_path, vid_out, 1, 1, 1)
        return

    video_paths = _collect_video_paths(input_path)
    total_videos = len(video_paths)
    if total_videos == 0:
        print(f" ! No supported videos found in: {input_path}")
        return

    for video_counter, vid_path in enumerate(video_paths, start=1):
        if args.collate:
            global_counter = _process_video(vid_path, output_root, global_counter, video_counter, total_videos)
        else:
            vid_name = os.path.splitext(os.path.basename(vid_path))[0]
            vid_out = os.path.join(output_root, vid_name)
            os.makedirs(vid_out, exist_ok=True)
            _process_video(vid_path, vid_out, 1, video_counter, total_videos)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Extract frames from video files. By default it grabs a fixed number "
            "of frames per second/minute/hour. If --random is supplied it instead "
            "samples up to --frames random frames from the whole clip. If --all is "
            "supplied it extracts every frame and overrides --frames/--time/--random. "
            "Use --buffer to compare nearby frames around each mark and save the best "
            "scoring one. Use --collate to dump everything into a single directory."
        )
    )

    parser.add_argument("--input", required=True, help="Path to a video file or a directory containing videos.")
    parser.add_argument("--output", required=True, help="Directory where frames will be saved.")
    parser.add_argument("--frames", type=int, default=1, help="Frames per time unit (or max random frames if --random).")
    parser.add_argument("--time", choices=["second", "minute", "hour"], default="second", help="Time unit basis.")
    parser.add_argument("--random", action="store_true", help="Enable random frame sampling mode.")
    parser.add_argument("--all", action="store_true", help="Extract all frames. Overrides --frames, --time, and --random.")
    parser.add_argument(
        "--buffer",
        type=int,
        default=0,
        help="Compare frames in a +/- buffer window around each mark and save the best scorer. Ignored by --all.",
    )
    parser.add_argument("--collate", action="store_true", help="Save all frames into the output directory itself.")
    parser.add_argument("--filename-width", type=int, default=6, help="Zero-pad width for output frame filenames.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Shortest side of saved frames (px). Omit to keep original frame size.",
    )

    main(parser.parse_args())
