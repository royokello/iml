import argparse
import os
import random
from typing import List, Optional

import cv2


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

    # Random mode: sample frame indices across the full clip.
    if random_mode:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            print(f" ! Skipping {video_path} (invalid frame count).")
            cap.release()
            return image_count

        n_random = frames_per_unit if frames_per_unit and frames_per_unit > 0 else total_frames
        n_random = min(n_random, total_frames)
        indices = sorted(random.sample(range(total_frames), n_random))

        print(f"[{video_counter}/{video_total}] Random mode: {video_path} - extracting {n_random} frame(s).")

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, frame = cap.read()
            if not success:
                print(f" ! Failed to read frame {idx} in {video_path}")
                continue

            frame = _resize_preserve_aspect(frame, resolution)
            cv2.imwrite(os.path.join(output_dir, f"{image_count:0{filename_width}d}.png"), frame)
            image_count += 1

        cap.release()
        return image_count

    # Time-grid mode: extract evenly at an interval based on FPS and time unit.
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print(f" ! Skipping {video_path} (cannot read FPS).")
        cap.release()
        return image_count

    multiplier = 1 if time_unit == "second" else 60 if time_unit == "minute" else 3600
    frames_in_unit = fps * multiplier
    frames_per_unit = frames_per_unit if frames_per_unit and frames_per_unit > 0 else 1
    interval = max(int(round(frames_in_unit / frames_per_unit)), 1)

    print(
        f"[{video_counter}/{video_total}] {video_path} - {frames_per_unit} frame(s) per "
        f"{time_unit} (fps={fps:.2f}) -> interval {interval}"
    )

    frame_count = 0
    success, frame = cap.read()
    while success:
        if frame_count % interval == 0:
            resized = _resize_preserve_aspect(frame, resolution)
            cv2.imwrite(os.path.join(output_dir, f"{image_count:0{filename_width}d}.png"), resized)
            image_count += 1
        frame_count += 1
        success, frame = cap.read()

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
            "Use --collate to dump everything into a single directory."
        )
    )

    parser.add_argument("--input", required=True, help="Path to a video file or a directory containing videos.")
    parser.add_argument("--output", required=True, help="Directory where frames will be saved.")
    parser.add_argument("--frames", type=int, default=1, help="Frames per time unit (or max random frames if --random).")
    parser.add_argument("--time", choices=["second", "minute", "hour"], default="second", help="Time unit basis.")
    parser.add_argument("--random", action="store_true", help="Enable random frame sampling mode.")
    parser.add_argument("--all", action="store_true", help="Extract all frames. Overrides --frames, --time, and --random.")
    parser.add_argument("--collate", action="store_true", help="Save all frames into the output directory itself.")
    parser.add_argument("--filename-width", type=int, default=6, help="Zero-pad width for output frame filenames.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Shortest side of saved frames (px). Omit to keep original frame size.",
    )

    main(parser.parse_args())
