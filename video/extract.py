import argparse
import cv2
import os
import random


def _resize_preserve_aspect(frame, target_short_side: int) -> "cv2.Mat":
    """Resize *frame* so that its **shortest** side equals *target_short_side* while
    preserving aspect‑ratio. Uses *INTER_AREA* for downscaling and *INTER_LINEAR*
    for up‑scaling.
    """
    h, w = frame.shape[:2]
    if target_short_side <= 0:
        return frame  # invalid or "no resize" sentinel

    short, long_ = (h, w) if h < w else (w, h)
    if short == target_short_side:
        return frame  # already the requested size

    scale = target_short_side / short
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(frame, (new_w, new_h), interpolation=interpolation)


def extract_frames(
    video_path: str,
    output_dir: str,
    frames_per_unit: int,
    resolution: int,
    time_unit: str,
    start_count: int,
    random_mode: bool,
):
    """Extract frames from *video_path* and write them into *output_dir*.

    When *random_mode* is **False** the function extracts *frames_per_unit* frames
    **per** *time_unit* (second / minute / hour) using an evenly spaced interval.

    When *random_mode* is **True**, up to *frames_per_unit* **random** frames are
    sampled from the entire clip (if *frames_per_unit* is 0 or negative, every
    frame is eligible).

    All saved frames are resized so that the shorter side equals *resolution*.
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f" ! Cannot open video: {video_path}")
        return start_count

    image_count = start_count

    # --------------------------------------------------
    # RANDOM‑MODE BRANCH
    # --------------------------------------------------
    if random_mode:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            print(f" ! Skipping {video_path} (invalid frame count).")
            cap.release()
            return image_count

        n_random = frames_per_unit if frames_per_unit and frames_per_unit > 0 else total_frames
        n_random = min(n_random, total_frames)

        indices = sorted(random.sample(range(total_frames), n_random))
        print(f" * Random mode: {video_path} – extracting {n_random} frame(s).")

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, frame = cap.read()
            if not success:
                print(f" ! Failed to read frame {idx} in {video_path}")
                continue

            frame = _resize_preserve_aspect(frame, resolution)
            cv2.imwrite(os.path.join(output_dir, f"{image_count}.png"), frame)
            image_count += 1

        cap.release()
        return image_count

    # --------------------------------------------------
    # TIME‑GRID BRANCH
    # --------------------------------------------------
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print(f" ! Skipping {video_path} (cannot read FPS).")
        cap.release()
        return image_count

    multiplier = 1 if time_unit == "second" else 60 if time_unit == "minute" else 3600
    frames_in_unit = fps * multiplier
    # Protect against division by 0 / negative
    frames_per_unit = frames_per_unit if frames_per_unit and frames_per_unit > 0 else 1
    interval = max(int(round(frames_in_unit / frames_per_unit)), 1)

    print(
        f" * Processing {video_path} – {frames_per_unit} frame(s) per {time_unit} (fps={fps:.2f}) → interval {interval}"
    )

    frame_count = 0
    success, frame = cap.read()
    while success:
        if frame_count % interval == 0:
            resized = _resize_preserve_aspect(frame, resolution)
            cv2.imwrite(os.path.join(output_dir, f"{image_count}.png"), resized)
            image_count += 1
        frame_count += 1
        success, frame = cap.read()

    cap.release()
    return image_count


def _initial_global_count(dir_path: str) -> int:
    """Return next image id for *dir_path* by counting existing image files."""
    existing = [f for f in os.listdir(dir_path) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
    return len(existing) + 1


def main(args):
    input_path = args.input
    output_root = args.output

    # Normalise resolution argument (may come in as str)
    try:
        resolution = int(args.resolution)
    except ValueError:
        print(f" ! Invalid resolution '{args.resolution}', defaulting to 512")
        resolution = 512

    # Collate mode prepares one shared directory and a global counter
    if args.collate:
        os.makedirs(output_root, exist_ok=True)
        global_counter = _initial_global_count(output_root)
    else:
        global_counter = 1

    # Helper to process a single video path
    def _process_video(path: str, out_dir: str, start_idx: int):
        return extract_frames(
            video_path=path,
            output_dir=out_dir,
            frames_per_unit=args.frames,
            resolution=resolution,
            time_unit=args.time,
            start_count=start_idx,
            random_mode=args.random,
        )

    # If the input is a single file, handle directly
    if os.path.isfile(input_path):
        if args.collate:
            _process_video(input_path, output_root, global_counter)
        else:
            video_name = os.path.splitext(os.path.basename(input_path))[0]
            vid_out = os.path.join(output_root, video_name)
            os.makedirs(vid_out, exist_ok=True)
            _process_video(input_path, vid_out, 1)
        return

    # Otherwise walk directory
    for root, _, files in os.walk(input_path):
        for file in files:
            if not file.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv")):
                continue
            vid_path = os.path.join(root, file)
            if args.collate:
                new_count = _process_video(vid_path, output_root, global_counter)
                global_counter = new_count  # update for next clip
            else:
                vid_name = os.path.splitext(file)[0]
                vid_out = os.path.join(output_root, vid_name)
                os.makedirs(vid_out, exist_ok=True)
                _process_video(vid_path, vid_out, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Extract frames from video files. By default it grabs a fixed number "
            "of frames per second/minute/hour. If --random is supplied it instead "
            "samples up to --frames random frames from the whole clip. Use --collate "
            "to dump everything into a single directory."
        )
    )

    parser.add_argument("--input", required=True, help="Path to a video file or a directory containing videos.")
    parser.add_argument("--output", required=True, help="Directory where frames will be saved.")
    parser.add_argument("--frames", type=int, default=1, help="Frames per time unit (or max random frames if --random).")
    parser.add_argument("--time", choices=["second", "minute", "hour"], default="second", help="Time unit basis.")
    parser.add_argument("--random", action="store_true", help="Enable random frame sampling mode.")
    parser.add_argument("--collate", action="store_true", help="Save all frames into the output directory itself.")
    parser.add_argument("--resolution", type=int, default=512, help="Shortest side of saved frames (px).")

    main(parser.parse_args())
