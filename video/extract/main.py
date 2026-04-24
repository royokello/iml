from __future__ import annotations

import argparse
import os

from .core import collect_video_paths, extract_frames, initial_global_count


def run(args: argparse.Namespace) -> None:
    input_path = args.input
    output_root = args.output
    resolution = args.resolution

    if args.buffer < 0:
        raise SystemExit("--buffer must be >= 0.")
    if args.all and args.buffer > 0:
        print(" ! Ignoring --buffer because --all extracts exact decode-order frames.")

    if args.collate:
        os.makedirs(output_root, exist_ok=True)
        global_counter = initial_global_count(output_root)
    else:
        global_counter = 1

    def process_video(path: str, out_dir: str, start_idx: int, video_counter: int, video_total: int) -> int:
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
            process_video(input_path, output_root, global_counter, 1, 1)
        else:
            video_name = os.path.splitext(os.path.basename(input_path))[0]
            vid_out = os.path.join(output_root, video_name)
            os.makedirs(vid_out, exist_ok=True)
            process_video(input_path, vid_out, 1, 1, 1)
        return

    video_paths = collect_video_paths(input_path)
    total_videos = len(video_paths)
    if total_videos == 0:
        print(f" ! No supported videos found in: {input_path}")
        return

    for video_counter, vid_path in enumerate(video_paths, start=1):
        if args.collate:
            global_counter = process_video(vid_path, output_root, global_counter, video_counter, total_videos)
        else:
            vid_name = os.path.splitext(os.path.basename(vid_path))[0]
            vid_out = os.path.join(output_root, vid_name)
            os.makedirs(vid_out, exist_ok=True)
            process_video(vid_path, vid_out, 1, video_counter, total_videos)


def build_parser() -> argparse.ArgumentParser:
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
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
