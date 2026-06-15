import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np


def _run_cmd(cmd, verbose=False):
    if verbose:
        print(" ".join(cmd))
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.returncode, p.stdout, p.stderr


def _resolve_tool(ffmpeg_path):
    p = Path(ffmpeg_path)
    if p.name.lower() == "ffmpeg.exe":
        ffprobe = str(p.parent / "ffprobe.exe")
    else:
        ffprobe = str(p.parent / "ffprobe")
    return str(p), ffprobe


def _selection_mode_label(selection_mode):
    labels = {
        "interval": "Frame interval",
        "segment_middle": "Segment midpoints",
    }
    return labels.get(selection_mode, selection_mode.replace("_", " ").title())


def _center_crop_to_aspect(frame, target_aspect):
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


def _resize_frame(frame, target_width, target_height):
    if target_width <= 0 or target_height <= 0:
        return frame
    height, width = frame.shape[:2]
    if height == 0 or width == 0:
        return frame
    scale = target_width / width
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(frame, (target_width, target_height), interpolation=interpolation)


def _build_grid_image(frames, grid_rows, grid_cols, target_aspect, cell_width, cell_height):
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


def _segment_middle_indices(total_frames, segment_count):
    if total_frames <= 0:
        raise ValueError("Video does not contain any frames.")
    if segment_count <= 0:
        raise ValueError("Segment count must be positive.")
    if total_frames < segment_count:
        raise ValueError(
            f"Video has only {total_frames} frames, but {segment_count} are required to fill the grid."
        )
    indices = []
    for segment_idx in range(segment_count):
        start = (segment_idx * total_frames) // segment_count
        end = ((segment_idx + 1) * total_frames) // segment_count
        middle = start + ((end - start - 1) // 2)
        indices.append(middle)
    return indices


def _ffprobe_meta(ffprobe, path):
    cmd = [ffprobe, "-v", "error", "-print_format", "json",
           "-show_format", "-show_streams", path]
    rc, out, err = _run_cmd(cmd)
    if rc != 0:
        raise RuntimeError(f"ffprobe failed: {err}")
    j = json.loads(out)
    v = next((s for s in j.get("streams", []) if s.get("codec_type") == "video"), {})
    fmt = j.get("format", {})
    dur = float(fmt.get("duration", 0) or 0)
    r = v.get("r_frame_rate") or v.get("avg_frame_rate") or "0/1"
    fps = 0.0
    try:
        a, b = r.split("/")
        fps = float(a) / (float(b) if float(b) != 0 else 1.0)
    except Exception:
        pass
    nb = v.get("nb_frames")
    return {
        "width": int(v.get("width") or 0),
        "height": int(v.get("height") or 0),
        "fps": fps,
        "duration_s": dur,
        "frame_count": int(nb) if nb else None,
    }


def _count_frames_ffprobe(ffprobe, path):
    cmd = [ffprobe, "-v", "error", "-count_frames",
           "-select_streams", "v:0",
           "-show_entries", "stream=nb_read_frames",
           "-of", "csv=p=0",
           path]
    rc, out, err = _run_cmd(cmd)
    if rc != 0:
        raise RuntimeError(f"ffprobe frame count failed: {err}")
    try:
        return int(out.strip())
    except ValueError:
        raise RuntimeError(f"Could not parse frame count: {out}")


def _extract_frames(ffmpeg, video_path, tmpdir, filter_expr, verbose=False):
    output_pattern = str(tmpdir / "frame_%06d.png")
    cmd = [ffmpeg, "-hide_banner", "-nostats",
           "-loglevel", "info" if verbose else "error",
           "-i", str(video_path),
           "-vf", filter_expr,
           "-vsync", "0",
           output_pattern]
    rc, _, err = _run_cmd(cmd, verbose=verbose)
    if rc != 0:
        raise RuntimeError(f"ffmpeg frame extraction failed: {err}")
    frames = sorted(tmpdir.glob("frame_*.png"))
    if not frames:
        raise RuntimeError("ffmpeg extracted 0 frames.")
    return frames


def create_video_grids(
    video_path,
    output_dir,
    ffmpeg,
    grid_rows,
    grid_cols,
    cell_ratio,
    cell_height,
    frame_interval_sec=None,
    alignment="center",
    selection_mode="interval",
    verbose=False,
):
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")
    if alignment != "center":
        raise ValueError("Only center alignment is supported.")

    ffmpeg_bin, ffprobe = _resolve_tool(ffmpeg)

    ratio_w, ratio_h = cell_ratio
    if ratio_w <= 0 or ratio_h <= 0:
        raise ValueError("Cell ratio must be positive.")
    if grid_rows <= 0 or grid_cols <= 0:
        raise ValueError("Grid rows/cols must be positive.")
    if cell_height <= 0:
        raise ValueError("Cell height must be greater than 0.")
    if selection_mode not in {"interval", "segment_middle"}:
        raise ValueError(f"Unsupported selection mode: {selection_mode}")
    if selection_mode == "interval" and (frame_interval_sec is None or frame_interval_sec <= 0):
        raise ValueError("Frame interval must be greater than 0.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = _ffprobe_meta(ffprobe, video_path)
    fps = meta["fps"]
    total_frames = meta["frame_count"]
    if fps <= 0:
        raise ValueError("Could not read FPS from the video.")

    frames_per_grid = grid_rows * grid_cols
    cell_width = max(int(round(cell_height * (ratio_w / ratio_h))), 1)
    target_aspect = ratio_w / ratio_h

    outputs = []
    buffer = []
    selected_frame_indices = []
    selected_frame_count = 0
    remainder_frames = 0
    interval_frames = None

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)

        if selection_mode == "interval":
            interval_frames = max(int(round(fps * frame_interval_sec)), 1)
            if total_frames and total_frames > 0:
                sampled_frames = ((total_frames - 1) // interval_frames) + 1
                grids_estimate = max(sampled_frames // frames_per_grid, 1)
            else:
                grids_estimate = 1
            pad_width = len(str(grids_estimate))

            filter_expr = f"select='not(mod(n,{interval_frames}))'"
            frame_paths = _extract_frames(ffmpeg_bin, path, tmpdir, filter_expr, verbose)

            for idx, frame_path in enumerate(frame_paths):
                frame = cv2.imread(str(frame_path))
                if frame is None:
                    raise RuntimeError(f"Failed to read extracted frame: {frame_path}")
                selected_frame_count += 1
                buffer.append(frame)
                if len(buffer) == frames_per_grid:
                    grid_image = _build_grid_image(
                        buffer, grid_rows, grid_cols, target_aspect, cell_width, cell_height
                    )
                    filename = f"{len(outputs) + 1:0{pad_width}d}.png"
                    output_path = output_dir / filename
                    if not cv2.imwrite(str(output_path), grid_image):
                        raise RuntimeError("Failed to write grid image.")
                    outputs.append({
                        "output_path": str(output_path),
                        "filename": filename,
                        "grid_index": len(outputs) + 1,
                    })
                    buffer = []

            remainder_frames = len(buffer)
            if not outputs:
                raise ValueError(
                    f"Video too short for a {grid_rows}x{grid_cols} grid "
                    f"at {frame_interval_sec}s intervals."
                )
        else:
            if not total_frames or total_frames <= 0:
                total_frames = _count_frames_ffprobe(ffprobe, video_path)
            selected_frame_indices = _segment_middle_indices(total_frames, frames_per_grid)
            filter_expr = "+".join(f"eq(n,{i})" for i in selected_frame_indices)
            frame_paths = _extract_frames(ffmpeg_bin, path, tmpdir, filter_expr, verbose)

            buffer = [cv2.imread(str(fp)) for fp in frame_paths]
            if any(f is None for f in buffer):
                raise RuntimeError("Failed to read one or more extracted frames.")

            grid_image = _build_grid_image(
                buffer, grid_rows, grid_cols, target_aspect, cell_width, cell_height
            )
            filename = "1.png"
            output_path = output_dir / filename
            if not cv2.imwrite(str(output_path), grid_image):
                raise RuntimeError("Failed to write grid image.")
            outputs.append({
                "output_path": str(output_path),
                "filename": filename,
                "grid_index": 1,
            })

    sampling_config = {
        "selection_mode": selection_mode,
        "selection_mode_label": _selection_mode_label(selection_mode),
        "selected_frame_count": selected_frame_count if selection_mode == "interval" else len(selected_frame_indices),
    }
    if selection_mode == "interval":
        sampling_config["frame_interval_seconds"] = frame_interval_sec
        sampling_config["interval_frames"] = interval_frames
    else:
        sampling_config["segment_count"] = frames_per_grid
        sampling_config["selected_frame_indices"] = selected_frame_indices

    config = {
        "video_path": str(path),
        "grid_rows": grid_rows,
        "grid_cols": grid_cols,
        "cell_ratio": {"width": ratio_w, "height": ratio_h},
        "cell_height": cell_height,
        "cell_width": cell_width,
        "selection_mode": selection_mode,
        "selection_mode_label": _selection_mode_label(selection_mode),
        "frame_interval_seconds": frame_interval_sec,
        "alignment": alignment,
        "frames_per_grid": frames_per_grid,
        "interval_frames": interval_frames if selection_mode == "interval" else None,
        "fps": fps,
        "total_frames": total_frames,
        "grids_made": len(outputs),
        "remainder_frames": remainder_frames,
        "sampling": sampling_config,
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
        "selection_mode": selection_mode,
        "selection_mode_label": _selection_mode_label(selection_mode),
        "interval_seconds": frame_interval_sec,
        "grids_made": len(outputs),
        "remainder_frames": remainder_frames,
        "config": config,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate video grid images")
    parser.add_argument("-i", "--input", required=True, help="Input video path")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument("--ffmpeg", required=True, help="Path to ffmpeg executable")
    parser.add_argument("--rows", type=int, default=2, help="Grid rows (default: 2)")
    parser.add_argument("--cols", type=int, default=2, help="Grid columns (default: 2)")
    parser.add_argument("--cell-ratio", default="1x1",
                        help="Cell aspect ratio WxH (default: 1x1)")
    parser.add_argument("--cell-height", type=int, default=384,
                        help="Cell height in pixels (default: 384)")
    parser.add_argument("--selection-mode", default="interval",
                        choices=["interval", "segment_middle"],
                        help="Frame selection mode (default: interval)")
    parser.add_argument("--frame-interval", type=float, default=1.0,
                        help="Frame interval in seconds (default: 1.0, interval mode only)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose ffmpeg output")
    args = parser.parse_args()

    parts = args.cell_ratio.lower().replace(":", "x").split("x")
    if len(parts) != 2:
        parser.error("--cell-ratio must be in WxH format (e.g. 16x9)")
    try:
        ratio_w, ratio_h = int(parts[0]), int(parts[1])
    except ValueError:
        parser.error("--cell-ratio must contain integers")

    if args.dry_run:
        print(f"Input: {args.input}")
        print(f"Output: {args.output}")
        print(f"FFmpeg: {args.ffmpeg}")
        print(f"Grid: {args.rows}x{args.cols}")
        print(f"Cell ratio: {ratio_w}x{ratio_h}")
        print(f"Cell height: {args.cell_height}")
        print(f"Selection mode: {args.selection_mode}")
        if args.selection_mode == "interval":
            print(f"Frame interval: {args.frame_interval}s")
        print("[dry-run] No frames extracted.")
        sys.exit(0)

    try:
        result = create_video_grids(
            video_path=args.input,
            output_dir=args.output,
            ffmpeg=args.ffmpeg,
            grid_rows=args.rows,
            grid_cols=args.cols,
            cell_ratio=(ratio_w, ratio_h),
            cell_height=args.cell_height,
            frame_interval_sec=args.frame_interval if args.selection_mode == "interval" else None,
            selection_mode=args.selection_mode,
            verbose=args.verbose,
        )
        print(f"Created {result['grids_made']} grid(s) in {args.output}")
        for o in result["outputs"]:
            print(f"  {o['filename']}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
