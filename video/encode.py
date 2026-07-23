import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from datetime import date
from pathlib import Path
from typing import List, Tuple

from utils.banner import print_banner

import numpy as np

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".mpg", ".mpeg"}
DEFAULT_CRFS = [16, 20, 24, 28, 32, 36, 40]


def _run_cmd(cmd: List[str], verbose: bool = False) -> Tuple[int, str, str]:
    if verbose:
        print(" ".join(cmd))
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.returncode, p.stdout, p.stderr


def _ffprobe_meta(ffprobe: str, path: str) -> dict:
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
    return {
        "width": int(v.get("width") or 0),
        "height": int(v.get("height") or 0),
        "fps": fps,
        "duration_s": dur,
    }


def _sample_offsets(duration: float, sample_length: float, num_samples: int) -> List[float]:
    if duration <= 0:
        return [0.0]
    count = max(1, num_samples)
    max_start = max(duration - max(1.0, sample_length), 0.0)
    if count == 1 or max_start == 0:
        return [0.0]
    offsets = sorted([random.uniform(0.0, max_start) for _ in range(count)])
    return offsets


def _compute_ssim(ffmpeg: str, distorted: str, reference: str,
                   width: int, height: int, verbose: bool = False) -> float:
    vf = (
        f"[0:v]scale={width}:{height}:flags=bicubic[dist];"
        f"[1:v]scale={width}:{height}:flags=bicubic[ref];"
        "[dist][ref]ssim"
    )
    cmd = [ffmpeg, "-hide_banner", "-nostats", "-loglevel", "info",
           "-i", distorted, "-i", reference,
           "-lavfi", vf, "-f", "null", "-"]
    rc, out, err = _run_cmd(cmd, verbose)
    if rc != 0:
        if verbose:
            print(f"SSIM command failed (rc={rc}): {err}", file=sys.stderr)
        return float("nan")
    m = re.search(r"All:\s*([0-9.]+)", err)
    if not m:
        if verbose:
            print(f"SSIM: could not parse 'All:' from stderr:\n{err}", file=sys.stderr)
        return float("nan")
    return float(m.group(1))


def _knee_from_dense(xs, ys):
    if xs is None or ys is None:
        return None
    xs_arr = np.asarray(xs, dtype=float)
    ys_arr = np.asarray(ys, dtype=float)
    if xs_arr.size < 3 or ys_arr.size < 3:
        return None
    x_min = float(xs_arr.min())
    x_max = float(xs_arr.max())
    if math.isclose(x_min, x_max):
        return None
    y_min = float(np.nanmin(ys_arr))
    y_max = float(np.nanmax(ys_arr))
    if math.isclose(y_min, y_max):
        return None
    x_norm = (xs_arr - x_min) / (x_max - x_min)
    y_norm = (ys_arr - y_min) / (y_max - y_min)
    diff = np.asarray(y_norm - x_norm, dtype=float)
    if diff.size > 4:
        diff[:2] = diff[2]
        diff[-2:] = diff[-3]
    try:
        idx = int(np.nanargmax(diff))
    except (ValueError, TypeError):
        return None
    idx = max(1, min(idx, diff.size - 2))
    return float(xs_arr[idx]), float(ys_arr[idx])


def _extract_reference_sample(ffmpeg: str, input_path: str, output_path: str,
                               offset: float, duration: float, verbose: bool) -> None:
    cmd = [
        ffmpeg, "-y",
        "-ss", f"{max(offset, 0.0):.3f}",
        "-i", input_path,
        "-t", f"{max(duration, 0.1):.3f}",
        "-map", "0:v:0",
        "-c:v", "copy",
        "-an",
        output_path,
    ]
    rc, out, err = _run_cmd(cmd, verbose)
    if rc != 0:
        raise RuntimeError(f"Failed to extract reference sample: {err}")


def _encode_analysis_sample(ffmpeg: str, input_path: str, output_path: str,
                             height: int, crf: int, preset: str, verbose: bool) -> None:
    scale = f"scale=-2:{height}:flags=bicubic"
    cmd = [
        ffmpeg, "-y",
        "-i", input_path,
        "-vf", scale,
        "-c:v", "hevc_nvenc",
        "-pix_fmt", "yuv420p",
        "-colorspace", "bt709",
        "-color_primaries", "bt709",
        "-color_trc", "bt709",
        "-color_range", "tv",
        "-preset", preset,
        "-rc", "vbr_hq",
        "-cq", str(crf),
        "-b:v", "0", "-maxrate", "0", "-bufsize", "0",
        "-fps_mode", "vfr",
        "-an",
        output_path,
    ]
    rc, out, err = _run_cmd(cmd, verbose)
    if rc != 0:
        raise RuntimeError(f"Failed to encode analysis sample at CRF {crf}: {err}")


def _find_optimal_crf(ffmpeg: str, ffprobe: str, input_path: str, resolution: int,
                       crfs: List[int], sample_length: float, num_samples: int,
                       preset: str, verbose: bool) -> int:
    meta = _ffprobe_meta(ffprobe, input_path)
    duration = meta["duration_s"]
    ref_width = meta["width"]
    ref_height = meta["height"]

    if duration < 1:
        raise RuntimeError(f"Video too short ({duration:.1f}s) for analysis")

    with tempfile.TemporaryDirectory(prefix="iml_encode_") as tmpdir:
        offsets = _sample_offsets(duration, sample_length, num_samples)
        if verbose:
            print(f"Analysis offsets: {[f'{o:.2f}s' for o in offsets]}")

        ref_samples = []
        for i, offset in enumerate(offsets):
            print(f"  Sample {i + 1}/{len(offsets)} at {offset:.1f}s...")
            ref_path = os.path.join(tmpdir, f"ref_{i:03d}.mp4")
            _extract_reference_sample(ffmpeg, input_path, ref_path, offset, sample_length, verbose)
            ref_samples.append(ref_path)

        crf_ssims = {crf: [] for crf in crfs}
        for i, ref_path in enumerate(ref_samples):
            for crf in crfs:
                enc_path = os.path.join(tmpdir, f"enc_{i:03d}_crf_{crf}.mp4")
                _encode_analysis_sample(ffmpeg, ref_path, enc_path, resolution, crf, preset, verbose)
                ssim = _compute_ssim(ffmpeg, enc_path, ref_path, ref_width, ref_height, verbose)
                crf_ssims[crf].append(ssim)
                if verbose:
                    print(f"  Sample {i}, CRF {crf}: SSIM = {ssim:.4f}")

        avg_ssims = {}
        for crf, values in crf_ssims.items():
            valid = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
            avg_ssims[crf] = sum(valid) / len(valid) if valid else 0.0

        if verbose:
            print("\nAverage SSIM per CRF:")
            for crf in sorted(avg_ssims):
                print(f"  CRF {crf}: {avg_ssims[crf]:.4f}")

        sorted_crfs = sorted(avg_ssims.keys())
        if len(sorted_crfs) >= 3:
            data_pairs = [(-float(c), float(avg_ssims[c])) for c in sorted_crfs]
            data_pairs.sort(key=lambda p: p[0])
            xs = [p[0] for p in data_pairs]
            ys = [p[1] for p in data_pairs]

            deg = min(3, len(xs) - 1)
            poly = np.poly1d(np.polyfit(xs, ys, deg))
            x_dense = np.linspace(xs[0], xs[-1], 200)
            y_dense = poly(x_dense)

            knee = _knee_from_dense(x_dense, y_dense)
            if knee:
                val = -knee[0]
                optimal_crf = int(val) if (val - int(val)) < 0.6 else int(val) + 1
                if verbose:
                    print(f"\nKnee point at CRF ~{optimal_crf} (SSIM: {knee[1]:.4f})")
                analysis_data = {
                    "crfs": list(sorted_crfs),
                    "ssims": [float(avg_ssims[c]) for c in sorted_crfs],
                    "x_dense": x_dense,
                    "y_dense": y_dense,
                    "knee": tuple(float(v) for v in knee),
                }
                return optimal_crf, analysis_data

        raise RuntimeError(
            f"Knee point detection failed: {len(sorted_crfs)} CRF values tested, "
            "no valid knee point found"
        )


def _plot_analysis(analysis: dict, optimal_crf: int, label: str, output_path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    crfs = analysis["crfs"]
    ssims = analysis["ssims"]
    x_dense = analysis["x_dense"]
    y_dense = analysis["y_dense"]
    knee = analysis["knee"]

    fig, ax = plt.subplots(figsize=(10, 6))

    raw_crfs = [-x for x in x_dense]
    ax.scatter(crfs, ssims, color="blue", s=60, zorder=5, label="Measured SSIM")
    ax.plot(raw_crfs, y_dense, color="blue", linewidth=1.5, label="Polyfit curve")

    if knee:
        k_crf = -knee[0]
        ax.axvline(x=k_crf, color="green", linestyle="--", linewidth=1.5,
                   label=f"Knee: CRF {k_crf:.1f}")
        ax.scatter([k_crf], [knee[1]], color="green", s=120, marker="*", zorder=10)

    ax.set_xlabel("CRF (lower = higher quality)")
    ax.set_ylabel("SSIM")
    ax.set_title(f"{label} — Optimal CRF {optimal_crf}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Plot saved: {output_path}")


def _encode_video(ffmpeg: str, ffprobe: str, input_path: str, output_path: str,
                   resolution: int, crf: int, preset: str, verbose: bool,
                   audio_bitrate: int | None = None,
                   audio_layout: dict | None = None) -> dict:
    scale = f"scale=-2:{resolution}:flags=bicubic"
    cmd = [ffmpeg]
    if verbose:
        cmd += ["-loglevel", "verbose"]
    cmd += ["-y",
        "-i", input_path,
        "-map", "0:v:0",
        "-map", "0:a?",
        "-map", "0:s?",
        "-map", "0:t?",
        "-dn",
        "-vf", scale,
        "-c:v", "hevc_nvenc",
        "-pix_fmt", "yuv420p",
        "-colorspace", "bt709",
        "-color_primaries", "bt709",
        "-color_trc", "bt709",
        "-color_range", "tv",
        "-preset", preset,
        "-rc", "vbr_hq",
        "-cq", str(crf),
        "-b:v", "0", "-maxrate", "0", "-bufsize", "0",
        "-fps_mode", "vfr",
    ]
    if audio_bitrate is not None:
        streams = _audio_stream_info(ffprobe, input_path)
        if streams:
            for i, info in enumerate(streams):
                if audio_layout and info["channels"] > audio_layout["count"]:
                    total = audio_bitrate * audio_layout["count"]
                    parts = [
                        f"-ac:a:{i}", str(audio_layout["count"]),
                        f"-channel_layout:a:{i}", audio_layout["layout"],
                        f"-c:a:{i}", "aac", f"-b:a:{i}", f"{total}k",
                    ]
                    cmd += parts
                    continue
                total = audio_bitrate * info["channels"]
                cmd += [f"-c:a:{i}", "aac", f"-b:a:{i}", f"{total}k"]
        else:
            cmd += ["-c:a", "copy"]
    else:
        cmd += ["-c:a", "copy"]
    cmd += ["-c:s", "copy", "-c:t", "copy", output_path]
    if verbose:
        print("Encoding:", " ".join(cmd))
    rc, out, err = _run_cmd(cmd, verbose)
    result = {
        "input": input_path,
        "output": output_path,
        "resolution": resolution,
        "crf": crf,
        "success": rc == 0,
        "error": err if rc != 0 else None,
    }
    if rc == 0:
        try:
            result["source_size"] = os.path.getsize(input_path)
            result["output_size"] = os.path.getsize(output_path)
        except Exception:
            pass
    return result


_LAYOUTS = {
    "mono":   {"count": 1, "layout": "mono",   "has_lfe": False},
    "1.0":    {"count": 1, "layout": "mono",   "has_lfe": False},
    "stereo": {"count": 2, "layout": "stereo", "has_lfe": False},
    "2.0":    {"count": 2, "layout": "stereo", "has_lfe": False},
    "2.1":    {"count": 3, "layout": "2.1",    "has_lfe": True},
    "5.0":    {"count": 5, "layout": "5.0",    "has_lfe": False},
    "5.1":    {"count": 6, "layout": "5.1",    "has_lfe": True},
    "7.0":    {"count": 7, "layout": "7.0",    "has_lfe": False},
    "7.1":    {"count": 8, "layout": "7.1",    "has_lfe": True},
}


def _parse_audio_layout(s: str) -> dict:
    s = s.strip().lower()
    if s in _LAYOUTS:
        return dict(_LAYOUTS[s])
    raise ValueError(f"Unknown audio layout: '{s}'. Supported: {', '.join(_LAYOUTS)}")


def _audio_stream_info(ffprobe: str, path: str) -> List[dict]:
    cmd = [ffprobe, "-v", "error", "-print_format", "json",
           "-show_streams", "-select_streams", "a", path]
    rc, out, err = _run_cmd(cmd)
    if rc != 0:
        return []
    j = json.loads(out)
    result = []
    for s in j.get("streams", []):
        ch = int(s.get("channels", 0) or 0)
        if ch <= 0:
            continue
        layout = (s.get("channel_layout") or "").lower().strip()
        result.append({
            "channels": ch,
            "layout": layout if layout else "unknown",
            "has_lfe": ".1" in layout or "lfe" in layout if layout else None,
        })
    return result


def _output_name(stem: str, ext: str, resolution: int, crf: int,
                 audio_bitrate: int | None = None,
                 audio_layout: dict | None = None) -> str:
    suffix = ""
    if audio_layout:
        suffix += f"_{audio_layout['layout']}ch"
    if audio_bitrate:
        suffix += f"_{audio_bitrate}kbps"
    return f"{stem}_hevc_{resolution}p_cq_{crf}{suffix}{ext}"


def _collect_videos(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    return sorted([
        f for f in path.iterdir()
        if f.is_file() and f.suffix.lower() in VIDEO_EXTS
    ])


def _output_exists(v: Path, output_path: Path, single_input: bool,
                    resolution: int, audio_bitrate: int | None,
                    audio_layout: dict | None) -> bool:
    if single_input:
        return output_path.is_file() if output_path.suffix.lower() in VIDEO_EXTS else False
    pattern = f"{v.stem}_hevc_{resolution}p_cq_*"
    return bool(list(output_path.glob(pattern)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Encode video(s) with analysis-guided optimal CRF (HEVC NVENC)"
    )
    parser.add_argument("-i", "--input", required=True, type=Path,
                        help="Input video file or directory")
    parser.add_argument("-o", "--output", required=True, type=Path,
                        help="Output file (single input) or directory (multiple inputs)")
    parser.add_argument("-r", "--resolution", required=True, type=int,
                        help="Shortest side in pixels")
    parser.add_argument("--root", required=True, type=Path,
                        help="Root directory (ffmpeg at <root>/ffmpeg/bin/ffmpeg.exe)")
    parser.add_argument("--crf-range",
                        default=",".join(str(c) for c in DEFAULT_CRFS),
                        help="Comma-separated CRF values for analysis "
                             f"(default: {','.join(str(c) for c in DEFAULT_CRFS)})")
    parser.add_argument("--sample-length", type=float, default=8.0,
                        help="Length of each analysis sample in seconds (default: 8.0)")
    parser.add_argument("--num-samples", type=int, default=8,
                        help="Number of analysis samples (default: 8)")
    parser.add_argument("--preset", default="slow",
                        help="NVENC preset (default: slow)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose output")
    parser.add_argument("--audio-bitrate", type=int, default=None,
                        help="Re-encode audio as AAC at this bitrate (kbps per channel, "
                             "e.g. 64). Appended to filename as _{bitrate}kbps.")
    parser.add_argument("--audio-channels", type=str, default=None,
                        help="Downmix audio to this layout (e.g. 'stereo', '2.1', "
                             "'5.1'). Only applies with --audio. Raises error if "
                             "source lacks LFE when target requires it.")
    parser.add_argument("--reset", action="store_true",
                        help="Re-encode even if output already exists")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show analysis result but skip encoding")
    args = parser.parse_args()

    print_banner("Video Encoder")

    root = args.root.resolve()
    ffmpeg = str(root / "ffmpeg" / "bin" / "ffmpeg.exe")
    ffprobe = str(root / "ffmpeg" / "bin" / "ffprobe.exe")

    if not os.path.isfile(ffmpeg):
        sys.exit(f"ffmpeg not found at {ffmpeg}")
    if not os.path.isfile(ffprobe):
        sys.exit(f"ffprobe not found at {ffprobe}")

    plot_dir = root / "video" / "encode"
    plot_dir.mkdir(parents=True, exist_ok=True)
    today_str = date.today().isoformat()
    existing = sorted(plot_dir.glob(f"{today_str}-*.png"))
    plot_counter = 1
    if existing:
        last = existing[-1].stem
        try:
            plot_counter = int(last.split("-")[-1]) + 1
        except (ValueError, IndexError):
            pass

    try:
        crfs = sorted({int(c.strip()) for c in args.crf_range.split(",") if c.strip()})
    except ValueError as e:
        sys.exit(f"Invalid --crf-range: {e}")
    if not crfs:
        sys.exit("No valid CRF values in --crf-range")

    audio_layout = None
    if args.audio_bitrate and args.audio_channels:
        try:
            audio_layout = _parse_audio_layout(args.audio_channels)
        except ValueError as e:
            sys.exit(e)
    elif args.audio_channels and not args.audio_bitrate:
        sys.exit("--audio-channels requires --audio")

    input_path = args.input.resolve()
    if not input_path.exists():
        sys.exit(f"Input not found: {input_path}")

    output_path = args.output.resolve()
    videos = _collect_videos(input_path)
    if not videos:
        sys.exit(f"No video files found in: {input_path}")

    single_input = input_path.is_file()

    if args.dry_run:
        print("Dry-run mode: analysis only, no encoding\n")

    results = []
    total = len(videos)
    for idx, v in enumerate(videos, 1):
        prefix = f"[{idx}/{total}] "
        start = time.time()
        print(f"\n{'-'*50}")
        print(f"File: {v.name}")

        if audio_layout:
            streams = _audio_stream_info(ffprobe, str(v))
            for i, info in enumerate(streams):
                if not info["layout"] or info["layout"] == "unknown":
                    sys.exit(f"{v.name} audio stream {i}: source layout unknown, "
                             f"cannot verify LFE for target '{args.audio_channels}'")
                if audio_layout["has_lfe"] and not info["has_lfe"]:
                    sys.exit(f"{v.name} audio stream {i}: target "
                             f"'{args.audio_channels}' requires LFE but source "
                             f"layout '{info['layout']}' has none")

        if not args.reset and _output_exists(
            v, output_path, single_input, args.resolution,
            args.audio_bitrate, audio_layout
        ):
            print(f"{prefix} Output exists, skipping")
            results.append({"input": str(v), "success": True, "skipped": True})
            continue

        print(f"{prefix}[1/4] Analyzing video...")

        try:
            optimal_crf, analysis_data = _find_optimal_crf(
                ffmpeg, ffprobe, str(v), args.resolution, crfs,
                args.sample_length, args.num_samples, args.preset, args.verbose
            )
        except RuntimeError as e:
            print(f"Analysis failed for {v.name}: {e}", file=sys.stderr)
            results.append({"input": str(v), "success": False, "error": str(e)})
            continue

        plot_path = plot_dir / f"{today_str}-{plot_counter:04d}.png"
        plot_counter += 1
        _plot_analysis(analysis_data, optimal_crf, v.stem, str(plot_path))

        print(f"{prefix}[2/4] Optimal CRF: {optimal_crf}")

        if args.dry_run:
            results.append({
                "input": str(v),
                "output": "dry-run",
                "resolution": args.resolution,
                "crf": optimal_crf,
                "success": True,
                "dry_run": True,
            })
            print()
            continue

        if single_input and output_path.suffix.lower() in VIDEO_EXTS:
            out = str(output_path)
        else:
            out_name = _output_name(v.stem, v.suffix, args.resolution, optimal_crf,
                                     args.audio_bitrate, audio_layout)
            output_path.mkdir(parents=True, exist_ok=True)
            out = str(output_path / out_name)

        print(f"{prefix}[3/4] Encoding...")
        result = _encode_video(ffmpeg, ffprobe, str(v), out, args.resolution,
                               optimal_crf, args.preset, args.verbose, args.audio_bitrate,
                               audio_layout)
        results.append(result)

        elapsed = time.time() - start
        mins, secs = divmod(int(elapsed), 60)
        time_str = f"{mins}m{secs:02d}s" if mins else f"{secs}s"
        print(f"{prefix}[4/4] Done in {time_str}:", end=" ")
        if result["success"]:
            src_mb = result.get("source_size", 0) / 1_048_576
            out_mb = result.get("output_size", 0) / 1_048_576
            pct = (out_mb / src_mb * 100) if src_mb else 0
            print(f"{src_mb:.1f} MB -> {out_mb:.1f} MB ({pct:.1f}%)")
        else:
            print(f"encode failed: {result.get('error', 'unknown')}")

    print(f"{'='*60}")
    print("Summary:")
    encoded = [r for r in results if r.get("success") and not r.get("skipped")]
    skipped = [r for r in results if r.get("skipped")]
    failures = [r for r in results if not r.get("success")]
    if encoded:
        print(f"  Encoded: {len(encoded)}")
        for r in encoded:
            if r.get("dry_run"):
                print(f"    {r['input']} -> dry-run (CRF {r['crf']})")
            else:
                print(f"    {r['input']} -> {r['output']} (CRF {r['crf']})")
    if skipped:
        print(f"  Skipped: {len(skipped)}")
        for r in skipped:
            print(f"    {r['input']}")
    if failures:
        print(f"  Failed: {len(failures)}")
        for r in failures:
            print(f"    {r['input']}: {r.get('error', 'unknown')}")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
