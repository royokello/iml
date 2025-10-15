import argparse, json, math, os, re, subprocess, csv, random
import numpy as np
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt

VIDEO_EXTS = {".mp4",".mkv",".mov",".avi",".webm",".m4v",".mpg",".mpeg"}

FFMPEG_BIN = "ffmpeg"
FFPROBE_BIN = "ffprobe"

VERBOSE = False

DEFAULT_RESOLUTIONS = [768, 1024]
DEFAULT_BITRATES = [768, 1024, 1536, 2048]


def _parse_int_list(raw, fallback):
    if raw is None:
        return list(fallback)
    if isinstance(raw, (list, tuple)):
        items = raw
    else:
        raw = str(raw).strip()
        if not raw:
            return list(fallback)
        raw = raw.replace(";", ",")
        items = raw.split(",")
    values = []
    for item in items:
        s = str(item).strip()
        if not s:
            continue
        sign = -1 if s.startswith("-") else 1
        token = s[1:] if sign == -1 else s
        if not token.isdigit():
            raise ValueError(f"Invalid integer value '{s}'")
        values.append(sign * int(token))
    return values if values else list(fallback)


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


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
    diff = y_norm - x_norm
    diff = np.asarray(diff, dtype=float)
    if diff.size > 4:
        diff[:2] = diff[2]
        diff[-2:] = diff[-3]
    try:
        idx = int(np.nanargmax(diff))
    except (ValueError, TypeError):
        return None
    idx = max(1, min(idx, diff.size - 2))
    return float(xs_arr[idx]), float(ys_arr[idx])


def _load_cached_ssim_map(output_dir):
    directory = Path(output_dir)
    if not directory.is_dir():
        return {}
    try:
        candidates = sorted(
            directory.glob("*-samples.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except Exception:
        return {}
    for csv_path in candidates:
        try:
            with open(csv_path, newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                mapping = {}
                for row in reader:
                    try:
                        res = int(float(row.get("resolution", 0)))
                        bitrate = int(float(row.get("target_bitrate_kbps", 0)))
                        sample_idx = int(float(row.get("sample_index", 0)))
                    except (TypeError, ValueError):
                        continue
                    ssim_str = row.get("ssim")
                    if ssim_str is None or ssim_str == "":
                        ssim_val = float("nan")
                    else:
                        try:
                            ssim_val = float(ssim_str)
                        except ValueError:
                            ssim_val = float("nan")
                    mapping[(res, bitrate, sample_idx)] = ssim_val
        except Exception:
            continue
        else:
            if mapping:
                vprint(f"Loaded cached SSIM values from {csv_path}")
            return mapping
    return {}


def _sample_offsets(duration, sample_length, num_samples):
    if duration <= 0:
        return [0.0]
    count = max(1, num_samples)
    effective_length = max(1.0, float(sample_length))
    max_start = max(duration - effective_length, 0.0)
    if count == 1:
        if max_start == 0:
            return [0.0]
        return [random.uniform(0.0, max_start)]
    if max_start == 0:
        return [0.0]
    offsets = [random.uniform(0.0, max_start) for _ in range(count)]
    offsets.sort()
    return offsets


def _mean(values):
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def _format_time(seconds):
    return f"{seconds:.3f}"


def _create_reference_sample(input_path, output_path, start, duration, ff_args):
    cmd = [
        FFMPEG_BIN,
        *ff_args,
        "-y",
        "-ss",
        _format_time(max(start, 0.0)),
        "-i",
        input_path,
        "-t",
        _format_time(max(duration, 0.1)),
        "-map",
        "0:v:0",
        "-c:v",
        "copy",
        "-an",
        output_path,
    ]
    vprint("Creating reference sample:", " ".join(cmd))
    rc, out, err = run(cmd)
    if rc != 0:
        raise RuntimeError(f"Failed to create reference sample: {err or out}")


def _encode_sample(reference_path, output_path, height, bitrate_kbps, ff_args):
    scale_filter = f"scale=-2:{height}:flags=bicubic"
    bitrate = int(bitrate_kbps)
    cmd = [
        FFMPEG_BIN,
        *ff_args,
        "-y",
        "-i",
        reference_path,
        "-vf",
        scale_filter,
        "-c:v",
        "hevc_nvenc",
        "-preset",
        "slow",
        "-b:v",
        f"{bitrate}k",
        "-maxrate",
        f"{bitrate}k",
        "-bufsize",
        f"{bitrate * 2}k",
        "-fps_mode",
        "vfr",
        "-an",
        output_path,
    ]
    vprint("Encoding sample:", " ".join(cmd))
    rc, out, err = run(cmd)
    if rc != 0:
        raise RuntimeError(f"Failed to encode sample at {height}p/{bitrate}k: {err or out}")


def _fmt_float(value, precision=3):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    return f"{value:.{precision}f}"


def _write_markdown(path, context):
    lines = [
        "# Video Quality Assessment",
        "",
        f"- Input video: `{context['input']}`",
        f"- Output directory: `{context['output']}`",
        f"- Resolutions: {', '.join(str(r) for r in context['resolutions'])}",
        f"- Bitrates (kbps): {', '.join(str(b) for b in context['bitrates'])}",
        f"- Sample length: {context['sample_length']} s",
        f"- Samples per combination: {context['num_samples']}",
    ]
    metric = context.get("metric")
    if metric:
        lines.append(f"- Plotted metric: `{metric}`")
    plot = context.get("plot")
    lines.append("")
    lines.append("| Resolution | Target Bitrate (kbps) | Avg Bitrate (kbps) | Mean Storage (bytes/s) | Mean SSIM |")
    lines.append("|------------|-----------------------|--------------------|------------------------|-----------|")
    for row in context["rows"]:
        lines.append(
            f"| {row['resolution']} | {row['target_bitrate_kbps']} | "
            f"{_fmt_float(row['avg_actual_bitrate_kbps'], 1)} | "
            f"{_fmt_float(row.get('mean_storage_bps'), 1)} | "
            f"{_fmt_float(row['mean_ssim'], 4)} |"
        )
    if plot:
        lines.append("")
        lines.append(f"![Quality Plot]({os.path.basename(plot)})")
    storage_plot = context.get("storage_plot")
    if storage_plot:
        lines.append("")
        lines.append(f"![Storage Plot]({os.path.basename(storage_plot)})")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def vprint(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)

def _resolve_tool(tool, base_dir=None):
    if base_dir:
        candidate = os.path.join(base_dir, tool)
        if os.name == "nt" and not candidate.lower().endswith(".exe"):
            candidate_exe = candidate + ".exe"
            if os.path.isfile(candidate_exe):
                return candidate_exe
        if os.path.isfile(candidate):
            return candidate
    return tool

def _set_ffmpeg_bins(base_dir):
    global FFMPEG_BIN, FFPROBE_BIN
    FFMPEG_BIN = _resolve_tool("ffmpeg", base_dir)
    FFPROBE_BIN = _resolve_tool("ffprobe", base_dir)

def _ff_verbosity_args():
    return ["-loglevel", "verbose"] if VERBOSE else ["-hide_banner","-nostats","-loglevel","error"]

def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.returncode, p.stdout, p.stderr

def ffprobe_meta(path):
    level = "info" if VERBOSE else "error"
    cmd = [FFPROBE_BIN,"-v",level,"-print_format","json","-show_format","-show_streams",path]
    rc,out,err = run(cmd)
    if rc != 0: raise RuntimeError(err)
    j = json.loads(out)
    v = next((s for s in j.get("streams",[]) if s.get("codec_type")=="video"),{})
    fmt = j.get("format",{})
    size = int(fmt.get("size",0)) if "size" in fmt else os.path.getsize(path)
    dur = float(fmt.get("duration",0) or 0)
    br = float(fmt.get("bit_rate",0) or 0)/1000.0
    return {
        "width": int(v.get("width") or 0),
        "height": int(v.get("height") or 0),
        "fps": _fps(v),
        "duration_s": dur,
        "bitrate_kbps": br,
        "size_bytes": size
    }

def _fps(vstream):
    r = vstream.get("r_frame_rate") or vstream.get("avg_frame_rate") or "0/1"
    try:
        a,b = r.split("/")
        a = float(a); b = float(b) if float(b)!=0 else 1.0
        return a/b
    except Exception:
        return 0.0

def list_videos(folder):
    p = Path(folder)
    return sorted(str(f) for f in p.iterdir() if f.is_file() and f.suffix.lower() in VIDEO_EXTS)

def pick_reference(files):
    return max(files, key=lambda f: os.path.getsize(f)) if files else None

def compute_ssim(distorted, reference, width=None, height=None):
    vf = []
    if width and height:
        vf += [f"[0:v]scale={width}:{height}:flags=bicubic[dist]",
               f"[1:v]scale={width}:{height}:flags=bicubic[ref]",
               "[dist][ref]ssim"]
    else:
        vf += ["[0:v][1:v]ssim"]
    cmd = [FFMPEG_BIN,*_ff_verbosity_args(),"-i",distorted,"-i",reference,"-lavfi",";".join(vf),"-f","null","-"]
    vprint("Running:", " ".join(cmd))
    rc,out,err = run(cmd)
    if rc != 0: raise RuntimeError(err)
    m = re.search(r"All:\s*([0-9.]+)", err)
    return float(m.group(1)) if m else float("nan")

def _to_kbps(val: float, unit: str) -> float:
    u = unit.lower()
    if u in ("k","kbps"): return float(val)
    if u in ("m","mbps"): return float(val) * 1000.0
    return float("nan")

def parse_intended_video_audio_kbps(filename: str):
    """Best-effort parse of intended video/audio bitrates from filename.

    Supports patterns like:
    - *_1080p_1500k_2ch_128k*
    - *...1024p_768kbps_aac_2ch_128kbps*
    - *... 1.5 m 2ch 128 k ...*

    Returns a tuple (video_kbps, audio_kbps) where values may be NaN if unknown.
    """
    s = os.path.basename(filename).lower()

    # Try a tolerant direct pattern first: resolution -> video rate -> (optional codec) -> 2ch -> audio rate
    m = re.search(
        r'_(?:\d{3,4})p_.*?(\d+(?:\.\d+)?)\s*(k|kbps|m|mbps).*?(?:aac|ac3|opus|mp3|flac|eac3|dts|truehd|audio)?_?\s*2ch_.*?(\d+(?:\.\d+)?)\s*(k|kbps|m|mbps)\b',
        s
    )
    if m:
        v_val, v_unit = float(m.group(1)), m.group(2)
        a_val, a_unit = float(m.group(3)), m.group(4)
        return _to_kbps(v_val, v_unit), _to_kbps(a_val, a_unit)

    # General token-based heuristic: collect all k/mbps values and classify by nearby audio hints
    tokens = list(re.finditer(r'(\d+(?:\.\d+)?)\s*(k|kbps|m|mbps)\b', s))
    video_candidates, audio_candidates = [], []
    audio_hints = [
        '2ch', 'stereo', 'mono', '5.1', '7.1', 'aac', 'ac3', 'eac3', 'ddp', 'dts', 'truehd',
        'opus', 'mp3', 'flac', 'vorbis', 'alac', 'pcm', 'audio'
    ]
    for t in tokens:
        val = float(t.group(1)); unit = t.group(2)
        kbps = _to_kbps(val, unit)
        start_idx = t.start(); end_idx = t.end()
        # Look around the token for audio context
        context = s[max(0, start_idx-24):min(len(s), end_idx+16)]
        is_audio = any(h in context for h in audio_hints)
        if is_audio:
            audio_candidates.append(kbps)
        else:
            video_candidates.append(kbps)

    video = max(video_candidates) if video_candidates else float("nan")
    audio = min(audio_candidates) if audio_candidates else float("nan")
    return video, audio

def _pareto_frontier(points):
    pts = sorted([p for p in points if p["bitrate_kbps"]>0 and not math.isnan(p["q"])], key=lambda r: r["bitrate_kbps"])
    frontier, best_q = [], -1e9
    for r in pts:
        if r["q"] > best_q + 1e-12:
            frontier.append(r); best_q = r["q"]
    return frontier

def _efficient_point(frontier):
    if not frontier: return None
    return max(frontier, key=lambda r: (r["q"]/r["bitrate_kbps"]))

def _knee_point(frontier):
    if not frontier or len(frontier) < 3: return None
    x0, y0 = frontier[0]["bitrate_kbps"], frontier[0]["q"]
    x1, y1 = frontier[-1]["bitrate_kbps"], frontier[-1]["q"]
    denom = math.hypot(x1-x0, y1-y0)
    if denom == 0: return None
    max_d, best = -1.0, None
    for r in frontier:
        x, y = r["bitrate_kbps"], r["q"]
        d = abs((y1 - y0)*x - (x1 - x0)*y + x1*y0 - y1*x0) / denom
        if d > max_d: max_d, best = d, r
    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to the input video")
    ap.add_argument("--output", required=True, help="Directory for generated samples and results")
    ap.add_argument("--ffmpeg-dir", default=None, help="Directory containing ffmpeg/ffprobe executables")
    ap.add_argument("--resolutions", default="1024", help="Comma separated target heights in pixels (e.g. 768,1024)")
    ap.add_argument("--bitrates", default="256,512,1024,1536,2048,3072,4096", help="Comma separated target bitrates in kbps (e.g. 768,1024,1536,2048)")
    ap.add_argument("--skip", action="store_true", default=False, help="Skip regenerating samples/encodes and reuse existing files")
    ap.add_argument("--num_samples", type=int, default=3, help="Number of snippets per resolution/bitrate combination")
    ap.add_argument("--sample_length", type=int, default=120, help="Length of each sample in seconds")
    ap.add_argument("-v","--verbose", action="store_true", help="Enable verbose logs and verbose ffmpeg output")
    args = ap.parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.isfile(input_path):
        raise SystemExit(f"Input video not found: {input_path}")

    output_dir = os.path.abspath(args.output)
    _ensure_dir(output_dir)

    global VERBOSE
    VERBOSE = bool(args.verbose)
    skip_requested = bool(args.skip)
    if args.ffmpeg_dir:
        _set_ffmpeg_bins(args.ffmpeg_dir)
        vprint(f"Using ffmpeg from: {FFMPEG_BIN}")
        vprint(f"Using ffprobe from: {FFPROBE_BIN}")
    else:
        vprint("Using ffmpeg/ffprobe from PATH")
    if skip_requested:
        vprint("Skip enabled: reusing existing samples and encodes.")

    try:
        resolutions = sorted({int(abs(r)) for r in _parse_int_list(args.resolutions, DEFAULT_RESOLUTIONS) if int(abs(r)) > 0})
    except ValueError as exc:
        raise SystemExit(f"Invalid --resolutions value: {exc}") from exc
    if not resolutions:
        raise SystemExit("No valid resolutions provided.")

    try:
        bitrates = sorted({int(abs(b)) for b in _parse_int_list(args.bitrates, DEFAULT_BITRATES) if int(abs(b)) > 0})
    except ValueError as exc:
        raise SystemExit(f"Invalid --bitrates value: {exc}") from exc
    if not bitrates:
        raise SystemExit("No valid bitrates provided.")

    num_samples = max(1, int(args.num_samples))
    sample_length = max(1, int(args.sample_length))

    ref_meta = ffprobe_meta(input_path)
    video_duration = ref_meta.get("duration_s", 0.0)
    if video_duration <= 0:
        raise SystemExit("Unable to determine video duration for sampling.")
    vprint(f"Input duration: {video_duration:.2f}s")

    if skip_requested:
        offsets = [0.0 for _ in range(num_samples)]
    else:
        offsets = _sample_offsets(video_duration, sample_length, num_samples)
    if offsets:
        vprint("Sample offsets (s):", ", ".join(f"{o:.3f}" for o in offsets))
    ff_args = _ff_verbosity_args()

    reference_dir = os.path.join(output_dir, "reference_samples")
    _ensure_dir(reference_dir)
    reference_samples = []
    for idx, start in enumerate(offsets):
        name = f"sample_{idx:02d}.mp4"
        path = os.path.join(reference_dir, name)
        if skip_requested:
            if not os.path.isfile(path):
                raise SystemExit(f"Missing cached reference sample: {path}. Rerun without --skip to regenerate.")
            sample_meta = ffprobe_meta(path)
            vprint(f"Reusing reference sample {name}")
            duration_val = sample_meta.get("duration_s") or float(sample_length)
            start_val = float(start)
        else:
            duration = min(sample_length, max(video_duration - start, 0.1))
            _create_reference_sample(input_path, path, start, duration, ff_args)
            sample_meta = ffprobe_meta(path)
            start_val = float(start)
            duration_val = sample_meta.get("duration_s") or float(duration)
        reference_samples.append({
            "index": idx,
            "start": start_val,
            "path": path,
            "duration": duration_val,
            "meta": sample_meta,
        })
    actual_sample_count = len(reference_samples)
    if actual_sample_count == 0:
        raise SystemExit("Failed to create reference samples.")
    if actual_sample_count < num_samples:
        vprint(f"Adjusted sample count from {num_samples} to {actual_sample_count} due to duration constraints.")

    cached_ssims = _load_cached_ssim_map(output_dir) if skip_requested else {}

    ref_width = ref_meta.get("width") or 0
    ref_height = ref_meta.get("height") or 0
    if ref_width <= 0 or ref_height <= 0:
        sample_meta = reference_samples[0].get("meta") or ffprobe_meta(reference_samples[0]["path"])
        ref_width = sample_meta.get("width") or ref_width
        ref_height = sample_meta.get("height") or ref_height
    if ref_width <= 0 or ref_height <= 0:
        raise SystemExit("Unable to determine reference dimensions for quality comparison.")

    summary_rows = []
    sample_rows = []
    for resolution in resolutions:
        for bitrate in bitrates:
            combo_dir = os.path.join(output_dir, f"{resolution}p_{bitrate}k")
            _ensure_dir(combo_dir)
            vprint(f"Processing {resolution}p @ {bitrate} kbps")
            combo_samples = []
            for sample in reference_samples:
                encoded_path = os.path.join(combo_dir, f"sample_{sample['index']:02d}.mp4")
                if skip_requested:
                    if not os.path.isfile(encoded_path):
                        raise SystemExit(f"Missing cached encode: {encoded_path}. Rerun without --skip to regenerate.")
                    encoded_meta = ffprobe_meta(encoded_path)
                    vprint(f"  Sample {sample['index']:02d}: reusing cached encode")
                else:
                    _encode_sample(sample["path"], encoded_path, resolution, bitrate, ff_args)
                    encoded_meta = ffprobe_meta(encoded_path)
                ssim = float("nan")
                reused_ssim = False
                if skip_requested and cached_ssims:
                    cached_ssim = cached_ssims.get((int(resolution), int(bitrate), int(sample["index"])))
                    if cached_ssim is not None:
                        ssim = cached_ssim
                        reused_ssim = True
                        if isinstance(ssim, float) and math.isnan(ssim):
                            vprint(f"  Sample {sample['index']:02d}: reusing cached SSIM (nan)")
                        else:
                            vprint(f"  Sample {sample['index']:02d}: reusing cached SSIM {ssim:.4f}")
                if not reused_ssim:
                    try:
                        ssim = compute_ssim(encoded_path, sample["path"], ref_width, ref_height)
                        vprint(f"  Sample {sample['index']:02d}: SSIM {ssim:.4f}")
                    except Exception:
                        vprint(f"  Sample {sample['index']:02d}: SSIM failed.")
                size_bytes = encoded_meta.get("size_bytes", float("nan"))
                duration = sample["duration"] or float("nan")
                bitrate_kbps = encoded_meta.get("bitrate_kbps")
                storage_bps = (size_bytes / duration) if duration and duration > 0 else (
                    bitrate_kbps * 1000.0 / 8.0 if bitrate_kbps and not math.isnan(bitrate_kbps) else float("nan")
                )
                entry = {
                    "resolution": resolution,
                    "target_bitrate_kbps": bitrate,
                    "sample_index": sample["index"],
                    "start_seconds": sample["start"],
                    "duration_seconds": sample["duration"],
                    "encoded_path": encoded_path,
                    "actual_bitrate_kbps": encoded_meta.get("bitrate_kbps", float("nan")),
                    "width": encoded_meta.get("width"),
                    "height": encoded_meta.get("height"),
                    "size_bytes": size_bytes,
                    "storage_bytes_per_second": storage_bps,
                    "ssim": ssim,
                }
                combo_samples.append(entry)
                sample_rows.append(entry)

            summary_rows.append({
                "resolution": resolution,
                "target_bitrate_kbps": bitrate,
                "avg_actual_bitrate_kbps": _mean([s["actual_bitrate_kbps"] for s in combo_samples]),
                "mean_storage_bps": _mean([s["storage_bytes_per_second"] for s in combo_samples]),
                "mean_ssim": _mean([s["ssim"] for s in combo_samples]),
            })

    ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    summary_csv = os.path.join(output_dir, f"{ts}-summary.csv")
    samples_csv = os.path.join(output_dir, f"{ts}-samples.csv")
    summary_fields = ["resolution","target_bitrate_kbps","avg_actual_bitrate_kbps","mean_storage_bps","mean_ssim"]
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    sample_fields = [
        "resolution","target_bitrate_kbps","sample_index","start_seconds","duration_seconds",
        "encoded_path","actual_bitrate_kbps","width","height","size_bytes","storage_bytes_per_second","ssim"
    ]
    with open(samples_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sample_fields)
        writer.writeheader()
        for row in sample_rows:
            writer.writerow(row)

    metric_key = None
    if any(not (isinstance(r["mean_ssim"], float) and math.isnan(r["mean_ssim"])) for r in summary_rows):
        metric_key = "mean_ssim"
    metric_label = "SSIM"

    storage_metric_key = None
    if any(not (isinstance(r["mean_storage_bps"], float) and math.isnan(r["mean_storage_bps"])) for r in summary_rows):
        storage_metric_key = "mean_storage_bps"
    storage_label = "Storage (bytes/s)"

    plot_path = None
    if metric_key:
        fig, ax = plt.subplots()
        for resolution in resolutions:
            data = [r for r in summary_rows if r["resolution"] == resolution and not (isinstance(r[metric_key], float) and math.isnan(r[metric_key]))]
            if not data:
                continue
            data = sorted(
                data,
                key=lambda r: (
                    r["avg_actual_bitrate_kbps"]
                    if not (isinstance(r["avg_actual_bitrate_kbps"], float) and math.isnan(r["avg_actual_bitrate_kbps"]))
                    else r["target_bitrate_kbps"]
                )
            )
            xs = []
            ys = []
            for row in data:
                x = (
                    row["avg_actual_bitrate_kbps"]
                    if not (isinstance(row["avg_actual_bitrate_kbps"], float) and math.isnan(row["avg_actual_bitrate_kbps"]))
                    else row["target_bitrate_kbps"]
                )
                xs.append(x)
                ys.append(row[metric_key])
            xs_arr = np.array(xs, dtype=float)
            ys_arr = np.array(ys, dtype=float)
            if len(xs_arr) >= 2:
                deg = min(3, len(xs_arr) - 1)
                poly = np.poly1d(np.polyfit(xs_arr, ys_arr, deg))
                dense_x = np.linspace(xs_arr.min(), xs_arr.max(), max(200, len(xs_arr) * 50))
                dense_y = poly(dense_x)
                line, = ax.plot(dense_x, dense_y, linewidth=2.0, label=f"{resolution}p")
                color = line.get_color()
                ax.scatter(xs_arr, ys_arr, color=color, edgecolors="white", zorder=5)
                for x, y, row in zip(xs_arr, ys_arr, data):
                    ax.text(x * 1.01 if x else x + 1, y, f"{resolution}p/{row['target_bitrate_kbps']}k", fontsize=8, color=color)

                knee_point = _knee_from_dense(dense_x, dense_y)
                if knee_point:
                    ax.scatter(
                        [knee_point[0]],
                        [knee_point[1]],
                        marker="x",
                        s=90,
                        linewidths=2.0,
                        color="black",
                        zorder=6,
                    )
            else:
                line, = ax.plot(xs_arr, ys_arr, linewidth=2.0, label=f"{resolution}p")
                color = line.get_color()
                ax.scatter(xs_arr, ys_arr, color=color, edgecolors="white", zorder=5)
                for x, y, row in zip(xs_arr, ys_arr, data):
                    ax.text(x * 1.01 if x else x + 1, y, f"{resolution}p/{row['target_bitrate_kbps']}k", fontsize=8, color=color)
        ax.set_xlabel("Bitrate (kbps)")
        ax.set_ylabel(metric_label)
        ax.set_title("Bitrate vs Quality (sample averages)")
        ax.set_ylim(top=1.0)
        ax.grid(True)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"{ts}-quality.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

    storage_plot_path = None
    if storage_metric_key:
        fig, ax = plt.subplots()
        for resolution in resolutions:
            data = [r for r in summary_rows if r["resolution"] == resolution and not (isinstance(r["mean_storage_bps"], float) and math.isnan(r["mean_storage_bps"]))]
            if not data:
                continue
            data = [
                r for r in data
                if not (isinstance(r["mean_ssim"], float) and math.isnan(r["mean_ssim"]))
            ]
            if not data:
                continue
            data = sorted(data, key=lambda r: r["mean_ssim"])
            storages = [row["mean_storage_bps"] for row in data]
            ssim_vals = [row["mean_ssim"] for row in data]
            scatter = ax.scatter(storages, ssim_vals, marker="o", linewidths=1.2, label=f"{resolution}p")
            color = tuple(scatter.get_facecolor()[0])
            for storage, ssim_val, target_row in zip(storages, ssim_vals, data):
                ax.annotate(
                    f"{target_row['target_bitrate_kbps']}k",
                    (storage, ssim_val),
                    textcoords="offset points",
                    xytext=(6, 4),
                    fontsize=8,
                    color=color,
                )
            if len(storages) >= 2:
                slopes = []
                for idx_pt in range(len(storages)):
                    if idx_pt == 0:
                        denom = storages[1] - storages[0]
                        slope_pt = (ssim_vals[1] - ssim_vals[0]) / denom if denom else 0.0
                    elif idx_pt == len(storages) - 1:
                        denom = storages[idx_pt] - storages[idx_pt - 1]
                        slope_pt = (ssim_vals[idx_pt] - ssim_vals[idx_pt - 1]) / denom if denom else 0.0
                    else:
                        denom = storages[idx_pt + 1] - storages[idx_pt - 1]
                        slope_pt = (ssim_vals[idx_pt + 1] - ssim_vals[idx_pt - 1]) / denom if denom else 0.0
                    slopes.append(slope_pt)
                idx_tangent = max(range(len(slopes)), key=lambda i: abs(slopes[i]))
                tan_storage = storages[idx_tangent]
                tan_ssim = ssim_vals[idx_tangent]
                ax.scatter([tan_storage], [tan_ssim], marker="x", s=100, linewidths=2.0, color=color, zorder=6)
        ax.set_xlabel(storage_label)
        ax.set_ylabel(metric_label)
        ax.set_title("SSIM vs Storage")
        ax.grid(True)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels)
        plt.tight_layout()
        storage_plot_path = os.path.join(output_dir, f"{ts}-storage.png")
        plt.savefig(storage_plot_path, dpi=150, bbox_inches="tight")
        plt.close()

    markdown_path = os.path.join(output_dir, f"{ts}-results.md")
    _write_markdown(markdown_path, {
        "input": input_path,
        "output": output_dir,
        "resolutions": resolutions,
        "bitrates": bitrates,
        "sample_length": sample_length,
        "num_samples": actual_sample_count,
        "rows": summary_rows,
        "plot": plot_path,
        "storage_plot": storage_plot_path,
        "metric": metric_label if metric_key else None,
    })

    print(summary_csv)
    print(samples_csv)
    if plot_path:
        print(plot_path)
    if storage_plot_path:
        print(storage_plot_path)
    print(markdown_path)

if __name__ == "__main__":
    main()
