import csv
import json
import math
import os
import random
import shutil
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

VERBOSE = True

def vprint(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)

def _get_ffmpeg_bin(root_dir: Path) -> Path:
    return root_dir / "ffmpeg" / "bin" / "ffmpeg.exe"

def _get_ffprobe_bin(root_dir: Path) -> Path:
    return root_dir / "ffmpeg" / "bin" / "ffprobe.exe"

def _run_cmd(cmd: List[str]) -> Tuple[int, str, str]:
    # vprint("Running:", " ".join(cmd))
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.returncode, p.stdout, p.stderr

def _get_video_meta(ffmpeg_exe: Path, input_path: Path) -> Dict[str, Any]:
    ffprobe_exe = ffmpeg_exe.parent / "ffprobe.exe"
    
    meta = {"duration": 0.0, "width": 0, "height": 0}

    if ffprobe_exe.exists():
        cmd = [
            str(ffprobe_exe), "-v", "error", "-print_format", "json", 
            "-show_format", "-show_streams", str(input_path)
        ]
        rc, out, err = _run_cmd(cmd)
        if rc == 0:
            try:
                j = json.loads(out)
                v = next((s for s in j.get("streams",[]) if s.get("codec_type")=="video"),{})
                fmt = j.get("format",{})
                meta["duration"] = float(fmt.get("duration",0) or 0)
                meta["width"] = int(v.get("width") or 0)
                meta["height"] = int(v.get("height") or 0)
                return meta
            except Exception:
                pass
    
    # Fallback to ffmpeg -i parsing
    cmd = [str(ffmpeg_exe), "-i", str(input_path)]
    rc, out, err = _run_cmd(cmd)
    
    match = re.search(r"Duration:\s+(\d+):(\d+):(\d+\.\d+)", err)
    if match:
        h, m, s = map(float, match.groups())
        meta["duration"] = h * 3600 + m * 60 + s
        
    match_res = re.search(r"Stream #\d+:\d+.*Video:.*,\s+(\d+)x(\d+)", err)
    if match_res:
        meta["width"] = int(match_res.group(1))
        meta["height"] = int(match_res.group(2))
        
    return meta

def _knee_from_dense(xs, ys):
    """
    Kneedle algorithm implementation to find the 'elbow' or 'knee'
    of the curve, i.e. the point of maximum curvature / distance
    from the chord connecting endpoints.
    """
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
        
    # Normalize to [0, 1]
    x_norm = (xs_arr - x_min) / (x_max - x_min)
    y_norm = (ys_arr - y_min) / (y_max - y_min)
    
    # Difference curve (distance from y=x line if we roughly assume monotonic)
    # The 'line' is y=x in normalized space (y_norm approx x_norm)
    # Actually, we want max distance from the line connecting (0,0) to (1,1) in norm space
    # which is simply Diff = Y_norm - X_norm
    diff = y_norm - x_norm
    diff = np.asarray(diff, dtype=float)
    
    # Smooth edges if possible
    if diff.size > 4:
        diff[:2] = diff[2]
        diff[-2:] = diff[-3]
        
    try:
        # We want the 'knee' where we get diminishing returns.
        # For SSIM (y) vs CRF (x):
        # Higher CRF = Lower Quality. Curve goes Down.
        # We want the point where Quality drops significantly for small bitrate gain?
        # Or usually: X=Bitrate, Y=SSIM. (Curve goes UP). Knee is top left.
        # Here: X=CRF (Lower is better/bigger file), Y=SSIM (Higher is better).
        # SSIM drops as CRF increases.
        # Normalized: CRF 0->1 (Low->High), SSIM 1->0 (High->Low).
        # We want the "Elbow" where it starts dropping fast.
        idx = int(np.nanargmax(diff)) # This assumes specific curve shape.
        
        # NOTE: quality.py assumes standard axes. Let's rely on finding the extremum of diff.
    except (ValueError, TypeError):
        return None
        
    idx = max(1, min(idx, diff.size - 2))
    return float(xs_arr[idx]), float(ys_arr[idx])


def run_analysis(
    root_dir: Path,
    source_path: str,
    resolution: int,
    start_crf: int,
    crfs: List[int],
    sample_len: float = 8.0,
    num_samples: int = 32
) -> Dict[str, Any]:
    
    source = Path(source_path)
    if not source.is_absolute():
        if (root_dir / source_path).exists():
            source = root_dir / source_path
        
    if not source.exists():
        return {"error": f"Source file not found: {source}"}

    ffmpeg_exe = _get_ffmpeg_bin(root_dir)
    if not ffmpeg_exe.exists():
        return {"error": f"FFmpeg not found at {ffmpeg_exe}"}

    quality_dir = root_dir / "quality"
    temp_dir = quality_dir / "temp"
    quality_dir.mkdir(exist_ok=True)
    
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()

    meta = _get_video_meta(ffmpeg_exe, source)
    duration = meta["duration"]
    
    if duration < 1:
        return {"error": "Could not determine video duration or duration too short."}

    max_start = max(0, duration - sample_len)
    offsets = []
    if max_start > 0:
        offsets = sorted([random.uniform(0, max_start) for _ in range(num_samples)])
    else:
        offsets = [0.0] * num_samples

    results = []
    
    # 1. Create Reference Samples (lossless scaling)
    ref_samples = []
    scale_filter_ref = f"scale=-2:{resolution}:flags=bicubic"
    
    for i, start_time in enumerate(offsets):
        ref_name = f"ref_{i:03d}.mp4"
        ref_path = temp_dir / ref_name
        
        cmd = [
            str(ffmpeg_exe), "-y",
            "-ss", f"{start_time:.3f}",
            "-i", str(source),
            "-t", f"{sample_len}",
            "-map", "0:v:0",
            "-vf", scale_filter_ref,
            "-c:v", "hevc_nvenc",
            "-preset", "slow",
            "-rc", "constqp",
            "-qp", "0",
            "-an",
            str(ref_path)
        ]
        rc, out, err = _run_cmd(cmd)
        if rc != 0:
            print(f"Ref sample {i} failed: {err}")
            continue
        ref_samples.append({"id": i, "path": ref_path})

    if not ref_samples:
        return {"error": "Failed to generate reference samples."}

    # 2. Process CRFs
    data_points = []
    analysis_results = {}

    for crf in crfs:
        current_ssims = []
        for sample in ref_samples:
            dist_name = f"dist_{sample['id']}_crf_{crf}.mp4"
            dist_path = temp_dir / dist_name
            
            # Encode
            cmd_enc = [
                str(ffmpeg_exe), "-y",
                "-i", str(sample["path"]),
                "-c:v", "hevc_nvenc",
                "-preset", "slow",
                "-rc", "vbr_hq",
                "-cq", str(crf),
                "-b:v", "0", "-maxrate", "0", "-bufsize", "0",
                "-fps_mode", "vfr",
                "-an",
                str(dist_path)
            ]
            _run_cmd(cmd_enc)
            
            # Compare
            cmd_ssim = [
                str(ffmpeg_exe), 
                "-i", str(dist_path),
                "-i", str(sample["path"]),
                "-lavfi", "[0:v][1:v]ssim",
                "-f", "null", "-"
            ]
            rc, out, err = _run_cmd(cmd_ssim)
            
            ssim_val = 0.0
            match = re.search(r"All:\s*([0-9.]+)", err)
            if match:
                ssim_val = float(match.group(1))
            
            current_ssims.append(ssim_val)
            
            data_points.append({
                "sample_index": sample["id"],
                "crf": crf,
                "ssim": ssim_val
            })
            
            if dist_path.exists():
                os.remove(dist_path)

        mean_ssim = sum(current_ssims) / len(current_ssims) if current_ssims else 0.0
        analysis_results[crf] = {
            "mean_ssim": mean_ssim,
            "samples": current_ssims
        }

    # 3. Knee Point Analysis (Refined)
    sorted_crfs = sorted(analysis_results.keys())
    knee_candidate = None
    
    if len(sorted_crfs) >= 3:
        # Kneedle expects X increasing -> Y increasing for the simple y-x diff logic.
        # CRF 15 (High Q) -> CRF 40 (Low Q).
        # We invert X: -40 (Low Q) -> -15 (High Q).
        # This gives us a trend -40->0.93 to -15->0.99.
        data_pairs = []
        for c in sorted_crfs:
            data_pairs.append((-float(c), float(analysis_results[c]["mean_ssim"])))
        
        # Sort by X (ascending)
        data_pairs.sort(key=lambda p: p[0])
        
        xs = np.array([p[0] for p in data_pairs], dtype=float)
        ys = np.array([p[1] for p in data_pairs], dtype=float)
        
        # Fit polynomial
        deg = min(3, len(xs) - 1)
        poly = np.poly1d(np.polyfit(xs, ys, deg))
        
        # Dense sampling for knee finding
        x_dense = np.linspace(xs.min(), xs.max(), 200)
        y_dense = poly(x_dense)
        
        # Knee finding
        knee_x, knee_y = _knee_from_dense(x_dense, y_dense) or (None, None)
        
        if knee_x is not None:
             # Invert X back to positive CRF
             knee_candidate = (-knee_x, knee_y)


    # 4. Generate Report
    report_filename = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".md"
    report_path = quality_dir / report_filename
    
    lines = []
    lines.append(f"# Video Quality Analysis Report")
    lines.append(f"")
    lines.append(f"## Info")
    lines.append(f"- **Source**: `{source_path}`")
    lines.append(f"- **Target Resolution**: {resolution}p")
    lines.append(f"- **Qualities (CRF)**: {', '.join(map(str, crfs))}")
    lines.append(f"- **Encoder**: HEVC NVENC (Slow)")
    lines.append(f"- **Reference Mode**: Lossless Scaled (constqp 0)")
    lines.append(f"- **Sample Length**: {sample_len}s")
    lines.append(f"- **Num Samples**: {num_samples}")
    lines.append(f"")
    
    lines.append(f"## Mean SSIM")
    lines.append(f"| CRF | Mean SSIM |")
    lines.append(f"| --- | --- |")
    for crf in sorted_crfs:
        lines.append(f"| {crf} | {analysis_results[crf]['mean_ssim']:.4f} |")
    lines.append(f"")

    if knee_candidate:
        lines.append(f"## Knee Point Analysis")
        # Round knee CRF to nearest integer or 0.5
        rounded_crf = round(knee_candidate[0] * 2) / 2
        lines.append(f"Estimated Knee: **CRF ~{rounded_crf}** (Calculated: {knee_candidate[0]:.2f}, SSIM: {knee_candidate[1]:.4f})")
        lines.append(f"> Note: This is a derived value from the smoothed curve.")
        lines.append(f"")

    lines.append(f"## Sample Details")
    lines.append(f"| Index | CRF | SSIM |")
    lines.append(f"| --- | --- | --- |")
    data_points.sort(key=lambda x: (x['sample_index'], x['crf']))
    for dp in data_points:
        lines.append(f"| {dp['sample_index']} | {dp['crf']} | {dp['ssim']:.4f} |")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    return {
        "report_path": str(report_path),
        "mean_table": [{"crf": k, "mean_ssim": v["mean_ssim"]} for k, v in analysis_results.items()],
        "knee_point": {"crf": knee_candidate[0], "ssim": knee_candidate[1]} if knee_candidate else None,
        "samples": data_points
    }
