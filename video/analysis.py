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
    
    # 1. Create Reference Samples 
    # Scale to TARGET RESOLUTION using Lossless HEVC NVENC
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
            
            # Encode: HEVC NVENC + Slow Preset + VBR_HQ/CQ
            # Reference is already at target resolution, no scaling needed here.
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
            
            # Compute SSIM: Direct comparison (both are same resolution)
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

    # 3. Knee Point Analysis
    sorted_crfs = sorted(analysis_results.keys())
    knee_candidate = None
    if len(sorted_crfs) > 2:
        p1 = (sorted_crfs[0], analysis_results[sorted_crfs[0]]["mean_ssim"])
        p2 = (sorted_crfs[-1], analysis_results[sorted_crfs[-1]]["mean_ssim"])
        max_dist = -1.0
        A = p1[1] - p2[1]
        B = p2[0] - p1[0]
        C = p1[0]*p2[1] - p2[0]*p1[1]
        denominator = math.sqrt(A*A + B*B)
        
        if denominator > 0:
            for crf in sorted_crfs:
                ssim = analysis_results[crf]["mean_ssim"]
                dist = abs(A*crf + B*ssim + C) / denominator
                if dist > max_dist:
                    max_dist = dist
                    knee_candidate = (crf, ssim)
    
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
        lines.append(f"Estimated Knee: **CRF {knee_candidate[0]}** (SSIM: {knee_candidate[1]:.4f})")
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
