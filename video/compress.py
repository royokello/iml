import shutil
import subprocess
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple

def _get_ffmpeg_bin(root_dir: Path) -> Path:
    return root_dir / "ffmpeg" / "bin" / "ffmpeg.exe"

def _run_cmd(cmd: List[str]) -> Tuple[int, str, str]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.returncode, p.stdout, p.stderr

def _compress_file(
    ffmpeg_exe: Path,
    source: Path,
    output_dir: Path,
    resolution: int,
    crf: int
) -> Dict[str, Any]:
    
    stem = source.stem
    ext = source.suffix
    new_filename = f"{stem}_hevc_{resolution}p_cq_{crf}{ext}"
    output_path = output_dir / new_filename

    scale_filter = f"scale=-2:{resolution}:flags=bicubic"
    
    cmd = [
        str(ffmpeg_exe), "-y",
        "-i", str(source),
        "-map", "0:v:0",       # First video stream only
        "-map", "0:a?",        # All audio streams
        "-map", "0:s?",        # All subtitle streams
        "-map", "0:t?",        # All attachment streams
        "-dn",                 # No data streams
        "-vf", scale_filter,
        "-c:v", "hevc_nvenc",
        "-pix_fmt", "yuv420p",
        "-colorspace", "bt709",
        "-color_primaries", "bt709",
        "-color_trc", "bt709",
        "-color_range", "tv",
        "-preset", "slow",
        "-rc", "vbr_hq",
        "-cq", str(crf),
        "-b:v", "0", "-maxrate", "0", "-bufsize", "0",
        "-fps_mode", "vfr",
        "-c:a", "copy",
        "-c:s", "copy",
        "-c:t", "copy",
        str(output_path)
    ]
    
    rc, out, err = _run_cmd(cmd)
    
    result = {
        "source": str(source),
        "output": str(output_path),
        "resolution": resolution,
        "crf": crf,
        "success": False,
        "error": None,
        "source_size": 0,
        "output_size": 0
    }

    if rc != 0:
        result["error"] = err
        return result
        
    try:
        source_size = source.stat().st_size
        output_size = output_path.stat().st_size
        result["success"] = True
        result["source_size"] = source_size
        result["output_size"] = output_size
    except Exception as e:
        result["error"] = str(e)
        
    return result

def run_batch_compression(
    root_dir: Path,
    source_input: str,
    output_dir_path: str,
    resolution: int,
    crf: int
) -> List[Dict[str, Any]]:
    
    source_path = Path(source_input)
    # Resolve source
    if not source_path.is_absolute():
        if (root_dir / source_input).exists():
            source_path = root_dir / source_input
            
    if not source_path.exists():
        return [{"error": f"Source not found: {source_path}", "success": False}]

    # Resolve Output Dir
    output_dir = Path(output_dir_path)
    if not output_dir.is_absolute():
        output_dir = root_dir / output_dir_path
        
    ffmpeg_exe = _get_ffmpeg_bin(root_dir)
    if not ffmpeg_exe.exists():
        return [{"error": f"FFmpeg not found at {ffmpeg_exe}", "success": False}]
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    files_to_process = []
    
    if source_path.is_file():
        files_to_process.append(source_path)
    elif source_path.is_dir():
        # extensions to look for
        exts = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv"}
        for f in source_path.iterdir():
            if f.is_file() and f.suffix.lower() in exts:
                files_to_process.append(f)
    
    results = []
    for f in files_to_process:
        res = _compress_file(ffmpeg_exe, f, output_dir, resolution, crf)
        results.append(res)
        
    if not results:
         return [{"error": "No video files found to process.", "success": False}]
         
    return results
