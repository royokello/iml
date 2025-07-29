#!/usr/bin/env python3
"""
loop.py
"""
from __future__ import annotations

import re
import subprocess
import argparse, json, os, sys, shutil
from pathlib import Path
from typing import List, Tuple, Sequence

import cv2, numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

# ── hyper‑params ──────────────────────────────────────────────────────────
ENDPOINT_THRESH = 0.618   # min SSIM(first,last)
MID_SEP_THRESH  = 0.900   # max SSIM(mid,first/last)
VAR_THRESH      = 0.100  # motion floor (set >0 to skip near‑static)

# ── helpers ──────────────────────────────────────────────────────────────

def resize_short(img: np.ndarray, side: int) -> np.ndarray:
    if side <= 0: return img
    h, w   = img.shape[:2]
    scale  = side / min(h, w)
    new_sz = (int(round(w*scale)), int(round(h*scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(img, new_sz, interpolation=interp)


def mean_var(frames: Sequence[np.ndarray], s:int, e:int) -> float:
    diffs = [1.0-ssim(frames[i], frames[i+1]) for i in range(s,e)]
    return float(np.mean(diffs)) if diffs else 0.0


def nms(cands: List[Tuple[int,int,float,float]])->List[Tuple[int,int,float,float]]:
    cands.sort(key=lambda x:(-x[2],-x[3]))
    kept=[]
    for s,e,ss,var in cands:
        if any(not(e<ks or s>ke) for ks,ke,_,_ in kept):
            continue
        kept.append((s,e,ss,var))
    return kept

# ── core loop detection on thumbnails ────────────────────────────────────

def find_loops(
    thumbs: List[np.ndarray], 
    lengths: List[int]
) -> List[Tuple[int,int,int,float,float]]:

    n = len(thumbs)
    lengths = sorted({l for l in lengths if l > 1})
    if not lengths:
        return []

    min_len = lengths[0]
    raw_loops = []

    # 1) collect all candidates
    for s in range(0, n - min_len):
        for gap in lengths:
            e = s + gap - 1
            if e >= n:
                break

            seam = ssim(thumbs[s], thumbs[e])
            if seam < ENDPOINT_THRESH:
                continue

            m = (s + e) // 2
            mid = max(ssim(thumbs[m], thumbs[s]),
                      ssim(thumbs[m], thumbs[e]))
            if mid > MID_SEP_THRESH:
                continue

            var = mean_var(thumbs, s, e)
            if var < VAR_THRESH:
                continue

            raw_loops.append((s, e, seam, var))

    # 2) non‑max suppression to prune overlaps
    kept = nms(raw_loops)  # returns list of (s, e, seam, var)

    # 3) assign loop IDs in the order they appear
    loops_with_id = [
        (i, s, e, seam, var)
        for i, (s, e, seam, var) in enumerate(kept)
    ]

    return loops_with_id


# ── video ingestion ──────────────────────────────────────────────────────


def build_ffmpeg_cmd(ffmpeg: str, path: Path, in_res: int,
                     out_fps: int, src_w: int, src_h: int) -> List[str]:
    w_arg, h_arg = ("-1", str(in_res)) if src_w >= src_h else (str(in_res), "-1")

    vf = (
        f"scale_cuda=w='if(gte(iw\\,ih)\\,{w_arg}\\,{h_arg})':"
        f"h='if(gte(iw\\,ih)\\,{h_arg}\\,{w_arg})',"
        "hwdownload,format=nv12,"
        f"fps={out_fps},"
        "format=gray"
    )
    return [
        ffmpeg, "-hide_banner", "-loglevel", "error",
        "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
        "-i", str(path),
        "-vf", vf,
        "-pix_fmt", "gray", "-f", "rawvideo", "-"
    ]

def ingest_video(path: Path,
                 in_res: int,
                 out_fps: int,
                 ffmpeg: str = "ffmpeg"
) -> List[np.ndarray]:
    """
    Extract grayscale thumbnails via NVDEC + scale_cuda at out_fps, short side = in_res.
    Returns only a list of NumPy arrays (thumbnails) in presentation order.
    """
    # ── 1. probe source geometry ──────────────────────────────────────────
    cap   = cv2.VideoCapture(str(path))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # ── 2. launch FFmpeg (GPU decode + resize) ────────────────────────────
    cmd  = build_ffmpeg_cmd(ffmpeg, path, in_res, out_fps, src_w, src_h)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**8)

    # ── 3. compute output frame size ──────────────────────────────────────
    if src_w >= src_h:
        dst_h = in_res
        dst_w = int(round(src_w * (in_res / src_h)))
    else:
        dst_w = in_res
        dst_h = int(round(src_h * (in_res / src_w)))

    # NV12 requires even dimensions
    dst_w += dst_w % 2
    dst_h += dst_h % 2
    frame_bytes = dst_w * dst_h     # gray8 ⇒ 1 byte/pixel

    # ── 4. read all thumbnails ─────────────────────────────────────────────
    thumbs: List[np.ndarray] = []
    read = proc.stdout.read

    while True:
        raw = read(frame_bytes)
        if len(raw) != frame_bytes:
            break
        thumbs.append(
            np.frombuffer(raw, np.uint8).reshape(dst_h, dst_w)
        )

    proc.stdout.close()
    proc.wait()
    return thumbs

# ── main ─────────────────────────────────────────────────────────────────

def build_argparser():
    p=argparse.ArgumentParser(description="Detect seamless loops from videos or frame folders")
    p.add_argument("--input", required=True, help="Video file or directory of videos/frames")
    p.add_argument("--output", required=True)
    p.add_argument("--lengths", nargs="+", type=int, default=[33])
    p.add_argument("--fps", type=int, required=True)
    p.add_argument("--in-res", type=int, default=64, help="Thumbnail resolution")
    p.add_argument("--out-res", type=int, default=0, help="Export resolution (0=original)")
    p.add_argument("--gif", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--skip", type=int, default=0)
    p.add_argument("--ffmpeg", required=True)
    return p
import tempfile

import tempfile
import subprocess
from pathlib import Path
from typing import List, Tuple

def process_single_video(vpath: Path, args, out_dir: Path, idx_offset: int) -> int:
    """
    Detect loops via thumbnails and emit them in one or more GPU‑cached passes,
    chunking ffmpeg calls so we never exceed Windows' CreateProcess limits by
    batching a fixed number of loops per invocation, forcing the desired FPS
    via a filter and dropping audio.
    """
    # 1) detect loops
    if args.verbose:
        print("ingesting video …")
    thumbs = ingest_video(vpath, args.in_res, args.fps, ffmpeg=args.ffmpeg)
    if args.verbose:
        print(f"finding loops in {len(thumbs)} frames …")
    loops = find_loops(thumbs, args.lengths)
    if not loops:
        return 0
    if args.verbose:
        print(f"found {len(loops)} loops – preparing ffmpeg chunks…")

    comma = r"\,"  # escaped comma for ffmpeg

    # build specs: one (filter_line, map_args) per loop
    specs: List[Tuple[str, List[str]]] = []
    for loop_id, s, e, seam, var in loops:
        label = f"v{loop_id}"
        if args.verbose:
            print(f"  • loop {loop_id}: frames {s}-{e}, seam={seam:.4f}, var={var:.4f}")
        filter_line = (
            f"[0:v]fps={args.fps},"
            f"trim=start_frame={s}:end_frame={e},"
            "setpts=PTS-STARTPTS,"
            f"scale_cuda="
            f"w='if(gte(iw{comma}ih){comma}-1{comma}{args.out_res})':"
            f"h='if(gte(iw{comma}ih){comma}{args.out_res}{comma}-1)',"
            "hwdownload,format=nv12,format=yuv420p"
            f"[{label}]"
        )
        out_path = out_dir / f"loop_{idx_offset + loop_id}.mkv"
        map_args = [
            "-map", f"[{label}]",
            "-c:v", "h264_nvenc", "-preset", "p7",
            "-rc", "constqp", "-qp", "0",
            "-pix_fmt", "yuv420p",
            str(out_path)
        ]
        specs.append((filter_line, map_args))

    # base ffmpeg command (filters+maps added later)
    base_cmd = [
        args.ffmpeg,
        "-hide_banner", "-loglevel", "error",
        "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
        "-i", str(vpath),
    ]

    # batch size by number of loops
    MAX_LOOPS_PER_CHUNK = 8
    total = 0
    cur_filters: List[str] = []
    cur_maps:    List[str] = []

    def flush_chunk():
        nonlocal cur_filters, cur_maps, total
        if not cur_filters:
            return
        # write only this chunk's filters
        with tempfile.NamedTemporaryFile('w', suffix='.ff', delete=False) as tf:
            tf.write(";".join(cur_filters))
            script_path = tf.name

        cmd = (
            base_cmd
            + ["-filter_complex_script", script_path, "-an"]
            + cur_maps
        )
        if args.verbose:
            print(f"→ running ffmpeg chunk: {len(cur_filters)} loops, cmd size approx {len(' '.join(cmd))} chars")
        subprocess.run(cmd, check=True)

        total += len(cur_filters)
        cur_filters = []
        cur_maps    = []

    # accumulate specs, flush when we hit MAX_LOOPS_PER_CHUNK
    for filter_line, map_args in specs:
        if len(cur_filters) >= MAX_LOOPS_PER_CHUNK:
            flush_chunk()
        cur_filters.append(filter_line)
        cur_maps.extend(map_args)

    # final flush of any remaining loops
    flush_chunk()
    return total


def main():
    args=build_argparser().parse_args()
    in_path=Path(args.input).expanduser().resolve()

    out_dir = Path(args.output).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    total_loops = sum(1 for p in out_dir.iterdir() if p.is_file())
    print(f"loop offset: {total_loops}")
    
    video_exts={".mp4",".mov",".mkv",".avi"}

    targets=[in_path] if in_path.is_file() else sorted(in_path.iterdir())
    for vid_idx, p in enumerate(targets[args.skip:]):
        print(f"[{vid_idx}/{len(targets)}] {p.name}")
        if p.suffix.lower() in video_exts:
            total_loops+=process_single_video(p,args,out_dir, total_loops)
    print(f"[done] {total_loops} loops exported to {out_dir}")

if __name__=="__main__":
    main()
