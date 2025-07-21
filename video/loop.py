#!/usr/bin/env python3
"""loop.py — Detect seamless loops of *specific lengths in frames*.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

# ────────────────────────────────────────────────────────────
# Tunables
# -----------------------------------------------------------
ENDPOINT_THRESH = 1 - (3/8)  # min SSIM(start,end)
MID_SEP_THRESH  = 0.99  # max SSIM(mid,start/end)
VAR_THRESH      = 0.01   # set >0 to enforce motion
SUPPORTED_EXTS  = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# ────────────────────────────────────────────────────────────
# Utilities
# -----------------------------------------------------------

def _resize_short_side(img: np.ndarray, target: int) -> np.ndarray:
    if target <= 0:
        return img
    h, w = img.shape[:2]
    short = min(h, w)
    if short == target:
        return img
    scale = target / short
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


def _natural_sort_key(p: Path):
    return p.stem.zfill(16)


def _write_gif(frame_paths: List[Path], fps: int, out_path: Path):
    imgs = [Image.open(p) for p in frame_paths]
    if not imgs:
        return
    duration_ms = int(1000 / fps)
    imgs[0].save(out_path, save_all=True, append_images=imgs[1:], duration=duration_ms, loop=0)


def load_thumbs(folder: Path, target: int) -> Tuple[List[Path], List[np.ndarray]]:
    files = sorted([p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED_EXTS], key=_natural_sort_key)
    if not files:
        raise RuntimeError(f"No images in {folder}")
    thumbs = []
    for p in files:
        im = cv2.imread(str(p))
        if im is None:
            raise IOError(f"Failed to read {p}")
        im = _resize_short_side(im, target)
        thumbs.append(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
    return files, thumbs


def _mean_variance(frames: List[np.ndarray], s: int, e: int) -> float:
    diffs = [1.0 - ssim(frames[i], frames[i+1]) for i in range(s, e)]
    return float(np.mean(diffs)) if diffs else 0.0


def _nms(cands: List[Tuple[int,int,float,float]]) -> List[Tuple[int,int,float,float]]:
    # sort by seam quality then variance
    cands.sort(key=lambda x: (-x[2], -x[3]))
    kept: List[Tuple[int,int,float,float]] = []
    for s,e,seam,var in cands:
        if any(not (e < ks or s > ke) for ks,ke,_,_ in kept):
            continue
        kept.append((s,e,seam,var))
    return kept


def find_loops(thumbs: List[np.ndarray], lengths: List[int]) -> List[Tuple[int,int,float,float]]:
    n = len(thumbs)
    lengths = sorted(set([l for l in lengths if l > 1]))
    if not lengths:
        return []
    min_len = lengths[0]
    loops: List[Tuple[int,int,float,float]] = []

    for start in range(0, n - min_len):
        for gap in lengths:
            end = start + gap - 1  # inclusive index
            if end >= n:
                break
            seam = ssim(thumbs[start], thumbs[end])
            if seam < ENDPOINT_THRESH:
                continue
            mid = (start + end) // 2
            mid_sep = max(ssim(thumbs[mid], thumbs[start]), ssim(thumbs[mid], thumbs[end]))
            if mid_sep > MID_SEP_THRESH:
                continue
            var = _mean_variance(thumbs, start, end)
            if var < VAR_THRESH:
                continue
            loops.append((start, end, seam, var))
    return _nms(loops)

# ────────────────────────────────────────────────────────────
# I/O helpers
# -----------------------------------------------------------

def export_loop(
    paths: List[Path],
    start: int,
    end: int,
    root: Path,
    idx: int,
    fps: int,
    copy: bool = True,
    gif: bool = False,
):
    loop_dir = root / f"loop_{idx:03d}"
    loop_dir.mkdir(parents=True, exist_ok=True)
    subset = []
    for new_i, orig in enumerate(range(start, end+1)):
        src = paths[orig]
        dst = loop_dir / f"{new_i:06d}{src.suffix.lower()}"
        if copy or os.name == "nt":
            shutil.copy2(src, dst)
        else:
            try:
                os.symlink(src.resolve(), dst)
            except FileExistsError:
                pass
        subset.append(src)

    if gif:
        gif_path = root / f"{loop_dir.name}.gif"
        _write_gif(subset, fps, gif_path)

# ────────────────────────────────────────────────────────────
# CLI / main
# -----------------------------------------------------------

def argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Detect seamless loops of specified frame lengths.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, help="Folder with sequential frames (000001.png …)")
    p.add_argument("--output", required=True, help="Destination directory for loops")
    p.add_argument("--lengths", nargs="+", type=int, default=[17, 33, 65],
                   help="Target loop lengths in frames (e.g. 17 33 65)")
    p.add_argument("--fps", type=int, required=True, help="Frame‑rate of frames on disk (GIF timing)")
    p.add_argument("--resolution", type=int, default=64,
                   help="Short side of thumbnails for SSIM analysis")
    p.add_argument("--copy", action="store_true", help="Copy frames instead of symlinks (always on Windows)")
    p.add_argument("--gif", action="store_true", help="Write loop.gif for each loop")
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv: List[str] | None = None):
    args = argparser().parse_args(argv)
    in_dir = Path(args.input).expanduser().resolve()
    out_dir = Path(args.output).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.verbose:
        print(f"[loop.py] Loading thumbnails from {in_dir} …", file=sys.stderr)
    paths, thumbs = load_thumbs(in_dir, args.resolution)

    if args.verbose:
        print(f"[loop.py] {len(paths)} frames loaded → searching lengths {args.lengths} …", file=sys.stderr)
    loops = find_loops(thumbs, args.lengths)

    if not loops:
        print("[loop.py] No loops found.", file=sys.stderr)
        return 0

    if args.verbose:
        print(f"[loop.py] {len(loops)} loops accepted → exporting …", file=sys.stderr)
    for i, (s,e,seam,var) in enumerate(loops):
        if args.verbose:
            print(f"  loop_{i:03d}: frames {s}-{e} (len={e-s+1})  seam={seam:.3f}  var={var:.3f}", file=sys.stderr)
        export_loop(paths, s, e, out_dir, i, fps=args.fps,
                    copy=args.copy or os.name=="nt", gif=args.gif)

    manifest = [
        {
            "loop_idx": i,
            "start_frame": s,
            "end_frame": e,
            "frames": e-s+1,
            "seconds": (e-s)/args.fps,
            "seam_ssim": seam,
            "mean_var": var,
        }
        for i,(s,e,seam,var) in enumerate(loops)
    ]
    with open(out_dir/"loops.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    if args.verbose:
        print(f"[loop.py] Done. Loops exported to {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
