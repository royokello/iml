import argparse, json, math, os, re, subprocess, tempfile, csv
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt

VIDEO_EXTS = {".mp4",".mkv",".mov",".avi",".webm",".m4v",".mpg",".mpeg"}

FFMPEG_BIN = "ffmpeg"
FFPROBE_BIN = "ffprobe"

VERBOSE = False

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

def compute_vmaf(distorted, reference, width=None, height=None):
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        log = tmp.name
    vf = []
    if width and height:
        vf += [f"[0:v]scale={width}:{height}:flags=bicubic,format=yuv420p[dist]",
               f"[1:v]scale={width}:{height}:flags=bicubic,format=yuv420p[ref]",
               "[dist][ref]libvmaf=log_fmt=json:log_path="+log]
    else:
        vf += ["[0:v]format=yuv420p[dist]","[1:v]format=yuv420p[ref]","[dist][ref]libvmaf=log_fmt=json:log_path="+log]
    cmd = [FFMPEG_BIN,*_ff_verbosity_args(),"-i",distorted,"-i",reference,"-lavfi",";".join(vf),"-f","null","-"]
    vprint("Running:", " ".join(cmd))
    rc,out,err = run(cmd)
    if rc != 0:
        try: os.unlink(log)
        except: pass
        raise RuntimeError(err)
    with open(log,"r") as f:
        j = json.load(f)
    try:
        s = j["pooled_metrics"]["vmaf"]["mean"]
    except Exception:
        s = j.get("aggregate",{}).get("VMAF_score")
    try: os.unlink(log)
    except: pass
    if s is None: raise RuntimeError("Unable to parse VMAF")
    return float(s)

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

def compute_psnr(distorted, reference, width=None, height=None):
    vf = []
    if width and height:
        vf += [f"[0:v]scale={width}:{height}:flags=bicubic[dist]",
               f"[1:v]scale={width}:{height}:flags=bicubic[ref]",
               "[dist][ref]psnr"]
    else:
        vf += ["[0:v][1:v]psnr"]
    cmd = [FFMPEG_BIN,*_ff_verbosity_args(),"-i",distorted,"-i",reference,"-lavfi",";".join(vf),"-f","null","-"]
    vprint("Running:", " ".join(cmd))
    rc,out,err = run(cmd)
    if rc != 0: raise RuntimeError(err)
    m = re.search(r"average:\s*([0-9.]+)", err)
    return float(m.group(1)) if m else float("nan")

def _to_kbps(val: float, unit: str) -> float:
    u = unit.lower()
    if u in ("k","kbps"): return float(val)
    if u in ("m","mbps"): return float(val) * 1000.0
    return float("nan")

def parse_intended_video_audio_kbps(filename: str):
    s = os.path.basename(filename).lower()
    m = re.search(r'_(\d{3,4})p_(\d+(?:\.\d+)?)\s*(k|kbps|m|mbps)_2ch_(\d+(?:\.\d+)?)\s*(k|kbps|m|mbps)\b', s)
    if m:
        v_val, v_unit = float(m.group(2)), m.group(3)
        a_val, a_unit = float(m.group(4)), m.group(5)
        return _to_kbps(v_val, v_unit), _to_kbps(a_val, a_unit)
    tokens = list(re.finditer(r'(\d+(?:\.\d+)?)\s*(k|kbps|m|mbps)\b', s))
    video_candidates, audio_candidates = [], []
    for t in tokens:
        val = float(t.group(1)); unit = t.group(2)
        kbps = _to_kbps(val, unit)
        start_idx = t.start()
        prefix = s[max(0, start_idx-8):start_idx]
        if "2ch_" in prefix or "stereo" in prefix or "aac" in prefix:
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
    ap.add_argument("--input", required=True, help="Folder containing videos")
    ap.add_argument("--ffmpeg-dir", default=None, help="Directory containing ffmpeg/ffprobe executables")
    ap.add_argument("-v","--verbose", action="store_true", help="Enable verbose logs and verbose ffmpeg output")
    args = ap.parse_args()

    in_dir = args.input
    global VERBOSE
    VERBOSE = bool(args.verbose)
    if args.ffmpeg_dir:
        _set_ffmpeg_bins(args.ffmpeg_dir)
        vprint(f"Using ffmpeg from: {FFMPEG_BIN}")
        vprint(f"Using ffprobe from: {FFPROBE_BIN}")
    else:
        vprint("Using ffmpeg/ffprobe from PATH")

    vids = list_videos(in_dir)
    if not vids: raise SystemExit("No videos found")
    vprint(f"Found {len(vids)} video(s) in: {in_dir}")

    ref = pick_reference(vids)
    ref_meta = ffprobe_meta(ref)
    vprint(f"Reference: {os.path.basename(ref)} ({ref_meta['width']}x{ref_meta['height']}, {ref_meta['bitrate_kbps']:.0f} kbps)")
    rows = []
    for idx, v in enumerate(vids, start=1):
        meta = ffprobe_meta(v)
        vmaf = ssim = psnr = float("nan")
        if v != ref:
            vprint(f"[{idx}/{len(vids)}] {os.path.basename(v)}: computing quality vs reference…")
            try:
                vmaf = compute_vmaf(v, ref, ref_meta["width"], ref_meta["height"])
                vprint(f"    VMAF: {vmaf:.3f}")
            except Exception:
                vprint("    VMAF failed. Trying SSIM/PSNR fallbacks…")
                try:
                    ssim = compute_ssim(v, ref, ref_meta["width"], ref_meta["height"]); vprint(f"    SSIM: {ssim:.4f}")
                except Exception:
                    vprint("    SSIM failed.")
                try:
                    psnr = compute_psnr(v, ref, ref_meta["width"], ref_meta["height"]); vprint(f"    PSNR: {psnr:.3f} dB")
                except Exception:
                    vprint("    PSNR failed.")
        intended_video_kbps, intended_audio_kbps = parse_intended_video_audio_kbps(os.path.basename(v))
        rows.append({
            "file": os.path.basename(v),
            "is_reference": v==ref,
            **meta,
            "intended_video_kbps": intended_video_kbps,
            "intended_audio_kbps": intended_audio_kbps,
            "vmaf": vmaf,
            "ssim": ssim,
            "psnr": psnr
        })

    ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    csv_path = os.path.join(in_dir, f"{ts}-result.csv")
    with open(csv_path,"w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    metric_name = "vmaf" if any(not math.isnan(r["vmaf"]) for r in rows) else ("ssim" if any(not math.isnan(r["ssim"]) for r in rows) else "psnr")

    points = []
    for r in rows:
        if r["is_reference"] and all(math.isnan(r[k]) for k in ["vmaf","ssim","psnr"]): continue
        q = r[metric_name]
        if q is None or (isinstance(q, float) and math.isnan(q)): continue
        points.append({
            "bitrate_kbps": r["bitrate_kbps"],
            "q": q,
            "height": r["height"],
            "intended_video_kbps": r.get("intended_video_kbps", float("nan"))
        })

    frontier = _pareto_frontier(points)
    efficient = _efficient_point(frontier) if frontier else None
    knee = _knee_point(frontier) if frontier else None

    ratios = []
    for r in rows:
        iv = r.get("intended_video_kbps"); act = r.get("bitrate_kbps")
        if iv and iv > 0 and act and act > 0:
            ratios.append(act / iv)
    overshoot_mean = sum(ratios)/len(ratios) if ratios else float("nan")
    overshoot_median = (sorted(ratios)[len(ratios)//2] if ratios else float("nan"))
    if ratios and len(ratios)%2==0:
        s = sorted(ratios); overshoot_median = (s[len(s)//2 - 1] + s[len(s)//2]) / 2.0

    def _req_for(target):
        if not target: return (float("nan"), float("nan"))
        achieved = target["bitrate_kbps"]
        req_mean = achieved / overshoot_mean if overshoot_mean and overshoot_mean>0 else float("nan")
        req_median = achieved / overshoot_median if overshoot_median and overshoot_median>0 else float("nan")
        return (req_mean, req_median)

    eff_req_mean, eff_req_median = _req_for(efficient)
    knee_req_mean, knee_req_median = _req_for(knee)

    xs2 = [p["bitrate_kbps"] for p in points]
    ys2 = [p["q"] for p in points]
    plt.figure(); plt.scatter(xs2, ys2)
    if efficient: plt.scatter([efficient["bitrate_kbps"]],[efficient["q"]], marker="x", s=100, label="Efficient")
    if knee: plt.scatter([knee["bitrate_kbps"]],[knee["q"]], marker="D", s=60, label="Knee")
    plt.xlabel("Bitrate (kbps)"); plt.ylabel(metric_name.upper()); plt.title("Bitrate vs Quality"); plt.grid(True)
    if efficient or knee: plt.legend()
    png_path = os.path.join(in_dir, f"{ts}-curve.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")

    # ANALYSIS: RATES ONLY (no filenames)
    analysis_rows = [{
        "metric": metric_name,
        "efficient_bitrate_kbps": (round(efficient["bitrate_kbps"],3) if efficient else ""),
        "efficient_quality": (round(efficient["q"],6) if efficient else ""),
        "knee_bitrate_kbps": (round(knee["bitrate_kbps"],3) if knee else ""),
        "knee_quality": (round(knee["q"],6) if knee else ""),
        "overshoot_mean": (round(overshoot_mean,6) if ratios else ""),
        "overshoot_median": (round(overshoot_median,6) if ratios else ""),
        "request_video_kbps_efficient_mean": (round(eff_req_mean,1) if not math.isnan(eff_req_mean) else ""),
        "request_video_kbps_efficient_median": (round(eff_req_median,1) if not math.isnan(eff_req_median) else ""),
        "request_video_kbps_knee_mean": (round(knee_req_mean,1) if not math.isnan(knee_req_mean) else ""),
        "request_video_kbps_knee_median": (round(knee_req_median,1) if not math.isnan(knee_req_median) else ""),
    }]
    analysis_path = os.path.join(in_dir, f"{ts}-analysis.csv")
    with open(analysis_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(analysis_rows[0].keys()))
        w.writeheader(); w.writerows(analysis_rows)

    # Console outputs
    print(csv_path)
    print(png_path)
    print(analysis_path)

if __name__ == "__main__":
    main()
