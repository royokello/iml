# IML

Author: Roy Okello, Stelar Labs

IML is a toolkit for image, video, and diffusion model workflows. It includes:
- Image preprocessing (group, shape), dataset curation (cull), and YOLO-based cropping (crop)
- Video frame extraction, grouping, loop detection, and quality analysis
- SDXL
- Flux 2

## Installation

Prerequisites
- Python 3.10+ and a virtual environment
- NVIDIA GPU + CUDA for GPU-accelerated tools
- FFmpeg/FFprobe on PATH for video modules

Create and activate a venv
- Windows
  - `python -m venv venv`
  - `venv\Scripts\activate`
- macOS/Linux
  - `python3 -m venv .venv`
  - `source .venv/bin/activate`

Install dependencies
- `pip install -r requirements.txt`

Notes
- Some modules expect CUDA (e.g., `video.loop`, YOLO inference/training).
- FFmpeg/FFprobe binaries are required by `video.group`, `video.loop`, and `video.quality`.

## Groups

### Video

#### `video.extract` – Extract frames
- Features
  - Evenly-spaced frames per time unit or random sampling
  - Short-side resize while preserving aspect ratio
  - Collated output or per-video subfolders
- Examples
  - `python -m video.extract --input "videos" --output "frames" --frames 2 --time second --resolution 768`
  - `python -m video.extract --input "videos" --output "frames_all" --frames 50 --random --collate --resolution 512`

#### `video.group` – Organize by orientation + size
- Features
  - Uses `ffprobe` to read width/height
  - Moves files into `_<orientation>/_<short-side>` buckets and prefixes non-group path to filename
  - Dry-run support
- Example
  - `python -m video.group --root "videos" --ffprobe "C:\\tools\\ffprobe.exe" --sizes 256 384 512 768 1024 --dry-run`

#### `video.loop` – Detect seamless loops (CUDA + FFmpeg)
- Features
  - GPU-decoded thumbnails via FFmpeg (`scale_cuda`), SSIM-based loop detection, batched exports
  - Exports lossless H.264 `.mkv` loops at target `--fps` and `--out-res`
- Example
  - `python -m video.loop --input "videos" --output "loops" --lengths 33 49 --fps 30 --in-res 64 --out-res 768 --ffmpeg "C:\\tools\\ffmpeg.exe"`

#### `video.quality` – VMAF/SSIM/PSNR + bitrate curve
- Features
  - Compares all videos in a folder to the largest file as reference
  - Produces CSVs and a bitrate vs. quality plot; tries VMAF first, falls back to SSIM/PSNR
- Example
  - `python -m video.quality --input "videos" --ffmpeg-dir "C:\\tools\\ffmpeg" -v`

### Image

#### `image.group` – Bucket images
- Features
  - Group by `width`, `height`, or `longest` side into size thresholds
  - Prefixes filename with path segments outside grouping folders
- Example
  - `python -m image.group --root "images" --orientation longest --sizes 256 384 512 768 --dry-run`

#### `image.collect` – Aggregate into one folder
- Features
  - Collects images from a directory tree into a single output folder
  - Converts to PNG, optional resize by width/height, optional square padding
  - Copies matching `.txt` captions alongside images
- Example
  - `python -m image.collect --input_dir "images" --output_dir "images_flat" --mode file`

#### `image.shape` – Non-crop reshape
- Features
  - Resizes only the chosen side (`width` or `height`), keeps the other side
  - Format-aware saves with sane defaults (JPEG/WebP/PNG/TIFF)
- Example
  - `python -m image.shape --input "images" --side width --size 1024`

#### `image.cull` – Keep/Cull classifier
- Features
  - Train a DeiT-based binary classifier on labeled images per stage
  - Inference copies kept images to next stage; logs and metrics saved
- Usage pattern (stages live under project root)
  - Train: `python -m image.cull.train --project "C:\\proj" --stage 1 --batch-size 32`
  - Predict: `python -m image.cull.main --project "C:\\proj" --stage 1`
  - Expects `stage_<N>` with images and `stage_<N>_cull_labels.csv` for training; writes `stage_<N>_cull_model.pth`

#### `image.crop` – YOLO-based cropping
- Features
  - Train a YOLO detector from CSV labels, then crop one best box per class
  - Crops are expanded to fixed aspect ratios per class and resized by long side
- Typical flow
  1) Label boxes (first step)
     - `python -m image.crop.label.main --project "C:\\proj" --stage 1` then open `http://localhost:5051`
     - Writes `stage_<N>_crop_labels.csv` under the project root (resumes if present)
  2) Prepare YOLO dataset from labels CSV
     - `python -m image.crop.prepare --project "C:\\proj" --stage 1 --val_split 0.2`
  3) Train a detector (Ultralytics)
     - `python -m image.crop.train --project "C:\\proj" --stage 1 --variant yolo11n --epochs 100 --batch 16`
  4) Run cropper using the trained weights
     - Native crop size (no resize): `python -m image.crop.main --project "C:\\proj" --stage 1 --classes 0 1 2 3`
     - Resize long side: `python -m image.crop.main --project "C:\\proj" --stage 1 --resolution 768 --classes 0 1 2 3`

### SDXL

### Flux 2 Klein 4b 

### Utils

#### `utils.stages` – Stage discovery
- Feature: `find_latest_stage(project)` returns the highest `stage_<N>` folder

#### `utils.cap.label` – Loop labeling UI
- Features
  - Minimal Flask app to tag exported loops; writes `caption.txt` in each loop folder
- Example
  - `python -m utils.cap.label --input "C:\\loops" --port 7860` then open `http://localhost:7860`
