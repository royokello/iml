# IML

Author: Roy Okello, Stelar Labs

IML is a toolkit for image, video, and diffusion model workflows. It includes:
- Image preprocessing (group, shape), dataset curation (cull), and YOLO-based cropping (crop)
- Video frame extraction, grouping, loop detection, and quality analysis
- Stable Diffusion 1.5 utilities (inference, LoRA training, tensor info, quantization)
- SDXL checkpoint profiling and a lightweight web UI for loop labeling

## Installation

Prerequisites
- Python 3.10+ and a virtual environment
- NVIDIA GPU + CUDA for GPU-accelerated tools (recommended)
- FFmpeg/FFprobe on PATH for video modules

Create and activate a venv
- Windows
  - `py -m venv venv`
  - `venv\Scripts\activate`
- macOS/Linux
  - `python3 -m venv .venv`
  - `source .venv/bin/activate`

Install dependencies
- `pip install -r requirements.txt`

Notes
- Some modules expect CUDA (e.g., `video.loop`, `sd15.*`, YOLO inference/training). CPU may work for a subset but is not the target path.
- FFmpeg/FFprobe binaries are required by `video.group`, `video.loop`, and `video.quality`.

## Groups

### Video

#### `video.extract` – Extract frames
- Features
  - Evenly-spaced frames per time unit or random sampling
  - Short-side resize while preserving aspect ratio
  - Collated output or per-video subfolders
- Examples
  - `py -m video.extract --input "videos" --output "frames" --frames 2 --time second --resolution 768`
  - `py -m video.extract --input "videos" --output "frames_all" --frames 50 --random --collate --resolution 512`

#### `video.group` – Organize by orientation + size
- Features
  - Uses `ffprobe` to read width/height
  - Moves files into `_<orientation>/_<short-side>` buckets and prefixes non-group path to filename
  - Dry-run support
- Example
  - `py -m video.group --root "videos" --ffprobe "C:\\tools\\ffprobe.exe" --sizes 256 384 512 768 1024 --dry-run`

#### `video.loop` – Detect seamless loops (CUDA + FFmpeg)
- Features
  - GPU-decoded thumbnails via FFmpeg (`scale_cuda`), SSIM-based loop detection, batched exports
  - Exports lossless H.264 `.mkv` loops at target `--fps` and `--out-res`
- Example
  - `py -m video.loop --input "videos" --output "loops" --lengths 33 49 --fps 30 --in-res 64 --out-res 768 --ffmpeg "C:\\tools\\ffmpeg.exe"`

#### `video.quality` – VMAF/SSIM/PSNR + bitrate curve
- Features
  - Compares all videos in a folder to the largest file as reference
  - Produces CSVs and a bitrate vs. quality plot; tries VMAF first, falls back to SSIM/PSNR
- Example
  - `py -m video.quality --input "videos" --ffmpeg-dir "C:\\tools\\ffmpeg" -v`

### Image

#### `image.group` – Bucket images
- Features
  - Group by `width`, `height`, or `longest` side into size thresholds
  - Prefixes filename with path segments outside grouping folders
- Example
  - `py -m image.group --root "images" --orientation longest --sizes 256 384 512 768 --dry-run`

#### `image.shape` – Non-crop reshape
- Features
  - Resizes only the chosen side (`width` or `height`), keeps the other side
  - Format-aware saves with sane defaults (JPEG/WebP/PNG/TIFF)
- Example
  - `py -m image.shape --input "images" --side width --size 1024`

#### `image.cull` – Keep/Cull classifier
- Features
  - Train a DeiT-based binary classifier on labeled images per stage
  - Inference copies kept images to next stage; logs and metrics saved
- Usage pattern (stages live under project root)
  - Train: `py -m image.cull.train --project "C:\\proj" --stage 1 --batch-size 32`
  - Predict: `py -m image.cull.main --project "C:\\proj" --stage 1`
  - Expects `stage_<N>` with images and `stage_<N>_cull_labels.csv` for training; writes `stage_<N>_cull_model.pth`

#### `image.crop` – YOLO-based cropping
- Features
  - Train a YOLO detector from CSV labels, then crop one best box per class
  - Crops are expanded to fixed aspect ratios per class and resized by long side
- Typical flow
  1) Prepare YOLO dataset from labels CSV
     - `py -m image.crop.prepare --project "C:\\proj" --stage 1 --val_split 0.2`
  2) Train a detector (Ultralytics)
     - `py -m image.crop.train --project "C:\\proj" --stage 1 --variant yolo11n --epochs 100 --batch 16`
  3) Run cropper using the trained weights
     - `py -m image.crop.main --project "C:\\proj" --stage 1 --resolution 768 --classes 0 1 2 3`

### SD15

#### `sd15.infer` – Text-to-image
- Features
  - CPU text encoding + GPU UNet/decoder for efficient VRAM use
  - DDIM, CFG scale, seed, and size control
- Example
  - `py -m sd15.infer --model "C:\\models\\sd15" --output "C:\\out" --prompt "analogue photo, ultrawide waterfall" --steps 20 --scale 7.5 --size 512,512`

#### `sd15.train` – LoRA training (manual injection)
- Features
  - Injects LoRA into UNet without PEFT; caches text embeddings; VAE on CPU
  - Presets for face/person/object/style; periodic samples and checkpoints
- Example
  - `py -m sd15.train --model "C:\\models\\sd15" --project "C:\\proj" --preset face --name alice`

#### `sd15.info` – Tensor inventory to CSV
- Features
  - Reads Diffusers directory or single `.safetensors`, filters by module(s)
  - Writes `name,shape,precision` CSV for quick audits
- Example
  - `py -m sd15.info --model "C:\\models\\sd15" --modules unet vae text --output "C:\\out"`

#### `sd15.quantize` – FP8 weight export (UNet)
- Features
  - Mixed-precision FP8 encoding per tensor category with optional per-tensor scaling
  - Produces `_quantized.safetensors` and metadata
- Example
  - `py -m sd15.quantize --model "C:\\models\\sd15" --scaling-mode tensor --scaling-precision e8m0 --proj-precision e2m1`

### SDXL

#### `sdxl.profile` – FP8 profile summary
- Features
  - Profiles checkpoint tensors by section and layer type; reports FP8/FP16 byte budgets
- Example
  - `py -m sdxl.profile --model "C:\\models\\sdxl.safetensors"`

### Utils

#### `utils.stages` – Stage discovery
- Feature: `find_latest_stage(project)` returns the highest `stage_<N>` folder

#### `utils.cap.label` – Loop labeling UI
- Features
  - Minimal Flask app to tag exported loops; writes `caption.txt` in each loop folder
- Example
  - `py -m utils.cap.label --input "C:\\loops" --port 7860` then open `http://localhost:7860`
