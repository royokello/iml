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
- If you run into PyTorch GPU memory-caching issues, set `PYTORCH_NO_CUDA_MEMORY_CACHING=1`.
- FFmpeg/FFprobe binaries are required by `video.group`, `video.loop`, and `video.quality`.

## Groups

### Video

#### `video.extract`
- Features
  - Evenly-spaced frames per time unit, random sampling, or full extraction with `--all`
  - `--all` overrides `--frames`, `--time`, and `--random`
  - Optional `--buffer` compares the marked frame with nearby frames and saves the best-scoring result
  - `--buffer 0` saves the exact marked frame; `--all` ignores `--buffer`
  - Zero-padded output names via `--filename-width` (default `000001.png` style)
  - Optional short-side resize while preserving aspect ratio (omit `--resolution` to keep original frame size)
  - Collated output or per-video subfolders
- Examples
  - `python -m video.extract --input "videos" --output "frames" --frames 2 --time second --resolution 768`
  - `python -m video.extract --input "videos" --output "frames_best" --frames 2 --time second --buffer 3 --resolution 768`
  - `python -m video.extract --input "videos" --output "frames_all" --frames 50 --random --collate --resolution 512`
  - `python -m video.extract --input "videos" --output "frames_native" --frames 2 --time second`
  - `python -m video.extract --input "videos" --output "all_frames" --all --collate`

#### `video.group`
- Features
  - Uses `ffprobe` to read width/height
  - Moves files into `_<orientation>/_<short-side>` buckets and prefixes non-group path to filename
  - Dry-run support
- Example
  - `python -m video.group --root "videos" --ffprobe "C:\\tools\\ffprobe.exe" --sizes 256 384 512 768 1024 --dry-run`

#### `video.loop`
- Features
  - GPU-decoded thumbnails via FFmpeg (`scale_cuda`), SSIM-based loop detection, batched exports
  - Exports lossless H.264 `.mkv` loops at target `--fps` and `--out-res`
- Example
  - `python -m video.loop --input "videos" --output "loops" --lengths 33 49 --fps 30 --in-res 64 --out-res 768 --ffmpeg "C:\\tools\\ffmpeg.exe"`

#### `video.quality`
- Features
  - Compares all videos in a folder to the largest file as reference
  - Produces CSVs and a bitrate vs. quality plot; tries VMAF first, falls back to SSIM/PSNR
- Example
  - `python -m video.quality --input "videos" --ffmpeg-dir "C:\\tools\\ffmpeg" -v`

### Image

#### `image.group`
- Features
  - Group by `width`, `height`, `longest`, or `shortest` side into size thresholds
  - Uses flat `_<size>` folders by default; add `--orientation-split` to nest `horizontal` / `vertical` folders for `longest` and `shortest`
  - Prefixes filename with path segments outside grouping folders
- Example
  - `python -m image.group --input "images" --orientation shortest --sizes 256 384 512 768 --dry-run`
  - `python -m image.group --input "images" --orientation shortest --orientation-split --sizes 256 384 512 768 --dry-run`

#### `image.collect`
- Features
  - Collects images from a directory tree into a single output folder
  - Converts to PNG, optional resize by width/height, optional square padding
  - Copies matching `.txt` captions alongside images
- Example
  - `python -m image.collect --input_dir "images" --output_dir "images_flat" --mode file`

#### `image.shape`
- Features
  - Writes reshaped images to a required output directory
    - `side` mode sets the chosen side (`width` or `height`) and scales the other side to preserve aspect ratio
    - `ratio` mode picks the closest ratio by aspect, then rounds down to the nearest size that matches it and enforces width/height multiples
  - Format-aware saves with sane defaults (JPEG/WebP/PNG/TIFF)
- Examples
  - `python -m image.shape --input "images" --output "out" --mode side --side width --size 1024`
  - `python -m image.shape --input "images" --output "out" --mode ratio --ratios 1x1 3x4 4x3 1x2 --length-multiple 64`

#### `image.tag.main`
- Features
  - Recursively scans the input folder for common image formats
  - Downloads the WD14 tagger model (SmilingWolf) into a local cache (`wd14_model` by default)
  - Writes comma-separated tags to `.txt` files beside each image (same basename)
  - Example
    - `python -m image.tag.main --input "images" --thresh 0.35 --max-tags 50`
  - Useful args
    - `--thresh`, `--general-thresh`, `--character-thresh`, `--max-tags` (0 for none), `--model`, `--prefix`

#### `image.tag.extract`
- Features
  - Reads comma-separated tags from `.txt` captions beside each image
  - Moves matching image + caption pairs into `<input>/<tag1>__<tag2>`
  - All tags must be present (case-insensitive)
- Example
  - `python -m image.tag.extract --input "images" --tags "1girl, blonde_hair"`

#### `image.tag.add`
- Features
  - Appends tags to every `.txt` caption in the input directory
  - Avoids duplicates and can place tags at the start or end
  - Uses comma-separated tag format
- Example
  - `python -m image.tag.add --input "images" --tags "1girl, blonde_hair"`
- Useful args
  - `--position` (`start` or `end`)

#### `image.tag.group`
- Features
  - Reads comma-separated tags from `.txt` captions beside each image
  - Groups image + caption pairs by caption tag count
  - Moves each pair into `<input>/<caption_count>_<image_count_in_bucket>`
  - Example folder names: `0_12`, `3_57`, `8_4`
- Example
  - `python -m image.tag.group --input "images"`

#### `image.dedup`
- Features
  - By default compares the whole image set across resolutions, then clusters with dHash
  - Optional `--first-set` / `--second-set` limits comparisons to matching `height`, `width`, `longest`, `shortest`, or `ratio`
  - Prints summary stats, including estimated unique images after dedup
  - Optional auto mode searches threshold for unique count closest to a target
  - When auto mode is used with sets, target analysis runs per set and allocates the target with `--target-set balanced|weighted` (default: `weighted`)
  - Optional output mode copies deduplicated images and matching `.txt` captions
  - In output mode, selects one keeper per duplicate group using weighted quality scoring:
    - sharpness score (higher is better)
    - exposure penalty (lower is better)
    - noise penalty (lower is better)
    - compression artifact penalty (lower is better)
- Example
  - `python -m image.dedup.main --input "images" --dhash-threshold 8`
  - `python -m image.dedup.main --input "images" --first-set ratio --second-set longest --dhash-threshold 8`
  - `python -m image.dedup.main --input "images" --target 500`
  - `python -m image.dedup.main --input "images" --target 500 --first-set ratio --target-set balanced`
  - `python -m image.dedup.main --input "images" --target 500 --output "images_dedup"`
  - `python -m image.dedup.main --input "images" --output "images_dedup" --sharpness-weight 1.0 --exposure-weight 0.4 --noise-weight 0.3 --artifact-weight 0.55`
- Useful args
  - `--input`, `-o/--output`
  - `--dhash-size`, `--dhash-threshold`
  - `--target`
  - `--target-set` (default `weighted`; only relevant with `--target` and sets)
  - `--first-set`, `--second-set` (`--second-set` requires `--first-set`)
  - `--min-group-size`
  - `--sharpness-weight` (default `1.0`)
  - `--exposure-weight` (default `0.4`)
  - `--noise-weight` (default `0.3`)
  - `--artifact-weight` (default `0.55`)

#### `image.cull`
- Features
  - Train a DeiT-based binary classifier on labeled images per stage
  - Inference supports `--mode copy|link|inplace` for kept-image output handling; logs and metrics saved
- Typical flow (stages live under project root)
  1) Label keep/cull (first step)
     - `python -m image.cull.label.main --project "C:\\proj" --stage 1` then open `http://localhost:5050`
     - Writes `stage_<N>_cull_labels.csv` under the project root (resumes if present)
  2) Train
     - `python -m image.cull.train --project "C:\\proj" --stage 1 --batch-size 32`
  3) Random cull preview sample (optional)
     - `python -m image.cull.sample --project "C:\\proj" --stage 1 --count 1000` (runs the stage cull model on a random sample and hard-links predicted keeps to `stage_<N>_cull_samples`)
  4) Predict
     - `python -m image.cull.main --project "C:\\proj" --stage 1`
     - Link instead of copy: `python -m image.cull.main --project "C:\\proj" --stage 1 --mode link`
     - In-place cull (move keeps, delete culls): `python -m image.cull.main --project "C:\\proj" --stage 1 --mode inplace`
     - Resume from numeric filename (inclusive): `python -m image.cull.main --project "C:\\proj" --resume 14487`
     - `--mode` defaults to `copy`
     - `--resume` compares the numeric filename stem (e.g., `014486.png` -> `14486`) and handles discontinuities
     - With `--resume` and no `--stage`, source stage is inferred as `latest - 1` (resume into the latest stage)
  - Expects `stage_<N>` with images and `stage_<N>_cull_labels.csv` for training; writes `stage_<N>_cull_model.pth`

#### `image.rank`
- Features
  - Pairwise preference labeling UI (left/right) for images in a stage
  - Writes `stage_<N>_rank_labels.csv` under the project root (resumes if present)
  - If `--stage` omitted, uses the latest `stage_<N>` found
- Example
  - `python -m image.rank.label.main --project "C:\\proj" --stage 1` then open `http://localhost:5000`

#### `image.crop`
- Features
  - Train a YOLO detector from CSV labels, then crop one best box per class
  - Crops are expanded to fixed aspect ratios per class and resized by long side
  - Output crops are zero-padded (default width 6, starting at `000001.png` via `--filename-width`)
- Typical flow
  1) Label boxes (first step)
     - `python -m image.crop.label.main --project "C:\\proj" --stage 1` then open `http://localhost:5051`
     - Custom port: `python -m image.crop.label.main --project "C:\\proj" --stage 1 --port 5001`
     - Writes `stage_<N>_crop_labels.csv` under the project root (resumes if present)
  2) Prepare YOLO dataset from labels CSV
      - `python -m image.crop.prepare --project "C:\\proj" --stage 1 --val_split 0.2`
      - Add `--balance` only if you explicitly want to downsample every class to the smallest class count
  3) Train a detector (Ultralytics)
     - `python -m image.crop.train --project "C:\\proj" --stage 1 --variant yolo11n --epochs 100 --batch 16`
  4) Run cropper using the trained weights
     - Native crop size (no resize): `python -m image.crop.main --project "C:\\proj" --stage 1 --classes 0 1 2 3 4`
     - Resize long side: `python -m image.crop.main --project "C:\\proj" --stage 1 --resolution 768 --classes 0 1 2 3 4`

### SDXL

### Flux 2 Klein 4b 

#### `flux_2_klein_4b.train`
- Current scope
  - Early scaffold for the Flux 2 Klein 4B training flow
  - Loads the tokenizer and text encoder from `<root>/flux_2_klein_4b/base`
  - Encodes captions from a local image-caption dataset into `prompt_embeds` and `text_ids`
  - Keeps text encodings on CPU while the text encoder is active, then frees the text encoder and moves the encodings to VRAM
  - Loads the denoiser from `<root>/flux_2_klein_4b/base/transformer`
  - Uses text encoder quantization args from the CLI
  - Uses fixed denoiser quantization `int4 / fp16 / block_size 64`
  - Prints encoded text memory size in MB
- Current CLI
  - `--root`
  - `--dataset`
  - `--text-quantization-precision` (default: `int8`)
  - `--text-scale-precision` (default: `fp16`)
  - `--text-block-size` (default: `128`)
- Example
  - `python -m flux_2_klein_4b.train --root "/models/flux" --dataset "/data/flux_dataset"`

#### Flux 2 Klein 4b Fine-Tuning Guidance
- Black Forest Labs guidance for LoRA
  - Learning rate: `8e-5` to `1e-4`
  - Style training steps: `1500-2500`
  - Character training steps: `1500-3000`
  - Start at `512px`, then move higher later
- Official AI-Toolkit example settings
  - `batch_size: 1`
  - `optimizer: "adamw8bit"`
  - `quantize: true`
  - `content_or_style: "balanced"`
  - Dataset resolutions should be treated as buckets, not one fixed size
- Timestep settings in official examples
  - Default example: `timestep_type: "weighted"`
  - Graphic Impressions example: `timestep_type: "shift"`

#### Flux 2 Klein 4b Dataset
- Current expected format
  - A flat folder containing images and matching caption files
  - Supported image extensions: `.png`, `.jpg`, `.jpeg`, `.webp`
  - Each image must have a `.txt` caption with the same basename
- Example
  - `000001.png`
  - `000001.txt`
  - `000002.jpg`
  - `000002.txt`
- Notes
  - The training module currently reads the captions as raw text and encodes them during the dataset step
  - Missing caption files raise an error
  - This README section only covers the features currently implemented in `train.py`

#### Flux 2 Klein 4b Dataset Guidance
- Style datasets
  - Recommended sample count: `20-40` images
  - Use a consistent trigger word in every caption
  - Describe the visible subject, composition, lighting, clothing, pose, and scene content
  - Do not explicitly name the style in every caption if the goal is to teach the style from the images themselves
  - Keep the images varied so the model learns the style instead of memorizing one composition
- Character datasets
  - Recommended sample count: `10-15` images
  - Use a consistent character trigger word or name in every caption
  - Caption stable identity traits directly: hair, face, outfit, body features, accessories, and other distinguishing details
  - Vary pose, camera angle, framing, background, and lighting across the dataset
  - Keep the identity consistent across images so the captions map to one character rather than a mixed concept

#### `flux_2_klein_4b.quantize.denoiser`
- Features
  - Quantizes the Flux 2 Klein 4B denoiser transformer weights to `fp16`, `int8`, or `int4`
  - Writes quantized output under `<root>/flux_2_klein_4b/quant/transformer`
  - Uses `<root>/flux_2_klein_4b/base/transformer/diffusion_pytorch_model.safetensors` by default
  - Accepts `--input` to quantize from an external `.safetensors` checkpoint while still using the root transformer config and root output directory
- Examples
  - `python -m flux_2_klein_4b.quantize.denoiser --root "/models/flux"`
  - `python -m flux_2_klein_4b.quantize.denoiser --root "/models/flux" --input "/external/transformer/diffusion_pytorch_model.safetensors"`
  - `python -m flux_2_klein_4b.quantize.denoiser --root "/models/flux" --input "/external/transformer/custom_name.safetensors" --quantization-precision int4 --scale-precision fp16 --block-size 128`
- Useful args
  - `--root`
  - `--input`
  - `--quantization-precision`
  - `--scale-precision`
  - `--block-size`
  - `--target-linear-names`

### Utils

#### `utils.stages`
- Feature: `find_latest_stage(project)` returns the highest `stage_<N>` folder

#### `utils.eval_dequant`
- Features
  - Benchmarks payload-to-`fp16` conversion only for the Flux 2 Klein 4B denoiser linear tensor shapes
  - Covers fixed `int4` and `int8` cases, `fp16` scales, and block sizes `32`, `64`, and `128`
  - Compares reference dequantization against the prebuilt CUDA full-dequant kernels:
    - `int4_reference_dequant`
    - `int4_cuda_dequant_kernel`
    - `int8_reference_dequant`
    - `int8_cuda_dequant_kernel`
  - Validates each method against the matching reference path and reports `max_abs_diff` plus pass/fail status
  - Prints per-case timing while running, then prints a final full table, per-case winners, and per-method averages
  - Loads prebuilt `int4_dequant_cuda` and `int8_dequant_cuda` extensions from `utils/cuda/*_dequant`; build them first before running
  - Uses fixed constants in the file; there are no CLI args
- Example
  - `python -m utils.eval_dequant`

#### `utils.cap.label`
- Features
  - Minimal Flask app to tag exported loops; writes `caption.txt` in each loop folder
- Example
  - `python -m utils.cap.label --input "C:\\loops" --port 7860` then open `http://localhost:7860`


#### `utils.cuda.int4_dequant`
- Features
  - CUDA full dequant kernel for packed signed `int4 -> fp16`
  - Uses a `256`-entry byte LUT stored in CUDA constant memory, where each byte decodes to one `half2`
  - Multiplies decoded values by `fp16` per-block scales inside the kernel
  - Tuned for GTX 1060 6GB / GP106 / compute capability `6.1`
  - Supports block sizes `32`, `64`, and `128`
- Files
  - Header: [int4_dequant_lut.cuh](utils/cuda/int4_dequant/int4_dequant_lut.cuh)
  - CUDA source: [int4_dequant_lut.cu](utils/cuda/int4_dequant/int4_dequant_lut.cu)
  - Python extension shim: [int4_dequant_extension.cpp](utils/cuda/int4_dequant/int4_dequant_extension.cpp)
  - Build script: [setup.py](utils/cuda/int4_dequant/setup.py)
- Build
  - Open a shell with your venv active and CUDA/MSVC available
  - Build the prebuilt CUDA Python module in place:
```bash
cd utils/cuda/int4_dequant
python setup.py build_ext --inplace
```
  - This build path uses `BuildExtension.with_options(use_ninja=False)`, so it does not require `ninja`
- Python usage after build
```python
import torch
import int4_dequant_cuda

packed = torch.empty((1024,), device="cuda", dtype=torch.uint8)
scales = torch.empty((64,), device="cuda", dtype=torch.float16)
out = torch.empty((2048,), device="cuda", dtype=torch.float16)
int4_dequant_cuda.dequantize_int4_fp16(packed, scales, out, out.numel(), 32)
```
- Public API
  - `iml::cuda::int4_dequant::init_byte_to_half2_lut(cudaStream_t stream = nullptr)`
    - Initializes the constant-memory LUT once before dequant launches
  - `iml::cuda::int4_dequant::launch_byte_lut_dequant(const uint8_t* packed, const __half* scales, __half* out, int64_t original_numel, int block_size, cudaStream_t stream = nullptr)`
    - Launches the full dequant kernel
- Input / output
  - `packed` is a packed signed-int4 buffer with two values per byte, using the same two's-complement nibble encoding as `utils.quantize`
  - `scales` is a contiguous CUDA `fp16` tensor with one scale per quantization block
  - `out` must point to a device buffer large enough for `original_numel` `fp16` values
  - `original_numel` is the number of decoded scalar outputs, not the number of packed bytes
- Notes
  - This module performs full dequant for the `fp16` scale-precision case only.
  - The LUT must be initialized before the first dequant launch in the current CUDA context.
  - The kernel assumes CUDA device pointers for `packed`, `scales`, and `out`.

#### `utils.cuda.int8_dequant`
- Features
  - CUDA full dequant kernel for plain `int8 -> fp16`
  - Reads contiguous `int8` values directly; there is no packed payload decode stage
  - Multiplies values by `fp16` per-block scales inside the kernel
  - Tuned for GTX 1060 6GB / GP106 / compute capability `6.1`
  - Supports block sizes `32`, `64`, and `128`
- Files
  - Header: [int8_dequant.cuh](utils/cuda/int8_dequant/int8_dequant.cuh)
  - CUDA source: [int8_dequant.cu](utils/cuda/int8_dequant/int8_dequant.cu)
  - Python extension shim: [int8_dequant_extension.cpp](utils/cuda/int8_dequant/int8_dequant_extension.cpp)
  - Build script: [setup.py](utils/cuda/int8_dequant/setup.py)
- Build
  - Open a shell with your venv active and CUDA/MSVC available
  - Build the prebuilt CUDA Python module in place:
```bash
cd utils/cuda/int8_dequant
python setup.py build_ext --inplace
```
  - This build path uses `BuildExtension.with_options(use_ninja=False)`, so it does not require `ninja`
- Python usage after build
```python
import torch
import int8_dequant_cuda

quantized = torch.empty((2048,), device="cuda", dtype=torch.int8)
scales = torch.empty((64,), device="cuda", dtype=torch.float16)
out = torch.empty((2048,), device="cuda", dtype=torch.float16)
int8_dequant_cuda.dequantize_int8_fp16(quantized, scales, out, out.numel(), 32)
```
- Public API
  - `iml::cuda::int8_dequant::launch_int8_dequant(const int8_t* quantized, const __half* scales, __half* out, int64_t original_numel, int block_size, cudaStream_t stream = nullptr)`
    - Launches the full int8 dequant kernel
- Input / output
  - `quantized` is a contiguous CUDA `int8` tensor
  - `scales` is a contiguous CUDA `fp16` tensor with one scale per quantization block
  - `out` must point to a device buffer large enough for `original_numel` `fp16` values
  - `original_numel` is the number of scalar outputs
- Notes
  - This module performs full dequant for the `fp16` scale-precision case only.
  - The kernel assumes CUDA device pointers for `quantized`, `scales`, and `out`.
