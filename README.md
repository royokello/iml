# IML

Author: Roy Okello, Stelar Labs

IML is a toolkit for image, video, and diffusion model workflows. It includes:
- Image preprocessing (group, shape), dataset curation (cull), and YOLO-based cropping (crop)
- Video frame extraction, grouping, loop detection, and quality analysis
- SDXL
- Flux 2

## Symlink Setup

Keep large model files outside the repository and symlink them into the expected project layout.

General pattern
- source: `<model_storage>\<model_name>\model\...`
- target: `<project_root>\<model_name>\model\...`

PowerShell examples
- Create a directory symlink:
  - `New-Item -ItemType SymbolicLink -Path "<project_root>\<model_name>\model\text_encoder" -Target "<model_storage>\<model_name>\model\text_encoder"`
- Link every file in a folder:
  - ```powershell
    $src = "<model_storage>\<model_name>\model\text_encoder"
    $dst = "<project_root>\<model_name>\model\text_encoder"
    New-Item -ItemType Directory -Force -Path $dst | Out-Null
    Get-ChildItem -Path $src -File | ForEach-Object {
        New-Item -ItemType SymbolicLink -Path (Join-Path $dst $_.Name) -Target $_.FullName
    }
    ```
- If you are linking a whole folder, create one directory symlink.
- If you only want selected files, use the file-link loop above.

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
- If you run into PyTorch GPU memory-caching issues, `set PYTORCH_NO_CUDA_MEMORY_CACHING=1`.
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
  - Standalone selector app via `python -m video.extract.select --root "/tools"` for previewing interval marks, optionally limiting work to time spans like `1:00 - 1:10`, `32:40 - 40:00`, or `00:15:30 - 00:18:00`, and saving best nearby native-res frames
  - The selector resolves FFmpeg from `<root>/ffmpeg/bin/ffmpeg.exe`
  - Preview images are generated efficiently with FFmpeg into `<output>/preview`, and that folder is cleared before each new preview build
- Examples
  - `python -m video.extract --input "videos" --output "frames" --frames 2 --time second --resolution 768`
  - `python -m video.extract --input "videos" --output "frames_best" --frames 2 --time second --buffer 3 --resolution 768`
  - `python -m video.extract --input "videos" --output "frames_all" --frames 50 --random --collate --resolution 512`
  - `python -m video.extract --input "videos" --output "frames_native" --frames 2 --time second`
  - `python -m video.extract --input "videos" --output "all_frames" --all --collate`
  - `python -m video.extract.select --root "/tools" --port 5052`

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

#### `video.grid`
- Features
  - Web UI at `/grids` for building image grids from a source video
  - `Frame interval` mode samples frames at a fixed seconds interval and emits as many full grids as possible
  - `Segment midpoints` mode divides the full video into exactly `rows * cols` segments and picks the middle frame from each segment to build one grid
  - Center-crops each frame to the requested cell ratio before resizing into the grid

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
  - Writes resized/cropped images to an output directory or updates files in place with `--inplace`
    - `side` mode sets the chosen side (`width`, `height`, or `longest`) and scales the other side to preserve aspect ratio
    - `ratio` mode picks the closest ratio by aspect, computes the target size, then applies a best-fit center crop; `--min` sets the shortest side exactly, `--max` sets the longest side exactly, and `--mid` caps square outputs at that side while keeping square inputs at or above `--min` aligned to `--length-multiple`
    - `--min`, `--max`, and `--mid` must be multiples of `--length-multiple`
  - Format-aware saves with sane defaults (JPEG/WebP/PNG/TIFF)
- Examples
  - `python -m image.shape --input "images" --output "out" --mode side --side width --size 1024`
  - `python -m image.shape --input "images" --output "out" --mode ratio --ratios 1x1 3x4 4x3 1x2 --length-multiple 64`
  - `python -m image.shape --input "images" --inplace --mode side --side height --size 768`
  - `python -m image.shape --input "images" --inplace --mode side --side longest --size 768`
  - `python -m image.shape --input "images" --inplace --mode ratio --ratios 1x1 2x3 3x4 1x2 2x1 4x3 3x2 --length-multiple 64 --min 512`
  - `python -m image.shape --input "images" --inplace --mode ratio --ratios 1x1 2x3 3x4 1x2 2x1 4x3 3x2 --length-multiple 64 --max 768`
  - `python -m image.shape --input "images" --inplace --mode ratio --ratios 1x1 2x3 3x4 1x2 2x1 4x3 3x2 --length-multiple 64 --min 512 --mid 640`
    - `--mid` will not upscale square inputs below 640x640; square inputs at or above `--min` are snapped to the nearest `--length-multiple`
  - `python -m image.shape --input "images" --inplace --mode ratio --ratios 1x1 2x3 3x4 1x2 2x1 4x3 3x2 --length-multiple 64 --max 768 --mid 640`

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


### Flux 2

The unified `flux2` package provides the version-aware entrypoints for generation and quantization.

#### `flux2.gen`
- Features
  - Unified Flux 2 generation entrypoint for `4b` and `9b`
  - Uses `--version` to select `<root>/flux_2_klein_4b` or `<root>/flux2_9b`
  - Generates a single image from a prompt and saves it to the selected model output folder
  - Requires local Flux 2 assets under the selected model's `model` directory:
    - `tokenizer`
    - `text_encoder`
    - `scheduler`
    - `vae`
    - `transformer/distill`
    - `transformer/base`
  - CUDA-only runtime; exits if CUDA is unavailable
  - Distilled mode is the default: `4` steps and guidance scale `1.0`
  - `--base` switches to base-model defaults: `50` steps and CFG guidance scale `4.0`
  - Supports optional image conditioning with `--images "img1.png, img2.png"` by encoding reference images into latent tokens
  - Reference images are resized so their longest side is at most `--ref-size`, then cropped to the nearest valid VAE multiple before encoding
  - Uses `--text-quant-method` and `--denoiser-quant-method` to choose `none`, `single`, or `double`
  - Passing `--text-quant-method none` or `--denoiser-quant-method none` keeps the original fp16 checkpoint weights
  - Saved quantized checkpoints are loaded automatically when present:
    - text encoder: `<root>/<model>/model/text_encoder/single_quant.safetensors` or `double_quant.safetensors`
    - denoiser: `<root>/<model>/model/transformer/<variant>/single_quant.safetensors` or `double_quant.safetensors`
  - Supports merging one or more LoRA checkpoints at load time with `--loras "path1:1.0,path2:0.7"`
- Examples
  - Prompt-only distilled generation:
    - `python -m flux2.gen --root "/models/flux" --version 4b --prompt "cinematic portrait, 85mm photo, rim lighting"`
  - Prompt-only base generation:
    - `python -m flux2.gen --root "/models/flux" --version 9b --base --prompt "fashion editorial, full body, studio backdrop" --width 768 --height 1024`
  - Image-conditioned generation:
    - `python -m flux2.gen --root "/models/flux" --version 4b --images "/data/ref1.png, /data/ref2.png" --prompt "same outfit, new pose, soft daylight"`
  - Apply LoRAs during generation:
    - `python -m flux2.gen --root "/models/flux" --version 9b --prompt "stylized character art" --loras "/models/lora/style.safetensors:0.8,/models/lora/character.safetensors:1.0"`
  - Disable denoiser quantization and force full-precision checkpoint weights as loaded:
    - `python -m flux2.gen --root "/models/flux" --version 4b --prompt "product render on white seamless" --denoiser-quant-method none`
- Useful args
  - `--root`
  - `--version`
  - `--prompt`
  - `--images`
  - `--width`, `--height`
  - `--steps`
  - `--seed`
  - `--base`
  - `--ref-size`
  - `--guidance-scale`
  - `--text-quant-method`
  - `--denoiser-quant-method`
  - `--loras`
  - `--max-length`

#### `flux2.train`
- Current scope
  - Unified Flux 2 training entrypoint
  - Current implementation trains against the Flux 2 Klein 4B model layout under `<root>/flux_2_klein_4b/model`
  - Loads the tokenizer and text encoder from the selected model directory
  - Encodes captions from a local image-caption dataset into `prompt_embeds` and `text_ids`
  - Keeps text encodings on CPU while the text encoder is active, then frees the text encoder and moves the encodings to VRAM
  - Loads the base denoiser checkpoint from the selected model's `model/transformer/base` directory
  - Writes LoRA checkpoints to `<output>/models/epoch_<n>.safetensors`
  - Writes training losses to `<output>/logs.csv`
  - Uses text encoder quantization method from the CLI
  - Uses denoiser quantization method from the CLI
  - Prints encoded text memory size in MB
- Current CLI
  - `--root`
  - `--dataset`
  - `--output`
  - `--steps`
  - `--resume`
  - `--text-quant-method` (default: `single`)
  - `--denoiser-quant-method` (default: `double`)
  - `--trigger`
- Example
  - `python -m flux2.train --root "/models/flux" --dataset "/data/flux_dataset" --output "/runs/flux_style_a"`

#### `flux2.quant.text_encoder`
- Features
  - Quantizes the Flux 2 text encoder for `4b` or `9b`
  - Uses `model-00001-of-00002.safetensors` for `4b`
  - Uses `model-00001-of-00004.safetensors` through `model-00004-of-00004.safetensors` for `9b`
  - Builds the target tensor list from the shared Qwen linear suffixes and the version layer count
  - Saves `<root>/<model>/model/text_encoder/<method>_quant.safetensors`
- Examples
  - `python -m flux2.quant.text_encoder --root "/models/flux" --version 4b --method single`
  - `python -m flux2.quant.text_encoder --root "/models/flux" --version 9b --method double`
- Useful args
  - `--root`
  - `--version`
  - `--method`

#### `flux2.quant.denoiser`
- Features
  - Quantizes the Flux 2 denoiser for `4b` or `9b`
  - Uses `diffusion_pytorch_model.safetensors` for `4b`
  - Uses `diffusion_pytorch_model-00001-of-00002.safetensors` and `diffusion_pytorch_model-00002-of-00002.safetensors` for `9b`
  - Quantizes `5` double blocks and `20` single blocks for `4b`
  - Quantizes `8` double blocks and `24` single blocks for `9b`
  - Saves `<root>/<model>/model/transformer/<variant>/<method>_quant.safetensors`
- Examples
  - `python -m flux2.quant.denoiser --root "/models/flux" --version 4b --variant distill --method single`
  - `python -m flux2.quant.denoiser --root "/models/flux" --version 9b --variant base --method double`
- Useful args
  - `--root`
  - `--version`
  - `--variant`
  - `--method`

#### Flux 2 Fine-Tuning Guidance
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

#### Flux 2 Dataset
- Current expected format
  - A flat folder containing training images
  - Supported image extensions: `.png`, `.jpg`, `.jpeg`, `.webp`
  - Captioned mode: each image has a `.txt` caption with the same basename
  - Captionless mode: no image has a `.txt` caption and `--trigger "<text>"` is required
- Example
  - Captioned dataset
  - `000001.png`
  - `000001.txt`
  - `000002.jpg`
  - `000002.txt`
  - Captionless dataset
  - `000001.png`
  - `000002.jpg`
- Notes
  - Captioned datasets are encoded per image and cached on CPU before training
  - Captionless datasets encode the shared trigger once and keep that conditioning on GPU for every image
  - Mixed datasets are rejected: either every image has a caption or none do

#### Flux 2 Dataset Guidance
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

### Wan 2.2 TI2V 5b

#### `wan22.quant.text_encoder`
- Features
  - Quantizes the Wan 2.2 TI2V 5B T5 text encoder linear weights with method `single` or `double`
  - Loads the base checkpoint from `<root>/wan22/model/text_encoder/t5_umt5-xxl-enc-bf16.pth` by default
  - `--input` can point directly at a `text_encoder` model directory and bypass the `<root>/wan22/model/text_encoder` default
  - Targets all `24` T5 blocks for attention and feed-forward linear weights:
    - `attn.q.weight`, `attn.k.weight`, `attn.v.weight`, `attn.o.weight`
    - `ffn.gate.0.weight`, `ffn.fc1.weight`, `ffn.fc2.weight`
  - Writes `<model_dir>/<method>_quant.pth`
  - Keeps non-target tensors in the output checkpoint, converting `float32` and `bfloat16` tensors to `float16`
- Examples
  - `python -m wan22.quant.text_encoder --root "/models/wan" --method single`
  - `python -m wan22.quant.text_encoder --root "/models/wan" --method double`
  - `python -m wan22.quant.text_encoder --root "/models/wan" --input "/models/custom/text_encoder" --method single`
- Useful args
  - `--root`
  - `--input`
  - `--method`

#### `wan22.quant.denoiser`
- Features
  - Quantizes the Wan 2.2 TI2V 5B denoiser linear weights with method `single` or `double`
  - Loads the base sharded denoiser checkpoint from `<root>/wan22/model/denoiser` by default:
    - `diffusion_pytorch_model-00001-of-00003.safetensors`
    - `diffusion_pytorch_model-00002-of-00003.safetensors`
    - `diffusion_pytorch_model-00003-of-00003.safetensors`
  - `--input` can point directly at a `denoiser` model directory and bypass the `<root>/wan22/model/denoiser` default
  - Targets all `30` denoiser blocks for self-attention, cross-attention, and feed-forward linear weights:
    - `self_attn.q.weight`, `self_attn.k.weight`, `self_attn.v.weight`, `self_attn.o.weight`
    - `cross_attn.q.weight`, `cross_attn.k.weight`, `cross_attn.v.weight`, `cross_attn.o.weight`
    - `ffn.0.weight`, `ffn.2.weight`
  - Writes `<model_dir>/<method>_quant.safetensors`
  - Stores safetensors metadata with `component=denoiser` and `method=<method>`
  - Keeps non-target tensors in the output checkpoint, converting `float32` and `bfloat16` tensors to `float16`
- Examples
  - `python -m wan22.quant.denoiser --root "/models/wan" --method single`
  - `python -m wan22.quant.denoiser --root "/models/wan" --method double`
  - `python -m wan22.quant.denoiser --root "/models/wan" --input "/models/custom/denoiser" --method double`
- Useful args
  - `--root`
  - `--input`
  - `--method`


### Utils

#### `utils.stages`
- Feature: `find_latest_stage(project)` returns the highest `stage_<N>` folder

#### `utils.quant.eval`
- Features
  - Benchmarks dequant-to-`fp16` only for the Flux 2 Klein 4B denoiser linear tensor shapes
  - Compares the current `utils.quant` dequantizers:
    - `single_block_dequant`
    - `double_block_dequant`
  - Prepares benchmark inputs with:
    - `quantize_to_single_block`
    - `quantize_to_double_block`
  - Measures reconstruction error against the original `fp16` weight tensor for each case
  - Reports:
    - `max_abs_diff`
    - `mean_abs_diff`
  - Prints per-case timing while running, then prints a full result table, best method per shape, and per-method averages
  - Reports input and output byte counts for each benchmark case
  - Computes accuracy metrics outside the timed loop, so they do not affect the speed measurements
  - Requires CUDA, because the double-block dequant path uses the CUDA extension in `utils/quant/double/cuda`
  - Uses fixed constants in the file; there are no CLI args
- Example
  - `python -m utils.quant.eval`

#### `utils.cap.label`
- Features
  - Minimal Flask app to tag exported loops; writes `caption.txt` in each loop folder
- Example
  - `python -m utils.cap.label --input "C:\\loops" --port 7860` then open `http://localhost:7860`

#### `utils.quant`
- Features
  - Package entry point for the block quantizers under `utils/quant`
  - Re-exports:
    - `quantize_to_single_block`
    - `quantize_to_double_block`
- Files
  - Package exports: [__init__.py](utils/quant/__init__.py)
  - Single-block quantizer: [utils/quant/single](utils/quant/single)
  - Double-block quantizer: [utils/quant/double](utils/quant/double)

#### `utils.quant.single`
- Features
  - Symmetric per-block `int8` quantization with one `fp16` scale per block
  - Uses a fixed block size of `16`
  - Preserves the original tensor shape for the quantized payload; only the internal math path pads to block boundaries
  - Supports round-trip conversion with:
    - `quantize_to_single_block`
    - `dequantize_from_single_block`
- Files
  - Package exports: [__init__.py](utils/quant/single/__init__.py)
  - Quantizer: [to.py](utils/quant/single/to.py)
  - Dequantizer: [fro.py](utils/quant/single/fro.py)
- Public API
  - `utils.quant.single.BLOCK_SIZE`
    - Fixed at `16`
  - `utils.quant.single.quantize_to_single_block(tensor)`
    - Returns `(quantized, scales)`
  - `utils.quant.single.dequantize_from_single_block(tensor, scales)`
    - Returns the reconstructed `fp16` tensor
- Input / output
  - `tensor` for quantization must be floating-point
  - `quantized` is returned as `torch.int8` with the same shape as the source tensor
  - `scales` is a flat `torch.float16` tensor with one scale per `16` values in flattened order
- Example
```python
import torch
from utils.quant.single import dequantize_from_single_block, quantize_to_single_block

weight = torch.randn(128, 64, device="cuda", dtype=torch.float16)
qweight, scales = quantize_to_single_block(weight)
restored = dequantize_from_single_block(qweight, scales)
```

#### `utils.quant.double`
- Features
  - Hierarchical double-block quantization using packed signed `int4` payloads
  - Uses:
    - `SUPER_BLOCK_SIZE = 128`
    - `SUB_BLOCK_SIZE = 16`
    - `SUB_BLOCKS_PER_SUPER = 8`
  - Stores one `fp16` `super_scale` per super-block and eight `int8` local scale codes per super-block
  - Local scale codes are derived from each local block's absmax using `ceil`, with true-zero blocks stored as `0` and nonzero blocks clamped to `1..127`
  - The packed weight stream stays padded to full `128`-value super-blocks so payload and metadata remain aligned
  - Packs two signed int4 values into each `torch.int8` output byte
  - Dequantization entry point uses the CUDA module in `utils/quant/double/cuda`
- Files
  - Package exports: [__init__.py](utils/quant/double/__init__.py)
  - Quantizer: [to.py](utils/quant/double/to.py)
  - Dequantizer wrapper: [fro.py](utils/quant/double/fro.py)
  - CUDA kernel module: [cuda](utils/quant/double/cuda)
- Public API
  - `utils.quant.double.SUPER_BLOCK_SIZE`
    - Fixed at `128`
  - `utils.quant.double.SUB_BLOCK_SIZE`
    - Fixed at `16`
  - `utils.quant.double.SUB_BLOCKS_PER_SUPER`
    - Fixed at `8`
  - `utils.quant.double.quantize_to_double_block(tensor)`
    - Returns `(packed, sub_scales, super_scales)`
  - `utils.quant.double.dequantize_from_double_block(tensor, sub_scales, super_scales, original_numel=None)`
    - Returns the reconstructed `fp16` tensor via the CUDA extension
- Input / output
  - `tensor` for quantization must be floating-point
  - `packed` is a flat `torch.int8` tensor containing two signed int4 values per byte, padded to a whole number of super-blocks
  - `sub_scales` is a `torch.int8` tensor with shape `(<num_super_blocks>, 8)` containing local scale codes in `0..127`
  - `super_scales` is a flat `torch.float16` tensor with one scale per super-block
  - `dequantize_from_double_block` requires CUDA tensors and the built extension in `utils/quant/double/cuda`
- Example
```python
import torch
from utils.quant.double import dequantize_from_double_block, quantize_to_double_block

weight = torch.randn(128, 64, device="cuda", dtype=torch.float16)
packed, sub_scales, super_scales = quantize_to_double_block(weight)
restored = dequantize_from_double_block(
    packed,
    sub_scales,
    super_scales,
    original_numel=weight.numel(),
).view_as(weight)
```

#### `utils.quant.double.cuda`
- Features
  - CUDA full dequant kernel for the double-block quantization format produced by `utils.quant.double.to.quantize_to_double_block`
  - Decodes packed signed `int4` weights into `fp16` with a `256`-entry constant-memory byte LUT, where each byte decodes to one `half2`
  - Applies the hierarchical scale path inside the kernel:
    - one `fp16` `super_scale` per `128` values
    - eight `int8` local scale codes per super-block, one per `16` values
    - effective scale per sub-block is `super_scale * local_scale_code`
  - Stages CTA-local effective scales for the four super-blocks covered by each `256`-thread launch block
  - Tuned for GTX 1060 6GB / GP106 / compute capability `6.1`
- Files
  - Header: [dequantize_from_double_block.cuh](utils/quant/double/cuda/dequantize_from_double_block.cuh)
  - CUDA source: [dequantize_from_double_block.cu](utils/quant/double/cuda/dequantize_from_double_block.cu)
  - Python extension shim: [dequantize_from_double_block_extension.cpp](utils/quant/double/cuda/dequantize_from_double_block_extension.cpp)
  - Build script: [setup.py](utils/quant/double/cuda/setup.py)
- Build
  - Open a shell with your venv active and CUDA/MSVC available
  - Build the prebuilt CUDA Python module in place:
```bash
cd utils/quant/double/cuda
python setup.py build_ext --inplace
```
  - This build path uses `BuildExtension.with_options(use_ninja=False)`, so it does not require `ninja`
- Python usage after build
```python
import torch
import dequantize_from_double_block_cuda

packed = torch.empty((1024,), device="cuda", dtype=torch.int8)
sub_scales = torch.empty((16, 8), device="cuda", dtype=torch.int8)
super_scales = torch.empty((16,), device="cuda", dtype=torch.float16)
out = torch.empty((2048,), device="cuda", dtype=torch.float16)

dequantize_from_double_block_cuda.dequantize_from_double_block_fp16(
    packed,
    sub_scales,
    super_scales,
    out,
    out.numel(),
)
```
- Public API
  - `iml::cuda::dequantize_from_double_block::init_byte_to_half2_lut(cudaStream_t stream = nullptr)`
    - Initializes the constant-memory LUT once before dequant launches
  - `iml::cuda::dequantize_from_double_block::launch_dequantize_from_double_block(const int8_t* packed, const int8_t* sub_scales, const __half* super_scales, __half* out, int64_t original_numel, cudaStream_t stream = nullptr)`
    - Launches the full double-block dequant kernel
- Input / output
  - `packed` is a contiguous CUDA `int8` tensor containing two signed int4 values per byte using the nibble packing from `utils.quant.double.to`
  - `sub_scales` is a contiguous CUDA `int8` tensor with shape `(<num_super_blocks>, 8)` or an equivalent contiguous flat layout, with values in `0..127`
  - `super_scales` is a contiguous CUDA `fp16` tensor with one scale per super-block
  - `out` must point to a device buffer large enough for `original_numel` `fp16` values
  - `original_numel` is the number of decoded scalar outputs, not the number of packed bytes
- Notes
  - This module performs full dequant for the current double-block `fp16 super_scale + int8 local-scale code + packed int4 payload` format only.
  - The LUT must be initialized before the first dequant launch in the current CUDA context.
  - The kernel assumes CUDA device pointers for `packed`, `sub_scales`, `super_scales`, and `out`.
