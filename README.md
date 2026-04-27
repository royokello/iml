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


### Gemma 4

#### `gemma4.quant`
- Features
  - Quantizes the Gemma 4 language component by default
  - Loads `<root>/gemma4/model.safetensors` by default
  - `--input` can point directly at a source `model.safetensors` file and bypass the default input path
  - Applies hardcoded mixed quantization to the language component
  - Writes `<root>/gemma4/language_mixed_quant.safetensors`, creating the output directory if needed
  - Add `--media` to also extract media tensors:
    - Saves audio tensors as fp16 to `<root>/gemma4/audio.safetensors`
    - Saves vision tensors as fp16 to `<root>/gemma4/vision.safetensors`
  - Uses `sym-high` quantization for:
    - `model.language_model.embed_tokens_per_layer.weight`
  - Uses `sym-low` quantization for:
    - `model.language_model.embed_tokens.weight`
    - `self_attn.q_proj.weight`, `self_attn.k_proj.weight`, `self_attn.v_proj.weight`, `self_attn.o_proj.weight`
    - `mlp.gate_proj.weight`, `mlp.up_proj.weight`, `mlp.down_proj.weight`
  - Keeps non-target language tensors in the language output, converting `float32` and `bfloat16` tensors to `float16`
  - Drops audio and vision tensors before processing the shared model quantization result for the language output
- Examples
  - `python -m gemma4.quant --root "/models"`
  - `python -m gemma4.quant --root "/models" --input "/models/custom/gemma4/model.safetensors"`
  - `python -m gemma4.quant --root "/models" --media`
- Useful args
  - `--root`
  - `--input`
  - `--media`

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
  - Uses `--text-quant-method` and `--denoiser-quant-method` to choose `none`, `sym-high`, `sym-low`, `aff-high`, or `aff-low`
  - Passing `--text-quant-method none` or `--denoiser-quant-method none` keeps the original fp16 checkpoint weights
  - Saved quantized checkpoints are loaded automatically when present:
    - text encoder: `<root>/<model>/model/text_encoder/symmetric_high_quant.safetensors`, `symmetric_low_quant.safetensors`, `affine_high_quant.safetensors`, or `affine_low_quant.safetensors`
    - denoiser: `<root>/<model>/model/transformer/<variant>/symmetric_high_quant.safetensors`, `symmetric_low_quant.safetensors`, `affine_high_quant.safetensors`, or `affine_low_quant.safetensors`
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
  - Uses `--version` to select `<root>/flux_2_klein_4b/model` or `<root>/flux2_9b/model`
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
  - `--version`
  - `--dataset`
  - `--output`
  - `--steps`
  - `--resume`
  - `--text-quant-method` (default: `sym-high`)
  - `--denoiser-quant-method` (default: `aff-low`)
  - `--trigger`
- Example
  - `python -m flux2.train --root "/models/flux" --version 4b --dataset "/data/flux_dataset" --output "/runs/flux_style_a"`

#### `flux2.quant.text_encoder`
- Features
  - Quantizes the Flux 2 text encoder for `4b` or `9b`
  - Uses `model-00001-of-00002.safetensors` for `4b`
  - Uses `model-00001-of-00004.safetensors` through `model-00004-of-00004.safetensors` for `9b`
  - `--input` can point directly at a source `text_encoder` model directory and bypass the `<root>/<model>/model/text_encoder` input default
  - Builds the target tensor list from the shared Qwen linear suffixes and the version layer count
  - Saves `<root>/<model>/model/text_encoder/<method>_quant.safetensors`, creating that output directory if needed
- Examples
  - `python -m flux2.quant.text_encoder --root "/models/flux" --version 4b --method sym-high`
  - `python -m flux2.quant.text_encoder --root "/models/flux" --version 9b --method aff-low`
  - `python -m flux2.quant.text_encoder --root "/models/flux" --version 9b --input "/models/custom/text_encoder" --method aff-high`
- Useful args
  - `--root`
  - `--input`
  - `--version`
  - `--method`

#### `flux2.quant.denoiser`
- Features
  - Quantizes the Flux 2 denoiser for `4b` or `9b`
  - Uses `diffusion_pytorch_model.safetensors` for `4b`
  - Uses `diffusion_pytorch_model-00001-of-00002.safetensors` and `diffusion_pytorch_model-00002-of-00002.safetensors` for `9b`
  - `--input` can point directly at a source transformer checkpoint directory and bypass the `<root>/<model>/model/transformer/<variant>` input default
  - Quantizes `5` double blocks and `20` single blocks for `4b`
  - Quantizes `8` double blocks and `24` single blocks for `9b`
  - Saves `<root>/<model>/model/transformer/<variant>/<method>_quant.safetensors`, creating that output directory if needed
- Examples
  - `python -m flux2.quant.denoiser --root "/models/flux" --version 4b --variant distill --method sym-high`
  - `python -m flux2.quant.denoiser --root "/models/flux" --version 9b --variant base --method aff-low`
  - `python -m flux2.quant.denoiser --root "/models/flux" --version 9b --variant base --input "/models/custom/transformer/base" --method aff-high`
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
  - Quantizes the Wan 2.2 TI2V 5B T5 text encoder linear weights with one of the shared quantization methods: `sym-high`, `sym-low`, `aff-high`, or `aff-low`
  - Loads the base checkpoint from `<root>/wan22/model/text_encoder/t5_umt5-xxl-enc-bf16.pth` by default
  - `--input` can point directly at a source `text_encoder` model directory and bypass the `<root>/wan22/model/text_encoder` input default
  - Targets all `24` T5 blocks for attention and feed-forward linear weights:
    - `attn.q.weight`, `attn.k.weight`, `attn.v.weight`, `attn.o.weight`
    - `ffn.gate.0.weight`, `ffn.fc1.weight`, `ffn.fc2.weight`
  - Writes `<root>/wan22/model/text_encoder/<method>_quant.safetensors`, creating that output directory if needed
  - Stores safetensors metadata with `component=text_encoder` and `method=<method>`
  - Keeps non-target tensors in the output checkpoint, converting `float32` and `bfloat16` tensors to `float16`
- Examples
  - `python -m wan22.quant.text_encoder --root "/models/wan" --method sym-high`
  - `python -m wan22.quant.text_encoder --root "/models/wan" --method aff-low`
  - `python -m wan22.quant.text_encoder --root "/models/wan" --input "/models/custom/text_encoder" --method aff-high`
- Useful args
  - `--root`
  - `--input`
  - `--method`

#### `wan22.quant.denoiser`
- Features
  - Quantizes the Wan 2.2 TI2V 5B denoiser linear weights with one of the shared quantization methods: `sym-high`, `sym-low`, `aff-high`, or `aff-low`
  - Loads the base sharded denoiser checkpoint from `<root>/wan22/model/denoiser` by default:
    - `diffusion_pytorch_model-00001-of-00003.safetensors`
    - `diffusion_pytorch_model-00002-of-00003.safetensors`
    - `diffusion_pytorch_model-00003-of-00003.safetensors`
  - `--input` can point directly at a source `denoiser` model directory and bypass the `<root>/wan22/model/denoiser` input default
  - Targets all `30` denoiser blocks for self-attention, cross-attention, and feed-forward linear weights:
    - `self_attn.q.weight`, `self_attn.k.weight`, `self_attn.v.weight`, `self_attn.o.weight`
    - `cross_attn.q.weight`, `cross_attn.k.weight`, `cross_attn.v.weight`, `cross_attn.o.weight`
    - `ffn.0.weight`, `ffn.2.weight`
  - Writes `<root>/wan22/model/denoiser/<method>_quant.safetensors`, creating that output directory if needed
  - Stores safetensors metadata with `component=denoiser` and `method=<method>`
  - Keeps non-target tensors in the output checkpoint, converting `float32` and `bfloat16` tensors to `float16`
- Examples
  - `python -m wan22.quant.denoiser --root "/models/wan" --method sym-high`
  - `python -m wan22.quant.denoiser --root "/models/wan" --method aff-low`
  - `python -m wan22.quant.denoiser --root "/models/wan" --input "/models/custom/denoiser" --method aff-high`
- Useful args
  - `--root`
  - `--input`
  - `--method`


### Utils

#### `utils.quant.eval`
- Features
  - Benchmarks dequant-to-`fp16` only for the Flux 2 Klein 4B denoiser linear tensor shapes
  - Compares the current `utils.quant` dequantizers:
    - `symmetric_high_dequant`
    - `symmetric_low_dequant`
    - `affine_high_dequant`
    - `affine_low_dequant`
  - Prepares benchmark inputs with:
    - `quantize_to_symmetric`
    - `quantize_to_affine`
  - Measures reconstruction error against the original `fp16` weight tensor for each case
  - Reports:
    - `max_abs_diff`
    - `mean_abs_diff`
  - Prints per-case timing while running, then prints a full result table, best method per shape, and per-method averages
  - Reports input and output byte counts for each benchmark case
  - Computes accuracy metrics outside the timed loop, so they do not affect the speed measurements
  - Uses CUDA when benchmarking the CUDA dequantizer variants in `utils/quant/cuda`
  - Uses fixed constants in the file; there are no CLI args
- Example
  - `python -m utils.quant.eval`
