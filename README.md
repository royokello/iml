# IML

Author: Roy Okello, Stelar Labs

IML is a toolkit for image, video, and diffusion model workflows. It includes:
- Image preprocessing (group, shape), dataset curation (cull), and YOLO-based cropping (crop)
- Video frame extraction, grouping, loop detection, and quality analysis
- Flux 2

## Installation

1. Install Python 3.10+.
   - Download: [Python releases](https://www.python.org/downloads/)

2. Install CUDA 12.3 for GPU-accelerated tools.
   - Download: [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)
   - Documentation: [CUDA Toolkit 12.3](https://docs.nvidia.com/cuda/archive/12.3.0/)

3. Clone the repo.
   - `git clone <repo-url>`
   - `cd iml`

4. Create a virtual environment.
   - Windows: `python -m venv venv`
   - macOS/Linux: `python3 -m venv .venv`

5. Activate the virtual environment.
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source .venv/bin/activate`

6. Install Python dependencies.
   - `pip install -r requirements.txt`

7. Run the root directory creator.
   - This script is not available yet. Until it exists, create the model/tool roots manually, for example:
     - `/models/flux`
     - `/models/gemma4`
     - `/tools/ffmpeg`
     - `/tools/llama.cpp`

8. Install FFmpeg.
   - Download: [FFmpeg release builds](https://github.com/BtbN/FFmpeg-Builds/releases)
   - Make sure `ffmpeg` and `ffprobe` are on PATH, or place them under the configured tools root.

9. Install llama.cpp.
   - Download: [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases)
   - Place the binaries under the configured tools root, or add them to PATH.

10. Download FLUX.2 weights.
    - Distilled 4B full weights: [black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
    - 4B base full weights: [black-forest-labs/FLUX.2-klein-base-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B)
    - Distilled 9B full weights: [black-forest-labs/FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B)
    - 9B base full weights: [black-forest-labs/FLUX.2-klein-base-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B)

11. Download Gemma 4 weights.
    - Full E2B instruction-tuned weights: [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-e2b-it)
    - Full E4B instruction-tuned weights: [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it)
    - E2B GGUF Q8_0 and `mmproj.bf16` from ggml-org: [ggml-org/gemma-4-E2B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF)
    - E4B GGUF Q4_K_M and `mmproj.fp16` from Unsloth: [unsloth/gemma-4-E4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF)

Notes:
- Some modules expect CUDA, including `video.loop` and YOLO inference/training.
- FFmpeg/FFprobe binaries are required by `video.group`, `video.loop`, and `video.quality`.
- If you run into PyTorch GPU memory-caching issues on Windows, run `set PYTORCH_NO_CUDA_MEMORY_CACHING=1`.

## Usage

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
  - Moves matching `.txt` captions alongside images
- Example
  - `python -m image.group --input "images" --orientation shortest --sizes 255 383 511 767 --dry-run`
  - `python -m image.group --input "images" --orientation shortest --orientation-split --sizes 256 384 512 768 --dry-run`

#### `image.collect`
- Features
  - Collects images from a directory tree into a single output folder
  - Converts to PNG, optional resize by width/height, optional square padding
  - Copies matching `.txt` captions alongside images
- Example
  - `python -m image.collect --input_dir "images" --output_dir "images_flat" --mode file`

#### `image.dataset`
- Features
  - Appends input images to an output dataset using the next numeric filename found in the output folder
  - Randomly selects reference image + `.txt` caption pairs without repeating within a run
  - Copies the input image to `<output>/<number>.<ext>` and the selected reference caption to `<output>/<number>.txt`
  - Creates `<output>/<number>/` and copies the selected reference image into that subfolder
- Example
  - `python -m image.dataset --inputs "images" --refs "refs" --output "dataset"`
- Useful args
  - `--filename-width` (default `6`), `--seed`, `--dry-run`

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
  - `python -m image.shape --inplace --mode ratio --ratios 1x1 2x3 1x2 2x1 3x2 --length-multiple 64 --min 512 --mid 512 --input "images"`

#### `image.tag.main`
- Features
  - Recursively scans the input folder for common image formats
  - Downloads the WD14 tagger model (SmilingWolf) into a local cache (`wd14_model` by default)
  - Writes comma-separated tags to `.txt` files beside each image (same basename)
  - Example
    - `python -m image.tag.main --input "images" --thresh 0.35 --max-tags 50`
  - Useful args
    - `--thresh`, `--general-thresh`, `--character-thresh`, `--max-tags` (0 for none), `--model`, `--prefix`

#### `image.caption`
- Features
  - Recursively scans the input folder for common image formats
  - Uses a local OpenAI-compatible vision chat endpoint to write natural-language `.txt` captions beside each image
  - Scales images down before upload only when their longest side exceeds `--max-res` (`768` by default)
  - Skips images that already have a matching `.txt` caption
  - Example
    - `python -m image.caption --input "images" --trigger "trxxr" --class woman`
    - `python -m image.caption --input "images" --trigger "trxxr" --class man`
    - `python -m image.caption --input "images" --trigger "objxx" --class object`
    - `python -m image.caption --input "images" --trigger "mystyle" --class style`
  - Useful args
    - `--trigger`, `--class`, `--max-res`, `--max-tokens`, `--server-url`, `--model`, `--print-response`

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

#### `image.tag.replace`
- Features
  - Replaces matching text in every `.txt` caption under a dataset directory
  - Matches whole comma-separated tags, ignoring spaces around commas
- Example
  - `python -m image.tag.replace --dataset "images" --old "old tag" --new "new tag"`

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

#### `image.similar` — Web GUI

A Flask-based web interface for finding and exporting near-duplicate images using dHash and Union-Find clustering.

- **Features**
  - **Scan**: Point the GUI at a directory; images are hashed and quality-scored in the background with a live progress bar
  - **Filter**: Adjust a similarity threshold slider (0–64, Hamming distance). Groups update in real time via debounced slider input
  - **Visualize**: Group cards show thumbnails with lazy loading. The best-quality image per group is highlighted with a green border and star badge. Hover tooltips show filename, dimensions, and composite quality score
  - **Save**: Each group has a "Save Group" button that copies the images (and companion `.txt` captions) to a user-specified output folder
  - **Advanced Quality**: Collapsible section with sharpness, exposure, noise, and artifact weight sliders that influence best-image selection
  - **Thumbnail caching**: 256px WebP thumbnails are generated on demand and cached to `{root}/image/similar/{timestamp}/thumbs/`
- **Workflow:** Enter directory path → Scan → Adjust threshold → Find groups → Save groups to output folders
- **Access:** `python -m image.similar.main --root /path/to/data --port 5051`
- **No file deletion.** The GUI only copies images; it never modifies or deletes input files.

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
      - `python -m image.crop.prepare.main --model yolo --project "C:\\proj" --stage 1 --val_split 0.2`
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
  - Applies a hardcoded mixed quantization preset to the language component
  - Writes `<root>/gemma4/language_<method>_quant.safetensors`, creating the output directory if needed
  - Add `--media` to also extract media tensors:
    - Saves audio tensors as fp16 to `<root>/gemma4/audio.safetensors`
    - Saves vision tensors as fp16 to `<root>/gemma4/vision.safetensors`
  - Uses `sym-high` quantization for:
    - `model.language_model.embed_tokens_per_layer.weight`
  - Uses `sym-med` quantization for:
    - `model.language_model.embed_tokens.weight`
    - `self_attn.q_proj.weight`, `self_attn.k_proj.weight`, `self_attn.v_proj.weight`, `self_attn.o_proj.weight`
    - `mlp.gate_proj.weight`, `mlp.up_proj.weight`, `mlp.down_proj.weight`
  - The `aff-med-mini` method uses `aff-med` quantization for:
    - `model.language_model.embed_tokens_per_layer.weight`
    - `self_attn.o_proj.weight`, `self_attn.v_proj.weight`
    - `mlp.down_proj.weight`
  - The `aff-med-mini` method uses `sym-low` quantization for:
    - `model.language_model.embed_tokens.weight`
    - `self_attn.k_proj.weight`, `self_attn.q_proj.weight`
    - `mlp.gate_proj.weight`, `mlp.up_proj.weight`
  - The `aff-high-mini` method uses `aff-high` quantization for:
    - `model.language_model.embed_tokens_per_layer.weight`
    - `self_attn.o_proj.weight`, `self_attn.v_proj.weight`
    - `mlp.down_proj.weight`
  - The `aff-high-mini` method uses `aff-med` quantization for:
    - `model.language_model.embed_tokens.weight`
    - `self_attn.k_proj.weight`, `self_attn.q_proj.weight`
    - `mlp.gate_proj.weight`, `mlp.up_proj.weight`
  - Keeps non-target language tensors in the language output, converting `float32` and `bfloat16` tensors to `float16`
  - Drops audio and vision tensors before processing the shared model quantization result for the language output
- Examples
  - `python -m gemma4.quant --root "/models"`
  - `python -m gemma4.quant --root "/models" --method aff-med-mini`
  - `python -m gemma4.quant --root "/models" --method aff-high-mini`
  - `python -m gemma4.quant --root "/models" --input "/models/custom/gemma4/model.safetensors"`
  - `python -m gemma4.quant --root "/models" --media`
- Useful args
  - `--root`
  - `--input`
  - `--method`
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
  - Uses `--text-quant-method` and `--denoiser-quant-method` to choose `none`, `sym-high`, `sym-med`, `aff-high`, `aff-med`, or `aff-low`
  - Passing `--text-quant-method none` or `--denoiser-quant-method none` keeps the original fp16 checkpoint weights
  - Saved quantized checkpoints are loaded automatically when present:
    - text encoder: `<root>/<model>/model/text_encoder/symmetric_high_quant.safetensors`, `symmetric_med_quant.safetensors`, `affine_high_quant.safetensors`, `affine_med_quant.safetensors`, or `affine_low_quant.safetensors`
    - denoiser: `<root>/<model>/model/transformer/<variant>/symmetric_high_quant.safetensors`, `symmetric_med_quant.safetensors`, `affine_high_quant.safetensors`, `affine_med_quant.safetensors`, or `affine_low_quant.safetensors`
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
  - Uses `--project` as the dataset directory and training run directory
  - Writes LoRA checkpoints to `<project>/models/epoch_<n>.safetensors`
  - Writes training losses to `<project>/logs/steps.csv`
  - Uses text encoder quantization method from the CLI
  - Uses denoiser quantization method from the CLI
  - Prints encoded text memory size in MB
- Current CLI
  - `--root`
  - `--version`
  - `--project`
  - `--steps`
  - `--resume`
  - `--checkpoint`
  - `--text-quant-method` (default: `sym-high`)
  - `--denoiser-quant-method` (default: `sym-med`)
  - `--trigger`
- Example
  - `python -m flux2.train --root "/models/flux" --version 4b --project "/data/flux_dataset"`

#### `flux2.sample`
- Features
  - Renders sample images for LoRA checkpoints saved by `flux2.train`
  - Uses `--project` as the training project directory
  - Scans `<project>/models` for `epoch_<n>.safetensors` and writes images to `<project>/samples`
  - Uses `--start` to skip earlier epochs and only render checkpoints at or after the selected epoch number
  - Skips any sample image that already exists, so reruns only render missing epoch/sample pairs
  - Supports two modes:
    - dataset-backed sampling from `--project`
    - prompt-only sampling with `--prompt`
  - Dataset-backed mode loads prompts and reference images from the Flux 2 dataset loader
  - Captioned datasets encode each selected caption independently
  - Captionless datasets require `--trigger`, which is encoded once and reused for all samples
  - `--samples` spreads selections evenly across the dataset from image `1` to the last image
  - `--sample-indices` overrides `--samples` with explicit 1-based dataset indices such as `"1,25,100"`
  - `--force-size` is dataset-only and rescales each selected sample so its longest side matches the requested size while preserving aspect ratio
  - Prompt-only mode renders a single logical sample using `--width` and `--height`
  - Sample seeds are deterministic per dataset index via `--sample-seed + sample_index`
  - Uses the distilled denoiser defaults from `flux2.gen`
  - Uses `--text-quant-method` and `--denoiser-quant-method` to choose `none`, `sym-high`, `sym-med`, `aff-high`, `aff-med`, or `aff-low`
  - Requires local Flux 2 assets under the selected model's `model` directory:
    - `tokenizer`
    - `text_encoder`
    - `scheduler`
    - `vae`
    - `transformer/distill`
  - CUDA-only runtime
- Output naming
  - Files are written as `<project>/samples/000012_3.png`
  - Example: `000012_3.png` is sample `3` rendered from step checkpoint `models/12.safetensors`
- Examples
  - Render one evenly-selected dataset sample for every checkpoint from epoch 1 onward:
    - `python -m flux2.sample --root "/models/flux" --version 4b --project "/data/flux_dataset"`
  - Render four evenly-spaced dataset samples for checkpoints starting at epoch 10:
    - `python -m flux2.sample --root "/models/flux" --version 4b --project "/data/flux_dataset" --samples 4 --start 10`
  - Render explicit dataset indices instead of evenly-spaced sampling:
    - `python -m flux2.sample --root "/models/flux" --version 9b --project "/data/flux_dataset" --sample-indices "1,25,100"`
  - Render captionless dataset samples using a shared trigger and resize the longest side to 768 before latent encoding:
    - `python -m flux2.sample --root "/models/flux" --version 4b --project "/data/flux_dataset" --sample-indices "5,9" --trigger "TOK" --force-size 768`
  - Render prompt-only samples for each checkpoint without loading a dataset:
    - `python -m flux2.sample --root "/models/flux" --version 9b --project "/data/flux_dataset" --prompt "fashion portrait, soft key light, editorial pose" --width 768 --height 1024`
- Useful args
  - `--root`
  - `--version`
  - `--project`
  - `--prompt`
  - `--samples`
  - `--sample-indices`
  - `--start`
  - `--sample-seed`
  - `--trigger`
  - `--force-size`
  - `--width`, `--height`
  - `--text-quant-method`
  - `--denoiser-quant-method`

#### `flux2.text_encoder.quant`
- Features
  - Quantizes the Flux 2 text encoder for `4b` or `9b`
  - Uses `model-00001-of-00002.safetensors` for `4b`
  - Uses `model-00001-of-00004.safetensors` through `model-00004-of-00004.safetensors` for `9b`
  - `--input` must point directly at the source `text_encoder` model directory
  - Builds the target tensor list from the shared Qwen linear suffixes and the version layer count
  - Saves `<iml_root>/<model>/model/text_encoder/<method>_quant.safetensors`, creating that output directory if needed
- Examples
  - `python -m flux2.text_encoder.quant --root "/iml_root" --version 4b --input "/models/flux2_4b/text_encoder" --method sym-high-mini`
- Useful args
  - `--root`
  - `--input`
  - `--version`
  - `--method`

#### `flux2.denoiser.quant`
- Features
  - Quantizes the Flux 2 denoiser for `4b` or `9b`
  - Uses `diffusion_pytorch_model.safetensors` for `4b`
  - Uses `diffusion_pytorch_model-00001-of-00002.safetensors` and `diffusion_pytorch_model-00002-of-00002.safetensors` for `9b`
  - `--input` can point directly at a source transformer checkpoint directory and bypass the `<root>/<model>/model/transformer/<variant>` input default
  - Quantizes `5` double blocks and `20` single blocks for `4b`
  - Quantizes `8` double blocks and `24` single blocks for `9b`
  - Saves `<root>/<model>/model/transformer/<variant>/<method>_quant.safetensors`, creating that output directory if needed
- Examples
  - `python -m flux2.denoiser.quant --root "/iml_root" --version 4b --variant base --method sym-med-mini`
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
- Timestep settings
  - Default example: `timestep_type: "weighted"`
  - Graphic Impressions example: `timestep_type: "shift"`

#### Flux 2 Dataset
- Current expected format
  - `--project` is the training run folder
  - Training images must be under `<project>/dataset`
  - Supported image extensions: `.png`, `.jpg`, `.jpeg`, `.webp`
  - Captioned mode: each image has a `.txt` caption with the same basename
  - Captionless mode: no image has a `.txt` caption and `--trigger "<text>"` is required
- Example
  - Captioned project layout
  - `dataset/000001.png`
  - `dataset/000001.txt`
  - `dataset/000002.jpg`
  - `dataset/000002.txt`
  - Captionless project layout
  - `dataset/000001.png`
  - `dataset/000002.jpg`
  - `models/`
  - `logs/`
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
  - Quantizes the Wan 2.2 TI2V 5B T5 text encoder linear weights with one of the shared quantization methods: `sym-high`, `sym-med`, `aff-high`, `aff-med`, or `aff-low`
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
  - `python -m wan22.quant.text_encoder --root "/models/wan" --method aff-med`
  - `python -m wan22.quant.text_encoder --root "/models/wan" --input "/models/custom/text_encoder" --method aff-high`
- Useful args
  - `--root`
  - `--input`
  - `--method`

#### `wan22.quant.denoiser`
- Features
  - Quantizes the Wan 2.2 TI2V 5B denoiser linear weights with one of the shared quantization methods: `sym-high`, `sym-med`, `aff-high`, `aff-med`, or `aff-low`
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
  - `python -m wan22.quant.denoiser --root "/models/wan" --method aff-med`
  - `python -m wan22.quant.denoiser --root "/models/wan" --input "/models/custom/denoiser" --method sym-med`
- Useful args
  - `--root`
  - `--input`
  - `--method`


### Utils

#### `utils.quant.estimator.main`
- Features
  - Estimates the final quantized `.safetensors` output size without running CUDA quantization
  - Reads safetensors shape/dtype metadata without materializing tensor data
  - Supports `.pth` source checkpoints for PyTorch-backed model components
  - Combines sharded source checkpoints into one estimated output size
  - Uses the same target tensor lists and prefix filters as the quantization scripts
  - Supports model presets:
    - `flux2_4b_text_encoder`
    - `flux2_9b_text_encoder`
    - `flux2_4b_denoiser`, `flux2_9b_denoiser`
    - `gemma4_2b`
    - `wan22_5b_text_encoder`, `wan22_5b_denoiser`
  - Prints estimates for every supported quantization method, ordered by the main quantization method bit width
- Examples
  - `python -m utils.quant.estimator.main --model flux2_4b_text_encoder --source "/models/flux/flux_2_klein_4b/model/text_encoder"`
  - `python -m utils.quant.estimator.main --model flux2_9b_denoiser --source "/models/flux/flux2_9b/model/transformer/base"`
  - `python -m utils.quant.estimator.main --model gemma4_2b --source "/models/gemma4"`
  - `python -m utils.quant.estimator.main --model wan22_5b_text_encoder --source "/models/wan/wan22/model/text_encoder"`
- Useful args
  - `--model`
  - `--source`

#### `utils.quant.eval`
- Features
  - Benchmarks dequant-to-`fp16` only for the Flux 2 Klein 4B denoiser linear tensor shapes
  - Compares the current `utils.quant` dequantizers:
    - `symmetric_high_dequant`
    - `symmetric_med_dequant`
    - `affine_high_dequant`
    - `affine_med_dequant`
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


### References

- [Black Forest Labs](https://blackforestlabs.ai/) for releasing the FLUX.2 [klein] model family and open-weight tooling.
  - [FLUX.2 documentation](https://docs.bfl.ai/flux_2)
  - [FLUX.2 GitHub repository](https://github.com/black-forest-labs/flux2)
- [Google DeepMind](https://deepmind.google/) and the Gemma 4 team for releasing the Gemma 4 open models and tooling.
  - [Gemma documentation](https://ai.google.dev/gemma)
  - [Gemma GitHub repository](https://github.com/google-gemma)
- [ggml](https://github.com/ggml-org) for the quantization formats and low-level tensor/runtime work that informed the quantization utilities.
- [Unsloth](https://huggingface.co/unsloth) for efficient quantization recipes, GGUF model releases, and practical low-memory model workflows.
