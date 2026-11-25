# IML

Author: Roy Okello, Stelar Labs

IML is a toolkit for image, video, and diffusion model workflows. It includes:
- Image preprocessing (group, shape), dataset curation (cull), and YOLO-based cropping (crop)
- Video frame extraction, grouping, loop detection, and quality analysis
- Flux 2 and Ideogram 4

## Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Image](#image)
    - [`image.group`](#imagegroup)
    - [`image.collect`](#imagecollect)
    - [`image.dataset`](#imagedataset)
    - [`image.shape`](#imageshape)
    - [`image.tag.main`](#imagetagmain)
    - [`image.caption`](#imagecaption)
    - [`image.tag.extract`](#imagetagextract)
    - [`image.tag.add`](#imagetagadd)
    - [`image.tag.replace`](#imagetagreplace)
    - [`image.tag.group`](#imagetaggroup)
    - [`image.dedup`](#imagededup)
    - [`image.generate` — Web GUI](#imagegenerate--web-gui)
    - [`image.similar` — Web GUI](#imagesimilar--web-gui)
    - [`image.cull`](#imagecull)
    - [`image.rank`](#imagerank)
    - [`image.crop`](#imagecrop)
  - [Video](#video)
    - [`video.extract`](#videoextract)
    - [`video.group`](#videogroup)
    - [`video.loop`](#videoloop)
    - [`video.grid.generate`](#videogridgenerate)
    - [`video.grid.animate`](#videogridanimate)
  - [Flux 2](#flux-2)
    - [`flux2.gen`](#flux2gen)
    - [`flux2.train.main`](#flux2trainmain)
    - [`flux2.sample`](#flux2sample)
    - [`flux2.text_encoder.quant`](#flux2text_encoderquant)
    - [`flux2.denoiser.quant`](#flux2denoiserquant)
    - [Flux 2 Fine-Tuning Guidance](#flux-2-fine-tuning-guidance)
    - [Flux 2 Dataset](#flux-2-dataset)
    - [Flux 2 Dataset Guidance](#flux-2-dataset-guidance)
  - [Flux 3 Klein](#flux-3-klein)
    - [`flux3.*.copy_files`](#flux3text_encodercopy_files-flux3denoisercopy_files-flux3vaecopy_files)
    - [`flux3.denoiser.quant`](#flux3denoiserquant)
    - [`flux3.text_encoder.quant`](#flux3text_encoderquant)
    - [`flux3.gen`](#flux3gen)
    - [`flux3.train.main`](#flux3trainmain)
    - [`flux3.sample`](#flux3sample)
  - [Krea 2](#krea-2)
    - [`krea2.*.copy_files`](#krea2text_encodercopy_files-krea2denoisercopy_files-krea2vaecopy_files)
    - [`krea2.denoiser.quant`](#krea2denoiserquant)
    - [`krea2.text_encoder.quant`](#krea2text_encoderquant)
    - [`krea2.gen`](#krea2gen)
  - [Qwen Image 2.1](#qwen-image-21)
    - [`qwen2.*.copy_files`](#qwen2text_encodercopy_files-qwen2denoisercopy_files-qwen2vaecopy_files)
    - [`qwen2.denoiser.quant`](#qwen2denoiserquant)
    - [`qwen2.text_encoder.quant`](#qwen2text_encoderquant)
    - [`qwen2.gen`](#qwen2gen)
  - [SDXL](#sdxl)
    - [`sdxl.denoiser.quant`](#sdxldenoiserquant)
    - [`sdxl.gen`](#sdxlgen)
    - [`sdxl.train.main`](#sdxltrainmain)
    - [`sdxl.sample`](#sdxlsample)
  - [Ideogram 4](#ideogram-4)
    - [`ideogram.gen`](#ideogramgen)
    - [`ideogram.text_encoder.quant`](#ideogramtext_encoderquant)
    - [`ideogram.denoiser.quant`](#ideogramdenoiserquant)
  - [Utils](#utils)
    - [`utils.quant.estimator.main`](#utilsquantestimatormain)
    - [`utils.quant.eval`](#utilsquanteval)
- [References](#references)

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
   - `python -c "from utils.startup import prepare_directories; prepare_directories('/models', '/tools')"`
   - This creates the expected model/tool directory structure. Create the roots manually if you prefer:
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
- FFmpeg/FFprobe binaries are required by `video.group` and `video.loop`.
- If you run into PyTorch GPU memory-caching issues on Windows, run `set PYTORCH_NO_CUDA_MEMORY_CACHING=1`.

## Usage

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
    - `python -m image.caption --input "images" --trigger "trxxr" --prompt-file "prompt_woman.txt"`
    - `python -m image.caption --input "images" --trigger "trxxr" --prompt-file "prompt_man.txt"`
    - `python -m image.caption --input "images" --trigger "objxx" --prompt-file "prompt_object.txt"`
    - `python -m image.caption --input "images" --trigger "mystyle" --prompt-file "prompt_style.txt"`
  - Useful args
    - `--trigger`, `--prompt-file`, `--max-res`, `--max-tokens`, `--server-url`, `--model`, `--print-response`, `--reset`

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

#### `image.generate` — Web GUI

A Flask-based web interface for generating images, supporting Flux 2 (4b/9b, distilled/base), Flux 3 Klein (distill/base), Anima and Ideogram models.

- **Features**
  - Prompt-based generation with configurable width, height, steps, guidance scale, and seed
  - Image-conditioned generation via reference image upload
  - Flux 3 Klein reads LoRA checkpoints from `<root>/flux3/loras/<project>/models/*.safetensors`, and quantized weights from `<root>/flux3/model`. The base variant also shows a negative prompt
  - Quantization method selection for text encoder and denoiser
  - Async job queue with live status tracking
  - Job history panel showing past generations with embedded metadata in PNG outputs
  - Model config schemas loaded dynamically from `static/config/`
- **Access:** `python -m image.generate --root /path/to/data` then open `http://localhost:5053`

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
  - Best checkpoint is selected on `val_bal_acc` (balanced accuracy = mean per-class recall), tie-broken by `val_auroc` then `val_acc`
  - Train/val split is stratified per class so both labels are represented in validation
  - Inference supports `--mode copy|link|inplace` for kept-image output handling; logs and metrics saved
- Typical flow (stages live under project root)
  1) Label keep/cull (first step)
     - `python -m image.cull.label.main --project "/proj" --stage 1` then open `http://localhost:5050`
     - Writes `stage_<N>_cull_labels.csv` under the project root (resumes if present)
  2) Train
     - `python -m image.cull.train --project "/proj" --stage 1 --batch-size 32`
  3) Random cull preview sample (optional)
     - `python -m image.cull.sample --project "/proj" --stage 1 --count 1000` (runs the stage cull model on a random sample and hard-links predicted keeps to `stage_<N>_cull_samples`)
  4) Predict
     - `python -m image.cull.main --project "/proj" --stage 1`
     - Link instead of copy: `python -m image.cull.main --project "/proj" --stage 1 --mode link`
     - In-place cull (move keeps, delete culls): `python -m image.cull.main --project "/proj" --stage 1 --mode inplace`
     - Resume from numeric filename (inclusive): `python -m image.cull.main --project "/proj" --resume 14487`
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
  - `python -m image.rank.label.main --project "/proj" --stage 1` then open `http://localhost:5000`

#### `image.crop`
- Features
  - Labels carry a class and an independent ratio; class is used for balancing/statistics, ratio drives the crop shape
  - Ratios: `1x1`, `1x2`, `2x3`, `2x1`, `3x2`
  - Each distinct `class_ratio` pair (e.g. `face_1x1`, `half_body_2x3`) becomes a YOLO class
  - Train a YOLO detector from CSV labels, then crop one best box per class/ratio combination
  - Classes: `0 face, 1 head_shot, 2 half_body, 3 cowboy_shot, 4 three_quarter, 5 full_body`
  - Crops are expanded to the labeled ratio and resized by long side
  - The cropper reads class names from the trained weights, so training on a subset stays consistent
  - Output crops are zero-padded (default width 6, starting at `000001.png` via `--filename-width`)
- Typical flow
  1) Label boxes (first step)
     - `python -m image.crop.label.main --project "/proj" --stage 1` then open `http://localhost:5051`
     - Custom port: `python -m image.crop.label.main --project "/proj" --stage 1 --port 5001`
     - Writes `stage_<N>_crop_labels.csv` (`img,class,ratio,x1,y1,x,y`) under the project root (resumes if present)
  2) Prepare YOLO dataset from labels CSV
      - `python -m image.crop.prepare.main --model yolo --project "/proj" --stage 1 --val_split 0.2`
      - Add `--balance` only if you explicitly want to downsample every class/ratio combination to the smallest count
      - `--min-samples N` drops combinations with fewer than N labels; `--combos head_shot_1x1 half_body_2x3` keeps only the listed combinations
      - Recommended when data is thin: train only well-populated combinations, e.g. `--min-samples 20`
  3) Train a detector (Ultralytics)
     - `python -m image.crop.train --project "/proj" --stage 1 --variant yolo11n --epochs 100 --batch 16`
  4) Run cropper using the trained weights
     - Native crop size (no resize): `python -m image.crop.main --project "/proj" --stage 1 --classes 0 1 2 3 4 5`
     - Resize long side: `python -m image.crop.main --project "/proj" --stage 1 --resolution 768 --classes 0 1 2 3 4 5`
     - `--classes` filters by base class id (all of its ratios)

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
  - `python -m video.group --root "videos" --ffprobe "/tools/ffprobe" --sizes 256 384 512 768 1024 --dry-run`

#### `video.loop`
- Features
  - GPU-decoded thumbnails via FFmpeg (`scale_cuda`), SSIM-based loop detection, batched exports
  - Exports lossless H.264 `.mkv` loops at target `--fps` and `--out-res`
- Example
  - `python -m video.loop --input "videos" --output "loops" --lengths 33 49 --fps 30 --in-res 64 --out-res 768 --ffmpeg "/tools/ffmpeg"`

#### `video.grid.generate`
- Features
  - CLI tool to build image grids from a source video
  - `interval` mode samples frames at a fixed seconds interval and emits as many full grids as possible
  - `segment_midpoints` mode divides the full video into exactly `rows * cols` segments and picks the middle frame from each segment to build one grid
  - Center-crops each frame to the requested cell ratio before resizing into the grid
  - Output: `<dir>/000001.png ...` + `config.json`
- Example
  - `python -m video.grid.generate -i "video.mp4" -o "grids" --ffmpeg "/tools/ffmpeg" --rows 3 --cols 3 --selection-mode interval --frame-interval 30`

#### `video.grid.animate`
- Features
  - Splits a grid image into cells and creates a cell-by-cell animation
  - Supports GIF and MP4 output
- Example
  - `python -m video.grid.animate -i "grids/000001.png" -o "animation.gif" --ffmpeg "/tools/ffmpeg" --length 5 --fps 12 --format gif`


### Flux 2

The unified `flux2` package provides the version-aware entrypoints for generation and quantization.

#### `flux2.gen`
- Features
  - Unified Flux 2 generation entrypoint for `4b` and `9b`
  - Uses `--version` to select `<root>/flux2/4b` or `<root>/flux2/9b`
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
  - Uses `--text-quant-method` and `--denoiser-quant-method` to choose `fp16`, `sym-high`, `sym-med`, `aff-high`, `aff-med`, or `aff-low`
  - Passing `--text-quant-method fp16` or `--denoiser-quant-method fp16` keeps the original fp16 checkpoint weights
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
    - `python -m flux2.gen --root "/models/flux" --version 4b --prompt "product render on white seamless" --denoiser-quant-method fp16`
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
  - `--stream-denoiser`

#### `flux2.train.main`
- Current scope
  - Unified Flux 2 training entrypoint
  - Uses `--version` to select `<root>/flux2_4b/model` or `<root>/flux2_9b/model`
  - Loads the tokenizer and text encoder from the selected model directory
  - Encodes captions from a local image-caption dataset into `prompt_embeds` and `text_ids`
  - Keeps text encodings on CPU while the text encoder is active, then frees the text encoder and moves the encodings to VRAM
  - Loads the base denoiser checkpoint from the selected model's `model/transformer/base` directory
  - Uses `--project` as the dataset directory and training run directory
  - Writes LoRA checkpoints to `<project>/models/<step>.safetensors`
  - Writes training losses to `<project>/models/steps.csv`
  - Uses text encoder quantization method from the CLI
  - Uses denoiser quantization method from the CLI
  - `--stream-denoiser` keeps the quantized weights in pinned host memory and copies each layer onto the GPU for its own multiply. It leaves gradient checkpointing on
  - Prints encoded text memory size in MB
- Current CLI
  - `--root`
  - `--version`
  - `--project`
  - `--steps` (default: `1024`)
  - `--resume`
  - `--model` (LoRA init checkpoint file, optional)
  - `--text-quant` (default: `aff-high-mini`)
  - `--denoiser-quant` (default: `aff-high-mini`)
  - `--stream-denoiser`
  - `--trigger`
  - `--cache-text` (pre-encode captions to CPU)
  - `--cache-images` (pre-encode image latents to disk)
  - `--rank` (default: `32`)
  - `--alpha` (default: `32`)
  - `--target_res` (default: `384`)
  - `--ref_res` (default: `384`)
  - `--lr` (default: `1e-4`)
- Example
  - `python -m flux2.train.main --root "/models/flux" --version 4b --project "/data/flux_dataset" --trigger "TOK"`

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
  - Uses `--text-quant-method` and `--denoiser-quant-method` to choose `fp16`, `sym-high`, `sym-med`, `aff-high`, `aff-med`, or `aff-low`
  - Requires local Flux 2 assets under the selected model's `model` directory:
    - `tokenizer`
    - `text_encoder`
    - `scheduler`
    - `vae`
    - `transformer/distill`
  - CUDA-only runtime
- Output naming
  - Files are written as `<project>/samples/<step>_<width>x<height>.png` for single samples, or `<step>_<index>_<width>x<height>.png` for multi-samples
  - Example: `000012_768x768.png` is rendered from step checkpoint `models/12.safetensors` at 768x768 resolution
  - Example: `000012_3_768x768.png` is sample `3` from the same checkpoint
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
  - `--target-res`, `--ref-res`
  - `--steps`, `--distill`
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
  - `--input` is required; point directly at a source transformer checkpoint directory
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
  - Training images must be under `<project>/images`
  - Supported image extensions: `.png`, `.jpg`, `.jpeg`, `.webp`
  - Captioned mode: each image has a `.txt` caption with the same basename
  - Captionless mode: no image has a `.txt` caption and `--trigger "<text>"` is required
- Example
  - Captioned project layout
  - `images/000001.png`
  - `images/000001.txt`
  - `images/000002.jpg`
  - `images/000002.txt`
  - Captionless project layout
  - `images/000001.png`
  - `images/000002.jpg`
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

### Flux 3 Klein

The `flux3` package runs the Flux 3 Klein base and distill models. The model folder is `<root>/flux3/model`:

```
<root>/flux3/model/
  text_encoder/          config, tokenizer files, model-*.safetensors
  transformer/
    config.json
    base/                diffusion_pytorch_model-*.safetensors
    distill/             diffusion_pytorch_model-*.safetensors
  vae/
```

The quant CLIs write only the quant file. They create the missing folders in `<root>/flux3/model`. The copy modules below put the config, tokenizer, and VAE files there. A loader never quantizes on the fly. It raises `FileNotFoundError` when the `<method>_quant.safetensors` file is missing.

#### `flux3.text_encoder.copy_files`, `flux3.denoiser.copy_files`, `flux3.vae.copy_files`
- Features
  - Copy the small files that `flux3.gen` reads, from the source model folders into `<root>/flux3/model`
  - Text encoder: `config.json`, `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`, `chat_template.json`
  - Denoiser: `transformer/config.json`. `--input` is the `transformer` folder, the parent of `base` and `distill`
  - VAE: `config.json` and `diffusion_pytorch_model.safetensors` (the VAE is not quantized)
  - Never copy the text encoder or denoiser weight shards
  - Skip a file that is already in the target with the same size. Raise `FileNotFoundError` if a listed file is missing
- Examples
  - `python -m flux3.text_encoder.copy_files --root "/iml_root" --input "/models/Flux 3 Image Klein/text_encoder"`
  - `python -m flux3.denoiser.copy_files --root "/iml_root" --input "/models/Flux 3 Image Klein/transformer"`
  - `python -m flux3.vae.copy_files --root "/iml_root" --input "/models/Flux 3 Image Klein/vae"`
- Useful args
  - `--root`
  - `--input`

#### `flux3.denoiser.quant`
- Features
  - Quantizes the Flux 3 denoiser for the `base` or `distill` variant
  - `--input` is optional; it defaults to `<root>/flux3/model/transformer/<variant>`
  - Saves `<root>/flux3/model/transformer/<variant>/<method>_quant.safetensors`
- Examples
  - `python -m flux3.denoiser.quant --root "/iml_root" --input "/models/Flux 3 Image Klein/transformer/distill" --variant distill --method sym-med-nano`
- Useful args
  - `--root`
  - `--input`
  - `--variant`
  - `--method`

#### `flux3.text_encoder.quant`
- Features
  - Quantizes the Flux 3 text encoder (Qwen3-VL-4B language model, layers 0 to 32)
  - Skips the visual tower, layers 33 to 35, and the final norm
  - `--input` is optional; it defaults to `<root>/flux3/model/text_encoder`
  - Saves `<root>/flux3/model/text_encoder/<method>_quant.safetensors`
- Examples
  - `python -m flux3.text_encoder.quant --root "/iml_root" --input "/models/Flux 3 Image Klein/text_encoder" --method sym-high-mini`
- Useful args
  - `--root`
  - `--input`
  - `--method`

#### `flux3.gen`
- Features
  - Generates one image from a prompt and saves it to `<root>/flux3/outputs`
  - CUDA-only runtime; exits if CUDA is unavailable
  - Distill is the default: `16` steps, no guidance
  - `--base` uses the base model: `50` steps, CFG `3.5`, and a negative prompt
  - `--images "a.png, b.png"` adds reference images. Their short side is resized to `--ref-size`
  - `--text-quant-method` and `--denoiser-quant-method` pick a saved quant file. `fp16` loads the fp16 weights
  - `--stream-denoiser` streams the quantized weights from pinned host memory, one layer at a time
  - `--loras "path1:1.0,path2:0.7"` applies LoRA checkpoints
- Examples
  - `python -m flux3.gen --root "/iml_root" --prompt "a cat holding a sign that says hello world"`
  - `python -m flux3.gen --root "/iml_root" --base --steps 30 --negative-prompt "blurry"`
  - `python -m flux3.gen --root "/iml_root" --images "/data/ref.png" --prompt "same character, new pose"`
- Useful args
  - `--root`
  - `--prompt`
  - `--images`
  - `--width`, `--height`
  - `--steps`
  - `--seed`
  - `--base`
  - `--guidance-scale`
  - `--negative-prompt`
  - `--ref-size`
  - `--text-quant-method`
  - `--denoiser-quant-method`
  - `--loras`
  - `--max-length`
  - `--stream-denoiser`

#### `flux3.train.main`
- Features
  - Trains a LoRA on the Flux 3 Klein base denoiser. CUDA-only
  - Uses `--project` as the dataset and run folder. The layout is the same as `flux2.train.main`
    - Targets: `<project>/images/<name>.<ext>`. Captions: `<name>.txt`
    - References: `<project>/images/<name>/<ref>.<ext>`
  - Samples are sorted by name. A target or reference below its resolution is skipped, unless `--target-upscale` or `--ref-upscale` is set
  - A sample without a caption uses the `--trigger` text
  - Samples sigma with the Flux 3 shift (6.93). Stops with an error when the loss or the gradient is inf or NaN
  - Writes LoRA checkpoints to `<project>/models/<step>.safetensors`, the step log to `<project>/models/steps.csv`, and the loss plot to `<project>/models/loss.png`
  - Delete `<project>/models/stop.txt` to stop cleanly after the current step
  - `--cache-images` and `--cache-text` read and write caches next to the images. The cache names carry a `flux3` tag, so Flux 2 caches stay untouched
  - `--text-quant` and `--denoiser-quant` pick a saved quant file. `fp16` loads the fp16 weights
  - `--stream-denoiser` keeps the quantized weights in pinned host memory and copies each layer onto the GPU for its own multiply. It leaves gradient checkpointing on
  - The checkpoints load with `flux3.gen --loras`
- Examples
  - `python -m flux3.train.main --root "/iml_root" --project "/data/flux3_dataset" --trigger "TOK"`
  - `python -m flux3.train.main --root "/iml_root" --project "/data/flux3_dataset" --trigger "TOK" --resume --steps 2048`
- Useful args
  - `--root`
  - `--project`
  - `--trigger`
  - `--steps` (default: `1024`)
  - `--resume`
  - `--model` (LoRA start checkpoint)
  - `--text-quant` (default: `sym-high-mini`)
  - `--denoiser-quant` (default: `sym-med-nano`)
  - `--stream-denoiser`
  - `--cache-text`, `--cache-images`
  - `--rank`, `--alpha` (default: `32`, `32`)
  - `--target-res`, `--ref-res` (default: `512`, `512`)
  - `--target-upscale`, `--ref-upscale`
  - `--lr` (default: `1e-4`)
  - `--seed`
  - `--moving-averages` (default: the sample count and the checkpoint interval)
  - `--checkpoint-steps` (default: `256`)
  - `--max-length` (default: `512`)

#### `flux3.sample`
- Features
  - Renders sample images for the LoRA step checkpoints that `flux3.train.main` saves. CUDA-only
  - Scans `<project>/models` for numeric `<step>.safetensors` files and writes images to `<project>/samples`
  - `--start` skips earlier steps
  - Skips a sample whose output file already exists. Two samples with the same name in one run get the suffix `_2`, `_3`
  - Dataset-backed mode uses the captions in `<project>/images`. A sample without a caption needs `--trigger`
  - `--prompt` replaces the captions. Without `--sample-indices`, it renders `--width` x `--height`
  - `--sample-indices "1,25,100"` picks 1-based dataset positions. The output size comes from the aspect ratio of each target and `--target-res`, rounded down to a multiple of 16
  - The noise is deterministic per sample name: `--sample-seed` and the file stem
  - The base denoiser is the default and uses CFG with an empty negative prompt (`--guidance-scale`, default `3.5`). `1.0` turns the negative pass off
  - `--distill` loads the distill denoiser, with no guidance
  - Reads existing latent and text caches from the project. It does not write caches
  - `--stream-denoiser` streams the quantized weights from pinned host memory, one layer at a time
- Examples
  - `python -m flux3.sample --root "/iml_root" --project "/data/flux3_dataset" --denoiser-quant-method aff-med --sample-indices "1,2"`
  - `python -m flux3.sample --root "/iml_root" --project "/data/flux3_dataset" --prompt "TOK on a beach" --steps 30`
- Useful args
  - `--root`
  - `--project`
  - `--prompt`
  - `--width`, `--height`
  - `--target-res`, `--ref-res` (default: `512`, `512`)
  - `--target-upscale`, `--ref-upscale`
  - `--start`
  - `--steps` (default: `16`)
  - `--distill`
  - `--guidance-scale`
  - `--sample-seed`
  - `--sample-indices`
  - `--trigger`
  - `--text-quant-method` (default: `sym-high-mini`)
  - `--denoiser-quant-method` (default: `sym-med-nano`)
  - `--max-length`
  - `--stream-denoiser`

### Krea 2

The `krea_2` package quantizes and runs the Krea 2 text-to-image model. The model folder is `<root>/krea_2/model`:

```
<root>/krea_2/model/
  denoiser/
    config.json
    turbo/                 <method>_quant.safetensors
    raw/                   <method>_quant.safetensors
  text_encoder/
    config.json
    tokenizer.json
    tokenizer_config.json
    chat_template.jinja
    <method>_quant.safetensors
  vae/
    config.json
    diffusion_pytorch_model.safetensors
```

The quant CLIs write only the quant file. The copy modules below put the config, tokenizer, and VAE files there. A loader never quantizes on the fly. It raises `FileNotFoundError` when the quant file is missing.

Krea 2 runs the Qwen3-VL-4B language model as the text encoder and the Qwen-Image VAE. The vision tower is not quantized and not built. It is text-to-image only.

#### `krea2.text_encoder.copy_files`, `krea2.denoiser.copy_files`, `krea2.vae.copy_files`
- Features
  - Copy the small files that `krea_2.gen` reads, from the source model folders into `<root>/krea_2/model`
  - Text encoder: `config.json` from `--input`, and `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` from `--tokenizer`
  - Denoiser: `transformer/config.json`. `--input` is the `transformer` folder, the parent of `turbo` and `raw`
  - VAE: `diffusion_pytorch_model.safetensors`, and a `config.json` written from the known Qwen-Image VAE values
  - Never copy the text encoder or denoiser weight shards
  - Skip a file that is already in the target with the same size. Raise `FileNotFoundError` if a listed file is missing
- Examples
  - `python -m krea_2.text_encoder.copy_files --root "/iml_root" --input "/models/Krea 2/text_encoder" --tokenizer "/models/Krea 2/tokenizer"`
  - `python -m krea_2.denoiser.copy_files --root "/iml_root" --input "/models/Krea 2/transformer"`
  - `python -m krea_2.vae.copy_files --root "/iml_root" --input "/models/Krea 2/vae"`
- Useful args
  - `--root`
  - `--input`
  - `--tokenizer` (text encoder only)

#### `krea2.denoiser.quant`
- Features
  - Quantizes the Krea 2 denoiser for the `turbo` or `raw` variant
  - Saves `<root>/krea_2/model/denoiser/<variant>/<method>_quant.safetensors`
  - `--variant` is required, so a raw run cannot overwrite turbo
  - Default method: `sym-med-nano`
- Examples
  - `python -m krea_2.denoiser.quant --root "/iml_root" --input "/models/Krea 2/transformer/turbo" --variant turbo --method sym-med-nano`
- Useful args
  - `--root`
  - `--input`
  - `--variant`
  - `--method`

#### `krea2.text_encoder.quant`
- Features
  - Quantizes the Krea 2 text encoder (Qwen3-VL-4B language model, 36 layers)
  - Skips the visual tower
  - Quantizes the embedding table with `sym-high`, which the loader reads with a row gather
  - Saves `<root>/krea_2/model/text_encoder/<method>_quant.safetensors`
  - Default method: `sym-high-mini`
- Examples
  - `python -m krea_2.text_encoder.quant --root "/iml_root" --input "/models/Krea 2/text_encoder" --method sym-high-mini`
- Useful args
  - `--root`
  - `--input`
  - `--method`

#### `krea2.gen`
- Features
  - Generates an image from a prompt and saves it to `<root>/krea_2/outputs`
  - CUDA-only runtime; exits if CUDA is unavailable
  - Turbo is the default: `8` steps, no guidance, `mu=1.15`
  - `--raw` uses the raw model: `28` steps, CFG `4.5`, resolution-derived `mu`, and a negative prompt
  - Turbo and raw read `denoiser/turbo/` and `denoiser/raw/`
  - `--text-quant-method` and `--denoiser-quant-method` pick a saved quant file. `fp16` reads the fp16 quant file
  - `--text-positions {diffusers,absolute}` picks the text position ids. Default: `diffusers` (cumulative valid tokens)
  - `--stream-denoiser` keeps the quantized weights in pinned host memory and copies one layer at a time
  - `--vae-dtype {fp32,fp16}` sets the VAE compute dtype. Default: `fp32`
  - On a 6 GB GPU, use `--stream-denoiser` at 1024, or generate at 512
- Examples
  - `python -m krea_2.gen --root "/iml_root" --prompt "a red fox in the snow" --stream-denoiser`
  - `python -m krea_2.gen --root "/iml_root" --width 512 --height 512 --steps 8`
  - `python -m krea_2.gen --root "/iml_root" --raw --prompt "a red fox in the snow" --negative-prompt "blurry"`
- Useful args
  - `--root`
  - `--prompt`
  - `--width`, `--height`
  - `--steps`
  - `--seed`
  - `--num-images`
  - `--raw`
  - `--guidance-scale`
  - `--negative-prompt`
  - `--text-quant-method`
  - `--denoiser-quant-method`
  - `--max-length`
  - `--stream-denoiser`
  - `--vae-dtype`
  - `--text-positions`

### Qwen Image 2.1

The `qwen_2` package quantizes and runs the Qwen Image 2.1 model. The model folder is `<root>/qwen_2/model`:

```
<root>/qwen_2/model/
  denoiser/
    config.json
    diffusion_pytorch_model.safetensors.index.json
    turbo/                 <method>_quant.safetensors
    base/                  <method>_quant.safetensors
  text_encoder/
    config.json
    generation_config.json
    tokenizer.json
    tokenizer_config.json
    chat_template.jinja
    processor_config.json
    <method>_quant.safetensors
  vae/
    config.json
    diffusion_pytorch_model.safetensors
```

The quant CLIs write only the quant file. The copy modules below put the config, tokenizer, and VAE files there. A loader never quantizes on the fly. It raises `FileNotFoundError` when the quant file is missing.

Qwen Image 2.1 runs a Qwen3-VL-8B text encoder and a 7.1B parameter denoiser. The text encoder quant file holds the whole vision tower as 351 bf16 tensors, because the edit pipeline reads it. Text-to-image reads the language model only, so the tower costs disk space and no time. `lm_head` stays out of the quant file, because neither path reads logits.

Reference sizes: text encoder 6.85 GB at `sym-med-nano`, denoiser 4.43 GB at `aff-high-nano`.

#### `qwen2.text_encoder.copy_files`, `qwen2.denoiser.copy_files`, `qwen2.vae.copy_files`
- Features
  - Copy the small files that `qwen_2.gen` reads, from the source model folders into `<root>/qwen_2/model`
  - Text encoder: `config.json` and `generation_config.json` from `--input`, and `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `processor_config.json` from `--processor`
  - Denoiser: `config.json` and `diffusion_pytorch_model.safetensors.index.json`. `--input` is the `transformer` folder, the parent of `turbo` and `base`
  - VAE: `config.json` and `diffusion_pytorch_model.safetensors`
  - Never copy the text encoder or denoiser weight shards
  - Skip a file that is already in the target with the same size. Raise `FileNotFoundError` if a listed file is missing
- Examples
  - `python -m qwen_2.text_encoder.copy_files --root "/iml_root" --input "/models/Qwen Image 2.1/text_encoder" --processor "/models/Qwen Image 2.1/processor"`
  - `python -m qwen_2.denoiser.copy_files --root "/iml_root" --input "/models/Qwen Image 2.1/transformer"`
  - `python -m qwen_2.vae.copy_files --root "/iml_root" --input "/models/Qwen Image 2.1/vae"`
- Useful args
  - `--root`
  - `--input`
  - `--processor` (text encoder only)

#### `qwen2.denoiser.quant`
- Features
  - Quantizes the Qwen Image 2.1 denoiser for the `turbo` or `base` variant
  - Saves `<root>/qwen_2/model/denoiser/<variant>/<method>_quant.safetensors`
  - `--variant` is required, so a base run cannot overwrite turbo
  - Splits the block tensors into three ranks: `attn.to_q` and `attn.to_out.0` first, `attn.to_v` and the image MLP second, `attn.to_k` third
  - Keeps every global at bf16: `modulation.1`, `norm_out.linear`, `txt_in`, and the timestep embedders
  - Default method: `sym-med-nano`, 5.24 GB. `aff-high-nano` gives 4.43 GB and runs resident at 1024 on a 6 GB card
  - `--denoiser-quant-method fp16`, `bf16`, or `fp32` skips quantization and reads the original shards from the same folder, cast to that dtype
- Examples
  - `python -m qwen_2.denoiser.quant --root "/iml_root" --input "/models/Qwen Image 2.1/transformer/Turbo" --variant turbo --method aff-high-nano`
- Useful args
  - `--root`
  - `--input`
  - `--variant`
  - `--method`

#### `qwen2.text_encoder.quant`
- Features
  - Quantizes the Qwen3-VL-8B language model (36 layers) and copies the 351 vision tensors at bf16
  - Quantizes the embedding table with `sym-high`, which the loader reads with a row gather
  - Splits the language tensors into three ranks: `v_proj` and `down_proj` first, `q_proj`, `k_proj`, `gate_proj`, and `up_proj` second, `o_proj` third
  - Never writes `lm_head`
  - Saves `<root>/qwen_2/model/text_encoder/<method>_quant.safetensors`
  - Default method: `sym-high-mini`, 7.99 GB. `sym-med-nano` gives 6.85 GB and is what the 6 GB commands below use
  - `--text-quant-method fp16`, `bf16`, or `fp32` skips quantization and reads `model.safetensors` from the same folder, cast to that dtype
- Examples
  - `python -m qwen_2.text_encoder.quant --root "/iml_root" --input "/models/Qwen Image 2.1/text_encoder" --method sym-med-nano`
- Useful args
  - `--root`
  - `--input`
  - `--method`

#### `qwen2.gen`
- Features
  - Generates one image from a prompt and saves it to `<root>/qwen_2/outputs`
  - The file name is a timestamp. The prompt, seed, size, variant, and quant methods go into the PNG text chunk under the key `qwen_2`
  - CUDA-only runtime; exits if CUDA is unavailable
  - Turbo is the default: 8 steps, guidance 1.0, and the fixed 8-step sigma list
  - `--variant base` uses 40 steps, a dynamic shift, and CFG
  - Turbo and base read `denoiser/turbo/` and `denoiser/base/`
  - `--text-quant-method` and `--denoiser-quant-method` pick a saved quant file. `fp16`, `bf16`, or `fp32` read the original checkpoint instead
  - The prefix KV cache is on by default. `--no-kv-cache` runs the text and image tokens together on every step
  - `--stream-text-encoder` and `--stream-denoiser` keep the quantized weights in pinned host memory and copy one layer at a time
  - `--vae-dtype {fp32,fp16}` sets the VAE compute dtype. Default: `fp16`
  - `--rgb` flattens alpha over white. The default keeps RGBA
  - Width and height round up to a multiple of 32
- Measured on a 6 GB card with the text encoder at `sym-med-nano` and the denoiser at `aff-high-nano`: at 1024, 47.5 s per step and about 7 minutes in total, with the text encoder streamed and the denoiser resident. At 512, 14.7 s per step. Streaming the denoiser adds about 27% to the step time, so stream it only when the quant file does not fit
- Examples
  - `python -m qwen_2.gen --root "/iml_root" --prompt "a red fox in fresh snow, morning light"`
  - `python -m qwen_2.gen --root "/iml_root" --prompt "a red fox in fresh snow" --width 512 --height 512 --stream-text-encoder --stream-denoiser`
  - `python -m qwen_2.gen --root "/iml_root" --prompt "a red fox in fresh snow" --variant base --guidance-scale 4.0 --negative-prompt "blurry"`
- Useful args
  - `--root`
  - `--prompt`
  - `--negative-prompt`
  - `--width`, `--height`
  - `--steps`
  - `--guidance-scale`
  - `--seed`
  - `--variant`
  - `--text-quant-method`
  - `--denoiser-quant-method`
  - `--stream-text-encoder`, `--stream-denoiser`
  - `--no-kv-cache`
  - `--vae-dtype`
  - `--rgb`
  - `--output`

### SDXL

The `sdxl` package prepares, quantizes, and runs SDXL checkpoints. The model folder is `<root>/sdxl/<name>`:

```
<root>/sdxl/
  base/
    unet/                config.json, <method>_quant.safetensors
                         and diffusion_pytorch_model.safetensors with --keep-fp16
    text_encoder/        CLIP-L, fp16
    text_encoder_2/      CLIP-G, fp16
    vae/                 fp32
    tokenizer/  tokenizer_2/  scheduler/
  outputs/
```

The loader never quantizes on the fly. It raises `FileNotFoundError` when the `<method>_quant.safetensors` file is missing.

#### `sdxl.denoiser.quant`
- Features
  - Turns one single-file SDXL checkpoint into a complete model folder at `<root>/sdxl/<name>`
  - Converts the UNet, the VAE, both text encoders, both tokenizers, and the Euler A scheduler
  - Quantizes the UNet in memory during the same pass and writes only `<name>/unet/<method>_quant.safetensors`. The 5.1 GB fp16 UNet file is not written, unless `--keep-fp16` is set
  - The quantized file holds every UNet tensor, so it is complete on its own
  - `--method` defaults to `sym-high`
  - Attention and projection weights go to `high`. Feed-forward weights go to `low`. The time and additive embeddings stay fp16
  - Fails when the model folder is not empty, unless `--overwrite` is set
  - CUDA-only runtime; exits if CUDA is unavailable
- Examples
  - `python -m sdxl.denoiser.quant --input "/models/Illustrious-XL-v1.0.safetensors" --root "/iml_root" --name illustrious`
  - `python -m sdxl.denoiser.quant --input "/models/Illustrious-XL-v1.0.safetensors" --root "/iml_root" --name illustrious --method sym-med --keep-fp16`
- Useful args
  - `--input`
  - `--root`, `--name`
  - `--method`
  - `--keep-fp16`
  - `--overwrite`

#### `sdxl.gen`
- Features
  - Generates one image from a prompt and saves it as a PNG in `<root>/sdxl/outputs`
  - `--output` overrides the output folder
  - Defaults: 1024×1024, 28 steps, CFG 5.0, seed 19930625
  - Reads `<root>/sdxl/<name>/unet/<method>_quant.safetensors`. `--no-quant` reads the fp16 UNet file instead
  - Writes the generation settings into the PNG under the `sdxl` key as JSON
  - Moves the text encoders off the GPU after prompt encoding, then moves the UNet off before VAE decode
  - `--stream-denoiser` keeps the quantized weights in pinned host memory and copies each layer to the GPU for its own multiply. It needs a quantized checkpoint
  - `--loras` loads kohya or native LoRA checkpoints into the UNet and the text encoders. Each entry is `path:strength`, and the strength multiplies that checkpoint only
  - Several LoRAs add together. The checkpoints may use different ranks: every module is injected at the largest rank of the list, and a smaller checkpoint writes its factors into the leading block, so its result is exact
  - Decodes the VAE with tiling
  - CUDA-only runtime; exits if CUDA is unavailable
- Examples
  - `python -m sdxl.gen --root "/iml_root" --prompt "a cat holding a sign that says hello world"`
  - `python -m sdxl.gen --root "/iml_root" --width 832 --height 1216 --steps 30`
  - `python -m sdxl.gen --root "/iml_root" --no-quant`
  - `python -m sdxl.gen --root "/iml_root" --loras "/models/loras/style.safetensors:1.0"`
  - `python -m sdxl.gen --root "/iml_root" --loras "/models/loras/person.safetensors:1.0,/models/loras/style.safetensors:0.6"`
- Useful args
  - `--root`
  - `--name`
  - `--output`
  - `--prompt`
  - `--negative`
  - `--width`, `--height`
  - `--steps`
  - `--cfg`
  - `--seed`
  - `--method`
  - `--no-quant`
  - `--stream-denoiser`
  - `--loras`

#### `sdxl.train.main`
- Features
  - Trains a LoRA on `<project>/images/<id>.png`, with an optional `<id>.txt` caption, on the `sym-high` quantized UNet
  - Resizes every image so the short side is `--target-res` and the aspect ratio is kept. Sides round to 32
  - Skips an image below `--target-res` unless `--target-upscale` is set
  - Encodes text and latents first, then frees the text encoders and the VAE. The loop holds only the UNet and the LoRA state
  - Gradient checkpointing on `down_blocks`, `mid_block`, and `up_blocks` is on by default. `--no-gradient-checkpointing` keeps the activations and is faster, but needs about 2 GB more VRAM
  - `--stream-denoiser` keeps the quantized weights in pinned host memory and copies each layer onto the GPU for its own multiply. It trades about 1 s per step for 2.2 GiB of VRAM, which is what fits 768 on a 6 GB card. It needs about 2.2 GiB of free host RAM and leaves gradient checkpointing on
  - `--cache-text` writes `<id>.text.safetensors`. `--cache-images` writes `<id>.<res>.safetensors`. Both sit next to the images
  - Saves `<step>.safetensors` and `<step>.optimizer.pt` under `<project>/models`, plus `steps.csv`, `stop.txt`, and `loss.png`
  - Prints one line per step in the Flux 3 format: step time, the load/forward/backward split, the gradient norm, and the ETA
  - Logs the loss and refreshes `loss.png` every 8 steps. `--ma` overrides the moving-average windows, which default to the sample count and the checkpoint interval
  - Delete `models/stop.txt` to stop cleanly after the current step
  - Checkpoint metadata: `rank`, `alpha`, `step`, `base`, `targets`
  - `--resume` continues from the highest step in `<project>/models`
  - The output loads straight into `sdxl.gen --loras`
  - CUDA-only runtime; exits if CUDA is unavailable
- Examples
  - `python -m sdxl.train.main --root "/iml_root" --project "/iml_root/sdxl/loras/niji" --cache-images --cache-text`
  - `python -m sdxl.train.main --root "/iml_root" --project "/iml_root/sdxl/loras/niji" --steps 512 --rank 16 --alpha 8`
  - `python -m sdxl.train.main --root "/iml_root" --project "/iml_root/sdxl/loras/niji" --resume`
- Useful args
  - `--root`, `--name`, `--project`
  - `--steps`, `--lr`, `--seed`
  - `--rank`, `--alpha`
  - `--noise-offset`, `--min-snr-gamma`, `--max-grad-norm`
  - `--target-res`, `--target-upscale`
  - `--no-gradient-checkpointing`
  - `--stream-denoiser`
  - `--cache-images`, `--cache-text`
  - `--checkpoint-steps`, `--resume`

#### `sdxl.sample`
- Features
  - Renders sample images for the LoRA step checkpoints that `sdxl.train.main` saves. CUDA-only
  - Scans `<project>/models` for numeric `<step>.safetensors` files and writes images to `<project>/samples`
  - Loads the model once, then loads each checkpoint into the same LoRA modules. Only the weights change between steps
  - `--start` skips earlier steps
  - Skips a sample whose output file already exists
  - Dataset-backed mode uses the captions in `<project>/images`. A sample without a caption raises
  - `--prompt` replaces the captions. Without `--sample-indices`, it renders `--width` x `--height`
  - `--sample-indices "1,2,3"` picks 1-based dataset positions, in the order the trainer uses. The output size comes from the aspect ratio of each image and `--target-res`, rounded to a multiple of 32
  - The noise is deterministic per sample name: `--sample-seed` and the file stem
  - Writes the checkpoint, step, seed, prompt, and sampler settings into the PNG under the `sdxl` key as JSON
  - `--stream-denoiser` keeps the quantized weights in pinned host memory and copies each layer to the GPU for its own multiply
- Examples
  - `python -m sdxl.sample --root "/iml_root" --project "/iml_root/sdxl/loras/niji" --sample-indices "1,2,3"`
  - `python -m sdxl.sample --root "/iml_root" --project "/iml_root/sdxl/loras/niji" --start 256 --steps 30`
  - `python -m sdxl.sample --root "/iml_root" --project "/iml_root/sdxl/loras/niji" --prompt "niji on a beach"`
- Useful args
  - `--root`, `--name`, `--project`
  - `--prompt`, `--width`, `--height`
  - `--target-res`
  - `--start`
  - `--steps` (default: `32`)
  - `--cfg`, `--negative`
  - `--sample-seed`, `--sample-indices`
  - `--method`, `--no-quant`
  - `--stream-denoiser`

### Ideogram 4

#### `ideogram.gen`
- Features
  - CLI text-to-image generation using local Ideogram 4 weights (F8 checkpoint + custom quantization)
  - Qwen3-VL text encoder captures hidden states from specific activation layers and stacks them into text features
  - Asymmetric CFG with conditional/unconditional transformer pair
  - Configurable width, height, steps, guidance scale, and seed
  - Quantization method selection for text encoder and denoiser
  - Block offloading support to reduce peak GPU memory
  - Dimensions must be divisible by `patch_size * ae_scale_factor` (default 16)
- Examples
  - `python -m ideogram.gen --root "/models" --prompt "A cat holding a sign" --output "cat.png"`
  - `python -m ideogram.gen --root "/models" --prompt "cinematic portrait" --width 768 --height 768 --steps 64 --guidance-scale 5.0`
  - `python -m ideogram.gen --root "/models" --prompt "futuristic city" --text-quant-method sym-high --denoiser-quant-method sym-med`
- Useful args
  - `--root`, `--prompt`, `--output`
  - `--width`, `--height`, `--steps`, `--guidance-scale`, `--seed`
  - `--text-quant-method`, `--denoiser-quant-method`
  - `--offload`

#### `ideogram.text_encoder.quant`
- Features
  - Quantizes the Ideogram 4 text encoder (Qwen3-VL) from its native F8 checkpoint
  - Loads `<input>/model.safetensors`
  - Dequantizes F8 scales back to fp32, then re-quantizes target tensors with the chosen method
  - Strips the `language_model.` prefix in the saved output
  - Saves `<root>/ideogram/text_encoder/<method>_quant.safetensors`
- Examples
  - `python -m ideogram.text_encoder.quant --root "/models" --input "/path/to/f8/text_encoder" --method sym-high-nano`
- Useful args
  - `--root`, `--input`, `--method`

#### `ideogram.denoiser.quant`
- Features
  - Quantizes the Ideogram 4 denoiser (conditional or unconditional transformer) from its native F8 checkpoint
  - Cond input: `<input>/transformer/diffusion_pytorch_model.safetensors`
  - Uncond input: `<input>/unconditional_transformer/diffusion_pytorch_model.safetensors`
  - Dequantizes F8 scales back to fp32, then re-quantizes target tensors with the chosen method
  - Saves `<root>/ideogram/transformer/<variant>/<method>_quant.safetensors`
- Examples
  - `python -m ideogram.denoiser.quant --root "/models" --input "/path/to/f8/root" --variant cond --method sym-med-mini`
- Useful args
  - `--root`, `--input`, `--method`, `--variant` (`cond` or `uncond`)


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
    - `qwen_2_text_encoder`, `qwen_2_denoiser`
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
