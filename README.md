# ILM

**Author:** Roy Okello, Stelar Labs

## Features

ILM is a Python library designed for video processing and management. It includes modules for extracting frames, grouping videos based on attributes, detecting seamless loops in videos, and labeling these loops through an interactive web interface.

### Current Modules:

* **extract.py**: Extract frames from videos, either at regular intervals or randomly, while preserving the aspect ratio.
* **group.py**: Organize videos into directories based on resolution and orientation, ensuring structured storage and easy retrieval.
* **loop.py**: Detect seamless loops in videos using GPU-accelerated methods and structural similarity (SSIM) metrics, optimizing for efficient processing.
* **label.py**: A lightweight Flask application providing a simple web UI for tagging and captioning detected video loops, storing metadata conveniently alongside content.

## Usage

### Extract Frames

Extract frames from videos at specified intervals or randomly.

```bash
python extract.py --input "videos/" --output "frames/" --frames 1 --time second --resolution 512 --random
```

### Group Videos

Organize videos by resolution and orientation.

```bash
python group.py --root "videos/" --ffprobe "path/to/ffprobe" --dry-run
```

### Detect Loops

Detect seamless loops in videos using GPU acceleration.

```bash
python loop.py --input "videos/" --output "loops/" --lengths 33 --fps 30 --in-res 64 --ffmpeg "path/to/ffmpeg"
```

### Label Loops

Start the Flask UI to label and tag video loops interactively.

```bash
python label.py --input "loops/" --port 7860
```

Visit `http://localhost:7860` to access the labeling interface.
