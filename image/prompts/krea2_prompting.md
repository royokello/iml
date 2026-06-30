# KREA 2 Prompting Guide

## Overview

KREA 2 is Krea's first foundation image model, trained from scratch. Available in two API variants (Medium and Large) plus an open-source release (RAW and Turbo). KREA 2 uses natural language prompts with structured API parameters for control — no JSON prompt format, but tunable creativity, style transfer, moodboards, and generative sliders.

## Variants

| Variant         | Best For                                   | Pricing   |
|-----------------|--------------------------------------------|-----------|
| Krea 2 Medium   | Illustration, anime, painting, expressive  | $0.030    |
| Krea 2 Large    | Photorealism, raw textures, artistic       | $0.060    |
| Krea 2 RAW      | Fine-tuning, LoRA training (open-source)   | Free (NCL)|
| Krea 2 Turbo    | 8-step distilled fast inference (open-source)| Free (NCL)|

## API Request Parameters

| Field                    | Type    | Required | Description                                         |
|--------------------------|---------|----------|-----------------------------------------------------|
| `prompt`                 | string  | Yes      | Text prompt describing the image                    |
| `aspect_ratio`           | string  | Yes      | `1:1`, `4:3`, `3:2`, `16:9`, `2.35:1`, `4:5`, `2:3`, `9:16` |
| `resolution`             | string  | Yes      | Currently only `1K`                                 |
| `creativity`             | enum    | No       | `raw`, `low`, `medium` (default), `high`            |
| `seed`                   | number  | No       | Reproducibility                                     |
| `image_style_references` | array   | No       | Style transfer references (up to 10)                |
| `moodboards`             | array   | No       | Moodboard ID + strength (max 1)                     |
| `styles`                 | array   | No       | Trained styles / LoRAs (id + strength)              |
| `intensity`              | integer | No       | -100 to 100, default 0 (stylization strength)       |
| `complexity`             | integer | No       | -100 to 100, default 0 (composition density)        |
| `movement`               | integer | No       | -100 to 100, default 0 (motion/pose energy)         |

## Creativity Parameter

Controls how literally the model follows your prompt vs. expanding creatively:

| Value    | Behavior                                                  |
|----------|-----------------------------------------------------------|
| `raw`    | No expansion — renders only what you explicitly describe  |
| `low`    | Minimal expansion — fills obvious gaps                    |
| `medium` | Balanced (default) — reasonable interpretation            |
| `high`   | Strong expansion — takes creative liberty with style/mood |

## Generative Sliders

Three numeric controls that shape visual character independently of creativity:

| Slider       | Negative (−100)        | Positive (+100)         |
|--------------|------------------------|-------------------------|
| `intensity`  | Bland, muted images    | Intensely stylized      |
| `complexity` | Minimal, clean comps   | Chaotic, dense scenes   |
| `movement`   | Static images          | Dynamic pose/camera motion |

### Suggested Starting Points

| Use case                            | Intensity | Complexity | Movement |
|-------------------------------------|-----------|------------|----------|
| Clean design, icons, editorial      | 0         | -60        | 0        |
| Cinematic / fashion / character     | +60       | 0          | +30      |
| Worlds and expressive scenes        | +60       | +50        | +20      |
| Exploring a new prompt              | 0         | 0          | 0        |

## Style Transfer

Pass one or more reference images via `image_style_references[]` array:

```json
{
  "image_style_references": [
    { "url": "https://...", "strength": 0.6 }
  ]
}
```

- `strength`: -2 to 2 (0.6 is a good starting point)
- Up to 10 references per request
- Multiple references blend additively

## Moodboards

Reference a Krea moodboard (created in webapp) for overall visual direction:

```json
{
  "moodboards": [
    { "id": "uuid-from-krea", "strength": 0.35 }
  ]
}
```

- `strength`: -0.5 to 1.5
- Max 1 moodboard per request
- Moodboard price: $0.040 (Medium), $0.070 (Large)

## Complete Request Example

```json
{
  "prompt": "A cinematic glass cabin beside a frozen lake at sunrise",
  "aspect_ratio": "16:9",
  "resolution": "1K",
  "creativity": "medium",
  "intensity": 40,
  "complexity": -20,
  "movement": 0,
  "image_style_references": [
    { "url": "https://assets.krea.ai/ref1.png", "strength": 0.5 }
  ]
}
```

## Open-Source Licensing

| License          | Use Case                                       |
|------------------|------------------------------------------------|
| Community License| Research, personal, non-commercial, commercial under revenue threshold |
| Commercial       | Enterprise use above revenue threshold         |

Both checkpoints on Hugging Face:
- `krea/Krea-2-Raw` — undistilled base (fine-tuning, LoRA)
- `krea/Krea-2-Turbo` — 8-step distilled (fast inference)

## Sources

- docs.krea.ai — Krea 2 API Overview, Style Transfer, Moodboards, Generative Sliders
- github.com/krea-ai/krea-2 — Open-source model code
- huggingface.co/krea — Krea 2 RAW and Turbo checkpoints
