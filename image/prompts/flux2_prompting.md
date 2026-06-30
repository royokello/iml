# FLUX 2 Prompting Guide

## Overview

FLUX 2 by Black Forest Labs offers multiple tiers ([klein], [pro], [flex], [max], [dev]) ranging from sub-second inference to highest quality with grounding search. It supports natural language, structured JSON, and multi-reference editing.

## Natural Language Prompting

### Recommended Template

```
[SUBJECT], [LOCATION],
[STYLE], [CAMERA SETTINGS], [LIGHTING], [COLORS], [EFFECT],
[ADDITIONAL ELEMENTS]
```

### Key Principles

- Use natural language — describe the image clearly
- Place exact text to render in quotation marks
- Refine iteratively: start simple, tweak one detail at a time
- English produces most precise results but multilingual works
- No negative prompts — describe what you want, not what you don't

### Example

```
A cinematic long shot with the camera positioned half underwater and half above the surface in the open ocean. Beneath the water, a large northern whale is diving smoothly, its tail rising above the surface while the rest of its body descends into the deep blue. The ocean is calm and clear, with bubbles drifting upward from the whale and soft sun rays piercing through the water.
```

## Structured (JSON) Prompting

FLUX 2 supports JSON-structured prompts for precise control. The JSON is passed directly as the prompt string.

### JSON Schema

```json
{
  "subject": "<main subject description>",
  "background": "<setting and environment>",
  "lighting": "<lighting conditions>",
  "style": "<visual style, medium, aesthetic>",
  "camera_angle": "<viewpoint or perspective>",
  "composition": "<framing, layout, orientation>"
}
```

### Example

```json
{
  "subject": "Mona Lisa painting by Leonardo da Vinci",
  "background": "museum gallery wall, ornate gold frame",
  "lighting": "soft gallery lighting, warm spotlights",
  "style": "digital art, high contrast",
  "camera_angle": "eye level view",
  "composition": "centered, portrait orientation"
}
```

## Exact Color Control

Specify brand colors via hex codes directly in the prompt. No JSON required.

```
A vase on a table in living room, the color of the vase is a gradient of color, starting with color #02eb3c and finishing with color #edfa3c. The flowers inside the vase have the color #ff0088
```

## Text in Images

Place exact text in quotation marks. The text will be rendered visibly in the image.

```
A photorealistic still life of matchsticks on beige paper. Bold serif typography reads "The Importance Of Being Non-Aligned".
```

## Multi-Reference Editing (API)

FLUX 2 accepts up to 8 (or 10 on playground) reference images via the API for:
- Character consistency across generations
- Style transfer
- Image compositing and editing

## API Parameters

| Parameter     | Type     | Description                                         |
|---------------|----------|-----------------------------------------------------|
| `prompt`      | string   | Text prompt or JSON-structured prompt               |
| `width`       | integer  | Image width (up to 4MP)                             |
| `height`      | integer  | Image height                                        |
| `num_images`  | integer  | Number of images to generate                        |
| `image`       | string   | URL for image editing input                         |
| `references`  | array    | Up to 8-10 reference image URLs                     |

## Model Tiers

| Tier     | Best For                        | Pricing             |
|----------|---------------------------------|---------------------|
| [klein]  | Real-time, high-volume, local   | from $0.014/image   |
| [pro]    | Production at scale             | from $0.03/MP       |
| [flex]   | Typography/small detail control | $0.06/MP            |
| [max]    | Highest quality, grounding      | from $0.07/MP       |
| [dev]    | Local development (open-weight) | Free (non-commercial)|

## Sources

- docs.bfl.ai — FLUX.2 Overview, Prompting Guide, Text to Image API
- replicate.com/black-forest-labs/flux-2-pro
