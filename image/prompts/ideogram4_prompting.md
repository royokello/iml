# Ideogram 4.0 Prompting Guide

## Overview

Ideogram 4.0 supports two prompting modes: **natural language** (also works on v2.0 and v3.0) and **structured JSON** (v4.0 only). Magic Prompt can auto-expand natural language into JSON.

## Natural Language Prompting

### Recommended Structure

```
[Image summary]. [Main subject details], [Pose or action],
[Secondary elements], [Setting & Background],
[Lighting & Atmosphere], [Framing & Composition], [Technical enhancers]
```

### Sections

1. **Image summary** — single sentence describing the whole image
2. **Main subject details** — color, shape, material, texture; quote text to render
3. **Pose or action** — what the subject is doing
4. **Secondary elements** — props, accessories, ambient detail
5. **Setting & background** — location, time of day, environment
6. **Lighting & atmosphere** — mood, light quality, emotional tone
7. **Framing & composition** — camera angle, shot type, placement
8. **Technical enhancers** — lens, bokeh, brush texture, rendering style

### Example

```
A product photo of a men's perfume bottle named "Nightlife for men" in a sleek studio setup. The bottle is tall and rectangular with dark glass, a matte black cap, and silver lettering. The text "Nightlife for men" appears on the label in bold, modern font. The bottle stands upright with a slight reflection on the surface below. A wristwatch and a pair of sunglasses sit nearby, adding a masculine vibe. The scene is set on a smooth black surface with blurred city lights in the background. Lighting is moody and cool, with soft blue highlights and deep shadows. The bottle is centered in the frame, captured at eye level. A shallow depth of field gives the image a polished, professional look.
```

### Important: Ideogram does not understand negation. Describe the positive visual opposite.

- "an empty room" instead of "no people"
- "a bald figure with smooth skin" instead of "no hair"
- "an empty beach at sunrise" instead of "a beach without people"

Max prompt length: ~150-160 words / ~200 tokens.

## Structured JSON Prompting (Ideogram 4.0)

Ideogram 4.0 was trained exclusively on structured JSON captions. JSON gives exact color control (hex), bounding-box layout, precise text placement, and repeatable outputs.

### Top-Level Schema

```json
{
  "high_level_description": "One or two sentences summarizing the full image",
  "style_description": {
    "...style fields..."
  },
  "compositional_deconstruction": {
    "background": "Description of background/environment",
    "elements": [
      "...element objects..."
    ]
  }
}
```

### Required Key Order (MUST be preserved)

**Photo style:**
`high_level_description` > `style_description` > `aesthetics` > `lighting` > `photo` > `medium` > `color_palette` > `compositional_deconstruction` > `background` > `elements`

**Non-photo style:**
`high_level_description` > `style_description` > `aesthetics` > `lighting` > `medium` > `art_style` > `color_palette` > `compositional_deconstruction` > `background` > `elements`

### `style_description` Fields

| Field           | Type            | Description                                                         |
|-----------------|-----------------|---------------------------------------------------------------------|
| `aesthetics`    | string          | Aesthetic keywords (e.g. "moody, cinematic, desaturated")          |
| `lighting`      | string          | Lighting description                                                |
| `photo`         | string          | Camera/lens for photos (use OR `art_style`, not both)               |
| `medium`        | string          | "photograph", "illustration", "3d_render", "painting", etc.         |
| `art_style`     | string          | Art style for non-photo (use OR `photo`, not both)                  |
| `color_palette` | list of strings | Up to 16 uppercase `#RRGGBB` hex codes                              |

Must have exactly one of `photo` or `art_style`.

### `compositional_deconstruction` Elements

| Field        | Type   | Description                                          |
|--------------|--------|------------------------------------------------------|
| `background` | string | Description of the background/environment            |
| `elements`   | list   | List of element objects (obj or text)                |

### Element Types

**Object element:**
```json
{
  "type": "obj",
  "bbox": [y_min, x_min, y_max, x_max],
  "desc": "Detailed description of the element",
  "color_palette": ["#RRGGBB"]
}
```

**Text element:**
```json
{
  "type": "text",
  "bbox": [y_min, x_min, y_max, x_max],
  "text": "Literal text to render",
  "desc": "Description of the text styling",
  "color_palette": ["#RRGGBB"]
}
```

- `type`: `"obj"` or `"text"`
- `bbox`: optional, `[y_min, x_min, y_max, x_max]` in normalized 0–1000 coordinates (top-left origin)
- `desc`: required, detailed description
- `text`: required for text elements
- `color_palette`: optional, up to 5 hex codes per element

### Bounding Box System

Normalized 0–1000 coordinate system. `[0, 0]` is top-left, `[1000, 1000]` is bottom-right.

- `[0, 0, 500, 1000]` → top half of image
- `[250, 250, 750, 750]` → roughly centered
- If omitted, model places element freely

### Complete JSON Example (Poster with Bounding Boxes)

```json
{
  "high_level_description": "A bold event poster for a jazz night called 'Blue Note Sessions' at The Velvet Room, Saturday August 9th.",
  "style_description": {
    "aesthetics": "moody, retro, sophisticated, 1960s jazz club aesthetic",
    "lighting": "dramatic, deep shadows, warm spotlight glow",
    "medium": "graphic_design",
    "art_style": "vintage poster design, textured paper, bold typography, muted color palette with warm accents",
    "color_palette": ["#1A1A2E", "#16213E", "#E8C97A", "#D4A843", "#F5F0E8", "#8B4513"]
  },
  "compositional_deconstruction": {
    "background": "Deep navy and near-black background with subtle aged paper texture and faint horizontal grain lines.",
    "elements": [
      {
        "type": "obj",
        "bbox": [150, 200, 650, 800],
        "desc": "A silhouetted jazz trumpeter in side profile, mid-performance, instrument raised. Warm golden spotlight illuminates from above, casting dramatic shadows."
      },
      {
        "type": "text",
        "bbox": [30, 50, 140, 950],
        "text": "BLUE NOTE SESSIONS",
        "desc": "Large bold all-caps serif headline in warm golden-yellow, spanning the full width near the top of the poster."
      },
      {
        "type": "text",
        "bbox": [660, 100, 760, 900],
        "text": "Live jazz every Saturday night",
        "desc": "Medium-weight italic serif subheading in off-white, centered beneath the main title."
      }
    ]
  }
}
```

## Magic Prompt

When enabled, Magic Prompt auto-expands natural language into a full JSON caption. Recommended for casual use. For production/repeatable workflows, write JSON directly.

## When to Use What

| Use case                              | Recommended approach     |
|---------------------------------------|--------------------------|
| Quick exploration, creative ideas     | Natural language         |
| Expanding a short idea                | Natural + Magic Prompt   |
| Design work, posters, branded graphics| JSON                     |
| Exact text placement                  | JSON                     |
| Specific color palette                | JSON                     |
| Repeatable layouts                    | JSON                     |
| Photorealism where placement isn't critical | Natural language   |

## API Generation Parameters

| Parameter       | Type   | Description                                      |
|-----------------|--------|--------------------------------------------------|
| `prompt`        | string | Text prompt or JSON caption                      |
| `aspect_ratio`  | string | Width:Height format (e.g. "1:1", "16:9", "4:5") |
| `model`         | string | "V_4" for Ideogram 4.0                           |
| `magic_prompt`  | bool   | Auto-enhance prompt to JSON (auto-off if JSON)   |
| `seed`          | int    | Seed for reproducibility                         |
| `style`         | string | Style preset                                     |
| `color_palette` | list   | Hex color palette                                |

## Sources

- docs.ideogram.ai — Prompting Guide (Section 3: Prompt Structure, Section 4: JSON Prompting)
- docs.ideogram.ai — Magic Prompt, Prompt Builder
