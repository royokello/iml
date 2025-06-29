# Image Machine Learning (IML) Project

The **Image Machine Learning (IML)** mono‑repo houses several standalone tools that can be chained together to prepare, clean, and rank large image datasets **and** a lightweight Stable‑Diffusion 1.5 pipeline.

---

## Table of Contents
1. [IML Toolkit](#iml-toolkit)
2. [Stable‑Diffusion 1.5 — FP8 Quant + INT8 Inference](#stable‑diffusion-15)
3. [CUDA INT8 Extension](#cuda-int8-extension)
4. [End‑to‑End Examples](#end‑to‑end-examples)

---

## IML Toolkit

### Components
| Tool | Purpose |
|------|---------|
| **iml‑aggregate** | Recursively collect images, convert to PNG, copy matching captions |
| **iml‑extract** | Extract `n` frames per video (sequential or random) |
| **iml‑cull** | Flask UI + ViT helper for manual / AI‑assisted culling |
| **iml‑crop** | Train a crop model, predict boxes, or batch‑crop images |
| **iml‑rank** | Gather pairwise preferences, train an Elo‑style ranker |

Suggested workflow → *Aggregator → Cull → Cropper → Ranker*.

---

## Stable‑Diffusion 1.5

### 1 – Quantise the UNet to FP8 (E5M2)
```bash
python sd15/quant.py -i "models/stable-diffusion-v1-5" -o "models/stable-diffusion-v1-5-fp8-e5m2"
```

### 2 – Build the CUDA INT8 extension
```bash
python setup.py install
```

### 3 – Run inference
```bash
python sd15/infer.py --model "/models/stable-diffusion-v1-5-fp8-e5m2" --prompt "A scenic mountain landscape" --output "/pictures/out.png"
```
---

## CUDA INT8 Extension
* **Kernel:** naïve DP4A GEMM (row‑major) ⇒ replace with CUTLASS for >2× speed.
* **Architecture flag:** `-gencode arch=compute_61,code=sm_61` (Pascal).
* **Interface:** `int32 = int8_gemm(a_int8, b_int8)`; used by custom `Int8Linear / Int8Conv2d` layers in `sd15/model.py`.

---

## End‑to‑End Examples
| Task | Command |
|------|---------|
| Quantise SD1.5 | `python sd15/quant.py -i /models/v1-5 -o /models/v1-5-fp8` |
| Build CUDA op  | `python setup.py install` |
| 512×512 image  | `python sd15/infer.py --model /models/v1-5-fp8 --prompt "cat" --output cat.png` |
| 768×512 wide   | `python sd15/infer.py --model /models/v1-5-fp8 --prompt "sunset" --output sunset.png --size 768,512` |

---

### Requirements
* Python ≥3.10, PyTorch ≥2.2 with CUDA, Diffusers ≥0.27
* A Pascal‑class GPU (`sm_61`) or newer. For other arches adjust `-gencode` in **setup.py**.

> **Tip** If you only need CPU preprocessing (IML toolkit) you can skip the CUDA build.
