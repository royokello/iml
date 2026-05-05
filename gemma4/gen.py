#!/usr/bin/env python3
"""
Minimal text generation for Gemma 4 using the official processor and causal LM.
Works with local checkpoints (gemma4_2b, gemma4_e2b, etc.)
"""

import argparse
from pathlib import Path

import torch

from gemma4.loaders.text import load_gemma4_text_model
from gemma4.models.casual import Gemma4ForCausalLM
from gemma4.models.processor import Gemma4Processor


def main(root_dirpath: Path, version: str, quant_method: str | None) -> None:
    model_dir = root_dirpath / f"gemma4_{version}"

    # 1. Load processor (handles tokenization + chat template)
    processor = Gemma4Processor.from_pretrained(model_dir)

    # 2. Load text model (handles PLE, quantization)
    model = load_gemma4_text_model(model_dir, quant_method=quant_method)
    config = model.config
    causal = Gemma4ForCausalLM(config=config, model=model)
    causal.eval()

    # 3. Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    causal = causal.to(device)

    # 4. Build conversation
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a short joke about saving RAM."},
    ]

    # 5. Apply chat template and tokenize
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,   # Gemma 4 supports thinking, disable for simple jokes
    )
    inputs = processor(text=text, return_tensors="pt").to(device)
    inputs.pop("mm_token_type_ids", None)
    input_len = inputs["input_ids"].shape[-1]

    # 6. Generate
    with torch.no_grad():
        outputs = causal.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,      # greedy; set to True with temperature for sampling
        )

    # 7. Decode only new tokens
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=True)

    # 8. Print
    print(response)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text with a local Gemma 4 checkpoint.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/models"),
        help="Root directory containing gemma4_<version> directories.",
    )
    parser.add_argument(
        "--version",
        default="2b",
        help="Checkpoint version suffix (e.g., 2b, e2b, 4b, e4b).",
    )
    parser.add_argument(
        "--quant-method",
        default=None,
        choices=("high", "aff-med-mini", "aff-high-mini"),
        help=(
            "Quantization preset to load, e.g. 'high', 'aff-med-mini', or 'aff-high-mini'."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.root, args.version, args.quant_method)
