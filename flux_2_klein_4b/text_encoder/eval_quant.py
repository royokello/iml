from __future__ import annotations

import argparse
import gc
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import torch
from transformers import Qwen2TokenizerFast

from .loader import load_qwen3_text_encoder

LAYER_INDICES = (9, 18, 27)
DEFAULT_PROMPTS = (
    "A cinematic portrait of a woman in a red coat standing in rain at night.",
    "An isometric city block with tiny cafes, bicycles, and warm window light.",
    "A product photo of a matte black camera on a pale stone surface.",
)


@dataclass(frozen=True)
class QuantizationConfig:
    label: str
    quantization_precision: str | None = None
    scale_precision: str | None = None
    block_size: int | None = None


DEFAULT_CONFIGS = (
    QuantizationConfig(label="baseline-fp16"),
    # QuantizationConfig(label="int8-fp32-b128", quantization_precision="int8", scale_precision="fp32", block_size=128),
    QuantizationConfig(label="int8-fp16-b128", quantization_precision="int8", scale_precision="fp16", block_size=128),
    QuantizationConfig(label="int8-fp16-b64", quantization_precision="int8", scale_precision="fp16", block_size=64),
    QuantizationConfig(label="int8-fp16-b32", quantization_precision="int8", scale_precision="fp16", block_size=32),
    # QuantizationConfig(label="int8-e8m0-b128", quantization_precision="int8", scale_precision="e8m0", block_size=128),
    # QuantizationConfig(label="int8-e8m0-b64", quantization_precision="int8", scale_precision="e8m0", block_size=64),
    # QuantizationConfig(label="int8-e8m0-b32", quantization_precision="int8", scale_precision="e8m0", block_size=32),
    
)
BASELINE_CONFIG_LABEL = "baseline-fp16"


def _serialize_value(value: float | int | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.6f}"
    return str(value)


def _tensor_metrics(baseline: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    baseline_flat = baseline.detach().to(dtype=torch.float64).reshape(-1)
    candidate_flat = candidate.detach().to(dtype=torch.float64).reshape(-1)
    diff = candidate_flat - baseline_flat
    baseline_norm = torch.linalg.vector_norm(baseline_flat)
    candidate_norm = torch.linalg.vector_norm(candidate_flat)
    diff_norm = torch.linalg.vector_norm(diff)
    cosine_denominator = (baseline_norm * candidate_norm).clamp_min(1e-12)
    cosine = torch.dot(baseline_flat, candidate_flat) / cosine_denominator
    cosine = cosine.clamp(min=-1.0, max=1.0).item()
    return {
        "cosine_similarity": cosine,
        "mae": diff.abs().mean().item(),
        "max_abs_error": diff.abs().max().item(),
        "relative_l2_error": (diff_norm / baseline_norm.clamp_min(1e-12)).item(),
    }


def _estimate_model_size_mb(model: torch.nn.Module) -> float:
    total_bytes = 0
    for parameter in model.parameters():
        total_bytes += parameter.numel() * parameter.element_size()
    for buffer in model.buffers():
        total_bytes += buffer.numel() * buffer.element_size()
    return total_bytes / (1024 * 1024)


def _load_prompts(prompts_file: str | Path | None) -> tuple[str, ...]:
    if prompts_file is None:
        return DEFAULT_PROMPTS
    lines = [line.strip() for line in Path(prompts_file).read_text(encoding="utf-8").splitlines()]
    return tuple(line for line in lines if line)


def _tokenize_prompt(
    tokenizer: Qwen2TokenizerFast,
    prompt: str,
    max_length: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    encoded = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    return {key: value.to(device) for key, value in encoded.items()}


def _extract_prompt_data(
    text_encoder: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], float]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    encode_start = time.perf_counter()
    with torch.inference_mode():
        output = text_encoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
            use_cache=False,
            compute_logits=False,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    encode_seconds = time.perf_counter() - encode_start
    hidden_states = output.hidden_states
    if hidden_states is None:
        raise RuntimeError("Text encoder did not return hidden_states. The evaluation requires output_hidden_states.")
    selected_layers = [hidden_states[index].detach().cpu() for index in LAYER_INDICES]
    stacked = torch.stack([hidden_states[index] for index in LAYER_INDICES], dim=1)
    prompt_embeds = stacked.permute(0, 2, 1, 3).reshape(stacked.shape[0], stacked.shape[2], -1).detach().cpu()
    return (
        {
            "layers": selected_layers,
            "prompt_embeds": prompt_embeds,
        },
        encode_seconds,
    )


def _load_text_encoder(
    model_path: Path,
    config: QuantizationConfig,
    device: torch.device,
) -> tuple[torch.nn.Module, float]:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    print(f"  loading checkpoint for {config.label}")
    text_encoder = load_qwen3_text_encoder(
        str(model_path),
        quantization_precision=config.quantization_precision,
        scale_precision=config.scale_precision,
        block_size=config.block_size,
    )
    print(f"  moving {config.label} to {device}")
    text_encoder = text_encoder.to(device)
    print(f"  {config.label} is on {device}")
    model_size_mb = _estimate_model_size_mb(text_encoder)
    return text_encoder, model_size_mb


def _average_metrics(rows: Iterable[dict[str, float | int | str | None]], keys: tuple[str, ...]) -> dict[str, float]:
    rows = list(rows)
    if not rows:
        return {key: float("nan") for key in keys}
    return {key: sum(float(row[key]) for row in rows) / len(rows) for key in keys}


def _markdown_table(rows: list[dict[str, float | int | str | None]], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(_serialize_value(row.get(key)) for key, _ in columns) + " |")
    if not body:
        body.append("| " + " | ".join("" for _ in columns) + " |")
    return "\n".join([header, divider, *body])


def _write_markdown_summary(
    path: Path,
    *,
    prompts: tuple[str, ...],
    device: torch.device,
    aggregate_rows: list[dict[str, float | int | str | None]],
    layer_rows: list[dict[str, float | int | str | None]],
    prompt_rows: list[dict[str, float | int | str | None]],
) -> None:
    summary_columns = [
        ("label", "Config"),
        ("model_size_mb", "Model Size (MB)"),
        ("avg_prompt_encode_seconds", "Avg Encode (s)"),
        ("prompt_embed_cosine", "Prompt Cosine"),
        ("prompt_embed_mae", "Prompt MAE"),
        ("prompt_embed_max_abs_error", "Prompt Max Abs"),
        ("prompt_embed_relative_l2_error", "Prompt Rel L2"),
    ]
    layer_columns = [
        ("label", "Config"),
        ("layer", "Layer"),
        ("prompt_index", "Prompt"),
        ("cosine_similarity", "Cosine"),
        ("mae", "MAE"),
        ("max_abs_error", "Max Abs"),
        ("relative_l2_error", "Rel L2"),
    ]
    prompt_columns = [
        ("label", "Config"),
        ("prompt_index", "Prompt"),
        ("cosine_similarity", "Cosine"),
        ("mae", "MAE"),
        ("max_abs_error", "Max Abs"),
        ("relative_l2_error", "Rel L2"),
    ]
    prompt_lines = "\n".join(f"- {prompt}" for prompt in prompts)
    markdown = "\n".join(
        [
            "# Text Encoder Quantization Evaluation",
            "",
            f"Device: `{device}`",
            "",
            "## Prompts",
            prompt_lines,
            "",
            "## Aggregate Summary",
            _markdown_table(aggregate_rows, summary_columns),
            "",
            "## Layer Metrics",
            _markdown_table(layer_rows, layer_columns),
            "",
            "## Prompt Metrics",
            _markdown_table(prompt_rows, prompt_columns),
            "",
            "## Notes",
            "- `baseline-fp16` is the reference output for all comparisons.",
            "- Layer metrics are averaged over the configured prompt set.",
            "- Prompt embedding metrics compare the stacked Flux conditioning embedding built from layers 9, 18, and 27.",
            "- `Model Size (MB)` is estimated from parameter and buffer storage after the loader finishes quantization.",
            "- `Avg Encode (s)` is the average wall-clock time to encode one prompt for that config.",
        ]
    )
    path.write_text(markdown + "\n", encoding="utf-8")


def _timestamped_summary_path(output_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    return output_dir / f"{timestamp}.md"


def evaluate_quantization_configs(
    *,
    root: str | Path,
    prompts: tuple[str, ...] = DEFAULT_PROMPTS,
    device: str | None = None,
    configs: tuple[QuantizationConfig, ...] = DEFAULT_CONFIGS,
) -> dict[str, list[dict[str, float | int | str | None]]]:
    output_path = Path(root) / "flux_2_klein_4b" / "eval"
    output_path.mkdir(parents=True, exist_ok=True)

    model_path = Path(root) / "flux_2_klein_4b" / "base" / "text_encoder"
    tokenizer_path = Path(root) / "flux_2_klein_4b" / "base" / "tokenizer"

    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Evaluating text encoder quantization on {target_device}")
    print(f"Writing results to {output_path}")
    tokenizer = Qwen2TokenizerFast.from_pretrained(str(tokenizer_path))
    tokenized_prompts = [
        _tokenize_prompt(tokenizer, prompt, max_length=512, device=target_device) for prompt in prompts
    ]

    baseline_data: list[dict[str, torch.Tensor]] | None = None
    aggregate_rows: list[dict[str, float | int | str | None]] = []
    per_layer_rows: list[dict[str, float | int | str | None]] = []
    per_prompt_rows: list[dict[str, float | int | str | None]] = []

    for config_index, config in enumerate(configs, start=1):
        print(f"[{config_index}/{len(configs)}] Loading {config.label}")
        text_encoder, model_size_mb = _load_text_encoder(model_path, config, target_device)
        text_encoder.eval()

        current_prompt_data = []
        encode_seconds = []
        for prompt_index, prompt in enumerate(prompts, start=1):
            print(f"  prompt {prompt_index}/{len(prompts)}: {prompt}")
            prompt_data, prompt_encode_seconds = _extract_prompt_data(
                text_encoder,
                tokenized_prompts[prompt_index - 1],
                device=target_device,
            )
            current_prompt_data.append(prompt_data)
            encode_seconds.append(prompt_encode_seconds)

        if config.label == BASELINE_CONFIG_LABEL:
            baseline_data = current_prompt_data
        if baseline_data is None:
            raise RuntimeError(f"Missing baseline config '{BASELINE_CONFIG_LABEL}' in this evaluation run.")

        prompt_metric_rows = []
        layer_metric_rows = []
        for prompt_index, prompt in enumerate(prompts):
            baseline_prompt = baseline_data[prompt_index]
            current_prompt = current_prompt_data[prompt_index]

            embed_metrics = _tensor_metrics(baseline_prompt["prompt_embeds"], current_prompt["prompt_embeds"])
            per_prompt_rows.append(
                {
                    "label": config.label,
                    "prompt_index": prompt_index + 1,
                    "prompt": prompt,
                    **embed_metrics,
                }
            )
            prompt_metric_rows.append(embed_metrics)

            for layer_name, baseline_layer, current_layer in zip(LAYER_INDICES, baseline_prompt["layers"], current_prompt["layers"]):
                layer_metrics = _tensor_metrics(baseline_layer, current_layer)
                per_layer_rows.append(
                    {
                        "label": config.label,
                        "layer": layer_name,
                        "prompt_index": prompt_index + 1,
                        "prompt": prompt,
                        **layer_metrics,
                    }
                )
                layer_metric_rows.append({"layer": layer_name, **layer_metrics})

        averaged_prompt = _average_metrics(
            prompt_metric_rows,
            ("cosine_similarity", "mae", "max_abs_error", "relative_l2_error"),
        )
        aggregate_rows.append(
            {
                "label": config.label,
                "model_size_mb": model_size_mb,
                "avg_prompt_encode_seconds": sum(encode_seconds) / len(encode_seconds),
                "prompt_embed_cosine": averaged_prompt["cosine_similarity"],
                "prompt_embed_mae": averaged_prompt["mae"],
                "prompt_embed_max_abs_error": averaged_prompt["max_abs_error"],
                "prompt_embed_relative_l2_error": averaged_prompt["relative_l2_error"],
            }
        )

        for layer_index in LAYER_INDICES:
            layer_slice = [row for row in layer_metric_rows if row["layer"] == layer_index]
            averaged_layer = _average_metrics(
                layer_slice,
                ("cosine_similarity", "mae", "max_abs_error", "relative_l2_error"),
            )
            per_layer_rows.append(
                {
                    "label": config.label,
                    "layer": layer_index,
                    **averaged_layer,
                }
            )

        print(f"  tearing down {config.label}")
        del text_encoder
        gc.collect()
        if target_device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  finished teardown for {config.label}")

    summary_path = _timestamped_summary_path(output_path)
    _write_markdown_summary(
        summary_path,
        prompts=prompts,
        device=target_device,
        aggregate_rows=aggregate_rows,
        layer_rows=per_layer_rows,
        prompt_rows=per_prompt_rows,
    )
    print(f"Wrote markdown summary to {summary_path}")

    return {
        "aggregate_rows": aggregate_rows,
        "per_layer_rows": per_layer_rows,
        "per_prompt_rows": per_prompt_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Qwen3 text encoder quantization quality for Flux 2 Klein 4B.")
    parser.add_argument(
        "--root",
        required=True,
        help="Root folder that contains flux_2_klein_4b/base/tokenizer and flux_2_klein_4b/base/text_encoder.",
    )
    parser.add_argument(
        "--prompts-file",
        help="Optional newline-delimited prompt file. Defaults to an internal prompt set.",
    )
    parser.add_argument(
        "--device",
        help='Optional device override, for example "cuda" or "cpu".',
    )
    parser.add_argument(
        "--config",
        action="append",
        help="Optional config label filter. Repeat to run multiple specific configs only.",
    )
    return parser.parse_args()


def _filter_configs(selected_labels: list[str] | None) -> tuple[QuantizationConfig, ...]:
    if not selected_labels:
        return DEFAULT_CONFIGS
    allowed = {label.strip() for label in selected_labels if label and label.strip()}
    filtered = tuple(config for config in DEFAULT_CONFIGS if config.label in allowed)
    if not filtered:
        raise ValueError(f"No matching quantization configs for: {', '.join(sorted(allowed))}")
    return filtered


def main() -> None:
    args = parse_args()
    prompts = _load_prompts(args.prompts_file)
    evaluate_quantization_configs(
        root=args.root,
        prompts=prompts,
        device=args.device,
        configs=_filter_configs(args.config),
    )


if __name__ == "__main__":
    main()
