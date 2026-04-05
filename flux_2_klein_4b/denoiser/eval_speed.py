from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from .loader import load_qwen3_denoiser

DEFAULT_WIDTH = 256
DEFAULT_HEIGHT = 256
DEFAULT_SCHEDULER_STEPS = 25
DEFAULT_TIMESTEP_INDEX = 0
DEFAULT_REPEATS = 1
DEFAULT_WARMUP_STEPS = 1
DEFAULT_TEXT_SEQ_LEN = 512
BENCHMARK_BATCH_SIZE = 1
BENCHMARK_DTYPE = torch.float16
DEFAULT_SEED = 0


@dataclass(frozen=True)
class QuantizationConfig:
    label: str
    quantization_precision: str | None = None
    scale_precision: str | None = None
    block_size: int | None = None


DEFAULT_CONFIGS = (
    # QuantizationConfig(label="baseline-fp16"),
    QuantizationConfig(label="int8-fp16-b128", quantization_precision="int8", scale_precision="fp16", block_size=128),
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


def _estimate_model_size_mb(model: torch.nn.Module) -> float:
    total_bytes = 0
    for parameter in model.parameters():
        total_bytes += parameter.numel() * parameter.element_size()
    for buffer in model.buffers():
        total_bytes += buffer.numel() * buffer.element_size()
    return total_bytes / (1024 * 1024)


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_transformer_config(path: Path) -> dict[str, object]:
    return _load_json(path / "config.json")


def _load_vae_scale_factor(path: Path) -> int:
    config = _load_json(path / "config.json")
    block_out_channels = config["block_out_channels"]
    if not isinstance(block_out_channels, list):
        raise ValueError("Unexpected VAE config: block_out_channels must be a list.")
    return 2 ** (len(block_out_channels) - 1)


def _prepare_latent_ids(latents: torch.Tensor) -> torch.Tensor:
    batch_size, _, height, width = latents.shape
    latent_ids = torch.cartesian_prod(
        torch.arange(1),
        torch.arange(height),
        torch.arange(width),
        torch.arange(1),
    )
    return latent_ids.unsqueeze(0).expand(batch_size, -1, -1)


def _pack_latents(latents: torch.Tensor) -> torch.Tensor:
    batch_size, num_channels, height, width = latents.shape
    return latents.reshape(batch_size, num_channels, height * width).permute(0, 2, 1)


def _prepare_text_ids(batch_size: int, seq_len: int) -> torch.Tensor:
    token_ids = torch.cartesian_prod(
        torch.arange(1),
        torch.arange(1),
        torch.arange(1),
        torch.arange(seq_len),
    )
    return token_ids.unsqueeze(0).expand(batch_size, -1, -1)


def _compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


def _retrieve_timesteps(
    scheduler: FlowMatchEulerDiscreteScheduler,
    num_inference_steps: int,
    *,
    device: torch.device,
    sigmas: list[float] | None,
    mu: float,
) -> torch.Tensor:
    if sigmas is None:
        scheduler.set_timesteps(num_inference_steps, device=device, mu=mu)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, sigmas=sigmas, mu=mu)
    return scheduler.timesteps


def _markdown_table(rows: list[dict[str, float | int | str | None]], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(_serialize_value(row.get(key)) for key, _ in columns) + " |")
    if not body:
        body.append("| " + " | ".join("" for _ in columns) + " |")
    return "\n".join([header, divider, *body])


def _timestamped_summary_path(output_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    return output_dir / f"{timestamp}.md"


def _write_markdown_summary(
    path: Path,
    *,
    device: torch.device,
    width: int,
    height: int,
    effective_width: int,
    effective_height: int,
    text_seq_len: int,
    image_seq_len: int,
    scheduler_steps: int,
    timestep_index: int,
    repeats: int,
    warmup_steps: int,
    summary_rows: list[dict[str, float | int | str | None]],
) -> None:
    columns = [
        ("label", "Config"),
        ("model_size_mb", "Model Size (MB)"),
        ("load_seconds", "Load (s)"),
        ("total_forward_seconds", "Total Forward (s)"),
        ("avg_step_ms", "Avg Step (ms)"),
        ("median_step_ms", "Median Step (ms)"),
        ("p95_step_ms", "P95 Step (ms)"),
        ("peak_memory_mb", "Peak Memory (MB)"),
    ]
    markdown = "\n".join(
        [
            "# Denoiser Speed Evaluation",
            "",
            f"Device: `{device}`",
            "",
            "## Benchmark Shape",
            f"- Requested image size: `{width}x{height}`",
            f"- Effective image size: `{effective_width}x{effective_height}`",
            f"- Batch size: `{BENCHMARK_BATCH_SIZE}`",
            f"- Text sequence length: `{text_seq_len}`",
            f"- Image sequence length: `{image_seq_len}`",
            f"- Scheduler steps: `{scheduler_steps}`",
            f"- Benchmarked timestep index: `{timestep_index}`",
            f"- Timed repeats: `{repeats}`",
            f"- Warmup steps: `{warmup_steps}`",
            "",
            "## Summary",
            _markdown_table(summary_rows, columns),
            "",
            "## Notes",
            "- `baseline-fp16` loads the dense fp16 checkpoint and is the reference configuration.",
            "- Timings cover a single denoiser forward step repeated multiple times, using `cache_context(\"cond\")` for each call.",
            "- The benchmark uses deterministic synthetic prompt embeddings and latents that match the pipeline tensor shapes.",
            "- The timed section excludes text encoder, scheduler state updates, and VAE work.",
            "- `Peak Memory (MB)` is CUDA peak allocated memory during the timed section; it is blank on non-CUDA devices.",
        ]
    )
    path.write_text(markdown + "\n", encoding="utf-8")


def _build_benchmark_fixture(
    *,
    root: Path,
    width: int,
    height: int,
    scheduler_steps: int,
    timestep_index: int,
    text_seq_len: int,
    seed: int,
    device: torch.device,
) -> dict[str, torch.Tensor | int]:
    transformer_path = root / "flux_2_klein_4b" / "base" / "transformer"
    scheduler_path = root / "flux_2_klein_4b" / "base" / "scheduler"
    vae_path = root / "flux_2_klein_4b" / "base" / "vae"

    transformer_config = _load_transformer_config(transformer_path)
    vae_scale_factor = _load_vae_scale_factor(vae_path)
    multiple_of = vae_scale_factor * 2
    effective_height = 2 * (int(height) // multiple_of)
    effective_width = 2 * (int(width) // multiple_of)
    if effective_height <= 0 or effective_width <= 0:
        raise ValueError(
            f"Image size {width}x{height} is too small for VAE scale factor {vae_scale_factor} and patch packing."
        )

    in_channels = int(transformer_config["in_channels"])
    joint_attention_dim = int(transformer_config["joint_attention_dim"])

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    latent_shape = (BENCHMARK_BATCH_SIZE, in_channels, effective_height // 2, effective_width // 2)
    latents = torch.randn(latent_shape, generator=generator, dtype=torch.float32).to(device=device, dtype=BENCHMARK_DTYPE)
    latent_ids = _prepare_latent_ids(latents).to(device=device)
    packed_latents = _pack_latents(latents)

    prompt_embeds = torch.randn(
        (BENCHMARK_BATCH_SIZE, text_seq_len, joint_attention_dim),
        generator=generator,
        dtype=torch.float32,
    ).to(device=device, dtype=BENCHMARK_DTYPE)
    text_ids = _prepare_text_ids(BENCHMARK_BATCH_SIZE, text_seq_len).to(device=device)

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(str(scheduler_path), local_files_only=True)
    sigmas = torch.linspace(1.0, 1 / scheduler_steps, scheduler_steps, dtype=torch.float32).tolist()
    if hasattr(scheduler.config, "use_flow_sigmas") and scheduler.config.use_flow_sigmas:
        sigmas = None

    image_seq_len = packed_latents.shape[1]
    mu = _compute_empirical_mu(image_seq_len=image_seq_len, num_steps=scheduler_steps)
    timesteps = _retrieve_timesteps(
        scheduler,
        scheduler_steps,
        device=device,
        sigmas=sigmas,
        mu=mu,
    )
    if timestep_index < 0 or timestep_index >= len(timesteps):
        raise ValueError(
            f"timestep_index must be between 0 and {len(timesteps) - 1} for scheduler_steps={scheduler_steps}."
        )

    return {
        "hidden_states": packed_latents,
        "encoder_hidden_states": prompt_embeds,
        "img_ids": latent_ids,
        "txt_ids": text_ids,
        "benchmark_timestep": timesteps[timestep_index],
        "effective_width": effective_width,
        "effective_height": effective_height,
        "image_seq_len": image_seq_len,
    }


def _load_denoiser(
    *,
    model_path: Path,
    config: QuantizationConfig,
    device: torch.device,
) -> tuple[torch.nn.Module, float, float]:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    load_start = time.perf_counter()
    denoiser = load_qwen3_denoiser(
        str(model_path),
        quantization_precision=config.quantization_precision,
        scale_precision=config.scale_precision,
        block_size=config.block_size,
    )
    denoiser = denoiser.to(device=device, dtype=BENCHMARK_DTYPE)
    denoiser.eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - load_start
    model_size_mb = _estimate_model_size_mb(denoiser)
    return denoiser, load_seconds, model_size_mb


def _run_forward_pass(
    denoiser: torch.nn.Module,
    *,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    txt_ids: torch.Tensor,
    img_ids: torch.Tensor,
    timestep_value: torch.Tensor,
) -> None:
    timestep = timestep_value.expand(hidden_states.shape[0]).to(hidden_states.dtype) / 1000
    with denoiser.cache_context("cond"):
        _ = denoiser(
            hidden_states=hidden_states,
            timestep=timestep,
            guidance=None,
            encoder_hidden_states=encoder_hidden_states,
            txt_ids=txt_ids,
            img_ids=img_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return float("nan")
    index = max(0, math.ceil(len(values) * percentile) - 1)
    return sorted(values)[index]


def _benchmark_denoiser(
    denoiser: torch.nn.Module,
    *,
    fixture: dict[str, torch.Tensor | int],
    repeats: int,
    warmup_steps: int,
    device: torch.device,
) -> dict[str, float]:
    hidden_states = fixture["hidden_states"]
    encoder_hidden_states = fixture["encoder_hidden_states"]
    txt_ids = fixture["txt_ids"]
    img_ids = fixture["img_ids"]
    benchmark_timestep = fixture["benchmark_timestep"]

    if not isinstance(hidden_states, torch.Tensor):
        raise TypeError("Benchmark fixture is missing hidden_states.")
    if not isinstance(encoder_hidden_states, torch.Tensor):
        raise TypeError("Benchmark fixture is missing encoder_hidden_states.")
    if not isinstance(txt_ids, torch.Tensor):
        raise TypeError("Benchmark fixture is missing txt_ids.")
    if not isinstance(img_ids, torch.Tensor):
        raise TypeError("Benchmark fixture is missing img_ids.")
    if not isinstance(benchmark_timestep, torch.Tensor):
        raise TypeError("Benchmark fixture is missing benchmark_timestep.")

    with torch.inference_mode():
        for warmup_index in range(warmup_steps):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            warmup_start = time.perf_counter()
            _run_forward_pass(
                denoiser,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                txt_ids=txt_ids,
                img_ids=img_ids,
                timestep_value=benchmark_timestep,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            warmup_seconds = time.perf_counter() - warmup_start
            print(f"  * WARMUP STEP {warmup_index + 1}/{warmup_steps}: {warmup_seconds:.3f}s")

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

        step_times: list[float] = []
        total_start = time.perf_counter()
        for repeat_index in range(repeats):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_start = time.perf_counter()
            _run_forward_pass(
                denoiser,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                txt_ids=txt_ids,
                img_ids=img_ids,
                timestep_value=benchmark_timestep,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_seconds = time.perf_counter() - step_start
            step_times.append(step_seconds)
            print(f"  * TEST STEP {repeat_index + 1}/{repeats}: {step_seconds:.3f}s")
        total_forward_seconds = time.perf_counter() - total_start

    peak_memory_mb: float | None = None
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    average_step_seconds = sum(step_times) / len(step_times)
    print(f"  * done with a test average of {average_step_seconds:.3f}s")

    return {
        "total_forward_seconds": total_forward_seconds,
        "avg_step_ms": average_step_seconds * 1000,
        "median_step_ms": statistics.median(step_times) * 1000,
        "p95_step_ms": _percentile(step_times, 0.95) * 1000,
        "peak_memory_mb": peak_memory_mb,
    }


def evaluate_denoiser_speed(
    *,
    root: str | Path,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    scheduler_steps: int = DEFAULT_SCHEDULER_STEPS,
    timestep_index: int = DEFAULT_TIMESTEP_INDEX,
    repeats: int = DEFAULT_REPEATS,
    warmup_steps: int = DEFAULT_WARMUP_STEPS,
    text_seq_len: int = DEFAULT_TEXT_SEQ_LEN,
    seed: int = DEFAULT_SEED,
    device: str | None = None,
) -> list[dict[str, float | int | str | None]]:
    if scheduler_steps <= 0:
        raise ValueError("scheduler_steps must be positive.")
    if timestep_index < 0:
        raise ValueError("timestep_index must be non-negative.")
    if repeats <= 0:
        raise ValueError("repeats must be positive.")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative.")
    if text_seq_len <= 0:
        raise ValueError("text_seq_len must be positive.")

    root_path = Path(root).expanduser().resolve()
    output_dir = root_path / "flux_2_klein_4b" / "eval" / "denoiser_speed"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = root_path / "flux_2_klein_4b" / "base" / "transformer"
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"Evaluating denoiser speed on {target_device}")
    print(f"Writing results to {output_dir}")

    fixture = _build_benchmark_fixture(
        root=root_path,
        width=width,
        height=height,
        scheduler_steps=scheduler_steps,
        timestep_index=timestep_index,
        text_seq_len=text_seq_len,
        seed=seed,
        device=target_device,
    )

    summary_rows: list[dict[str, float | int | str | None]] = []
    baseline_seen = False

    for config_index, config in enumerate(DEFAULT_CONFIGS, start=1):
        print(f"[{config_index}/{len(DEFAULT_CONFIGS)}] Loading {config.label}")
        if config.label == BASELINE_CONFIG_LABEL:
            baseline_seen = True

        denoiser, load_seconds, model_size_mb = _load_denoiser(
            model_path=model_path,
            config=config,
            device=target_device,
        )
        print("  * benchmarking ...")
        benchmark = _benchmark_denoiser(
            denoiser,
            fixture=fixture,
            repeats=repeats,
            warmup_steps=warmup_steps,
            device=target_device,
        )
        summary_rows.append(
            {
                "label": config.label,
                "model_size_mb": model_size_mb,
                "load_seconds": load_seconds,
                **benchmark,
            }
        )

        del denoiser
        gc.collect()
        if target_device.type == "cuda":
            torch.cuda.empty_cache()

    if not baseline_seen:
        raise RuntimeError(f"Missing baseline config '{BASELINE_CONFIG_LABEL}' in this evaluation run.")

    print("  * summarizing ...")
    summary_path = _timestamped_summary_path(output_dir)
    _write_markdown_summary(
        summary_path,
        device=target_device,
        width=width,
        height=height,
        effective_width=int(fixture["effective_width"]),
        effective_height=int(fixture["effective_height"]),
        text_seq_len=text_seq_len,
        image_seq_len=int(fixture["image_seq_len"]),
        scheduler_steps=scheduler_steps,
        timestep_index=timestep_index,
        repeats=repeats,
        warmup_steps=warmup_steps,
        summary_rows=summary_rows,
    )
    print(f"Wrote markdown summary to {summary_path}")
    return summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Flux 2 Klein 4B denoiser speed for quantized configs.")
    parser.add_argument(
        "--root",
        required=True,
        help="Root folder that contains flux_2_klein_4b/base/transformer, scheduler, and vae.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_WIDTH,
        help=f"Requested image width. Default: {DEFAULT_WIDTH}.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_HEIGHT,
        help=f"Requested image height. Default: {DEFAULT_HEIGHT}.",
    )
    parser.add_argument(
        "--scheduler-steps",
        type=int,
        default=DEFAULT_SCHEDULER_STEPS,
        help=f"Scheduler length used to pick the benchmark timestep. Default: {DEFAULT_SCHEDULER_STEPS}.",
    )
    parser.add_argument(
        "--timestep-index",
        type=int,
        default=DEFAULT_TIMESTEP_INDEX,
        help=f"Index of the single scheduler timestep to benchmark. Default: {DEFAULT_TIMESTEP_INDEX}.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help=f"Number of timed forward-pass repeats at the selected timestep. Default: {DEFAULT_REPEATS}.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=DEFAULT_WARMUP_STEPS,
        help=f"Warmup forward passes at the selected timestep before timing. Default: {DEFAULT_WARMUP_STEPS}.",
    )
    parser.add_argument(
        "--text-seq-len",
        type=int,
        default=DEFAULT_TEXT_SEQ_LEN,
        help=f"Synthetic text sequence length used for conditioning. Default: {DEFAULT_TEXT_SEQ_LEN}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Seed for deterministic synthetic inputs. Default: {DEFAULT_SEED}.",
    )
    parser.add_argument(
        "--device",
        help='Optional device override, for example "cuda" or "cpu".',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_denoiser_speed(
        root=args.root,
        width=args.width,
        height=args.height,
        scheduler_steps=args.scheduler_steps,
        timestep_index=args.timestep_index,
        repeats=args.repeats,
        warmup_steps=args.warmup_steps,
        text_seq_len=args.text_seq_len,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
