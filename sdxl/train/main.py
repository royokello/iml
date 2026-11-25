"""Train an SDXL LoRA on the `q8` quantized UNet.

Run from the repository root:

    python -m sdxl.train.main --root "/iml_root" --project "/iml_root/sdxl/loras/niji" --steps 1024 --cache-images --cache-text

The project folder uses the Flux layout: `images/` holds the images and their
`.txt` captions, `models/` holds `<step>.safetensors`, `<step>.optimizer.pt`,
`steps.csv`, and `stop.txt`. Pass a checkpoint to `sdxl.gen --loras` to sample
the result.
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import deque
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file as safe_load_file
from safetensors.torch import save_file

from flux2.lora.qmodel import (
    TrainableLoraLinear,
    build_lora_state_dict,
    inject_trainable_lora_modules,
)
from flux2.train.fork import STEP_CSV_HEADER, Step
from flux2.train.main import _parse_ma_windows, _plot_loss
from flux3.train.main import format_duration
from sdxl.config import DEFAULT_MODEL_NAME, resolve_model_root
from sdxl.denoiser.loader import load_sdxl_denoiser
from utils.quant.streaming import move_unstreamed_modules_to
from sdxl.lora.config import SDXL_LORA_TARGETS
from sdxl.train.dataset import DEFAULT_TARGET_RESOLUTION, SDXLDataset
from sdxl.train.noise import (
    add_noise,
    alphas_cumprod_from_betas,
    betas_from_config,
    min_snr_weight,
)

DEFAULT_QUANT_METHOD = "q8"
DEFAULT_STEPS = 1024
DEFAULT_SEED = 19930625
DEFAULT_LR = 1e-4
DEFAULT_RANK = 16
DEFAULT_ALPHA = 8
DEFAULT_NOISE_OFFSET = 0.1
DEFAULT_MIN_SNR_GAMMA = 5.0
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_CHECKPOINT_STEPS = 128
LOSS_LOG_FLUSH_STEPS = 8
ETA_WINDOW_STEPS = 32


def _read_metadata(path: Path) -> dict[str, str]:
    """Return the metadata of a safetensors file.

    Args:
        path: The checkpoint.

    Returns:
        The metadata, or an empty dict.
    """
    with safe_open(str(path), framework="pt") as handle:
        return dict(handle.metadata() or {})


def _read_step(path: Path) -> int:
    """Return the step recorded in a checkpoint.

    Args:
        path: The checkpoint.

    Returns:
        The step number.

    Raises:
        RuntimeError: If the file has no `step` metadata.
    """
    metadata = _read_metadata(path)
    if "step" not in metadata:
        raise RuntimeError(f"Checkpoint {path} has no step metadata.")
    return int(metadata["step"])


def _load_lora_state(unet: torch.nn.Module, path: Path) -> int:
    """Copy LoRA tensors from a checkpoint into the injected modules.

    A module that is absent from the file is zeroed, as `flux2/lora/loader.py`
    does. A module with only one factor raises.

    Args:
        unet: The UNet with injected `TrainableLoraLinear` modules.
        path: The checkpoint.

    Returns:
        The number of modules that matched.

    Raises:
        KeyError: If a module holds only one of the two factors.
    """
    state = safe_load_file(str(path), device="cpu")
    matched = 0

    for module_name, module in unet.named_modules():
        if not isinstance(module, TrainableLoraLinear):
            continue

        a_key = f"{module_name}.lora_A.weight"
        b_key = f"{module_name}.lora_B.weight"
        if a_key not in state and b_key not in state:
            module.lora_A.data.zero_()
            module.lora_B.data.zero_()
            continue
        if a_key not in state or b_key not in state:
            raise KeyError(f"Incomplete LoRA weights for {module_name} in {path}")

        module.lora_A.data.copy_(
            state[a_key].to(device=module.lora_A.device, dtype=module.lora_A.dtype)
        )
        module.lora_B.data.copy_(
            state[b_key].to(device=module.lora_B.device, dtype=module.lora_B.dtype)
        )

        alpha = state.get(f"{module_name}.alpha")
        if alpha is not None:
            value = float(alpha.reshape(-1)[0])
            module.alpha = int(value) if value.is_integer() else value
            module.scaling = value / module.rank

        matched += 1

    return matched


def _csv_has_rows(csv_path: Path) -> bool:
    """Return True when a CSV holds more than its header.

    Args:
        csv_path: The `steps.csv` file.

    Returns:
        True when at least one data row is present.
    """
    with csv_path.open(encoding="utf-8") as handle:
        lines = [line for line in handle.read().splitlines() if line.strip()]
    return len(lines) > 1


def _latest_checkpoint(models_dir: Path) -> Path:
    """Return the checkpoint with the highest numeric step in a folder.

    Args:
        models_dir: The `models` folder.

    Returns:
        The checkpoint path.

    Raises:
        FileNotFoundError: If the folder holds no numeric checkpoint.
    """
    checkpoints: list[tuple[int, Path]] = []
    for path in models_dir.glob("*.safetensors"):
        if path.stem.isdigit():
            checkpoints.append((int(path.stem), path))
    if not checkpoints:
        raise FileNotFoundError(f"No numeric step checkpoints in {models_dir}")
    return max(checkpoints, key=lambda item: item[0])[1]


def _truncate_loss_csv(csv_path: Path, step: int) -> None:
    """Drop loss rows after a step.

    Args:
        csv_path: The `steps.csv` file.
        step: The step to resume at.
    """
    if not csv_path.is_file():
        return

    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    kept = [row for row in rows if int(row["step"]) <= step]
    if len(kept) == len(rows):
        return

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(f"{STEP_CSV_HEADER}\n")
        writer = csv.DictWriter(handle, fieldnames=STEP_CSV_HEADER.split(","))
        for row in kept:
            writer.writerow(row)
    print(f"  * truncated steps.csv: removed {len(rows) - len(kept)} row(s)")


def _flush_loss_rows(
    csv_path: Path,
    pending_rows: list[tuple[int, int, int, str, float, float, float]],
    models_dir: Path,
    ma_windows: list[int],
) -> None:
    """Write the pending loss rows and redraw `loss.png`.

    Args:
        csv_path: The `steps.csv` file.
        pending_rows: One tuple per step: step, epoch, sample index, sample
            name, learning rate, loss, gradient norm.
        models_dir: The `models` folder that holds `loss.png`.
        ma_windows: The moving-average windows for the plot.
    """
    if not pending_rows:
        return

    rows: list[Step] = []
    for step, epoch, sample_index, sample_name, learning_rate, loss, grad_norm in pending_rows:
        row = Step()
        row.step = step
        row.epoch = epoch
        row.sample_index = sample_index
        row.sample_name = sample_name
        row.learning_rate = learning_rate
        row.loss = loss
        row.grad_norm = grad_norm

        rows.append(row)
        print(
            f"      logged epoch {row.epoch} sample {row.sample_index} ({row.sample_name}): "
            f"loss {row.loss:.6f}, grad_norm {row.grad_norm:.4f}",
            flush=True,
        )

    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        handle.write("\n".join(row.to_csv_row() for row in rows))
        handle.write("\n")
        handle.flush()

    try:
        _plot_loss(csv_path, models_dir / "loss.png", ma_windows=ma_windows)
    except Exception as exc:
        print(
            f"      warning: loss plot save failed ({exc}); continuing without plot",
            flush=True,
        )

    pending_rows.clear()


def _save_checkpoint(
    unet: torch.nn.Module,
    models_dir: Path,
    step: int,
    metadata: dict[str, str],
    optimizer: torch.optim.Optimizer,
) -> Path:
    """Save the LoRA tensors and the optimizer state as `<step>.safetensors`.

    Older optimizer files are removed, so `models` holds one of each.

    Args:
        unet: The UNet with injected LoRA modules.
        models_dir: The `models` folder.
        step: The step number.
        metadata: The safetensors metadata.
        optimizer: The optimizer to save.

    Returns:
        The checkpoint path.
    """
    state = {
        key: value.contiguous() for key, value in build_lora_state_dict(unet).items()
    }
    checkpoint_path = models_dir / f"{step:06d}.safetensors"
    optimizer_path = models_dir / f"{step:06d}.optimizer.pt"

    save_file(state, str(checkpoint_path), metadata=metadata)
    torch.save(optimizer.state_dict(), str(optimizer_path))

    for other_optimizer_path in models_dir.glob("*.optimizer.pt"):
        if other_optimizer_path != optimizer_path:
            other_optimizer_path.unlink()

    print(f"  * saved optimizer to {optimizer_path}")
    return checkpoint_path


def train(
    root: str | Path,
    project_dir: str | Path,
    name: str = DEFAULT_MODEL_NAME,
    steps: int = DEFAULT_STEPS,
    seed: int = DEFAULT_SEED,
    lr: float = DEFAULT_LR,
    rank: int = DEFAULT_RANK,
    alpha: int = DEFAULT_ALPHA,
    noise_offset: float = DEFAULT_NOISE_OFFSET,
    min_snr_gamma: float = DEFAULT_MIN_SNR_GAMMA,
    max_grad_norm: float = DEFAULT_MAX_GRAD_NORM,
    checkpoint_steps: int = DEFAULT_CHECKPOINT_STEPS,
    target_resolution: int = DEFAULT_TARGET_RESOLUTION,
    target_upscale: bool = False,
    gradient_checkpointing: bool = True,
    stream_denoiser: bool = False,
    cache_images: bool = False,
    cache_text: bool = False,
    text_dropout: float = 0.0,
    trigger: str | None = None,
    resume: bool = False,
    ma_windows: list[int] | None = None,
) -> Path:
    """Train a LoRA on a captioned image folder.

    The text encoders and the VAE run only during the cache pass. The loop
    holds the quantized UNet and the LoRA state.

    Args:
        root: The data root. The model folder is `<root>/sdxl/<name>`.
        project_dir: The LoRA folder. It holds `images` and `models`.
        name: The model folder name, which is also the checkpoint name.
        steps: The total number of optimizer steps.
        seed: The global seed.
        lr: The learning rate.
        rank: The LoRA rank.
        alpha: The LoRA alpha. Scaling is `alpha / rank`.
        noise_offset: The per-channel noise offset. Zero disables it.
        min_snr_gamma: The min-SNR gamma. Zero disables the weighting.
        max_grad_norm: The gradient norm clip.
        checkpoint_steps: Save every N steps.
        target_resolution: The short side of every encoded image.
        target_upscale: Encode images below `target_resolution` instead of
            skipping them.
        gradient_checkpointing: Recompute the block activations in `backward`.
            This lowers peak memory and costs step time.
        stream_denoiser: Keep the quantized weights in pinned host memory and
            copy each layer onto the GPU for its own multiply. Lowers peak
            memory at the cost of host-to-device traffic.
        cache_images: Write and reuse latents.
        cache_text: Write and reuse text embeddings.
        text_dropout: The probability that a tag other than the trigger is
            removed from a caption. The captions are encoded once, before the
            loop, at `ceil(steps / images)` per image, so the encoders stay
            out of the loop.
        trigger: The tag that tag dropout never removes. Required when
            `text_dropout` is above zero.
        resume: Continue from the highest step in `<models>`.
        ma_windows: The moving-average windows for `loss.png`. Defaults to the
            sample count and the checkpoint interval.

    Returns:
        The final checkpoint path.

    Raises:
        RuntimeError: If no CUDA device is available.
        FileNotFoundError: If the model folder, the dataset folder, or a resume
            checkpoint is missing.
        FileExistsError: If `steps.csv` exists and `resume` is False.
        ValueError: If a flag is out of range.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("SDXL training needs a CUDA device.")
    if steps <= 0:
        raise ValueError(f"--steps must be positive, got {steps}.")
    if rank <= 0 or alpha <= 0:
        raise ValueError(f"--rank and --alpha must be positive, got {rank} and {alpha}.")
    if checkpoint_steps <= 0:
        raise ValueError(f"--checkpoint-steps must be positive, got {checkpoint_steps}.")
    if not 0.0 <= text_dropout < 1.0:
        raise ValueError(f"--text-dropout must be in [0, 1), got {text_dropout}.")
    trigger = trigger.strip() if trigger else None
    if text_dropout > 0 and not trigger:
        raise ValueError("--text-dropout needs --trigger.")

    model_root = resolve_model_root(root, name)
    if not model_root.is_dir():
        raise FileNotFoundError(f"Model folder not found: {model_root}")

    project_path = Path(project_dir).expanduser().resolve()
    images_dir = project_path / "images"
    models_dir = project_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    loss_csv = models_dir / "steps.csv"
    stop_file = models_dir / "stop.txt"
    stop_file.write_text(
        "Delete this file to stop training cleanly after the current step.\n",
        encoding="utf-8",
    )
    device = torch.device("cuda")

    print("SDXL LoRA training ...", flush=True)
    print(f"  * model: {model_root}", flush=True)
    print(f"  * project: {project_path}", flush=True)
    print(f"  * quant method: {DEFAULT_QUANT_METHOD}", flush=True)
    print(f"  * stream denoiser: {'on' if stream_denoiser else 'off'}", flush=True)
    print(f"  * rank/alpha: {rank}/{alpha} (scaling {alpha / rank:.3f})", flush=True)
    print(f"  * steps: {steps}", flush=True)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    start_step = 0
    resume_checkpoint: Path | None = None
    if resume:
        resume_checkpoint = _latest_checkpoint(models_dir)
        metadata = _read_metadata(resume_checkpoint)
        if "rank" in metadata and int(metadata["rank"]) != rank:
            raise RuntimeError(
                f"Checkpoint rank {metadata['rank']} does not match --rank {rank}."
            )
        checkpoint_dropout = metadata.get("text_dropout")
        if checkpoint_dropout is not None and float(checkpoint_dropout) != text_dropout:
            print(
                f"  * warning: checkpoint text dropout {checkpoint_dropout} "
                f"differs from --text-dropout {text_dropout:g}",
                flush=True,
            )
        checkpoint_trigger = metadata.get("trigger")
        if checkpoint_trigger is not None and checkpoint_trigger != trigger:
            print(
                f"  * warning: checkpoint trigger {checkpoint_trigger!r} "
                f"differs from --trigger {trigger!r}",
                flush=True,
            )
        start_step = _read_step(resume_checkpoint)
        _truncate_loss_csv(loss_csv, start_step)
        if not loss_csv.exists():
            loss_csv.write_text(f"{STEP_CSV_HEADER}\n", encoding="utf-8")
        print(f"  * resuming at step {start_step} from {resume_checkpoint.name}", flush=True)
    else:
        if loss_csv.exists() and _csv_has_rows(loss_csv):
            raise FileExistsError(
                f"{loss_csv} already holds steps. Use --resume or remove it before a fresh run."
            )
        loss_csv.write_text(f"{STEP_CSV_HEADER}\n", encoding="utf-8")

    dataset = SDXLDataset(
        model_root=model_root,
        dataset_dir=images_dir,
        target_resolution=target_resolution,
        target_upscale=target_upscale,
        cache_images=cache_images,
        cache_text=cache_text,
        text_dropout=text_dropout,
        trigger=trigger,
        text_steps=steps,
        text_seed=seed,
        device=device,
    )
    if len(dataset) == 0:
        raise ValueError(f"No samples in {images_dir}.")

    if ma_windows is None:
        ma_windows = _parse_ma_windows(f"{len(dataset)},{checkpoint_steps}")

    scheduler_config = model_root / "scheduler" / "scheduler_config.json"
    betas = betas_from_config(scheduler_config)
    alphas_cumprod = alphas_cumprod_from_betas(betas).to(device=device)
    num_train_timesteps = len(betas)
    print(f"  * betas: {num_train_timesteps} steps from {scheduler_config.name}", flush=True)

    print("Loading UNet ...", flush=True)
    unet = load_sdxl_denoiser(
        model_root,
        quant_method=DEFAULT_QUANT_METHOD,
        stream_denoiser=stream_denoiser,
    )
    for parameter in unet.parameters():
        parameter.requires_grad = False

    injected_names = inject_trainable_lora_modules(
        unet,
        target_linear_names=SDXL_LORA_TARGETS,
        rank=rank,
        alpha=alpha,
    )
    unet.train()
    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    print(f"  * injected {len(injected_names)} LoRA modules", flush=True)
    print(f"  * gradient checkpointing: {unet.gradient_checkpointing}", flush=True)

    if resume:
        matched = _load_lora_state(unet, resume_checkpoint)
        print(f"  * resumed {matched} LoRA modules from {resume_checkpoint.name}", flush=True)

    if stream_denoiser:
        move_unstreamed_modules_to(unet, device)
    else:
        unet = unet.to(device=device)

    lora_parameters = [parameter for parameter in unet.parameters() if parameter.requires_grad]
    trainable_parameters = sum(parameter.numel() for parameter in lora_parameters)
    print(f"  * trainable parameters: {trainable_parameters:,}", flush=True)

    optimizer = torch.optim.AdamW(
        lora_parameters,
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )

    if resume:
        optimizer_path = resume_checkpoint.with_suffix(".optimizer.pt")
        if optimizer_path.is_file():
            optimizer.load_state_dict(torch.load(str(optimizer_path), map_location="cpu"))
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(device=device)
            print(f"  * resumed optimizer from {optimizer_path.name}", flush=True)

    metadata = {
        "rank": str(rank),
        "alpha": str(alpha),
        "step": "0",
        "base": name,
        "targets": ",".join(SDXL_LORA_TARGETS),
    }
    if text_dropout > 0:
        metadata["text_dropout"] = f"{text_dropout:g}"
        metadata["trigger"] = trigger

    sample_count = len(dataset)
    cursor = start_step % sample_count
    step = start_step
    last_saved_step = start_step
    pending_loss_rows: list[tuple[int, int, int, str, float, float, float]] = []
    recent_step_durations: deque[float] = deque(maxlen=ETA_WINDOW_STEPS)
    training_start = time.perf_counter()

    while step < steps:
        step_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)

        sample_start = time.perf_counter()
        index = cursor % sample_count
        variant = (
            (step // sample_count) % dataset.text_variants
            if dataset.text_variants > 1
            else 0
        )

        latents = dataset.latents[index].unsqueeze(0).to(device=device, dtype=torch.float16)
        prompt_embeds = (
            dataset.prompt_embeds[index][variant]
            .unsqueeze(0)
            .to(device=device, dtype=torch.float16)
        )
        pooled_embeds = (
            dataset.pooled[index][variant]
            .unsqueeze(0)
            .to(device=device, dtype=torch.float16)
        )
        time_ids = dataset.time_ids(index, dtype=prompt_embeds.dtype).to(device=device)

        noise = torch.randn_like(latents)
        timestep = torch.randint(0, num_train_timesteps, (1,), device=device)
        noisy_latents = add_noise(
            latents,
            noise,
            alphas_cumprod,
            timestep,
            noise_offset=noise_offset,
        )

        forward_start = time.perf_counter()
        load_seconds = forward_start - sample_start
        prediction = unet(
            noisy_latents,
            timestep,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs={"text_embeds": pooled_embeds, "time_ids": time_ids},
            return_dict=False,
        )[0]

        weight = min_snr_weight(alphas_cumprod, timestep, min_snr_gamma)
        loss = (weight * (prediction.float() - noise.float()) ** 2).mean()
        forward_seconds = time.perf_counter() - forward_start

        backward_start = time.perf_counter()
        loss.backward()
        backward_seconds = time.perf_counter() - backward_start

        grad_norm = torch.nn.utils.clip_grad_norm_(lora_parameters, max_grad_norm)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"Gradient is inf or NaN at step {step + 1}.")

        optimizer.step()
        step += 1
        cursor += 1

        step_duration = time.perf_counter() - step_start
        loss_value = float(loss.detach())
        epoch = (cursor - 1) // sample_count + 1
        sample_index = (cursor - 1) % sample_count

        pending_loss_rows.append((
            step,
            epoch,
            sample_index + 1,
            dataset.names[sample_index],
            lr,
            loss_value,
            float(grad_norm),
        ))
        if len(pending_loss_rows) >= LOSS_LOG_FLUSH_STEPS:
            _flush_loss_rows(loss_csv, pending_loss_rows, models_dir, ma_windows)

        recent_step_durations.append(step_duration)
        eta_seconds = sum(recent_step_durations) / len(recent_step_durations) * (steps - step)
        print(
            f"    * [{step}/{steps}]: Epoch {epoch}, "
            f"Step {step} done in {step_duration:.3f}s "
            f"(load={load_seconds:.2f}s, fwd={forward_seconds:.2f}s, "
            f"bck={backward_seconds:.2f}s, grad_norm={float(grad_norm):.4f}) "
            f"| eta {format_duration(eta_seconds)}",
            flush=True,
        )

        if step % checkpoint_steps == 0:
            _flush_loss_rows(loss_csv, pending_loss_rows, models_dir, ma_windows)
            metadata["step"] = str(step)
            _save_checkpoint(unet, models_dir, step, metadata, optimizer)
            last_saved_step = step

        if not stop_file.is_file():
            _flush_loss_rows(loss_csv, pending_loss_rows, models_dir, ma_windows)
            print("  * stop.txt deleted - saving final checkpoint ...", flush=True)
            metadata["step"] = str(step)
            final_path = _save_checkpoint(unet, models_dir, step, metadata, optimizer)
            print(f"  * stop checkpoint saved at step {step}. Exiting.", flush=True)
            print(f"  * peak VRAM: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
            return final_path

    _flush_loss_rows(loss_csv, pending_loss_rows, models_dir, ma_windows)
    metadata["step"] = str(step)
    if step == last_saved_step:
        final_path = models_dir / f"{step:06d}.safetensors"
    else:
        final_path = _save_checkpoint(unet, models_dir, step, metadata, optimizer)
    print(
        f"Training done in {time.perf_counter() - training_start:.1f}s. "
        f"Peak VRAM: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB. "
        f"Use `sdxl.gen --loras \"{final_path}:1.0\"`.",
        flush=True,
    )
    return final_path


def parse_args() -> argparse.Namespace:
    """Parse the command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train an SDXL LoRA on the q8 UNet.")
    parser.add_argument("--root", type=Path, required=True, help="The data root.")
    parser.add_argument(
        "--name",
        default=DEFAULT_MODEL_NAME,
        help="The model folder under <root>/sdxl (default: base).",
    )
    parser.add_argument(
        "--project",
        type=Path,
        required=True,
        help="The LoRA folder. It holds images/ and models/.",
    )
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--rank", type=int, default=DEFAULT_RANK)
    parser.add_argument("--alpha", type=int, default=DEFAULT_ALPHA)
    parser.add_argument(
        "--noise-offset",
        dest="noise_offset",
        type=float,
        default=DEFAULT_NOISE_OFFSET,
        help="Per-channel noise offset. Zero disables it (default: 0.1).",
    )
    parser.add_argument(
        "--min-snr-gamma",
        dest="min_snr_gamma",
        type=float,
        default=DEFAULT_MIN_SNR_GAMMA,
        help="Min-SNR gamma. Zero disables it (default: 5).",
    )
    parser.add_argument(
        "--max-grad-norm",
        dest="max_grad_norm",
        type=float,
        default=DEFAULT_MAX_GRAD_NORM,
    )
    parser.add_argument(
        "--checkpoint-steps",
        dest="checkpoint_steps",
        type=int,
        default=DEFAULT_CHECKPOINT_STEPS,
    )
    parser.add_argument(
        "--target-res",
        dest="target_res",
        type=int,
        default=DEFAULT_TARGET_RESOLUTION,
        help="Short side of every encoded image (default: 512).",
    )
    parser.add_argument(
        "--target-upscale",
        dest="target_upscale",
        action="store_true",
        help="Encode images below --target-res instead of skipping them.",
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
        help="Keep the block activations. Faster, but about 2 GB more VRAM.",
    )
    parser.add_argument(
        "--stream-denoiser",
        dest="stream_denoiser",
        action="store_true",
        help=(
            "Keep the quantized weights in pinned host memory and copy each layer "
            "to the GPU for its own multiply. Lowers peak VRAM and costs step time."
        ),
    )
    parser.add_argument("--cache-images", dest="cache_images", action="store_true")
    parser.add_argument("--cache-text", dest="cache_text", action="store_true")
    parser.add_argument(
        "--text-dropout",
        dest="text_dropout",
        type=float,
        default=0.0,
        help=(
            "Probability that a tag other than the trigger is removed from a "
            "caption. Captions are encoded once before the loop, one per image "
            "visit (default: 0)."
        ),
    )
    parser.add_argument(
        "--trigger",
        default=None,
        help="The tag that --text-dropout never removes. Required with it.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--moving-averages",
        "--ma",
        dest="ma_windows",
        type=str,
        default=None,
        help=(
            "Comma-separated moving average window sizes for the loss plot, up to 3 values "
            "(default: the sample count and the checkpoint interval)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run training from the command line."""
    args = parse_args()
    train(
        root=args.root,
        project_dir=args.project,
        name=args.name,
        steps=args.steps,
        seed=args.seed,
        lr=args.lr,
        rank=args.rank,
        alpha=args.alpha,
        noise_offset=args.noise_offset,
        min_snr_gamma=args.min_snr_gamma,
        max_grad_norm=args.max_grad_norm,
        checkpoint_steps=args.checkpoint_steps,
        target_resolution=args.target_res,
        target_upscale=args.target_upscale,
        gradient_checkpointing=args.gradient_checkpointing,
        stream_denoiser=args.stream_denoiser,
        cache_images=args.cache_images,
        cache_text=args.cache_text,
        text_dropout=args.text_dropout,
        trigger=args.trigger,
        resume=args.resume,
        ma_windows=_parse_ma_windows(args.ma_windows) if args.ma_windows else None,
    )


if __name__ == "__main__":
    main()
