"""LoRA training for Flux 3 Klein. Run with `python -m flux3.train.main`."""

from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import torch
from safetensors.torch import load_file as safe_load_file
from safetensors.torch import save_file
from torch.utils.checkpoint import checkpoint as checkpoint_fn

from flux2.lora.loader import load_checkpoint as load_lora_checkpoint
from flux2.lora.model import build_lora_state_dict, inject_trainable_lora_modules
from flux2.train.fork import DEFAULT_MAX_GRAD_NORM, STEP_CSV_HEADER, Step
from flux2.train.main import _ensure_grad_norm_column, _parse_ma_windows, _plot_loss, _truncate_logs_to_step
from flux3.config import SCHEDULER_SHIFT, resolve_model_root
from flux3.denoiser.loader import load_flux3_denoiser
from flux3.lora.config import FLUX3_LORA_ALPHA, FLUX3_LORA_RANK, FLUX3_LORA_TARGETS
from flux3.train.dataset import Flux3Dataset

LOSS_LOG_FLUSH_STEPS = 8
ETA_WINDOW_STEPS = 32
MODEL_METADATA_NAME = "flux 3 klein"


def format_duration(seconds: float) -> str:
    """Format a duration as `H:MM:SS`, or `D:HH:MM:SS` from 24 hours.

    Args:
        seconds: The duration in seconds.

    Returns:
        The formatted duration.
    """
    total = int(round(seconds))
    days, rest = divmod(total, 86400)
    hours, rest = divmod(rest, 3600)
    minutes, secs = divmod(rest, 60)
    if days:
        return f"{days}:{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours}:{minutes:02d}:{secs:02d}"


def sample_sigma(uniform: torch.Tensor) -> torch.Tensor:
    """Map uniform samples to shifted sigmas.

    Args:
        uniform: Samples from U(0, 1).

    Returns:
        `shift * u / (1 + (shift - 1) * u)` with the Flux 3 shift.
    """
    return SCHEDULER_SHIFT * uniform / (1 + (SCHEDULER_SHIFT - 1) * uniform)


def _epoch_order(dataset_size: int, shuffle: bool, seed: int, epoch: int) -> list[int]:
    if not shuffle:
        return list(range(dataset_size))
    generator = torch.Generator("cpu").manual_seed(seed + epoch)
    return torch.randperm(dataset_size, generator=generator).tolist()


def main(
    project_dir: Path,
    checkpoint_metadata: dict[str, str],
    model_root: Path,
    text_quant: str | None,
    denoiser_quant: str | None,
    trigger: str,
    steps: int,
    target_resolution: int,
    reference_resolution: int,
    target_upscale: bool,
    ref_upscale: bool,
    rank: int,
    alpha: int,
    resume: bool,
    cache_text: bool,
    cache_images: bool,
    lr: float,
    lora_init_checkpoint: Path | None = None,
    seed: int = 19930625,
    ma_windows: list[int] | None = None,
    checkpoint_steps: int = 256,
    max_length: int = 512,
    max_grad_norm: float = DEFAULT_MAX_GRAD_NORM,
    shuffle: bool = False,
) -> None:
    ma_windows = ma_windows or [32]
    print("Training Started ...")
    print(f"  * seed: {seed}")
    print(f"  * shuffle: {shuffle}")
    print(f"  * max grad norm: {max_grad_norm}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pending_loss_logs: list[tuple[int, int, int, str, float, torch.Tensor, torch.Tensor]] = []
    recent_step_durations: deque[float] = deque(maxlen=ETA_WINDOW_STEPS)

    current_step = 0
    current_epoch = 1
    learning_rate = lr

    models_dirpath = project_dir / "models"
    models_dirpath.mkdir(parents=True, exist_ok=True)
    stop_filepath = models_dirpath / "stop.txt"
    stop_filepath.write_text("Delete this file to stop training cleanly after the current step.\n")
    logs_filepath = models_dirpath / "steps.csv"
    resume_checkpoint_path: Path | None = None
    resume_optimizer_filepath: Path | None = None

    if logs_filepath.exists() and not resume:
        raise FileExistsError(
            f"{logs_filepath} already exists. Use --resume to continue from the latest checkpoint "
            "or remove the CSV before starting a fresh run."
        )

    if resume:
        checkpoints: list[tuple[int, Path]] = []
        for checkpoint_path in models_dirpath.glob("*.safetensors"):
            if not checkpoint_path.stem.isdigit():
                continue
            checkpoints.append((int(checkpoint_path.stem), checkpoint_path))
        checkpoints.sort(key=lambda item: item[0])
        if not checkpoints:
            raise FileNotFoundError(f"No numeric step checkpoints found in {models_dirpath}")

        current_step, resume_checkpoint_path = checkpoints[-1]
        resume_optimizer_filepath = resume_checkpoint_path.with_suffix(".optimizer.pt")
        print(f"  * resuming from checkpoint: {resume_checkpoint_path}")
        print(f"  * resume step: {current_step}")
        _ensure_grad_norm_column(logs_filepath)
        _truncate_logs_to_step(logs_filepath, current_step)
    else:
        with logs_filepath.open("w", encoding="utf-8", newline="") as logs_handle:
            logs_handle.write(f"{STEP_CSV_HEADER}\n")

    if current_step >= steps:
        print(f"  * latest checkpoint step {current_step} is already at max steps {steps}; nothing to train")
        _plot_loss(logs_filepath, models_dirpath / "loss.png", ma_windows=ma_windows)
        return

    # LOAD DATASET
    dataset = Flux3Dataset(
        model_root,
        text_quant,
        Path(project_dir) / "images",
        trigger,
        target_resolution,
        reference_resolution,
        cache_text=cache_text,
        cache_images=cache_images,
        target_upscale=target_upscale,
        ref_upscale=ref_upscale,
        max_length=max_length,
    )
    dataset_size = len(dataset.target_latents)
    if dataset_size == 0:
        raise ValueError("Training dataset has no target image latents.")
    current_epoch = (current_step // dataset_size) + 1
    first_sample_index = current_step % dataset_size

    # LOAD TRANSFORMER
    denoiser_load_start = time.perf_counter()
    transformer = load_flux3_denoiser(model_root / "transformer", denoiser_quant, "base")
    for parameter in transformer.parameters():
        parameter.requires_grad = False
    lora_module_names = inject_trainable_lora_modules(
        transformer,
        target_linear_names=FLUX3_LORA_TARGETS,
        rank=rank,
        alpha=alpha,
    )
    transformer.train()

    init_optimizer_filepath: Path | None = None
    if lora_init_checkpoint is not None:
        if not lora_init_checkpoint.is_file():
            raise FileNotFoundError(f"Init LoRA checkpoint not found: {lora_init_checkpoint}")

        init_state = safe_load_file(str(lora_init_checkpoint), device="cpu")
        first_lora_a = next((key for key in init_state if key.endswith(".lora_A.weight")), None)
        if first_lora_a is not None:
            checkpoint_rank = init_state[first_lora_a].shape[0]
            if checkpoint_rank != rank:
                print(f"  * WARNING: checkpoint rank {checkpoint_rank} != configured rank {rank}")
        del init_state

        load_lora_checkpoint(transformer, lora_init_checkpoint)
        print(f"  * initialized from checkpoint: {lora_init_checkpoint}")

        possible_optimizer = lora_init_checkpoint.with_suffix(".optimizer.pt")
        if possible_optimizer.is_file():
            init_optimizer_filepath = possible_optimizer
            print(f"  * found init optimizer: {init_optimizer_filepath}")

    if resume_checkpoint_path is not None:
        load_lora_checkpoint(transformer, resume_checkpoint_path)

    def non_reentrant_checkpoint(module, *inputs):
        return checkpoint_fn(module, *inputs, use_reentrant=False)

    transformer.enable_gradient_checkpointing(gradient_checkpointing_func=non_reentrant_checkpoint)
    print(f"  * transformer loaded in {time.perf_counter() - denoiser_load_start:.3f}s")

    print(f"  * lora targets: {', '.join(FLUX3_LORA_TARGETS)}")
    print(f"  * injected lora modules: {len(lora_module_names)}")
    print(f"  * lora rank/alpha: r={rank}, alpha={alpha}")
    print(f"  * gradient checkpointing: {transformer.is_gradient_checkpointing}")
    lora_parameters = [parameter for parameter in transformer.parameters() if parameter.requires_grad]
    print(f"  * trainable params: {sum(parameter.numel() for parameter in lora_parameters):,}")
    print(f"  * lora parameter dtype: {lora_parameters[0].dtype}")

    optimizer = torch.optim.AdamW(lora_parameters, lr=learning_rate)

    optimizer_state_dict = None
    optimizer_path = resume_optimizer_filepath or init_optimizer_filepath
    if optimizer_path is not None:
        if optimizer_path.is_file():
            optimizer_state_dict = torch.load(str(optimizer_path), map_location="cpu")
            print(f"  * loaded optimizer state: {optimizer_path}")
        else:
            print(f"  * optimizer state not found, using a new optimizer: {optimizer_path}")

    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to("cuda")

    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate

    transformer = transformer.to("cuda")

    def flush_pending_loss_logs() -> None:
        if not pending_loss_logs:
            return

        new_steps = []
        for (
            log_total_step,
            log_epoch_number,
            log_sample_number,
            log_sample_name,
            log_learning_rate,
            pending_loss,
            pending_grad_norm,
        ) in pending_loss_logs:
            step = Step()
            step.step = log_total_step
            step.epoch = log_epoch_number
            step.sample_index = log_sample_number
            step.sample_name = log_sample_name
            step.learning_rate = float(log_learning_rate)
            step.loss = float(pending_loss.item())
            step.grad_norm = float(pending_grad_norm.item())

            new_steps.append(step)
            print(
                f"      logged epoch {step.epoch} sample {step.sample_index} ({step.sample_name}): "
                f"loss {step.loss:.6f}, grad_norm {step.grad_norm:.4f}"
            )

        with logs_filepath.open("a", encoding="utf-8", newline="") as logs_handle:
            logs_handle.write("\n".join(step.to_csv_row() for step in new_steps))
            logs_handle.write("\n")
            logs_handle.flush()

        try:
            _plot_loss(logs_filepath, models_dirpath / "loss.png", ma_windows=ma_windows)
        except Exception as exc:
            print(f"      warning: loss plot save failed ({exc}); continuing without plot")

        pending_loss_logs.clear()

    def save_checkpoint() -> None:
        checkpoint_filepath = models_dirpath / f"{current_step:06d}.safetensors"
        save_file(build_lora_state_dict(transformer), str(checkpoint_filepath), metadata=checkpoint_metadata)
        optimizer_filepath = checkpoint_filepath.with_suffix(".optimizer.pt")
        torch.save(optimizer.state_dict(), str(optimizer_filepath))
        print(f"  * saved optimizer to {optimizer_filepath}")
        for other_optimizer_filepath in models_dirpath.glob("*.optimizer.pt"):
            if other_optimizer_filepath != optimizer_filepath:
                other_optimizer_filepath.unlink()

    while current_step < steps:

        epoch_order = _epoch_order(dataset_size, shuffle, seed, current_epoch)

        for sample_index in epoch_order[first_sample_index:]:

            sample_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)

            text_embedding = dataset.text_embeddings[sample_index]
            if text_embedding is None:
                text_embedding = dataset.trigger_embedding
            prompt_embeds = text_embedding[0].unsqueeze(0).to("cuda", torch.float16)
            text_ids = text_embedding[1].to("cuda")

            target_tokens, target_ids = dataset.target_latents[sample_index]
            x0 = target_tokens.unsqueeze(0).to("cuda", torch.float32)
            img_ids = target_ids.to("cuda")

            cond = None
            cond_ids = None
            ref_latents = dataset.ref_latents[sample_index]
            if ref_latents is not None:
                cond = ref_latents[0].unsqueeze(0).to("cuda", torch.float16)
                cond_ids = ref_latents[1].to("cuda")

            noise = torch.randn_like(x0)
            sigma = sample_sigma(torch.rand((1,), device="cuda", dtype=torch.float32))
            sigma_mix = sigma.view(-1, 1, 1)
            noisy = (1.0 - sigma_mix) * x0 + sigma_mix * noise
            target = noise - x0

            forward_start = time.perf_counter()
            load_duration = forward_start - sample_start
            prediction = transformer(
                noisy.to(torch.float16),
                encoder_hidden_states=prompt_embeds,
                timestep=sigma,
                img_ids=img_ids,
                txt_ids=text_ids,
                image_cond_hidden_states=cond,
                image_cond_ids=cond_ids,
                return_dict=False,
            )[0]

            loss = torch.nn.functional.mse_loss(prediction.float(), target)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Flux 3 training loss is inf or NaN at step {current_step + 1}.")

            backward_start = time.perf_counter()
            forward_duration = backward_start - forward_start
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(lora_parameters, max_grad_norm)
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"Flux 3 training gradient is inf or NaN at step {current_step + 1}.")
            backward_duration = time.perf_counter() - backward_start

            optimizer.step()
            sample_total_duration = time.perf_counter() - sample_start

            current_step += 1
            recent_step_durations.append(sample_total_duration)
            eta_seconds = sum(recent_step_durations) / len(recent_step_durations) * (steps - current_step)

            pending_loss_logs.append((
                current_step,
                current_epoch,
                sample_index + 1,
                dataset.sample_names[sample_index],
                learning_rate,
                loss.detach(),
                grad_norm.detach(),
            ))

            if len(pending_loss_logs) >= LOSS_LOG_FLUSH_STEPS:
                flush_pending_loss_logs()

            print(
                f"    * [{current_step}/{steps}]: Epoch {current_epoch}, "
                f"Step {current_step} done in {sample_total_duration:.3f}s "
                f"(load={load_duration:.2f}s, fwd={forward_duration:.2f}s, "
                f"bck={backward_duration:.2f}s, grad_norm={grad_norm.item():.4f}) "
                f"| eta {format_duration(eta_seconds)}"
            )

            if current_step % checkpoint_steps == 0:
                flush_pending_loss_logs()
                save_checkpoint()

            if not stop_filepath.is_file():
                flush_pending_loss_logs()
                print("  * stop.txt deleted - saving final checkpoint ...")
                save_checkpoint()
                print(f"  * stop checkpoint saved at step {current_step}. Exiting.")
                return

            del (
                prompt_embeds,
                text_ids,
                x0,
                img_ids,
                cond,
                cond_ids,
                noise,
                sigma,
                sigma_mix,
                noisy,
                target,
                prediction,
                loss,
                grad_norm,
            )

            if current_step >= steps:
                flush_pending_loss_logs()
                break

        first_sample_index = 0
        current_epoch += 1

    _plot_loss(logs_filepath, models_dirpath / "loss.png", ma_windows=ma_windows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flux 3 Klein LoRA training (base variant).")
    parser.add_argument("--root", type=Path, required=True, help="Root folder that contains flux3/model.")
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--text-quant", default="sym-high-mini", help='Text encoder quantization. "none" loads fp16.')
    parser.add_argument("--denoiser-quant", default="aff-med-mini", help='Denoiser quantization. "none" loads fp16.')
    parser.add_argument("--cache-text", action="store_true")
    parser.add_argument("--cache-images", action="store_true")
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the sample order in each epoch. The order comes from --seed plus the epoch number.",
    )
    parser.add_argument("--trigger", type=str, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="LoRA start checkpoint. Optionally loads optimizer from the same directory with a .optimizer.pt suffix.",
    )
    parser.add_argument("--rank", type=int, default=FLUX3_LORA_RANK)
    parser.add_argument("--alpha", type=int, default=FLUX3_LORA_ALPHA)
    parser.add_argument("--target-res", dest="target_res", type=int, default=512)
    parser.add_argument("--target-upscale", action="store_true")
    parser.add_argument("--ref-res", dest="ref_res", type=int, default=512)
    parser.add_argument("--ref-upscale", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 0.0001)")
    parser.add_argument(
        "--seed",
        type=int,
        default=19930625,
        help="Global seed for reproducible noise sampling (default: 19930625)",
    )
    parser.add_argument(
        "--moving-averages",
        "--ma",
        dest="ma_windows",
        type=str,
        default="32",
        help="Comma-separated moving average window sizes for the loss plot, up to 3 values (default: 32).",
    )
    parser.add_argument(
        "--checkpoint-steps", type=int, default=256, help="Save checkpoint every N steps (default: 256)"
    )
    parser.add_argument("--max-length", type=int, default=512, help="Maximum text tokens (default: 512)")
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=DEFAULT_MAX_GRAD_NORM,
        help=f"Clip the gradient norm to this value. Use inf to disable (default: {DEFAULT_MAX_GRAD_NORM})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ma_windows = _parse_ma_windows(args.ma_windows)

    print("1. Run training epochs ...")

    checkpoint_metadata = {
        "model": MODEL_METADATA_NAME,
        "rank": str(args.rank),
        "alpha": str(args.alpha),
        "trigger": args.trigger,
    }

    training_start = time.perf_counter()
    print(f"  * max steps: {args.steps}")

    main(
        project_dir=args.project,
        checkpoint_metadata=checkpoint_metadata,
        model_root=resolve_model_root(args.root),
        text_quant=None if args.text_quant == "none" else args.text_quant,
        denoiser_quant=None if args.denoiser_quant == "none" else args.denoiser_quant,
        trigger=args.trigger,
        steps=args.steps,
        resume=args.resume,
        cache_text=args.cache_text,
        cache_images=args.cache_images,
        lora_init_checkpoint=args.model,
        rank=args.rank,
        alpha=args.alpha,
        lr=args.lr,
        target_resolution=args.target_res,
        reference_resolution=args.ref_res,
        target_upscale=args.target_upscale,
        ref_upscale=args.ref_upscale,
        seed=args.seed,
        ma_windows=ma_windows,
        checkpoint_steps=args.checkpoint_steps,
        max_length=args.max_length,
        max_grad_norm=args.max_grad_norm,
        shuffle=args.shuffle,
    )

    print(f"  * training done in {time.perf_counter() - training_start:.3f}s")
