from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any, List

import torch
from torch.utils.checkpoint import checkpoint as checkpoint_fn

from flux2.denoiser.loader import _load_flux2_denoiser as load_flux2_denoiser
from flux2.lora.config import FLUX2_LORA_ALPHA, FLUX2_LORA_RANK, FLUX2_LORA_TARGETS
from flux2.lora.loader import load_checkpoint as load_lora_checkpoint
from flux2.lora.model import build_lora_state_dict, inject_trainable_lora_modules
from flux2.train.dataset import Flux2Dataset
from flux2.train.logs import save_training_graph

from safetensors.torch import load_file as safe_load_file
from safetensors.torch import save_file

CHECKPOINT_AFTER_STEPS = 64
LOSS_LOG_FLUSH_STEPS = 8

from flux2.train.fork import Checkpoint, STEP_CSV_HEADER, Step


def _load_steps_csv(logs_filepath: Path) -> list[Step]:
    if not logs_filepath.is_file():
        return []

    base_fields = {
        "datetime",
        "step",
        "epoch",
        "sample",
        "learning rate",
        "loss",
    }
    steps: list[Step] = []

    with logs_filepath.open("r", encoding="utf-8", newline="") as logs_handle:
        reader = csv.DictReader(logs_handle)
        fieldnames = set(reader.fieldnames or [])
        missing_base = base_fields - fieldnames
        if missing_base:
            missing_text = ", ".join(sorted(missing_base))
            raise ValueError(f"Missing columns in {logs_filepath}: {missing_text}")

        has_ma16_ma64 = "ma16" in fieldnames and "ma64" in fieldnames

        for row_number, row in enumerate(reader, start=2):
            if not any(row.values()):
                continue

            try:
                step = Step(row["datetime"])
                step.step = int(row["step"])
                step.epoch = int(row["epoch"])
                step.sample = int(row["sample"])
                step.learning_rate = float(row["learning rate"])
                step.loss = float(row["loss"])
                if has_ma16_ma64:
                    step.ma16 = float(row["ma16"])
                    step.ma64 = float(row["ma64"])
                else:
                    # ma64 missing — reconstruct both from losses.
                    # Covers: moving_average, ma16+ma32, and any legacy format.
                    step.ma16 = 0.0
                    step.ma64 = 0.0
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid step row {row_number} in {logs_filepath}: {row}") from exc

            steps.append(step)

    # Reconstruct ma16 and ma64 from raw losses if either column was missing
    if not has_ma16_ma64:
        for i, step in enumerate(steps):
            start_16 = max(0, i - 15)
            window_16 = [steps[j].loss for j in range(start_16, i + 1)]
            step.ma16 = float(sum(window_16) / len(window_16))
            start_64 = max(0, i - 63)
            window_64 = [steps[j].loss for j in range(start_64, i + 1)]
            step.ma64 = float(sum(window_64) / len(window_64))

    return steps


def _write_steps_csv(logs_filepath: Path, steps: list[Step]) -> None:
    with logs_filepath.open("w", encoding="utf-8", newline="") as logs_handle:
        logs_handle.write(f"{STEP_CSV_HEADER}\n")
        if steps:
            logs_handle.write("\n".join(step.to_csv_row() for step in steps))
            logs_handle.write("\n")


def run_training(
    transformer: Any,
    project_dir: Path,
    checkpoint_metadata: dict[str, str],
    root_path,
    model_version,
    text_quant_method,
    denoiser_quant_method,
    trigger,
    max_steps: int,
    resume: bool = False,
    cache_text: bool = False,
    cache_images: bool = False,
    lora_init_checkpoint: Path | None = None,
    prompt_step: int | None = None,
    high_res_step: int | None = None,
    ref_image_step: int | None = None,
    lora_rank: int = 32,
    lora_alpha: int = 32,
):
    print("Training Started ...")
    pending_loss_logs: list[tuple[int, int, int, float, torch.Tensor]] = []

    current_total_step = 0
    current_epoch = 1
    current_learning_rate = 1e-4
    checkpoints: List[Checkpoint] = []
    steps: List[Step] = []

    models_dirpath = project_dir / "models"
    models_dirpath.mkdir(parents=True, exist_ok=True)
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

        current_total_step, resume_checkpoint_path = checkpoints[-1]
        resume_optimizer_filepath = resume_checkpoint_path.with_suffix(".optimizer.pt")
        steps = [
            step
            for step in _load_steps_csv(logs_filepath)
            if step.step <= current_total_step
        ]
        _write_steps_csv(logs_filepath, steps)
        save_training_graph(steps, project_dir)
        print(f"  * resuming from checkpoint: {resume_checkpoint_path}")
        print(f"  * resume step: {current_total_step}")
        print(f"  * retained csv rows through step {current_total_step}: {len(steps)}")
    else:
        _write_steps_csv(logs_filepath, steps)

    if current_total_step >= max_steps:
        print(f"  * latest checkpoint step {current_total_step} is already at max steps {max_steps}; nothing to train")
        return

    # LOAD DATASET
    dataset = Flux2Dataset(
        model_rootpath=root_path,
        dataset_dirpath=Path(project_dir) / "images",
        text_quant_method=text_quant_method,
        trigger=trigger,
        cache_text=cache_text,
        cache_images=cache_images,
    )
    dataset_size = len(dataset.base_target_latents)
    if dataset_size == 0:
        raise ValueError("Training dataset has no target image latents.")
    current_epoch = (current_total_step // dataset_size) + 1
    first_sample_index = current_total_step % dataset_size

    # LOAD TRANSFORMER
    denoiser_load_start = time.perf_counter()
    transformer_path = root_path / "transformer"
    transformer = load_flux2_denoiser(
        str(transformer_path),
        quant_method=denoiser_quant_method,
        variant="base",
        version=model_version,
    )
    for parameter in transformer.parameters():
        parameter.requires_grad = False
    lora_module_names = inject_trainable_lora_modules(
        transformer,
        target_linear_names=FLUX2_LORA_TARGETS,
        rank=lora_rank,
        alpha=lora_alpha,
    )
    transformer.train()

    init_optimizer_filepath: Path | None = None
    if lora_init_checkpoint is not None:
        if not lora_init_checkpoint.is_file():
            raise FileNotFoundError(f"Init LoRA checkpoint not found: {lora_init_checkpoint}")

        # Validate rank matches the configured rank
        init_state = safe_load_file(str(lora_init_checkpoint), device="cpu")
        first_lora_a = next(
            (key for key in init_state if key.endswith(".lora_A.weight")),
            None,
        )
        if first_lora_a is not None:
            checkpoint_rank = init_state[first_lora_a].shape[0]
            if checkpoint_rank != lora_rank:
                print(
                    f"  * WARNING: checkpoint rank {checkpoint_rank} != "
                    f"configured rank {lora_rank}"
                )
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

    print(f"  * initial lora targets: {', '.join(FLUX2_LORA_TARGETS)}")
    print(f"  * injected lora modules: {len(lora_module_names)}")
    print(f"  * lora rank/alpha: r={lora_rank}, alpha={lora_alpha}")
    print(f"  * gradient checkpointing: {transformer.is_gradient_checkpointing}")
    trainable_params = sum(parameter.numel() for parameter in transformer.parameters() if parameter.requires_grad)
    print(f"  * trainable params: {trainable_params:,}")
    lora_parameters = [parameter for parameter in transformer.parameters() if parameter.requires_grad]
    print(f"  * lora parameter dtype: {lora_parameters[0].dtype}")

    optimizer = torch.optim.AdamW(
        [parameter for parameter in transformer.parameters() if parameter.requires_grad],
        lr=current_learning_rate,
    )

    optimizer_state_dict = None
    optimizer_path = resume_optimizer_filepath or init_optimizer_filepath
    if optimizer_path is not None:
        if optimizer_path.is_file():
            optimizer_state_dict = torch.load(str(optimizer_path), map_location="cpu")
            print(f"  * loaded optimizer state: {optimizer_path}")
        else:
            print(f"  * optimizer state not found, using a new optimizer: {optimizer_path}")

    def move_optimizer_state_to_cuda() -> None:
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to("cuda")

    def set_optimizer_learning_rate() -> None:
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_learning_rate

    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
        move_optimizer_state_to_cuda()

    set_optimizer_learning_rate()

    transformer = transformer.to("cuda")

    def flush_pending_loss_logs() -> None:
        if not pending_loss_logs:
            return

        logs_filepath = models_dirpath / "steps.csv"
        new_steps = []
        for (
            log_total_step,
            log_epoch_number,
            log_sample_number,
            log_learning_rate,
            pending_loss,
        ) in pending_loss_logs:
            loss_value = pending_loss.item()
            step = Step()
            step.step = log_total_step
            step.epoch = log_epoch_number
            step.sample = log_sample_number
            step.learning_rate = float(log_learning_rate)
            step.loss = float(loss_value)

            # MA16: average of last 16 losses (including this one)
            prev_16 = [s.loss for s in steps[-(16-1):]] if steps else []
            prev_16.append(step.loss)
            step.ma16 = float(sum(prev_16) / len(prev_16))

            # MA64: average of last 64 losses (including this one)
            prev_64 = [s.loss for s in steps[-(64-1):]] if steps else []
            prev_64.append(step.loss)
            step.ma64 = float(sum(prev_64) / len(prev_64))

            steps.append(step)
            new_steps.append(step)
            print(
                f"      logged epoch {step.epoch} sample {step.sample}: "
                f"loss {step.loss:.6f}, ma16 {step.ma16:.6f}, ma64 {step.ma64:.6f}"
            )

        with logs_filepath.open("a", encoding="utf-8", newline="") as logs_handle:
            logs_handle.write("\n".join(step.to_csv_row() for step in new_steps))
            logs_handle.write("\n")
            logs_handle.flush()

        save_training_graph(steps, project_dir)
        pending_loss_logs.clear()

    while current_total_step < max_steps:

        use_high = False
        if high_res_step is not None and current_total_step >= high_res_step:
            use_high = (current_epoch % 2 == 0)
            if use_high:
                has_high_any = any(h is not None for h in dataset.high_target_latents)
                if not has_high_any:
                    use_high = False

        new_lr = 5e-5 if use_high else 1e-4
        if current_learning_rate != new_lr:
            current_learning_rate = new_lr
            set_optimizer_learning_rate()
            print(f"    * lr adjusted to {current_learning_rate} (high={'yes' if use_high else 'no'})")

        for sample_index in range(first_sample_index, dataset_size):
            sample_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)

            embed = dataset.text_embeds_list[sample_index]
            text_id = dataset.text_ids_list[sample_index]
            if prompt_step is None or current_total_step < prompt_step or embed is None:
                embed = dataset.trigger_embed
                text_id = dataset.trigger_id
            prompt_embeds_batch = embed.unsqueeze(0).to(
                device="cuda",
                dtype=transformer.dtype,
            )
            text_ids_batch = text_id.unsqueeze(0).to(device="cuda")

            if use_high and dataset.high_target_latents[sample_index] is not None:
                image_latents = dataset.high_target_latents[sample_index]
                image_latent_ids = dataset.high_target_latent_ids[sample_index]
            else:
                image_latents = dataset.base_target_latents[sample_index]
                image_latent_ids = dataset.base_target_latent_ids[sample_index]
            image_latents_batch = image_latents.unsqueeze(0).to(device="cuda")
            image_latent_ids_batch = image_latent_ids.unsqueeze(0).to(device="cuda")

            reference_latents_batch = None
            reference_latent_ids_batch = None
            if ref_image_step is not None and current_total_step >= ref_image_step:
                if use_high and dataset.high_ref_latents[sample_index] is not None:
                    ref_latents = dataset.high_ref_latents[sample_index]
                    ref_ids = dataset.high_ref_latent_ids[sample_index]
                else:
                    ref_latents = dataset.base_ref_latents[sample_index]
                    ref_ids = dataset.base_ref_latent_ids[sample_index]
                if ref_latents is not None and ref_ids is not None:
                    reference_latents_batch = ref_latents.unsqueeze(0).to(
                        device="cuda",
                        dtype=image_latents_batch.dtype,
                    )
                    reference_latent_ids_batch = ref_ids.unsqueeze(0).to(device="cuda")

            noise = torch.randn_like(image_latents_batch)
            timestep = torch.rand((1,), device="cuda", dtype=image_latents_batch.dtype) * 1000.0
            sigma = (timestep / 1000.0).view(-1, 1, 1)

            noisy_latents = (1.0 - sigma) * image_latents_batch + sigma * noise
            target = noise - image_latents_batch
            model_hidden_states = noisy_latents
            model_img_ids = image_latent_ids_batch
            if reference_latents_batch is not None and reference_latent_ids_batch is not None:
                model_hidden_states = torch.cat([noisy_latents, reference_latents_batch], dim=1)
                model_img_ids = torch.cat([image_latent_ids_batch, reference_latent_ids_batch], dim=1)

            forward_start = time.perf_counter()
            load_duration = forward_start - sample_start
            noise_pred = transformer(
                hidden_states=model_hidden_states.to(dtype=transformer.dtype),
                timestep=timestep / 1000,
                guidance=None,
                encoder_hidden_states=prompt_embeds_batch,
                txt_ids=text_ids_batch,
                img_ids=model_img_ids,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]
            noise_pred = noise_pred[:, : image_latents_batch.size(1)]

            loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float())

            backward_start = time.perf_counter()
            forward_duration = backward_start - forward_start
            loss.backward()
            backward_duration = time.perf_counter() - backward_start

            optimizer.step()
            sample_total_duration = time.perf_counter() - sample_start

            current_total_step += 1

            pending_loss_logs.append((
                current_total_step,
                current_epoch,
                sample_index + 1,
                current_learning_rate,
                loss.detach(),
            ))

            if len(pending_loss_logs) >= LOSS_LOG_FLUSH_STEPS:
                flush_pending_loss_logs()

            print(
                f"    * [{current_total_step}/{max_steps}]: Epoch {current_epoch}, "
                f"Step {current_total_step} done in {sample_total_duration:.3f}s "
                f"(load={load_duration:.2f}s, fwd={forward_duration:.2f}s, "
                f"bck={backward_duration:.2f}s, total={sample_total_duration:.2f}s)"
            )

            if current_total_step % CHECKPOINT_AFTER_STEPS == 0:
                flush_pending_loss_logs()
                checkpoint_filepath = models_dirpath / f"{current_total_step:06d}.safetensors"
                lora_state_dict = build_lora_state_dict(transformer)
                save_file(lora_state_dict, str(checkpoint_filepath), metadata=checkpoint_metadata)
                latest_step = steps[-1]
                checkpoint = Checkpoint()
                checkpoint.steps = current_total_step
                checkpoint.ma16 = latest_step.ma16
                checkpoints.append(checkpoint)

                optimizer_filepath = checkpoint_filepath.with_suffix(".optimizer.pt")
                torch.save(optimizer.state_dict(), str(optimizer_filepath))
                print(f"  * saved optimizer to {optimizer_filepath}")
                for other_optimizer_filepath in models_dirpath.glob("*.optimizer.pt"):
                    if other_optimizer_filepath == optimizer_filepath:
                        continue
                    other_optimizer_filepath.unlink()

            del (
                prompt_embeds_batch,
                text_ids_batch,
                image_latents_batch,
                image_latent_ids_batch,
                reference_latents_batch,
                reference_latent_ids_batch,
                noise,
                timestep,
                sigma,
                noisy_latents,
                model_hidden_states,
                model_img_ids,
                target,
                noise_pred,
                loss,
            )

            if current_total_step >= max_steps:
                flush_pending_loss_logs()
                break

        first_sample_index = 0
        current_epoch += 1


MODEL_METADATA_NAMES = {
    "4b": "flux 2 klein 4b",
    "9b": "flux 2 klein 9b",
}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument(
        "--version",
        choices=("4b", "9b"),
        required=True,
        help="Model family to load.",
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--steps", type=_positive_int, default=1024)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest checkpoint.",
    )
    parser.add_argument("--text-quant-method", default="sym-high-mini")
    parser.add_argument("--denoiser-quant-method", default="sym-med-mini")
    parser.add_argument(
        "--cache-text",
        action="store_true",
        help="Cache text encodings as per-sample safetensors files and reuse them on later runs.",
    )
    parser.add_argument(
        "--cache-images",
        action="store_true",
        help="Cache image latents as per-sample safetensors files and reuse them on later runs.",
    )
    parser.add_argument(
        "--trigger",
        type=str,
        required=True,
        help="Trigger token. Always required — used for both captionless training and step-based recipe switching.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to a LoRA .safetensors checkpoint to initialise weights from "
        "(continues training from an existing model instead of random init). "
        "Optionally loads optimizer from the same directory with a .optimizer.pt suffix.",
    )
    parser.add_argument(
        "--prompt-step",
        type=int,
        default=None,
        help="Step at which to switch from trigger to per-sample captions. "
        "None (default): always use trigger.",
    )
    parser.add_argument(
        "--high-res-step",
        type=int,
        default=None,
        help="Step at which to start alternating high-res images every other epoch. "
        "None (default): always use base.",
    )
    parser.add_argument(
        "--ref-image-step",
        type=int,
        default=None,
        help="Step at which to start loading reference images into the forward pass. "
        "None (default): never load refs.",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=32,
        help="LoRA rank (default: 32).",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha scaling factor (default: 32).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_path = args.project

    trigger = None if args.trigger is None else args.trigger.strip()
    if trigger == "":
        trigger = None
    model_root = args.root / f"flux2_{args.version}" / "model"

    print("1. Run training epochs ...")

    checkpoint_metadata = {
        "model": MODEL_METADATA_NAMES[args.version.strip().lower()],
        "rank": str(args.lora_rank),
        "alpha": str(args.lora_alpha),
        "trigger": trigger or "",
    }
    
    training_start = time.perf_counter()
    print(f"  * max steps: {args.steps}")

    run_training(
        transformer=None,
        project_dir=project_path,
        checkpoint_metadata=checkpoint_metadata,
        root_path=model_root,
        model_version=args.version.strip().lower(),
        text_quant_method=args.text_quant_method,
        denoiser_quant_method=args.denoiser_quant_method,
        trigger=trigger,
        max_steps=args.steps,
        resume=args.resume,
        cache_text=args.cache_text,
        cache_images=args.cache_images,
        lora_init_checkpoint=args.model,
        prompt_step=args.prompt_step,
        high_res_step=args.high_res_step,
        ref_image_step=args.ref_image_step,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )

    print(f"  * training done in {time.perf_counter() - training_start:.3f}s")

if __name__ == "__main__":
    main()
