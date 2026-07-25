from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint as checkpoint_fn

from flux2.denoiser.loader import _load_flux2_denoiser as load_flux2_denoiser
from flux2.lora.config import FLUX2_LORA_TARGETS
from flux2.lora.loader import load_checkpoint as load_lora_checkpoint
from flux2.lora.model import build_lora_state_dict, inject_trainable_lora_modules
from flux2.train.dataset import Flux2Dataset

from safetensors.torch import load_file as safe_load_file
from safetensors.torch import save_file

CHECKPOINT_AFTER_STEPS = 128
LOSS_LOG_FLUSH_STEPS = 8

from flux2.train.fork import STEP_CSV_HEADER, Step

def main(
    project_dir: Path,
    checkpoint_metadata: dict[str, str],
    root_path,
    model_version,
    text_quant,
    denoiser_quant,
    trigger,
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
):
    print("Training Started ...")
    pending_loss_logs: list[tuple[int, int, int, float, torch.Tensor]] = []

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
    else:
        with logs_filepath.open("w", encoding="utf-8", newline="") as logs_handle:
            logs_handle.write(f"{STEP_CSV_HEADER}\n")

    if current_step >= steps:
        print(f"  * latest checkpoint step {current_step} is already at max steps {steps}; nothing to train")
        return

    # LOAD DATASET
    dataset = Flux2Dataset(
        model_rootpath=root_path,
        dataset_dirpath=Path(project_dir) / "images",
        text_quant_method=text_quant,
        trigger_str=trigger,
        cache_text=cache_text,
        cache_images=cache_images,
        target_resolution=target_resolution,
        reference_resolution=reference_resolution,
        target_upscale=target_upscale,
        ref_upscale=ref_upscale,
    )
    dataset_size = len(dataset.target_latents)
    if dataset_size == 0:
        raise ValueError("Training dataset has no target image latents.")
    current_epoch = (current_step // dataset_size) + 1
    first_sample_index = current_step % dataset_size

    # LOAD TRANSFORMER
    denoiser_load_start = time.perf_counter()
    transformer_path = root_path / "transformer"
    transformer = load_flux2_denoiser(
        str(transformer_path),
        quant_method=denoiser_quant,
        variant="base",
        version=model_version,
    )
    for parameter in transformer.parameters():
        parameter.requires_grad = False
    lora_module_names = inject_trainable_lora_modules(
        transformer,
        target_linear_names=FLUX2_LORA_TARGETS,
        rank=rank,
        alpha=alpha,
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
            if checkpoint_rank != rank:
                print(
                    f"  * WARNING: checkpoint rank {checkpoint_rank} != "
                    f"configured rank {rank}"
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
    print(f"  * lora rank/alpha: r={rank}, alpha={alpha}")
    print(f"  * gradient checkpointing: {transformer.is_gradient_checkpointing}")
    trainable_params = sum(parameter.numel() for parameter in transformer.parameters() if parameter.requires_grad)
    print(f"  * trainable params: {trainable_params:,}")
    lora_parameters = [parameter for parameter in transformer.parameters() if parameter.requires_grad]
    print(f"  * lora parameter dtype: {lora_parameters[0].dtype}")

    optimizer = torch.optim.AdamW(
        [parameter for parameter in transformer.parameters() if parameter.requires_grad],
        lr=learning_rate,
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
            param_group["lr"] = learning_rate

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

            new_steps.append(step)
            print(
                f"      logged epoch {step.epoch} sample {step.sample}: "
                f"loss {step.loss:.6f}"
            )

        with logs_filepath.open("a", encoding="utf-8", newline="") as logs_handle:
            logs_handle.write("\n".join(step.to_csv_row() for step in new_steps))
            logs_handle.write("\n")
            logs_handle.flush()

        pending_loss_logs.clear()

    while current_step < steps:

        for sample_index in range(first_sample_index, dataset_size):

            sample_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)

            text_embedding = dataset.text_embeddings[sample_index]
            if text_embedding is None:
                text_embedding = dataset.trigger_embedding

            prompt_embeds_batch = text_embedding[0].unsqueeze(0).to(
                device="cuda",
                dtype=transformer.dtype,
            )
            text_ids_batch = text_embedding[1].unsqueeze(0).to(device="cuda")

            target_latent = dataset.target_latents[sample_index]
            
            image_latents_batch = target_latent[0].unsqueeze(0).to(device="cuda")
            image_latent_ids_batch = target_latent[1].unsqueeze(0).to(device="cuda")

            ref_latents = dataset.ref_latents[sample_index]
            reference_latents_batch = None
            reference_latent_ids_batch = None
            if ref_latents is not None:
                ref_latent_tensor, ref_ids_tensor = ref_latents
                reference_latents_batch = ref_latent_tensor.unsqueeze(0).to(
                    device="cuda",
                    dtype=image_latents_batch.dtype,
                )
                reference_latent_ids_batch = ref_ids_tensor.unsqueeze(0).to(device="cuda")

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

            current_step += 1

            pending_loss_logs.append((
                current_step,
                current_epoch,
                sample_index + 1,
                learning_rate,
                loss.detach(),
            ))

            if len(pending_loss_logs) >= LOSS_LOG_FLUSH_STEPS:
                flush_pending_loss_logs()

            print(
                f"    * [{current_step}/{steps}]: Epoch {current_epoch}, "
                f"Step {current_step} done in {sample_total_duration:.3f}s "
                f"(load={load_duration:.2f}s, fwd={forward_duration:.2f}s, "
                f"bck={backward_duration:.2f}s, total={sample_total_duration:.2f}s)"
            )

            if current_step % CHECKPOINT_AFTER_STEPS == 0:
                flush_pending_loss_logs()
                checkpoint_filepath = models_dirpath / f"{current_step:06d}.safetensors"
                lora_state_dict = build_lora_state_dict(transformer)
                save_file(lora_state_dict, str(checkpoint_filepath), metadata=checkpoint_metadata)

                optimizer_filepath = checkpoint_filepath.with_suffix(".optimizer.pt")
                torch.save(optimizer.state_dict(), str(optimizer_filepath))
                print(f"  * saved optimizer to {optimizer_filepath}")
                for other_optimizer_filepath in models_dirpath.glob("*.optimizer.pt"):
                    if other_optimizer_filepath == optimizer_filepath:
                        continue
                    other_optimizer_filepath.unlink()

            if not stop_filepath.is_file():
                flush_pending_loss_logs()
                print("  * stop.txt deleted — saving final checkpoint ...")
                checkpoint_filepath = models_dirpath / f"{current_step:06d}.safetensors"
                lora_state_dict = build_lora_state_dict(transformer)
                save_file(lora_state_dict, str(checkpoint_filepath), metadata=checkpoint_metadata)
                optimizer_filepath = checkpoint_filepath.with_suffix(".optimizer.pt")
                torch.save(optimizer.state_dict(), str(optimizer_filepath))
                print(f"  * stop checkpoint saved at step {current_step}. Exiting.")
                return

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

            if current_step >= steps:
                flush_pending_loss_logs()
                break

        first_sample_index = 0
        current_epoch += 1


MODEL_METADATA_NAMES = {
    "4b": "flux 2 klein 4b",
    "9b": "flux 2 klein 9b",
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--version", choices=("4b", "9b"), required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--text-quant", default="sym-med-nano")
    parser.add_argument("--denoiser-quant", default="sym-med-nano")
    parser.add_argument("--cache-text", action="store_true")
    parser.add_argument("--cache-images", action="store_true")
    parser.add_argument("--trigger", type=str, required=True)
    parser.add_argument("--model", type=Path, default=None,
        help="Optionally loads optimizer from the same directory with a .optimizer.pt suffix.",
    )
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--target_res", type=int, default=512)
    parser.add_argument("--target-upscale", action="store_true")
    parser.add_argument("--ref_res", type=int, default=512)
    parser.add_argument("--ref-upscale", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4,
        help="Learning rate (default: 0.0001)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    project_path = args.project
    model_root = args.root / "flux2" / args.version / "model"

    print("1. Run training epochs ...")

    checkpoint_metadata = {
        "model": MODEL_METADATA_NAMES[args.version.strip().lower()],
        "rank": str(args.rank),
        "alpha": str(args.alpha),
        "trigger": args.trigger,
    }
    
    training_start = time.perf_counter()
    print(f"  * max steps: {args.steps}")

    main(
        project_dir=project_path,
        checkpoint_metadata=checkpoint_metadata,
        root_path=model_root,
        model_version=args.version,
        text_quant=args.text_quant,
        denoiser_quant=args.denoiser_quant,
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
    )

    print(f"  * training done in {time.perf_counter() - training_start:.3f}s")
