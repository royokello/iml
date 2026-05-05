from __future__ import annotations

import time
import shutil
import copy
from pathlib import Path
from typing import Any

import torch
from torch.utils.checkpoint import checkpoint as checkpoint_fn

from flux2.denoiser.loader import _load_flux2_denoiser as load_flux2_denoiser
from flux2.lora.model import build_lora_state_dict, inject_trainable_lora_modules
from flux2.train.dataset import Flux2Dataset
from flux2.lora.loader import load_checkpoint
from flux2.train.logs import save_training_graph

from safetensors.torch import save_file

INITIAL_LORA_TARGET_LINEAR_NAMES = (
    "transformer_blocks.attn.to_q",
    "transformer_blocks.attn.to_k",
    "transformer_blocks.attn.to_v",
    "transformer_blocks.attn.to_out",
    "transformer_blocks.ff.linear_in",
    "transformer_blocks.ff.linear_out",
    "single_transformer_blocks.attn.to_qkv_mlp_proj",
    "single_transformer_blocks.attn.to_out",
)
INITIAL_LORA_RANK = 32
INITIAL_LORA_ALPHA = 16

CHECKPOINT_AFTER_STEPS = 16
LOSS_LOG_FLUSH_STEPS = 16
ERA_STEPS = 64

from flux2.train.fork import Checkpoint, Fork, STEP_CSV_HEADER


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
):
    print("  * training ...")

    current_fork = None
    pending_loss_logs: list[tuple[int, int, int, int, float, torch.Tensor]] = []

    current_era = 1
    current_era_steps = 0
    current_total_step = 0

    current_epoch = 1
    current_learning_rate = 1e-4
    available_learning_rates = [1e-4, 7e-5, 5e-5, 3e-5, 2e-5, 1e-5]

    rollback_counter = 0

    models_dirpath = project_dir / "models"

    # LOAD DATASET
    dataset = Flux2Dataset(
        root_path,
        text_quant_method,
        project_dir,
        trigger,
    )
    current_encodings = dataset.low_res_encodings
    dataset_size = len(current_encodings.target_image_latents)

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
        target_linear_names=INITIAL_LORA_TARGET_LINEAR_NAMES,
        rank=INITIAL_LORA_RANK,
        alpha=INITIAL_LORA_ALPHA,
    )
    transformer.train()
    def non_reentrant_checkpoint(module, *inputs):
        return checkpoint_fn(module, *inputs, use_reentrant=False)

    transformer.enable_gradient_checkpointing(gradient_checkpointing_func=non_reentrant_checkpoint)
    print(f"  * transformer loaded in {time.perf_counter() - denoiser_load_start:.3f}s")
    
    print(f"  * initial lora targets: {', '.join(INITIAL_LORA_TARGET_LINEAR_NAMES)}")
    print(f"  * injected lora modules: {len(lora_module_names)}")
    print(f"  * lora rank/alpha: r={INITIAL_LORA_RANK}, alpha={INITIAL_LORA_ALPHA}")
    print(f"  * gradient checkpointing: {transformer.is_gradient_checkpointing}")
    trainable_params = sum(parameter.numel() for parameter in transformer.parameters() if parameter.requires_grad)
    print(f"  * trainable params: {trainable_params:,}")
    lora_parameters = [parameter for parameter in transformer.parameters() if parameter.requires_grad]
    print(f"  * lora parameter dtype: {lora_parameters[0].dtype}")

    optimizer = torch.optim.AdamW(
        [parameter for parameter in transformer.parameters() if parameter.requires_grad],
        lr=current_learning_rate,
    )

    # LOAD FORKS
    optimizer_state_dict = None
    if resume:
        current_fork = Fork.load_from_csv(models_dirpath)

        checkpoint_paths = sorted(
            (
                path
                for path in (models_dirpath).glob("*.safetensors")
                if path.stem.isdigit()
            ),
            key=lambda path: int(path.stem),
        )
        latest_checkpoint_path = checkpoint_paths[-1] if checkpoint_paths else None
        if latest_checkpoint_path is not None:
            load_checkpoint(transformer, latest_checkpoint_path)
        optimizer_checkpoint_path = (
            latest_checkpoint_path.with_suffix(".optimizer.pt")
            if latest_checkpoint_path is not None
            else None
        )

        if optimizer_checkpoint_path is not None and optimizer_checkpoint_path.is_file():
            optimizer_state_dict = torch.load(str(optimizer_checkpoint_path), map_location="cpu")

        if latest_checkpoint_path is not None:
            latest_checkpoint_step = int(latest_checkpoint_path.stem)
            
            current_fork.rollback(latest_checkpoint_step)
            if current_fork.steps:
                latest_step = current_fork.steps[-1]
                current_total_step = latest_step.step
                current_epoch = latest_step.epoch
                current_era = latest_step.era
                current_era_steps = sum(1 for step in current_fork.steps if step.era == current_era)

    else:
        current_fork = Fork()

        continuous_logs_filepath = models_dirpath / "steps.csv"
        with continuous_logs_filepath.open("w", encoding="utf-8", newline="") as logs_handle:
            logs_handle.write(f"{STEP_CSV_HEADER}\n")

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

    def write_steps_csv() -> None:
        logs_filepath = models_dirpath / "steps.csv"
        with logs_filepath.open("w", encoding="utf-8", newline="") as logs_handle:
            logs_handle.write(f"{STEP_CSV_HEADER}\n")
            logs_handle.write("\n".join(step.to_csv_row() for step in current_fork.steps))
            logs_handle.write("\n")

    def delete_checkpoints_after(max_step: int) -> None:
        model_paths = [
            path
            for path in models_dirpath.glob("*.safetensors")
            if path.stem.isdigit() and int(path.stem) > max_step
        ]
        optimizer_paths = [path.with_suffix(".optimizer.pt") for path in model_paths]
        for path in model_paths + [path for path in optimizer_paths if path.exists()]:
            path.unlink()

    def flush_pending_loss_logs() -> None:
        if not pending_loss_logs:
            return

        logs_filepath = models_dirpath / "steps.csv"
        steps = []
        for (
            log_total_step,
            log_epoch_number,
            log_sample_number,
            log_era_number,
            log_learning_rate,
            pending_loss,
        ) in pending_loss_logs:
            loss_value = pending_loss.item()
            step = current_fork.generate_next_step(
                log_total_step,
                log_epoch_number,
                log_sample_number,
                log_era_number,
                log_learning_rate,
                loss_value,
            )
            steps.append(step)
            print(
                f"      logged era {step.era} epoch {step.epoch} sample {step.sample}: "
                f"loss {step.loss:.6f}, moving average {step.moving_average:.6f}"
            )

        with logs_filepath.open("a", encoding="utf-8", newline="") as logs_handle:
            logs_handle.write("\n".join(step.to_csv_row() for step in steps))
            logs_handle.write("\n")
            logs_handle.flush()

        save_training_graph(current_fork, project_dir)
        pending_loss_logs.clear()

    if resume and current_fork.steps:
        recommendation, new_lr, rollback_step = current_fork.recommend_training_action(
            learning_rate=current_learning_rate,
            available_learning_rates=available_learning_rates,
        )
        print(
            f"  * resumed fork recommendation: {recommendation}, "
            f"new lr: {new_lr}, rollback: {rollback_step}"
        )

        if rollback_step:
            current_fork.rollback(rollback_step)
            current_total_step = current_fork.steps[-1].step if current_fork.steps else rollback_step
            delete_checkpoints_after(rollback_step)
            load_checkpoint(transformer, models_dirpath / f"{rollback_step:06d}.safetensors")

            rollback_optimizer_path = models_dirpath / f"{rollback_step:06d}.optimizer.pt"
            if rollback_optimizer_path.is_file():
                optimizer.load_state_dict(
                    torch.load(str(rollback_optimizer_path), map_location="cpu")
                )
                move_optimizer_state_to_cuda()
                set_optimizer_learning_rate()
            else:
                optimizer.state.clear()

        if new_lr != current_learning_rate:
            current_learning_rate = new_lr
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_learning_rate

        if current_fork.steps:
            latest_step = current_fork.steps[-1]
            current_total_step = latest_step.step
            current_epoch = latest_step.epoch + 1
            current_era = latest_step.era + 1
        else:
            current_total_step = 0
            current_epoch = 1
            current_era = 1
        current_era_steps = 0

        if current_learning_rate <= 2e-5:
            current_encodings = dataset.high_res_encodings
        else:
            current_encodings = dataset.low_res_encodings
        dataset_size = len(current_encodings.target_image_latents)

        write_steps_csv()
        save_training_graph(current_fork, project_dir)

    while current_total_step < max_steps or rollback_counter > 4:

        for sample_index in range(dataset_size):
            sample_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)

            if current_encodings.prompt_embeds is not None and current_encodings.text_ids is not None:
                prompt_embeds_batch = current_encodings.prompt_embeds[sample_index : sample_index + 1].to(
                    device="cuda",
                    dtype=transformer.dtype,
                )
                text_ids_batch = current_encodings.text_ids[sample_index : sample_index + 1].to(device="cuda")
            else:
                prompt_embeds_batch = dataset.shared_prompt_embeds
                text_ids_batch = dataset.shared_text_ids
            image_latents_batch = current_encodings.target_image_latents[sample_index].unsqueeze(0).to(device="cuda")
            image_latent_ids_batch = current_encodings.target_image_latent_ids[sample_index].unsqueeze(0).to(device="cuda")
            reference_latents_batch = None
            reference_latent_ids_batch = None
            if (
                current_encodings.reference_latents[sample_index] is not None
                and current_encodings.reference_latent_ids[sample_index] is not None
            ):
                reference_latents_batch = current_encodings.reference_latents[sample_index].unsqueeze(0).to(
                    device="cuda",
                    dtype=image_latents_batch.dtype,
                )
                reference_latent_ids_batch = current_encodings.reference_latent_ids[sample_index].unsqueeze(0).to(device="cuda")

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

            loss.backward()

            optimizer.step()

            current_era_steps += 1
            current_total_step += 1
            
            pending_loss_logs.append((
                current_total_step,
                current_epoch,
                sample_index + 1,
                current_era,
                current_learning_rate,
                loss.detach(),
            ))

            if len(pending_loss_logs) >= LOSS_LOG_FLUSH_STEPS:
                flush_pending_loss_logs()
            
            print(
                f"    * [{current_total_step}/{max_steps}]: Epoch {current_epoch}, Era {current_era}, "
                f"Step {current_era_steps} done in {time.perf_counter() - sample_start:.3f}s"
            )
            
            if current_era_steps % CHECKPOINT_AFTER_STEPS == 0:
                checkpoint_filepath = models_dirpath / f"{current_total_step:06d}.safetensors"
                lora_state_dict = build_lora_state_dict(transformer)
                save_file(lora_state_dict, str(checkpoint_filepath), metadata=checkpoint_metadata)
                latest_step = current_fork.steps[-1]
                checkpoint = Checkpoint()
                checkpoint.fork = current_fork
                checkpoint.era = current_era
                checkpoint.steps = current_total_step
                checkpoint.moving_average = latest_step.moving_average
                current_fork.checkpoints.append(checkpoint)

                if current_fork.is_best_era_loss(current_era, latest_step.loss):
                    optimizer_filepath = checkpoint_filepath.with_suffix(".optimizer.pt")
                    torch.save(optimizer.state_dict(), str(optimizer_filepath))
                    print(f"  * best era loss saving optimizer")
                    for checkpoint in current_fork.checkpoints:
                        if checkpoint.era != current_era or checkpoint.steps == current_total_step:
                            continue
                        other_optimizer_filepath = (
                            models_dirpath / f"{checkpoint.steps:06d}.optimizer.pt"
                        )
                        if other_optimizer_filepath.exists():
                            other_optimizer_filepath.unlink()

                print(f"  * saved lora to {checkpoint_filepath}")


            if current_era_steps == ERA_STEPS:
                flush_pending_loss_logs()
                recommendation, new_lr, rollback_step = current_fork.recommend_training_action(
                    learning_rate=current_learning_rate,
                    available_learning_rates=available_learning_rates,
                )
                print(f" * era reached, recommendation: {recommendation}, new lr: {new_lr}, rollback: {rollback_step}")

                # rollback fork if needed
                if rollback_step:
                    current_fork.rollback(rollback_step)
                    current_total_step = current_fork.steps[-1].step if current_fork.steps else rollback_step
                    delete_checkpoints_after(rollback_step)
                    write_steps_csv()

                    rollback_checkpoint_path = models_dirpath / f"{rollback_step:06d}.safetensors"
                    load_checkpoint(transformer, rollback_checkpoint_path)

                    rollback_optimizer_path = rollback_checkpoint_path.with_suffix(".optimizer.pt")
                    if rollback_optimizer_path.is_file():
                        optimizer.load_state_dict(
                            torch.load(str(rollback_optimizer_path), map_location="cuda")
                        )
                        move_optimizer_state_to_cuda()
                    else:
                        optimizer.state.clear()

                    rollback_counter += 1

                else:

                    rollback_counter = 0

                # update lr 
                if new_lr:
                    if new_lr <= 1e-5:
                        current_encodings = dataset.high_res_encodings
                    else:
                        current_encodings = dataset.low_res_encodings

                    current_learning_rate = new_lr

                    for param_group in optimizer.param_groups:
                        param_group["lr"] = current_learning_rate

                    print(f"  * reset optimizer lr={current_learning_rate}")

                current_era += 1
                current_era_steps = 0
            
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

        current_epoch += 1
