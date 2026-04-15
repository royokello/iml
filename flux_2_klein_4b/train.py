from __future__ import annotations

import argparse
import csv
from pathlib import Path

INITIAL_LORA_TARGET_LINEAR_NAMES = (
    "to_q",
    "to_k",
    "to_v",
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_add_out",
    "to_qkv_mlp_proj",
)
INITIAL_LORA_RANK = 16
INITIAL_LORA_ALPHA = 16
INITIAL_LORA_LEARNING_RATE = 1e-4


def _load_dataset_pairs(dataset_dir: Path) -> list[tuple[Path, str]]:
    image_paths = sorted(
        path for path in dataset_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )
    pairs: list[tuple[Path, str]] = []
    for image_path in image_paths:
        caption_path = image_path.with_suffix(".txt")
        if not caption_path.is_file():
            raise FileNotFoundError(f"Caption not found for image: {image_path}")
        caption = caption_path.read_text(encoding="utf-8").strip()
        pairs.append((image_path, caption))
    return pairs


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=_positive_int, default=1)
    parser.add_argument("--checkpoint", type=_positive_int, default=1)
    parser.add_argument("--text-quantization-precision", default="int8")
    parser.add_argument("--text-scale-precision", default="fp16")
    parser.add_argument("--text-block-size", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from flux_2_klein_4b.denoiser import load_qwen3_denoiser
    from flux_2_klein_4b.gen import (
        _encode_prompt_embeddings,
        _load_vae_scale_factor,
        _pack_latents,
        _patchify_latents,
        _prepare_latent_ids,
        _quantized_denoiser_path,
        _quantized_text_encoder_path,
        _retrieve_latents,
    )
    from flux_2_klein_4b.quantize.linear import QuantizedLinear
    from flux_2_klein_4b.text_encoder.loader import load_qwen3_text_encoder
    from PIL import Image
    from safetensors.torch import save_file
    import torch
    from diffusers import AutoencoderKLFlux2
    from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
    from transformers import Qwen2TokenizerFast

    class TrainableLoraLinear(torch.nn.Module):
        def __init__(self, base_module: torch.nn.Module, *, rank: int, alpha: int) -> None:
            super().__init__()
            self.base_module = base_module
            self.rank = rank
            self.alpha = alpha
            self.scaling = alpha / rank
            self.in_features = base_module.in_features
            self.out_features = base_module.out_features

            for parameter in self.base_module.parameters():
                parameter.requires_grad = False

            parameter_device = None
            parameter_dtype = None
            for tensor in list(self.base_module.parameters()) + list(self.base_module.buffers()):
                if tensor.is_floating_point():
                    parameter_device = tensor.device
                    parameter_dtype = tensor.dtype
                    break
            if parameter_device is None:
                parameter_device = torch.device("cpu")
                parameter_dtype = torch.float16

            self.lora_A = torch.nn.Parameter(
                torch.empty((rank, self.in_features), device=parameter_device, dtype=parameter_dtype)
            )
            self.lora_B = torch.nn.Parameter(
                torch.zeros((self.out_features, rank), device=parameter_device, dtype=parameter_dtype)
            )
            torch.nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            base_output = self.base_module(input)
            lora_hidden = torch.nn.functional.linear(input, self.lora_A.to(dtype=input.dtype))
            lora_output = torch.nn.functional.linear(lora_hidden, self.lora_B.to(dtype=input.dtype))
            return base_output + lora_output * self.scaling

    def inject_trainable_lora_modules(
        module: torch.nn.Module,
        *,
        target_linear_names: tuple[str, ...],
        rank: int,
        alpha: int,
    ) -> list[str]:
        injected_module_names: list[str] = []

        def _inject(parent: torch.nn.Module, prefix: str = "") -> None:
            for child_name, child in list(parent.named_children()):
                full_name = f"{prefix}.{child_name}" if prefix else child_name
                if child_name in target_linear_names and isinstance(child, (torch.nn.Linear, QuantizedLinear)):
                    setattr(parent, child_name, TrainableLoraLinear(child, rank=rank, alpha=alpha))
                    injected_module_names.append(full_name)
                    continue
                _inject(child, full_name)

        _inject(module)
        return injected_module_names

    def build_lora_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
        lora_state_dict: dict[str, torch.Tensor] = {}
        for module_name, child in module.named_modules():
            if not isinstance(child, TrainableLoraLinear):
                continue
            lora_state_dict[f"{module_name}.lora_A.weight"] = child.lora_A.detach().cpu()
            lora_state_dict[f"{module_name}.lora_B.weight"] = child.lora_B.detach().cpu()
            lora_state_dict[f"{module_name}.alpha"] = torch.tensor(float(child.alpha), dtype=torch.float32)
        return lora_state_dict

    device = torch.device("cuda")

    print("1. Load dataset into cpu ram ...")
    dataset_pairs = _load_dataset_pairs(args.dataset)
    images = [Image.open(image_path).convert("RGB") for image_path, _ in dataset_pairs]
    captions = [caption for _, caption in dataset_pairs]

    print("2. Load text encoder into gpu ...")
    tokenizer_path = args.root / "flux_2_klein_4b" / "base" / "tokenizer"
    text_encoder_path = args.root / "flux_2_klein_4b" / "base" / "text_encoder"
    tokenizer = Qwen2TokenizerFast.from_pretrained(str(tokenizer_path))
    quantized_state_path = _quantized_text_encoder_path(
        args.root,
        quantization_precision=args.text_quantization_precision,
        scale_precision=args.text_scale_precision,
        block_size=args.text_block_size,
    )
    if not quantized_state_path.is_file():
        quantized_state_path = None
    text_encoder = load_qwen3_text_encoder(
        str(text_encoder_path),
        quantization_precision=args.text_quantization_precision,
        scale_precision=args.text_scale_precision,
        block_size=args.text_block_size,
        quantized_state_path=quantized_state_path,
    )
    text_encoder = text_encoder.to(device)

    print("3. Prepare text encodings ...")
    prompt_embeds_list = []
    text_ids_list = []
    for caption_index, caption in enumerate(captions, start=1):
        print(f" * {caption_index} / {len(captions)} ...")
        prompt_embeds, text_ids = _encode_prompt_embeddings(
            torch,
            tokenizer,
            text_encoder,
            prompt=caption,
            device=device,
            max_length=512,
        )
        prompt_embeds_list.append(prompt_embeds.squeeze(0).cpu())
        text_ids_list.append(text_ids.squeeze(0).cpu())

    prompt_embeds = torch.stack(prompt_embeds_list, dim=0)
    text_ids = torch.stack(text_ids_list, dim=0)
    prompt_embeds_mb = prompt_embeds.numel() * prompt_embeds.element_size() / (1024 * 1024)
    text_ids_mb = text_ids.numel() * text_ids.element_size() / (1024 * 1024)
    print(f"  * prompt_embeds size: {prompt_embeds_mb:.2f} MB")
    print(f"  * text_ids size: {text_ids_mb:.2f} MB")
    del text_encoder
    torch.cuda.empty_cache()

    print("4. Load vae into gpu ...")
    vae_path = args.root / "flux_2_klein_4b" / "base" / "vae"
    vae_scale_factor = _load_vae_scale_factor(vae_path)
    vae = AutoencoderKLFlux2.from_pretrained(str(vae_path), local_files_only=True)
    vae = vae.to(device, dtype=torch.float16)

    print("5. Prepare image latents ...")
    image_processor = Flux2ImageProcessor(vae_scale_factor=vae_scale_factor * 2)
    image_latents = []
    image_latent_ids = []
    for image_index, image in enumerate(images, start=1):
        print(f" * {image_index} / {len(images)} ...")
        image_processor.check_image_input(image)

        image_width, image_height = image.size
        multiple_of = vae_scale_factor * 2
        image_width = (image_width // multiple_of) * multiple_of
        image_height = (image_height // multiple_of) * multiple_of

        image_tensor = image_processor.preprocess(image, height=image_height, width=image_width, resize_mode="crop")
        image_tensor = image_tensor.to(device=device, dtype=torch.float16)
        with torch.inference_mode():
            latent = _retrieve_latents(vae.encode(image_tensor))
        latent = _patchify_latents(latent)

        latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latent.device, latent.dtype)
        latents_bn_std = torch.sqrt(
            vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps
        ).to(latent.device, latent.dtype)
        latent = (latent - latents_bn_mean) / latents_bn_std

        image_latents.append(_pack_latents(latent).squeeze(0).cpu())
        image_latent_ids.append(_prepare_latent_ids(torch, latent).squeeze(0).cpu())

    image_latents_mb = sum(latent.numel() * latent.element_size() for latent in image_latents) / (1024 * 1024)
    image_latent_ids_mb = sum(latent_ids.numel() * latent_ids.element_size() for latent_ids in image_latent_ids) / (
        1024 * 1024
    )
    print(f"  * image_latents size: {image_latents_mb:.2f} MB")
    print(f"  * image_latent_ids size: {image_latent_ids_mb:.2f} MB")
    del vae
    torch.cuda.empty_cache()

    print("6. Load text encodings and image latents to gpu ...")
    prompt_embeds = prompt_embeds.to(device)
    text_ids = text_ids.to(device)
    image_latents = [latent.to(device) for latent in image_latents]
    image_latent_ids = [latent_ids.to(device) for latent_ids in image_latent_ids]

    print("7. Load denoiser ...")
    transformer_path = args.root / "flux_2_klein_4b" / "base" / "transformer"
    quantized_denoiser_path = _quantized_denoiser_path(
        args.root,
        quantization_precision="int4",
        scale_precision="fp16",
        block_size=64,
    )
    if not quantized_denoiser_path.is_file():
        quantized_denoiser_path = None
    transformer = load_qwen3_denoiser(
        str(transformer_path),
        quantization_precision="int4",
        scale_precision="fp16",
        block_size=64,
        quantized_state_path=quantized_denoiser_path,
    )
    transformer = transformer.to(device)
    for parameter in transformer.parameters():
        parameter.requires_grad = False
    lora_module_names = inject_trainable_lora_modules(
        transformer,
        target_linear_names=INITIAL_LORA_TARGET_LINEAR_NAMES,
        rank=INITIAL_LORA_RANK,
        alpha=INITIAL_LORA_ALPHA,
    )
    transformer.train()
    print(f"  * initial lora targets: {', '.join(INITIAL_LORA_TARGET_LINEAR_NAMES)}")
    print(f"  * injected lora modules: {len(lora_module_names)}")
    print(f"  * lora rank/alpha: r={INITIAL_LORA_RANK}, alpha={INITIAL_LORA_ALPHA}")
    trainable_params = sum(parameter.numel() for parameter in transformer.parameters() if parameter.requires_grad)
    print(f"  * trainable params: {trainable_params:,}")
    lora_parameters = [parameter for parameter in transformer.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(lora_parameters, lr=INITIAL_LORA_LEARNING_RATE)
    print(f"  * optimizer: AdamW lr={INITIAL_LORA_LEARNING_RATE}")
    optimizer_param_mb = sum(parameter.numel() * parameter.element_size() for parameter in lora_parameters) / (
        1024 * 1024
    )
    optimizer_state_mb = optimizer_param_mb * 2
    optimizer_total_mb = optimizer_param_mb + optimizer_state_mb
    print(f"  * optimizer params size: {optimizer_param_mb:.2f} MB")
    print(f"  * optimizer state size (estimated): {optimizer_state_mb:.2f} MB")
    print(f"  * optimizer total size (estimated): {optimizer_total_mb:.2f} MB")

    output_dir = args.output
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_path = output_dir / "logs.csv"

    def save_checkpoint(epoch_number: int) -> None:
        output_path = models_dir / f"epoch_{epoch_number}.safetensor"
        lora_state_dict = build_lora_state_dict(transformer)
        save_file(lora_state_dict, str(output_path))
        print(f"  * saved lora to {output_path}")

    print("8. Run training epochs ...")
    print(f"  * logs: {logs_path}")
    accumulated_loss = torch.zeros((), device=device)
    with logs_path.open("w", encoding="utf-8", newline="") as logs_handle:
        logs_writer = csv.writer(logs_handle)
        logs_writer.writerow(["epoch", "image(index)", "loss"])

        for epoch_index in range(args.epochs):
            epoch_number = epoch_index + 1
            print(f" * epoch {epoch_number} / {args.epochs} ...")
            for sample_index in range(len(image_latents)):
                print(f" * * sample {sample_index + 1} / {len(image_latents)} ...")
                optimizer.zero_grad(set_to_none=True)

                prompt_embeds_batch = prompt_embeds[sample_index : sample_index + 1]
                text_ids_batch = text_ids[sample_index : sample_index + 1]
                image_latents_batch = image_latents[sample_index].unsqueeze(0)
                image_latent_ids_batch = image_latent_ids[sample_index].unsqueeze(0)

                noise = torch.randn_like(image_latents_batch)
                timestep = torch.rand((1,), device=device, dtype=image_latents_batch.dtype) * 1000.0
                sigma = (timestep / 1000.0).view(-1, 1, 1)

                noisy_latents = (1.0 - sigma) * image_latents_batch + sigma * noise
                target = noise - image_latents_batch

                noise_pred = transformer(
                    hidden_states=noisy_latents.to(transformer.dtype),
                    timestep=timestep / 1000,
                    guidance=None,
                    encoder_hidden_states=prompt_embeds_batch.to(transformer.dtype),
                    txt_ids=text_ids_batch,
                    img_ids=image_latent_ids_batch,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0]

                loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float())
                accumulated_loss = accumulated_loss + loss
                loss_value = loss.item()
                print(f" * * * loss: {loss_value:.6f}")
                logs_writer.writerow([epoch_number, sample_index + 1, loss_value])
                logs_handle.flush()

                loss.backward()

                optimizer.step()

            if epoch_number % args.checkpoint == 0 or epoch_number == args.epochs:
                save_checkpoint(epoch_number)

    del transformer


if __name__ == "__main__":
    main()
