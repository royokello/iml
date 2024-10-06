import argparse
import os
import torch
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import UNet2DConditionModel, AutoencoderKL, LMSDiscreteScheduler
from PIL import Image
import uuid

def parse_args():
    parser = argparse.ArgumentParser(description="Generate a 1024x1024 image using SDXL with a single model path.")

    # Single model path
    parser.add_argument("--model_path", type=str, required=True, help="Path to the SDXL Base model directory.")

    # Generation parameters
    parser.add_argument("--positive_prompt", type=str, required=True, help="Positive text prompt for image generation.")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative text prompt for image generation.")
    parser.add_argument("--cfg", type=float, default=7.5, help="Classifier-Free Guidance scale.")
    parser.add_argument("--steps", type=int, default=50, help="Number of denoising steps.")

    # Output
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save the generated image.")

    return parser.parse_args()

def load_models(model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cpu_device = torch.device("cpu")

    # Define subdirectory paths for the second tokenizer and text encoder
    text_encoder_path = os.path.join(model_path, "text_encoder_2")
    tokenizer_path = os.path.join(model_path, "tokenizer_2")
    vae_path = os.path.join(model_path, "vae")
    unet_path = os.path.join(model_path, "unet")

    # Load the second Tokenizer
    print("Loading Tokenizer (Second) for Prompts...")
    tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)

    # Load the second Text Encoder
    print("Loading Text Encoder (Second) on CPU...")
    text_encoder = CLIPTextModel.from_pretrained(text_encoder_path).to(cpu_device)

    # Verify hidden_size of text_encoder_2
    hidden_size = text_encoder.config.hidden_size
    print(f"Text Encoder (Second) Hidden Size: {hidden_size}")
    if hidden_size != 1280:
        raise ValueError(f"Expected text_encoder_2 to have hidden_size=1280, but got {hidden_size}. Please verify the correct text encoder is being used.")

    # Load VAE on CPU
    print("Loading Autoencoder (VAE) on CPU...")
    vae = AutoencoderKL.from_pretrained(vae_path).to(cpu_device)

    # Load U-Net on GPU
    print("Loading U-Net on GPU...")
    unet = UNet2DConditionModel.from_pretrained(unet_path, torch_dtype=torch.float16).to(device)

    return tokenizer, text_encoder, vae, unet, device, cpu_device

def create_scheduler():
    print("Creating scheduler...")
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
    return scheduler

def ensure_output_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created output folder at: {folder_path}")

def generate_image(args, tokenizer, text_encoder, vae, unet, scheduler, device, cpu_device):
    # Tokenize prompts
    print("Tokenizing prompts...")
    # Unconditional (negative) prompt
    uncond_prompt = args.negative_prompt if args.negative_prompt else ""
    uncond_tokens = tokenizer(
        uncond_prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    ).to(cpu_device)

    # Conditional (positive) prompt
    cond_tokens = tokenizer(
        args.positive_prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    ).to(cpu_device)

    # Encode prompts
    print("Encoding prompts...")
    with torch.no_grad():
        # Encode unconditional (negative) prompt
        uncond_embeddings = text_encoder(uncond_tokens.input_ids).last_hidden_state.to(dtype=torch.float16)  # Shape: [1, seq_length, 1280]

        # Encode conditional (positive) prompt
        cond_embeddings = text_encoder(cond_tokens.input_ids).last_hidden_state.to(dtype=torch.float16)    # Shape: [1, seq_length, 1280]

        # Extract pooled embeddings (typically the first token's embedding, e.g., CLS token)
        uncond_pooled_embeddings = uncond_embeddings[:, 0, :]  # Shape: [1, 1280]
        cond_pooled_embeddings = cond_embeddings[:, 0, :]      # Shape: [1, 1280]

    # Verify embedding shapes
    print(f"Unconditional Embeddings Shape: {uncond_embeddings.shape}")
    print(f"Conditional Embeddings Shape: {cond_embeddings.shape}")

    # Concatenate embeddings for classifier-free guidance
    try:
        text_embeddings = torch.cat([uncond_embeddings, cond_embeddings], dim=0)  # Shape: [2, seq_length, 1280]
    except RuntimeError as e:
        print("Error during concatenation of text embeddings:", e)
        print("Ensure that both uncond_embeddings and cond_embeddings have the same hidden size.")
        raise e

    # Concatenate pooled embeddings
    add_text_embeds = torch.cat([uncond_pooled_embeddings, cond_pooled_embeddings], dim=0)  # Shape: [2, 1280]

    # Create time_ids (this is model-specific; adjust as necessary)
    # Here, assuming time_ids are fixed; adjust based on model's requirements
    add_time_ids = torch.tensor([[1024, 1024]] * 2, device=device, dtype=torch.float16)  # Shape: [2, 2]

    # Initialize latents on CPU
    print("Initializing latents...")
    latents = torch.randn((2, unet.config.in_channels, 128, 128), device=cpu_device, dtype=torch.float16)

    # Set timesteps
    scheduler.set_timesteps(args.steps)

    # Denoising loop
    print("Starting denoising process...")
    for i, t in enumerate(scheduler.timesteps):
        # Move necessary tensors to GPU
        latents = latents.to(device)
        text_embeddings_gpu = text_embeddings.to(device)
        add_text_embeds_gpu = add_text_embeds.to(device)
        add_time_ids_gpu = add_time_ids.to(device)

        # Prepare inputs
        latent_model_input = scheduler.scale_model_input(latents, t)

        # Predict noise residual
        with torch.no_grad():
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings_gpu,
                added_cond_kwargs={
                    "text_embeds": add_text_embeds_gpu,
                    "time_ids": add_time_ids_gpu
                }
            ).sample  # Shape: [2, in_channels, height, width]

        # Perform guidance
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + args.cfg * (noise_pred_cond - noise_pred_uncond)

        # Compute previous noisy sample
        latents = scheduler.step(noise_pred, t, latents).prev_sample

        # Move latents back to CPU to save GPU memory
        latents = latents.to(cpu_device)

        # Optional: Print progress
        if (i + 1) % 10 == 0 or (i + 1) == args.steps:
            print(f"Denoising step {i+1}/{args.steps} completed.")

    # Decode latents
    print("Decoding latents to image...")
    with torch.no_grad():
        latents = 1 / 0.18215 * latents.to(device)  # Move back to GPU for faster decoding
        image = vae.decode(latents).sample  # Shape: [2, 3, H, W]

    # Post-process image
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().float().numpy()
    image = (image * 255).round().astype("uint8")[0]  # Take the first image in the batch

    return Image.fromarray(image)

def main():
    args = parse_args()

    # Ensure output folder exists
    ensure_output_folder(args.output_folder)

    # Load models
    print("Loading models...")
    tokenizer, text_encoder, vae, unet, device, cpu_device = load_models(
        args.model_path
    )

    # Create scheduler
    scheduler = create_scheduler()

    # Generate image
    print("Generating image...")
    image = generate_image(
        args,
        tokenizer,
        text_encoder,
        vae,
        unet,
        scheduler,
        device,
        cpu_device
    )

    # Save image
    output_filename = f"generated_image_{uuid.uuid4().hex[:8]}.png"
    output_path = os.path.join(args.output_folder, output_filename)
    image.save(output_path)
    print(f"Image saved to {output_path}")

if __name__ == "__main__":
    main()
