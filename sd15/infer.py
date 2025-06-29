import argparse
import os
from datetime import datetime
import torch
import safetensors
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
import torchvision

# ================================ MAIN ==================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",  required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--steps",  choices=["low", "med", "high"], default="med")
    ap.add_argument("--size",   choices=["square", "portrait", "landscape"], default="square")
    args = ap.parse_args()

    STEP_CHOICES = {"low":16, "med":24, "high":32}
    SIZE_CHOICES = {"square":(512,512), "portrait":(384,512), "landscape":(512,384)}
    steps = STEP_CHOICES[args.steps]
    height, width = SIZE_CHOICES[args.size]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device: {device}")

    # 1) ---------- Tokeniser + text encoder run on CPU --------------
    tokenizer = CLIPTokenizer.from_pretrained(args.model, subfolder="tokenizer", local_files_only=True)
    tokens = tokenizer([args.prompt], padding="max_length", max_length=77, return_tensors="pt")

    text_enc = CLIPTextModel.from_pretrained(
        args.model, subfolder="text_encoder", torch_dtype=torch.float16, local_files_only=True, device_map="cpu"
    )
    with torch.no_grad():
        text_emb = text_enc(**tokens).last_hidden_state
    # Free text‑encoder weights from RAM
    del text_enc;  torch.cuda.empty_cache()
    text_emb = text_emb.to(device)

    # 2) ---------- Scheduler ----------------------------------------
    scheduler = PNDMScheduler.from_pretrained(args.model, subfolder="scheduler", local_files_only=True)
    scheduler.set_timesteps(steps)

    # 3) ---------- UNet ----------------------

    # --- load quantised UNet ---
    unet = 

    # --- prepare latent noise ---
    latents = torch.randn(
        (1, unet.config.in_channels, height // 8, width // 8),
        dtype=torch.float16, device=device
    )
    latents = latents * scheduler.init_noise_sigma

    # --- denoise ---
    with torch.no_grad():
        for t in scheduler.timesteps:
            noise_pred = unet(latents, t, encoder_hidden_states=text_emb).sample
            latents    = scheduler.step(noise_pred, t, latents).prev_sample

    del unet
    torch.cuda.empty_cache()

    # 4) ---------- Load VAE ---------------------
    vae = AutoencoderKL.from_pretrained(
        args.model, subfolder="vae",
        torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(device).eval()

    with torch.no_grad():
        images = vae.decode(latents / 0.18215).sample
        images = (images.clamp(-1, 1) + 1) / 2 

    # 5) ---------- Save output --------------------------------------
    os.makedirs(args.output, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    out_path  = os.path.join(args.output, f"{timestamp}.png")
    torchvision.utils.save_image(images, out_path)
    print(f"[INFO] saved image to {out_path}")

if __name__ == "__main__":
    main()
