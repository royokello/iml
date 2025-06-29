import argparse, os, time, torch
from diffusers import AutoencoderKL, DDIMScheduler, CLIPTextModel, CLIPTokenizer
from PIL import Image
from .model import load_quant_unet

@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--size", default="384,384")
    args = ap.parse_args()

    H,W = map(int, args.size.split(","))
    device = "cuda"

    # 1. tokenizer / text encoder on CPU
    tok = CLIPTokenizer.from_pretrained(os.path.join(args.model, "text_encoder"))
    txtenc = CLIPTextModel.from_pretrained(os.path.join(args.model, "text_encoder")).eval()
    # 2. quantised UNet on GPU
    unet = load_quant_unet(args.model, device)
    # 3. VAE on GPU (FP16)
    vae  = AutoencoderKL.from_pretrained(os.path.join(args.model, "vae")).half().to(device).eval()
    # 4. scheduler
    sched = DDIMScheduler.from_pretrained(args.model, subfolder="scheduler")
    sched.set_timesteps(args.steps)

    # prompt → embeddings
    emb = txtenc(tok([args.prompt], return_tensors="pt").input_ids)[0].to(device, dtype=torch.float16)

    # latents
    lat = torch.randn(1, 4, H//8, W//8, device=device, dtype=torch.float16)

    t0 = time.time()
    for t in sched.timesteps:
        inp = torch.cat([lat]*2)
        noise_pred = unet(inp, t, encoder_hidden_states=emb)["sample"]
        lat = sched.step(noise_pred, t, lat).prev_sample.half()
    img = vae.decode(lat / 0.18215)["sample"][0]
    img = (img.clamp(-1,1)+1)/2
    img = (img.cpu().permute(1,2,0).numpy()*255).round().astype("uint8")
    Image.fromarray(img).save(args.output)
    print(f"done in {time.time()-t0:.1f}s, peak VRAM {torch.cuda.max_memory_allocated()/1e6:.0f} MB")

if __name__ == "__main__":
    main()
