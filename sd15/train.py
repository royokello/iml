#!/usr/bin/env python
"""
LoRA Training Script for Stable Diffusion v1.5
==============================================

* **UNet** stays on the **GPU** for the entire run (training + sampling).
* **VAE** runs **only on the CPU** — both for encoding training images and
  decoding sample latents — keeping GPU VRAM free for the UNet.
* Text‑encoder stays on GPU, but sample‑prompt embeddings are cached once.
* Periodic sampling/checkpointing logic unchanged.

Folder layout (`--project`):
```
project/
├─ images/   # training *.jpg + *.txt
├─ samples/  # rendered sample_{step}.png
└─ models/   # ckpt_step####.safetensors + final.safetensors
```

Example:
```
py -m sd15.train --model "/models/sd15" --project "/my_project" --preset "face" --name "person"
```
"""
#!/usr/bin/env python
"""
LoRA Training Script for Stable Diffusion v1.5
==============================================

**Key changes (v2)**
--------------------
1. **Latent cache** – every training image is encoded **once** at start‑up;
   cached latents (fp16 on CPU) are reused every step → 40‑60 % speed‑up on
   small datasets.
2. UNet calls use `UNET.model(…)` to bypass PEFT’s NLP argument mapping.
3. Dataset loader still validates images/captions and supports jpg/jpeg/png/webp.
4. VAE decode thread prints `[decode]` / `[saved]` messages; logic unchanged.
"""
import argparse, os, random, threading, queue
from glob import glob

import torch, torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from peft.utils import TaskType
from safetensors.torch import save_file
from torch.optim import AdamW

# ────────────────── device & seed ──────────────────
GPU = torch.device('cuda') if torch.cuda.is_available() else None
if GPU is None:
    raise RuntimeError('CUDA GPU required')
CPU = torch.device('cpu')
random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)

# ────────────────── presets ──────────────────
PRESETS = {
    'face':   dict(lr=1e-4, steps=2000, rank=16, alpha=16,
                   prompts=["portrait photo of {name} in natural light",
                            "35 mm film close‑up of {name} smiling"]),
    'person': dict(lr=1e-4, steps=3000, rank=16, alpha=16,
                   prompts=["full‑body photo of {name} standing in a park",
                            "action shot of {name} jumping over a puddle"]),
    'object': dict(lr=1e-4, steps=3000, rank=32, alpha=32,
                   prompts=["studio shot of {name} on white background",
                            "macro photograph of {name} with dramatic lighting"]),
    'style':  dict(lr=5e-5, steps=1000, rank=8,  alpha=8,
                   prompts=["landscape painting in the style of {name}",
                            "portrait of a cat in the style of {name}"])
}

# ────────────────── CLI ──────────────────
cli = argparse.ArgumentParser()
cli.add_argument('--model',   required=True)
cli.add_argument('--project', required=True)
cli.add_argument('--preset',  choices=PRESETS.keys(), default='object')
cli.add_argument('--name',    required=True)
args = cli.parse_args()

proj      = args.project
img_dir   = os.path.join(proj, 'images')
smpl_dir  = os.path.join(proj, 'samples'); os.makedirs(smpl_dir, exist_ok=True)
ckpt_dir  = os.path.join(proj, 'models');  os.makedirs(ckpt_dir,  exist_ok=True)

cfg = PRESETS[args.preset]
print('Preset:', {k: cfg[k] for k in cfg if k!='prompts'})

# ────────────────── load tokenizer & text encoder ──────────────────
TOK = CLIPTokenizer.from_pretrained(os.path.join(args.model, 'tokenizer'))
TXT = CLIPTextModel.from_pretrained(os.path.join(args.model, 'text_encoder'), torch_dtype=torch.float16).to(GPU).eval()
for p in TXT.parameters(): p.requires_grad=False

# ────────────────── load UNet & insert LoRA ──────────────────
UNET = UNet2DConditionModel.from_pretrained(os.path.join(args.model,'unet'), torch_dtype=torch.float16).to(GPU).train()
UNET = get_peft_model(UNET, LoraConfig(r=cfg['rank'], lora_alpha=cfg['alpha'], target_modules=['to_q','to_k','to_v','to_out.0'], task_type=TaskType.FEATURE_EXTRACTION))

# ────────────────── load VAE (CPU) & scheduler ──────────────────
VAE  = AutoencoderKL.from_pretrained(os.path.join(args.model,'vae'), torch_dtype=torch.float32).to(CPU).eval()
SCHED= DDIMScheduler.from_pretrained(args.model, subfolder='scheduler')

# ────────────────── scan images & captions ──────────────────
IMG_EXT = ('.jpg','.jpeg','.png','.webp')
paths   = sorted([p for p in glob(os.path.join(img_dir,'*')) if p.lower().endswith(IMG_EXT)])
if not paths:
    raise ValueError(f"No images found in {img_dir}")

print('Encoding', len(paths), 'images to latents (fp32→fp16, CPU)…')
lat_cache=[]; id_cache=[]
prep = transforms.Compose([
    transforms.Resize((VAE.config.sample_size,VAE.config.sample_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3),
])
with torch.no_grad():
    for p in paths:
        img = prep(Image.open(p).convert('RGB'))
        latent = VAE.encode(img.unsqueeze(0)).latent_dist.sample() * VAE.config.scaling_factor
        lat_cache.append(latent.squeeze(0).half())  # CPU fp16
        cap_path = os.path.splitext(p)[0]+'.txt'
        if not os.path.exists(cap_path):
            raise FileNotFoundError(cap_path)
        caption = open(cap_path,'r',encoding='utf-8').read().strip()
        ids = TOK(caption, return_tensors='pt', padding='max_length', truncation=True, max_length=77).input_ids[0]
        id_cache.append(ids)
print('Caching done.')

# ────────────────── optimizer ──────────────────
OPT = AdamW(UNET.parameters(), lr=cfg['lr'])

# ────────────────── prompt embeddings cache ──────────────────
prompt_cache = []
for t in cfg['prompts']:
    prompt = t.format(name=args.name)
    hs = TXT(
        TOK(
          ["", prompt],
          return_tensors="pt",
          padding=True,
          truncation=True,
          max_length=77
        ).input_ids.to(GPU)
    )[0].half().chunk(2)
    prompt_cache.append((hs[0], hs[1]))

# ────────────────── training-caption embeddings cache ──────────────────
print('Caching', len(id_cache), 'training captions → text embeddings (CPU fp16)…')
text_emb_cache = []
with torch.no_grad():
    for ids in id_cache:
    # ids is a 1D tensor [seq_len]
        emb = TXT(ids.unsqueeze(0).to(GPU))[0].half()     # [1, seq_len, hidden]
        text_emb_cache.append(emb.squeeze(0).cpu())       # store [seq_len, hidden] on CPU
print('Cached', len(text_emb_cache), 'training text embeddings')
del TXT
torch.cuda.empty_cache()

# ────────────────── threaded VAE decode ──────────────────
work_q: queue.Queue[tuple[torch.Tensor,str]] = queue.Queue()

def decoder_worker():
    while True:
        lat, path = work_q.get()
        if lat is None:
            break
        print(f"[decode] → {os.path.basename(path)}", flush=True)
        with torch.no_grad():
            decoded = VAE.decode(lat).sample[0].clamp(-1, 1)
            img = ((decoded + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
            img = img.permute(1, 2, 0).cpu().numpy()
        Image.fromarray(img).save(path)
        print(f"[saved ] ✓ {os.path.basename(path)}", flush=True)
        work_q.task_done()

threading.Thread(target=decoder_worker, daemon=True).start()

# ────────────────── cadence / helpers ──────────────────
STEPS=cfg['steps']; GEN_STEPS=16
S_TOTAL=max(1,min(8,STEPS//500)); S_EVERY=max(1,STEPS//S_TOTAL)

@torch.no_grad()
def save_ckpt(i, final=False):
    save_file(get_peft_model_state_dict(UNET), os.path.join(ckpt_dir,'final.safetensors' if final else f'ckpt_step{i:04}.safetensors'))

@torch.no_grad()
def enqueue_sample(i, idx):
    UNET.eval()
    # unpack unconditional + conditional embeddings
    uncond_emb, cond_emb = prompt_cache[idx % len(prompt_cache)]
    emb = torch.cat([uncond_emb, cond_emb], dim=0)  # [2, seq_len, hidden]

    # prepare two identical latents
    z0 = torch.randn(
        1, 4,
        VAE.config.sample_size // 8,
        VAE.config.sample_size // 8,
        device=GPU,
        dtype=torch.float16
    )
    z = torch.cat([z0, z0], dim=0)  # [2,4,64,64]

    SCHED.set_timesteps(GEN_STEPS, device=GPU)
    for t in SCHED.timesteps:
        eps = UNET.model(
            sample=z,
            timestep=t,
            encoder_hidden_states=emb
        ).sample
        e0, e1 = eps.chunk(2)
        guided = e0 + 7.5 * (e1 - e0)
        z = SCHED.step(guided, t, z).prev_sample.half()

    work_q.put(
        (
            (z / 0.18215).to(CPU).float(),
            os.path.join(smpl_dir, f"step{i:04}.png")
        )
    )
    UNET.train()


# ────────────────── training loop ──────────────────
print('Training…'); step=0; n=len(lat_cache)
while step<STEPS:
    idx = step % n
    lat = lat_cache[idx].unsqueeze(0).to(GPU, non_blocking=True)
    ids = id_cache[idx].unsqueeze(0).to(GPU, non_blocking=True)
    t = torch.randint(0,SCHED.config.num_train_timesteps,(1,),device=GPU)
    noise=torch.randn_like(lat)
    noisy=SCHED.add_noise(lat,noise,t)
    emb = text_emb_cache[idx].unsqueeze(0).to(GPU)  # [1, seq_len, hidden]
    loss=F.mse_loss(UNET.model(sample=noisy, timestep=t, encoder_hidden_states=emb).sample, noise)
    OPT.zero_grad(); loss.backward(); OPT.step(); step+=1

    if step%100==0 or step==STEPS:
        print(f'step {step}/{STEPS} loss={round(loss.item(),4)}')
    
    if step%250==0 or step==STEPS:
        enqueue_sample(step, step//S_EVERY-1)
        save_ckpt(step)

# ────────────────── finish ──────────────────
work_q.join(); work_q.put((None,'')); save_ckpt(step,final=True)
print('Done. Models →',ckpt_dir,'| Samples →',smpl_dir)
