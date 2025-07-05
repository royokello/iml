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
LoRA Training Script for Stable Diffusion v1.5 (manual LoRA)
==============================================

This version injects LoRA modules manually (without PEFT) and saves both weights and metadata.
"""
import argparse, os, random, threading, queue, json
from glob import glob

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from safetensors.torch import save_file
from torch.optim import AdamW

# manual LoRA utilities
from .lora import inject_lora, extract_lora_state_dict, LoRALinear

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
TXT = CLIPTextModel.from_pretrained(
    os.path.join(args.model, 'text_encoder'), torch_dtype=torch.float16
).to(GPU).eval()
for p in TXT.parameters(): p.requires_grad = False

# ────────────────── load UNet & inject LoRA ──────────────────
UNET = UNet2DConditionModel.from_pretrained(
    os.path.join(args.model,'unet'), torch_dtype=torch.float16
).to(GPU).train()
# inject manual LoRA
TARGETS = ['to_q','to_k','to_v','to_out.0']
UNET = inject_lora(UNET, TARGETS, cfg['rank'], cfg['alpha'])
# freeze all weights, then unfreeze only LoRA parameters
for p in UNET.parameters():
    p.requires_grad = False
for module in UNET.modules():
    if isinstance(module, LoRALinear):
        module.A.requires_grad = True
        module.B.requires_grad = True

# ────────────────── load VAE (CPU) & scheduler ──────────────────
VAE  = AutoencoderKL.from_pretrained(
    os.path.join(args.model,'vae'), torch_dtype=torch.float32
).to(CPU).eval()
SCHED= DDIMScheduler.from_pretrained(args.model, subfolder='scheduler')

# ────────────────── scan images & captions ──────────────────
IMG_EXT = ('.jpg','.jpeg','.png','.webp')
paths   = sorted([p for p in glob(os.path.join(img_dir,'*')) if p.lower().endswith(IMG_EXT)])
if not paths:
    raise ValueError(f"No images in {img_dir}")

print('Encoding', len(paths), 'images to latents (fp32→fp16, CPU)…')
lat_cache=[]; id_cache=[]
prep = transforms.Compose([
    transforms.Resize((VAE.config.sample_size, VAE.config.sample_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3),
])
with torch.no_grad():
    for p in paths:
        img = prep(Image.open(p).convert('RGB'))
        latent = VAE.encode(img.unsqueeze(0)).latent_dist.sample() * VAE.config.scaling_factor
        lat_cache.append(latent.squeeze(0).half())
        cap_p = os.path.splitext(p)[0]+'.txt'
        if not os.path.exists(cap_p): raise FileNotFoundError(cap_p)
        caption = open(cap_p,'r',encoding='utf-8').read().strip()
        ids = TOK(caption, return_tensors='pt', padding='max_length', truncation=True, max_length=77).input_ids[0]
        id_cache.append(ids)
print('Caching done.')

# ────────────────── optimizer: only LoRA params ──────────────────
lora_params = [p for p in UNET.parameters() if p.requires_grad]
OPT = AdamW(lora_params, lr=cfg['lr'])

# ────────────────── prompt embeddings cache ──────────────────
prompt_cache=[]
for t in cfg['prompts']:
    prompt = t.format(name=args.name)
    ids = TOK(["", prompt], return_tensors="pt", padding=True, truncation=True, max_length=77).input_ids.to(GPU)
    hs = TXT(ids)[0].half().chunk(2)
    prompt_cache.append((hs[0], hs[1]))

# ────────────────── training-caption embeddings cache ──────────────────
print('Caching', len(id_cache), 'training captions → text embeddings…')
text_emb_cache=[]
with torch.no_grad():
    for ids in id_cache:
        emb = TXT(ids.unsqueeze(0).to(GPU))[0].half()
        text_emb_cache.append(emb.squeeze(0).cpu())
print('Cached', len(text_emb_cache), 'training embeddings')
del TXT; torch.cuda.empty_cache()

# ────────────────── threaded VAE decode ──────────────────
work_q: queue.Queue[tuple[torch.Tensor,str]] = queue.Queue()
def decoder_worker():
    while True:
        lat, path = work_q.get()
        if lat is None: break
        with torch.no_grad():
            img = ((VAE.decode(lat).sample[0].clamp(-1,1)+1)/2*255).clamp(0,255).to(torch.uint8)
            Image.fromarray(img.permute(1,2,0).cpu().numpy()).save(path)
        work_q.task_done()
threading.Thread(target=decoder_worker, daemon=True).start()

STEPS=cfg['steps']; GEN_STEPS=16

@torch.no_grad()
def save_ckpt(i, final=False):
    sd = extract_lora_state_dict(UNET)
    meta = {
        'r': str(cfg['rank']),
        'lora_alpha': str(cfg['alpha']),
        'target_modules': json.dumps(TARGETS),    }
    fname = 'final.safetensors' if final else f'ckpt_step{i:04}.safetensors'
    save_file(sd, os.path.join(ckpt_dir,fname), metadata=meta)

@torch.no_grad()
def enqueue_sample(i, idx):
    UNET.eval()
    uncond, cond = prompt_cache[idx % len(prompt_cache)]
    emb = torch.cat([uncond, cond], dim=0)
    z0 = torch.randn(1,4,VAE.config.sample_size//8,VAE.config.sample_size//8, device=GPU, dtype=torch.float16)
    z = torch.cat([z0, z0], dim=0)
    SCHED.set_timesteps(GEN_STEPS, device=GPU)
    for t in SCHED.timesteps:
        eps = UNET(z, t, emb).sample
        e0, e1 = eps.chunk(2)
        z = SCHED.step(e0 + 7.5*(e1-e0), t, z).prev_sample.half()
    work_q.put(((z/0.18215).to(CPU).float(), os.path.join(smpl_dir, f"step{i:04}.png")))
    UNET.train()

# ────────────────── training loop ──────────────────
print('Training…'); step=0; n=len(lat_cache)
while step<STEPS:
    idx = step % n
    lat = lat_cache[idx].unsqueeze(0).to(GPU)
    emb = text_emb_cache[idx].unsqueeze(0).to(GPU)
    t = torch.randint(0, SCHED.config.num_train_timesteps, (1,), device=GPU)
    noise = torch.randn_like(lat)
    noisy = SCHED.add_noise(lat, noise, t)
    loss = F.mse_loss(UNET(noisy, t, emb).sample, noise)
    OPT.zero_grad(); loss.backward(); OPT.step(); step+=1
    if step%100==0 or step==STEPS:
        print(f'step {step}/{STEPS} loss={loss.item():.4f}')
    if step%250==0 or step==STEPS:
        enqueue_sample(step, step//(STEPS//8 or 1))
        save_ckpt(step)
# finish
work_q.join(); work_q.put((None,'')); save_ckpt(step, final=True)
print('Done. Models→',ckpt_dir,'Samples→',smpl_dir)
