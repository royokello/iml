#!/usr/bin/env python
"""
LoRA Training Script for Stable Diffusion v1.5
==============================================

Trains low‑rank adaptation (LoRA) modules on a Stable Diffusion 1.5 EMA‑only model with
periodic **sample generation _and_ LoRA checkpointing during training.**

New behaviour
-------------
* Each time the script renders a sample image it writes a matching
  LoRA checkpoint: `ckpt_stepXXXX.safetensors`.
* At the very end a **final checkpoint** is saved as `final.safetensors`.
* You can now specify **`--name`** to replace placeholders in sample prompts.

Run example
-----------
```bash
python train.py \
  --model  /models/sd15_ema \
  --images ./train_images \
  --output ./output_lora \
  --preset face \
  --name JohnDoe
```
Yields prompts like “portrait photo of JohnDoe in natural light.”
"""
import argparse, os, random
from glob import glob

import torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from safetensors.torch import save_file
from torch.optim import AdamW

# ────────────────────────────────────────────────────────────
class ImageCaptionDataset(Dataset):
    def __init__(self, folder: str, tok: CLIPTokenizer, res: int = 512):
        self.paths = sorted(glob(os.path.join(folder, '*.jpg')))
        self.tok = tok
        self.prep = transforms.Compose([
            transforms.Resize((res, res)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        p = self.paths[i]
        img = self.prep(Image.open(p).convert('RGB'))
        cap = open(p.rsplit('.',1)[0]+'.txt','r',encoding='utf-8').read().strip()
        ids = self.tok(cap, return_tensors='pt', padding='max_length', truncation=True, max_length=77).input_ids[0]
        return img, ids

# ────────────────────────────────────────────────────────────
PRESETS = {
    'face':   {
        'lr':1e-4, 'steps':2000, 'rank':16, 'alpha':16,
        'prompts':[
            "portrait photo of {name} in natural light",
            "35 mm film close‑up of {name} smiling"
        ]
    },
    'person': {
        'lr':1e-4, 'steps':3000, 'rank':16, 'alpha':16,
        'prompts':[
            "full‑body photo of {name} standing in a park",
            "action shot of {name} jumping over a puddle"
        ]
    },
    'object': {
        'lr':1e-4, 'steps':3000, 'rank':32, 'alpha':32,
        'prompts':[
            "studio shot of {name} on white background",
            "macro photograph of {name} with dramatic lighting"
        ]
    },
    'style':  {
        'lr':5e-5, 'steps':1000, 'rank':8, 'alpha':8,
        'prompts':[
            "landscape painting in the style of {name}",
            "portrait of a cat in the style of {name}"
        ]
    },
}

# ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--model',  required=True, help='Diffusers‑style base model dir')
parser.add_argument('--images', required=True, help='Folder with JPG+TXT training pairs')
parser.add_argument('--output', required=True, help='Output directory for LoRA and samples')
parser.add_argument('--preset', choices=PRESETS.keys(), default='object', help='Training preset')
parser.add_argument('--name',   required=True, help='Name to inject into sample prompts')
args = parser.parse_args()

random.seed(42); torch.manual_seed(42)
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if DEVICE.type != 'cuda':
    raise RuntimeError('CUDA GPU required')

cfg = PRESETS[args.preset]
print('Preset config:', {k:v for k,v in cfg.items() if k!='prompts'})

# Load components
TOK = CLIPTokenizer.from_pretrained(os.path.join(args.model,'tokenizer'))
TXT = CLIPTextModel.from_pretrained(
    os.path.join(args.model,'text_encoder'), torch_dtype=torch.float16
).to(DEVICE).eval()
for p in TXT.parameters(): p.requires_grad=False

UNET = UNet2DConditionModel.from_pretrained(
    os.path.join(args.model,'unet'), torch_dtype=torch.float16
).to(DEVICE).train()
UNET = get_peft_model(
    UNET,
    LoraConfig(
        r=cfg['rank'],
        lora_alpha=cfg['alpha'],
        target_modules=['to_q','to_k','to_v','to_out'],
        task_type='CONDITIONING'
    )
)

VAE   = AutoencoderKL.from_pretrained(
    os.path.join(args.model,'vae'), torch_dtype=torch.float16
).to(DEVICE).eval()
SCHED = DDIMScheduler.from_pretrained(args.model, subfolder='scheduler')

# Data & optimizer
dataset = ImageCaptionDataset(args.images, TOK, VAE.config.sample_size)
loader  = DataLoader(dataset, batch_size=1, shuffle=True)
optimizer = AdamW(UNET.parameters(), lr=cfg['lr'])

# Sampling cadence
STEPS        = cfg['steps']
sample_total = max(1, min(8, STEPS//500))
sample_every = max(1, STEPS // sample_total)
GEN_STEPS    = 16

os.makedirs(args.output, exist_ok=True)
sample_dir = os.path.join(args.output,'samples')
os.makedirs(sample_dir, exist_ok=True)

# Helpers
@torch.no_grad()
def save_checkpoint(step_idx:int, final:bool=False):
    name = 'final.safetensors' if final else f'ckpt_step{step_idx:04}.safetensors'
    save_file(
        get_peft_model_state_dict(UNET),
        os.path.join(args.output, name)
    )

@torch.no_grad()
def sample_image(step_idx:int, prompt_template:str):
    UNET.eval()
    prompt = prompt_template.format(name=args.name)
    ids = TOK(["",prompt], return_tensors='pt', padding=True).input_ids.to(DEVICE)
    emb = TXT(ids)[0].half(); unc, con = emb.chunk(2); emb = torch.cat([unc,con])
    z = torch.randn(
        1,4,
        VAE.config.sample_size//8,
        VAE.config.sample_size//8,
        device=DEVICE, dtype=torch.float16
    )
    SCHED.set_timesteps(GEN_STEPS, device=DEVICE)
    for t in SCHED.timesteps:
        eps = UNET(torch.cat([z,z]), t, encoder_hidden_states=emb).sample
        e0,e1 = eps.chunk(2)
        eps = e0 + 7.5*(e1-e0)
        z = SCHED.step(eps, t, z).prev_sample.half()
    img = VAE.decode(z/0.18215).sample[0].clamp(-1,1)
    arr = ((img+1)/2).cpu().permute(1,2,0).numpy()
    Image.fromarray((arr*255).round().astype('uint8')).save(
        os.path.join(sample_dir, f'step{step_idx:04}.png')
    )
    UNET.train()

# Training loop
print('Training…')
step = 0
while step < STEPS:
    for imgs, ids in loader:
        imgs, ids = imgs.to(DEVICE), ids.to(DEVICE)
        with torch.no_grad():
            lat = VAE.encode(imgs*2-1).latent_dist.sample() * VAE.config.scaling_factor
        t = torch.randint(0, SCHED.config.num_train_timesteps, (1,), device=DEVICE)
        noise = torch.randn_like(lat)
        noisy_lat = SCHED.add_noise(lat, noise, t)
        with torch.no_grad():
            txt_emb = TXT(ids)[0].half()
        pred = UNET(noisy_lat, t, encoder_hidden_states=txt_emb).sample
        loss = F.mse_loss(pred, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step += 1
        if step % 100 == 0 or step == STEPS:
            print(f"step {step}/{STEPS}  loss={loss.item():.4f}")
        if step % sample_every == 0 or step == STEPS:
            tmpl = cfg['prompts'][ ((step//sample_every) -1) % len(cfg['prompts']) ]
            print(f"⍟ sample & ckpt @ {step}: '{tmpl.format(name=args.name)[:60]}…'")
            sample_image(step, tmpl)
            save_checkpoint(step)

        if step >= STEPS:
            break

# Final checkpoint
save_checkpoint(step, final=True)
print('Training complete; outputs in', args.output)
