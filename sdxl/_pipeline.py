import inspect
import time
from typing import Any, Dict, List, Union, Optional
import torch
from transformers import (
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)

from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.loaders import TextualInversionLoaderMixin
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from diffusers.utils.torch_utils import randn_tensor

from sdxl.unet import SDXLUNet

def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    r"""
    Rescales `noise_cfg` tensor based on `guidance_rescale` to improve image quality and fix overexposure. Based on
    Section 3.4 from [Common Diffusion Noise Schedules and Sample Steps are
    Flawed](https://huggingface.co/papers/2305.08891).

    Args:
        noise_cfg (`torch.Tensor`):
            The predicted noise tensor for the guided diffusion process.
        noise_pred_text (`torch.Tensor`):
            The predicted noise tensor for the text-guided diffusion process.
        guidance_rescale (`float`, *optional*, defaults to 0.0):
            A rescale factor applied to the noise predictions.

    Returns:
        noise_cfg (`torch.Tensor`): The rescaled noise prediction tensor.
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

class StableDiffusionXLPipeline(
    DiffusionPipeline,
    StableDiffusionMixin
):
    model_cpu_offload_seq = "text_encoder->text_encoder_2->unet->vae"
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        unet: SDXLUNet,
        scheduler: KarrasDiffusionSchedulers,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def encode_prompt(
        self,
        prompt: str,
        device: Optional[torch.device] = None,
        negative_prompt: Optional[str] = None,
        clip_skip: Optional[int] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str`):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            negative_prompt (`str`, *optional*):
                The prompt not to guide the image generation.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        """
        device = device or self._execution_device

        if not isinstance(prompt, str):
            raise TypeError("`prompt` must be a string when batching support is disabled.")
        prompt_texts = [prompt]

        # Define tokenizers and text encoders
        tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]
        text_encoders = (
            [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
        )

        # textual inversion: process multi-vector tokens if necessary
        prompt_embeds_list = []
        pooled_prompt_embeds = None
        prompts = [list(prompt_texts) for _ in range(len(tokenizers))]
        for current_prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
            if text_encoder is not None:
                text_encoder.to(device=device)
            if isinstance(self, TextualInversionLoaderMixin):
                current_prompt = self.maybe_convert_prompt(current_prompt, tokenizer)

            text_inputs = tokenizer(
                current_prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )

            text_input_ids = text_inputs.input_ids
            untruncated_ids = tokenizer(current_prompt, padding="longest", return_tensors="pt").input_ids

            encoder_output = text_encoder(text_input_ids.to(device), output_hidden_states=True)

            # We are only ALWAYS interested in the pooled output of the final text encoder
            if pooled_prompt_embeds is None and encoder_output[0].ndim == 2:
                pooled_prompt_embeds = encoder_output[0]

            if clip_skip is None:
                hidden_states = encoder_output.hidden_states[-2]
            else:
                # "2" because SDXL always indexes from the penultimate layer.
                hidden_states = encoder_output.hidden_states[-(clip_skip + 2)]

            prompt_embeds_list.append(hidden_states)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        # get unconditional embeddings for classifier free guidance
        negative_prompt_embeds = None
        negative_pooled_prompt_embeds = None
        zero_out_negative_prompt = negative_prompt is None
        if zero_out_negative_prompt:
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
        else:
            if negative_prompt is not None and not isinstance(negative_prompt, str):
                raise TypeError("`negative_prompt` must be a string when batching support is disabled.")

            negative_prompt = negative_prompt or ""
            negative_prompt_texts = [[negative_prompt] for _ in range(len(tokenizers))]

            negative_prompt_embeds_list = []
            for current_negative_prompt, tokenizer, text_encoder in zip(
                negative_prompt_texts, tokenizers, text_encoders
            ):
                if isinstance(self, TextualInversionLoaderMixin):
                    current_negative_prompt = self.maybe_convert_prompt(current_negative_prompt, tokenizer)

                max_length = prompt_embeds.shape[1]
                uncond_input = tokenizer(
                    current_negative_prompt,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                encoder_output = text_encoder(
                    uncond_input.input_ids.to(device),
                    output_hidden_states=True,
                )

                # We are only ALWAYS interested in the pooled output of the final text encoder
                if negative_pooled_prompt_embeds is None and encoder_output[0].ndim == 2:
                    negative_pooled_prompt_embeds = encoder_output[0]
                hidden_states = encoder_output.hidden_states[-2]

                negative_prompt_embeds_list.append(hidden_states)

            negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

        if self.text_encoder_2 is not None:
            prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
        else:
            prompt_embeds = prompt_embeds.to(dtype=self.unet.dtype, device=device)

        if negative_prompt_embeds is not None:
            if self.text_encoder_2 is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
            else:
                negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.unet.dtype, device=device)

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(image, return_tensors="pt").pixel_values

        image = image.to(device=device, dtype=dtype)
        if output_hidden_states:
            image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
            image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
            uncond_image_enc_hidden_states = self.image_encoder(
                torch.zeros_like(image), output_hidden_states=True
            ).hidden_states[-2]
            uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
                num_images_per_prompt, dim=0
            )
            return image_enc_hidden_states, uncond_image_enc_hidden_states
        else:
            image_embeds = self.image_encoder(image).image_embeds
            image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            uncond_image_embeds = torch.zeros_like(image_embeds)

            return image_embeds, uncond_image_embeds
        
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://huggingface.co/papers/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs
    
    def _get_add_time_ids(
        self, original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim=None
    ):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        passed_add_embed_dim = (
            self.unet.config["addition_time_embed_dim"] * len(add_time_ids) + text_encoder_projection_dim
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids


    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator):
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        height: int,
        width: int,
        num_inference_steps: int = 20,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[str] = None,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        gpu_device: Optional[torch.device] = None,
        
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str`):
                Text prompt to guide the image generation.
            height (`int`):
                Required height in pixels (a multiple of 8) for the generated image.
            width (`int`):
                Required width in pixels (a multiple of 8) for the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 5.0):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
            negative_prompt (`str`, *optional*):
                The prompt not to guide the image generation. Ignored when not using guidance (i.e., if `guidance_scale`
                is less than `1`).
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://huggingface.co/papers/2010.02502. Only
                applies to [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                Torch generator to make generation deterministic.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://huggingface.co/papers/2305.08891) `guidance_scale` is defined as `φ` in equation 16. of
                [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://huggingface.co/papers/2305.08891). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.
            gpu_device (`torch.device`, *optional*):
                Torch device to target for GPU workloads; defaults to the pipeline's `_execution_device` when unset.
        The pipeline always returns `PIL.Image.Image` instances.
        """

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        batch_size = 1

        device = gpu_device or self._execution_device

        # 3. Encode input prompt
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            device=device,
            negative_prompt=negative_prompt,
            clip_skip=self._clip_skip,
        )

        # Offload the text encoders now that embeddings are materialized.
        if self.text_encoder is not None:
            self.text_encoder.to(device="cpu")
        if self.text_encoder_2 is not None:
            self.text_encoder_2.to(device="cpu")

        torch.cuda.empty_cache()
        self.unet.to(device=device)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        # 5. Prepare latent variables
        latents = self.prepare_latents(
            batch_size=batch_size,
            num_channels_latents=self.unet.config["in_channels"],
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
        )
        print(f"Latent {tuple(latents.shape)} ...")

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        size_tuple = (height, width)
        add_time_ids = self._get_add_time_ids(
            size_tuple,
            (0, 0),
            size_tuple,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        negative_add_time_ids = add_time_ids

        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
        add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size, 1)

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        timestep_cond = None

        self._num_timesteps = len(timesteps)
        
        diffusion_start = time.perf_counter()
        for i, t in enumerate(timesteps):
            step_start = time.perf_counter()
            # always use classifier free guidance
            latent_model_input = torch.cat([latents, latents], dim=0)

            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=self._cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self._guidance_scale * (noise_pred_text - noise_pred_uncond)

            if self._guidance_rescale > 0.0:
                # Based on 3.4. in https://huggingface.co/papers/2305.08891
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self._guidance_rescale)

            # compute the previous noisy sample x_t -> x_t-1
            latents_dtype = latents.dtype
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
            if latents.dtype != latents_dtype:
                if torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                    latents = latents.to(latents_dtype)

            step_duration = time.perf_counter() - step_start
            print(f"Step {i + 1}/{len(timesteps)} ({step_duration:.2f}s)")

        total_duration = time.perf_counter() - diffusion_start
        print(f"Total diffusion time: {total_duration:.2f}s")

        # Move UNet off GPU before loading the VAE for decoding.
        self.unet.to(device="cpu")
        torch.cuda.empty_cache()
        self.vae.to(device=device, dtype=torch.float32)

        vae_dtype = next(iter(self.vae.post_quant_conv.parameters())).dtype
        latents = latents.to(dtype=vae_dtype)

        # unscale/denormalize the latents
        # denormalize with the mean and std if available and not None
        has_latents_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
        has_latents_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None
        if has_latents_mean and has_latents_std:
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
            )
            latents_std = (
                torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
            )
            latents = latents * latents_std / self.vae.config.scaling_factor + latents_mean
        else:
            latents = latents / self.vae.config.scaling_factor

        image = self.vae.decode(latents, return_dict=False)[0]

        image = self.image_processor.postprocess(image, output_type="pil")

        # Offload vae
        self.maybe_free_model_hooks()
        return image
