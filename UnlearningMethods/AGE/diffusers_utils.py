#!/usr/bin/env python3
"""Shared Diffusers helpers for AGE."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Sequence

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel


class StableDiffusionWrapper(torch.nn.Module):
    """Thin wrapper exposing the CompVis-style API expected by legacy AGE utilities."""

    def __init__(self, model_path: str, device: torch.device, dtype: torch.dtype):
        super().__init__()

        if Path(model_path).is_dir():
            pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=dtype)
        else:
            pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=dtype)

        pipe.set_progress_bar_config(disable=True)
        if pipe.safety_checker is not None:
            pipe.safety_checker = None
            pipe.requires_safety_checker = False

        self.pipe = pipe
        self.device = torch.device(device)
        self.dtype = dtype
        self.pipe.to(self.device)

        # Keep the text encoder in fp32 even when UNet/VAE run in fp16. CLIP becomes unstable in fp16
        # and quickly produces NaNs that then propagate into eps predictions.
        if self.pipe.text_encoder is not None:
            self.pipe.text_encoder = self.pipe.text_encoder.to(self.device, dtype=torch.float32)
            self.text_encoder_dtype = torch.float32
        else:
            self.text_encoder_dtype = self.dtype

        unet = UNet2DConditionModel.from_pretrained(
            model_path,
            subfolder="unet",
            torch_dtype=dtype,
            use_safetensors=True
        ).to("cuda")
        pipe.unet = unet
        
        self.unet = pipe.unet
        self.vae = pipe.vae
        self.text_encoder = pipe.text_encoder
        self.tokenizer = pipe.tokenizer
        self.scheduler_config = pipe.scheduler.config

        # Mimic the CompVis API used throughout AGE.
        self.model = SimpleNamespace(diffusion_model=self.unet)
        self.cond_stage_model = self.text_encoder

    def get_learned_conditioning(self, prompts: Sequence[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            list(prompts),
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.text_encoder(**inputs)
        return outputs[0].to(dtype=self.dtype)

    def decode_first_stage(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents / self.pipe.vae.config.scaling_factor
        decoded = self.vae.decode(latents).sample
        return decoded

    def apply_model(
        self, latents: torch.Tensor, timesteps: torch.Tensor, encoder_hidden_states: torch.Tensor
    ) -> torch.Tensor:
        return self.unet(latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample


def load_diffusers_model(model_path: str, device: torch.device, dtype: Optional[torch.dtype] = None) -> StableDiffusionWrapper:
    if dtype is None:
        dtype = torch.float16 if device.type == "cuda" and torch.cuda.is_available() else torch.float32
    return StableDiffusionWrapper(model_path, device, dtype)


def sample_latents(
    model: StableDiffusionWrapper,
    conditioning: torch.Tensor,
    height: int,
    width: int,
    ddim_steps: int,
    guidance_scale: float,
    eta: float = 0.0,
    start_code: torch.Tensor | None = None,
    start_step: Optional[int] = None,
    stop_step: Optional[int] = None,
) -> torch.Tensor:
    """DDIM sampling with optional partial denoising controls."""

    scheduler = DDIMScheduler.from_config(model.scheduler_config)
    scheduler.set_timesteps(ddim_steps, device=model.device)
    scheduler.eta = eta

    batch_size = conditioning.shape[0]
    if start_code is None:
        latents = torch.randn(
            batch_size,
            model.unet.in_channels,
            height // 8,
            width // 8,
            device=model.device,
            dtype=model.dtype,
        )
        latents = latents * scheduler.init_noise_sigma
    else:
        latents = start_code.to(device=model.device, dtype=model.dtype)

    uc = None
    if guidance_scale != 1.0:
        uc = model.get_learned_conditioning([""] * batch_size)

    conditioning = conditioning.to(device=model.device, dtype=model.dtype)
    total_steps = len(scheduler.timesteps)

    start_idx = 0
    if start_step is not None and start_step >= 0:
        start_idx = min(int(start_step), total_steps - 1)
    end_idx = total_steps if stop_step is None else max(0, min(int(stop_step), total_steps))

    for idx, t in enumerate(scheduler.timesteps):
        if idx < start_idx:
            continue
        if idx >= end_idx:
            break

        latent_input = latents
        text_embeddings = conditioning
        if uc is not None:
            latent_input = torch.cat([latents, latents], dim=0)
            text_embeddings = torch.cat([uc, conditioning], dim=0)

        latent_input = scheduler.scale_model_input(latent_input, t)
        noise_pred = model.apply_model(latent_input, t, text_embeddings)

        if uc is not None:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        latents = scheduler.step(noise_pred, t, latents).prev_sample

    return latents
