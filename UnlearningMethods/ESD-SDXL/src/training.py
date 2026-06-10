import random
from dataclasses import dataclass

import torch

from src.utils import truncate_for_log


@dataclass
class ConditioningCache:
    """Precomputed SDXL conditioning reused across ESD optimization steps."""

    concept_prompts: list
    erase_from_prompts: list
    concept_embeds: list
    concept_pooled_embeds: list
    null_embeds: torch.Tensor
    null_pooled_embeds: torch.Tensor
    erase_from_embeds: list
    erase_from_pooled_embeds: list
    add_time_ids: torch.Tensor
    timestep_cond: torch.Tensor | None

    @property
    def use_erase_from(self):
        return len(self.erase_from_prompts) > 0


def prepare_conditioning_cache(pipe, args, torch_dtype):
    """Encode target, null, and optional erase-from prompts once before training."""
    batch_size = 1
    concept_prompt_input = args.target_concepts if len(args.target_concepts) > 1 else args.target_concepts[0]
    concept_negative_prompt = [""] * len(args.target_concepts) if isinstance(concept_prompt_input, list) else ""

    with torch.no_grad():
        concept_embeds, null_embeds, concept_pooled, null_pooled = pipe.encode_prompt(
            prompt=concept_prompt_input,
            device=args.device,
            num_images_per_prompt=batch_size,
            do_classifier_free_guidance=True,
            negative_prompt=concept_negative_prompt,
        )

        concept_embeds_cache = []
        concept_pooled_cache = []
        for idx in range(len(args.target_concepts)):
            concept_embeds_cache.append(concept_embeds[idx : idx + 1].to(args.device))
            concept_pooled_cache.append(concept_pooled[idx : idx + 1].to(args.device))

        null_embeds = null_embeds[:batch_size].to(args.device)
        null_pooled = null_pooled[:batch_size].to(args.device)
        add_time_ids = _get_add_time_ids(pipe, args, concept_embeds).repeat(batch_size, 1)
        timestep_cond = _get_timestep_conditioning(pipe, args, torch_dtype, batch_size)

        erase_from_embeds_cache = []
        erase_from_pooled_cache = []
        if args.erase_from_prompts_list:
            erase_from_embeds, _, erase_from_pooled, _ = pipe.encode_prompt(
                prompt=args.erase_from_prompts_list,
                device=args.device,
                num_images_per_prompt=batch_size,
                do_classifier_free_guidance=False,
                negative_prompt="",
            )
            for idx in range(len(args.erase_from_prompts_list)):
                erase_from_embeds_cache.append(erase_from_embeds[idx : idx + 1].to(args.device))
                erase_from_pooled_cache.append(erase_from_pooled[idx : idx + 1].to(args.device))

    return ConditioningCache(
        concept_prompts=args.target_concepts,
        erase_from_prompts=args.erase_from_prompts_list,
        concept_embeds=concept_embeds_cache,
        concept_pooled_embeds=concept_pooled_cache,
        null_embeds=null_embeds,
        null_pooled_embeds=null_pooled,
        erase_from_embeds=erase_from_embeds_cache,
        erase_from_pooled_embeds=erase_from_pooled_cache,
        add_time_ids=add_time_ids,
        timestep_cond=timestep_cond,
    )


def run_esd_training_step(pipe, base_unet, esd_unet, cache, args, criteria):
    """Run one ESD-SDXL optimization step and return the base ESD loss plus log fields."""
    pipe.unet = base_unet
    batch_size = 1
    run_till_timestep = random.randint(0, args.num_inference_steps - 1)
    scheduler_timestep = pipe.scheduler.timesteps[run_till_timestep]
    seed_value = random.randint(0, 2**15)

    selected = _sample_conditioning(cache)

    with torch.no_grad():
        xt = pipe(
            selected["sampling_prompt"],
            num_images_per_prompt=batch_size,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            run_till_timestep=run_till_timestep,
            generator=_make_generator(args.device, seed_value),
            output_type="latent",
            height=args.height,
            width=args.width,
        ).images

        noise_pred_erase = _predict_noise(
            pipe.unet,
            xt,
            scheduler_timestep,
            selected["concept_embed"],
            selected["concept_pooled"],
            cache,
        )
        noise_pred_null = _predict_noise(
            pipe.unet,
            xt,
            scheduler_timestep,
            cache.null_embeds,
            cache.null_pooled_embeds,
            cache,
        )
        if cache.use_erase_from:
            noise_pred_erase_from = _predict_noise(
                pipe.unet,
                xt,
                scheduler_timestep,
                selected["erase_from_embed"],
                selected["erase_from_pooled"],
                cache,
            )
        else:
            noise_pred_erase_from = noise_pred_erase

    pipe.unet = esd_unet
    train_embed = selected["erase_from_embed"] if cache.use_erase_from else selected["concept_embed"]
    train_pooled = selected["erase_from_pooled"] if cache.use_erase_from else selected["concept_pooled"]
    noise_pred_esd = _predict_noise(
        pipe.unet,
        xt,
        scheduler_timestep,
        train_embed,
        train_pooled,
        cache,
    )

    target = noise_pred_erase_from - args.negative_guidance * (noise_pred_erase - noise_pred_null)
    esd_loss = criteria(noise_pred_esd, target.detach())
    logs = {
        "timestep": run_till_timestep,
        "erase": f"'{truncate_for_log(selected['concept_prompt'])}'",
        "prompt": f"'{truncate_for_log(selected['sampling_prompt'])}'",
    }
    return esd_loss, logs


def _sample_conditioning(cache):
    concept_idx = random.randrange(len(cache.concept_prompts))
    concept_prompt = cache.concept_prompts[concept_idx]
    selected = {
        "concept_prompt": concept_prompt,
        "sampling_prompt": concept_prompt,
        "concept_embed": cache.concept_embeds[concept_idx],
        "concept_pooled": cache.concept_pooled_embeds[concept_idx],
        "erase_from_embed": None,
        "erase_from_pooled": None,
    }
    if cache.use_erase_from:
        erase_idx = random.randrange(len(cache.erase_from_prompts))
        selected["sampling_prompt"] = cache.erase_from_prompts[erase_idx]
        selected["erase_from_embed"] = cache.erase_from_embeds[erase_idx]
        selected["erase_from_pooled"] = cache.erase_from_pooled_embeds[erase_idx]
    return selected


def _predict_noise(unet, latents, timestep, prompt_embeds, pooled_embeds, cache):
    return unet(
        latents,
        timestep,
        encoder_hidden_states=prompt_embeds,
        timestep_cond=cache.timestep_cond,
        cross_attention_kwargs=None,
        added_cond_kwargs={"text_embeds": pooled_embeds, "time_ids": cache.add_time_ids},
        return_dict=False,
    )[0]


def _get_add_time_ids(pipe, args, concept_embeds):
    if pipe.text_encoder_2 is None:
        projection_dim = int(concept_embeds.shape[-1])
    else:
        projection_dim = pipe.text_encoder_2.config.projection_dim
    return pipe._get_add_time_ids(
        (args.height, args.width),
        (0, 0),
        (args.height, args.width),
        dtype=concept_embeds.dtype,
        text_encoder_projection_dim=projection_dim,
    ).to(args.device)


def _get_timestep_conditioning(pipe, args, torch_dtype, batch_size):
    if pipe.unet.config.time_cond_proj_dim is None:
        return None
    guidance_scale_tensor = torch.tensor(args.guidance_scale - 1).repeat(batch_size)
    return pipe.get_guidance_scale_embedding(
        guidance_scale_tensor,
        embedding_dim=pipe.unet.config.time_cond_proj_dim,
    ).to(device=args.device, dtype=torch_dtype)


def _make_generator(device, seed):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        return torch.Generator(device=device).manual_seed(seed)
    return torch.Generator().manual_seed(seed)
