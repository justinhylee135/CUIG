# Standard Library
import os
from pathlib import Path

# Third Party
import torch
from pytorch_lightning import seed_everything
from tqdm import tqdm


# The celebrities used for retention calculation in our paper
DEFAULT_CELEB_SUBSET = [
    "Morgan_Freeman",
    "Keanu_Reeves",
    "George_Takei",
    "Aretha_Franklin",
    "Maya_Angelou",
    "Natalie_Portman",
]


def _build_celeb_subset(target_concepts):
    celeb_subset = list(DEFAULT_CELEB_SUBSET)
    for entry in target_concepts or []:
        # Training code may pass prompts or names with extra whitespace.
        name = str(entry).strip()
        if not name:
            continue
        if name not in celeb_subset:
            celeb_subset.append(name)
    if not celeb_subset:
        raise ValueError("No celebrity names provided in target_concepts; unable to sample.")
    return celeb_subset


def _prepare_output_dirs(iteration, model_save_path):
    # Keep the same output layout used by the other simultaneous regularizers.
    base_dir = Path(model_save_path).parent
    output_dir = os.path.join(base_dir, f"logs/log_{iteration}")
    img_dir = os.path.join(output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    return output_dir, img_dir


def _load_prompts_for_celeb(eval_prompt_dir: Path, celeb: str, max_prompts: int):
    celeb_key = celeb.replace(" ", "_")
    prompt_path = eval_prompt_dir / f"{celeb_key}.txt"

    if prompt_path.exists():
        with open(prompt_path, "r") as handle:
            prompts = [line.strip() for line in handle if line.strip()]
    else:
        print(f"[Simultaneous-Celeb] Missing prompt file for '{celeb}'. Using default portrait prompts.")
        prompts = [f"A portrait of {celeb}."]

    prompts = prompts[:max_prompts]
    if not prompts:
        prompts = [f"A portrait of {celeb}."]
    return prompts


def _build_prompt_cache(celeb_subset, eval_prompt_dir: Path, max_prompts: int, n_samples_per_prompt: int):
    prompt_cache = {}
    total_iterations = 0

    print(f"Using prompt directory: '{eval_prompt_dir}'")
    for celeb in celeb_subset:
        prompts = _load_prompts_for_celeb(eval_prompt_dir, celeb, max_prompts=max_prompts)
        prompt_cache[celeb] = prompts
        total_iterations += len(prompts) * n_samples_per_prompt

    return prompt_cache, total_iterations


def _sample_celeb_images(
    diffusion_pipeline,
    celeb_subset,
    prompt_cache,
    img_dir,
    device,
    n_samples_per_prompt,
    height,
    width,
    steps,
    guidance_scale,
):
    was_training = diffusion_pipeline.unet.training
    diffusion_pipeline.unet.eval()

    print(f"[Simultaneous-Celeb] Sampling for celebrities: {celeb_subset}")
    total_iterations = sum(len(prompt_cache[celeb]) * n_samples_per_prompt for celeb in celeb_subset)
    overall_bar = tqdm(total=total_iterations, desc="Celeb Images", unit="image")

    try:
        for celeb in celeb_subset:
            prompts = prompt_cache[celeb]
            slug = celeb.replace(" ", "_")
            print(f"[Simultaneous-Celeb] Generating images for '{celeb}' ({len(prompts)} prompts)")

            for seed in range(n_samples_per_prompt):
                seed_everything(seed)
                generator = torch.Generator(device=device).manual_seed(seed)

                for idx, prompt in enumerate(prompts):
                    img_save_path = os.path.join(img_dir, f"{slug}_prompt{idx+1}_seed{seed}.jpg")
                    if os.path.exists(img_save_path):
                        overall_bar.update(1)
                        continue

                    image = diffusion_pipeline(
                        prompt=prompt,
                        height=height,
                        width=width,
                        num_inference_steps=steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                    ).images[0]
                    image.save(img_save_path)
                    overall_bar.set_postfix({"celeb": celeb, "prompt": f"{idx+1}/{len(prompts)}", "seed": seed})
                    overall_bar.update(1)
    finally:
        overall_bar.close()
        if was_training:
            diffusion_pipeline.unet.train()


def _save_sampling_checkpoint(diffusion_pipeline, output_dir, iteration, celeb_subset, parameter_group):
    # Simultaneous celeb evaluation runs in a separate env, so this checkpoint is saved
    # here for later offline evaluation.
    ckpt_path = os.path.join(output_dir, f"{iteration}.ckpt")
    checkpoint = {
        "iteration": iteration,
        "global_step": iteration,
        "celebrities": celeb_subset,
    }

    # Save UNet Parameters
    if parameter_group != "text-emb" and hasattr(diffusion_pipeline, "unet"):
        diffusion_pipeline.unet.train()
        checkpoint["unet"] = {}

        # Iterate through UNet parameters and save based on parameter_group
        for name, params in diffusion_pipeline.unet.named_parameters():
            
            # Save only cross-attention key and value projection weights
            if parameter_group == "kv-xattn":
                if 'attn2.to_k' in name or 'attn2.to_v' in name:
                    checkpoint['unet'][name] = params.cpu().clone()

            elif parameter_group == "xattn": # Save all attention projection weights (query, key, value, output)
                if "to_q" in name or "to_k" in name or "to_v" in name or "to_out" in name:
                    checkpoint['unet'][name] = params.cpu().clone()

            elif parameter_group == "full": # Save all UNet parameters
                checkpoint['unet'][name] = params.cpu().clone()

            else: # Uknown parameter group
                raise ValueError(f"Parameter group '{parameter_group}' unrecognized. Code only supports '[kv-xattn, xattn, full, text-emb]'")

    # Save Text Encoder Parameters if that's what we're updating
    if parameter_group=="text-emb" and hasattr(diffusion_pipeline, "text_encoder"):
        diffusion_pipeline.text_encoder.train()
        checkpoint["text_encoder"] = diffusion_pipeline.text_encoder.state_dict()

    # Save checkpoint to disk
    torch.save(checkpoint, ckpt_path)
    print(f"[Simultaneous-Celeb] Checkpoint with sampled images saved to: {ckpt_path}")


def sample_celeb(diffusion_pipeline, iteration, model_save_path, target_concepts, device, eval_prompt_dir, parameter_group):
    celeb_subset = _build_celeb_subset(target_concepts)
    output_dir, img_dir = _prepare_output_dirs(iteration, model_save_path)
    print(f"[Simultaneous-Celeb] Saving sampled images to: {img_dir}")

    eval_prompt_dir = Path(eval_prompt_dir)
    if not eval_prompt_dir.exists():
        raise FileNotFoundError(f"Celebrity prompt directory not found: {eval_prompt_dir}")

    # Fixed sampling settings preserve historical celeb-monitoring behavior.
    max_prompts = 50
    n_samples_per_prompt = 1
    steps = 50
    guidance_scale = 7.5
    height = width = 512

    # Load prompts into memory
    prompt_cache, total_iterations = _build_prompt_cache(
        celeb_subset=celeb_subset,
        eval_prompt_dir=eval_prompt_dir,
        max_prompts=max_prompts,
        n_samples_per_prompt=n_samples_per_prompt,
    )
    if total_iterations == 0:
        print("[Simultaneous-Celeb] No prompts available; skipping sampling.")
        return 0.0

    _sample_celeb_images(
        diffusion_pipeline=diffusion_pipeline,
        celeb_subset=celeb_subset,
        prompt_cache=prompt_cache,
        img_dir=img_dir,
        device=device,
        n_samples_per_prompt=n_samples_per_prompt,
        height=height,
        width=width,
        steps=steps,
        guidance_scale=guidance_scale,
    )

    # Save ckpt used for sampling this iteration
    _save_sampling_checkpoint(
        diffusion_pipeline=diffusion_pipeline,
        output_dir=output_dir,
        iteration=iteration,
        celeb_subset=celeb_subset,
        parameter_group=parameter_group
    )
    return 0.0
