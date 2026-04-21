from __future__ import annotations

import ast
import os
from argparse import ArgumentParser, Namespace
from typing import Sequence

import torch
from diffusers import DDIMScheduler, DiffusionPipeline
from pytorch_lightning import seed_everything
from safetensors.torch import load_file
from tqdm import tqdm


DEFAULT_SD_PIPELINE_DIR = "CompVis/stable-diffusion-v1-4"
DEFAULT_SDXL_PIPELINE_DIR = "stabilityai/stable-diffusion-xl-base-1.0"


def build_arg_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--model_family", choices=["sd", "sdxl"], default="sd")
    parser.add_argument("--steps", default=50, type=int, help="Number of inference steps")
    parser.add_argument("--guidance_scale", default=7.5, type=float, help="Classifier-free guidance scale")
    parser.add_argument("--resolution", type=int, default=None, help="Image resolution (height and width)")
    parser.add_argument("--celeb_subset", type=str, default=None)
    parser.add_argument("--num_prompts", type=int, default=50, help="Max prompts to load per celeb")
    parser.add_argument(
        "--start_idx",
        type=int,
        default=None,
        help="Start from this prompt index (0-based) within each celeb file",
    )
    parser.add_argument("--n_samples_per_prompt", type=int, default=1, help="Samples per prompt")
    parser.add_argument(
        "--prompt_dir",
        type=str,
        required=False,
        default="prompts",
        help="Directory storing prompt txt files",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ckpt", type=str, required=True, help="Custom UNet checkpoint or 'base'")
    parser.add_argument(
        "--pipeline_dir",
        type=str,
        default=None,
        required=False,
        help="Diffusers pipeline directory (defaults depend on --model_family)",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["auto", "ddim", "pipeline_default"],
        default="auto",
        help="Scheduler to use. 'auto' matches previous behavior (DDIM for SD, default for SDXL).",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    return parser


def parse_cli_args() -> Namespace:
    return build_arg_parser().parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if torch.cuda.is_available():
        torch.cuda.set_device(device_arg)
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_pipeline_dir(model_family: str, pipeline_dir: str | None) -> str:
    if pipeline_dir:
        return pipeline_dir
    return DEFAULT_SDXL_PIPELINE_DIR if model_family == "sdxl" else DEFAULT_SD_PIPELINE_DIR


def resolve_resolution(model_family: str, resolution: int | None) -> int:
    if resolution is not None:
        return resolution
    return 1024 if model_family == "sdxl" else 512


def _normalize_unet_state_dict(unet_state_dict: dict | None) -> dict | None:
    if unet_state_dict is None:
        return None

    # Handle common checkpoint layouts used by training code.
    if "unet" in unet_state_dict and isinstance(unet_state_dict["unet"], dict):
        unet_state_dict = unet_state_dict["unet"]

    keys_list = list(unet_state_dict.keys())
    for key in keys_list:
        if key.startswith("unet."):
            unet_state_dict[key.replace("unet.", "")] = unet_state_dict.pop(key)
    return unet_state_dict


def _load_unet_state_dict(ckpt_path: str, device: torch.device) -> dict | None:
    if not ckpt_path or ckpt_path.lower() in {"sd", "base", "default"}:
        return None
    if not os.path.exists(ckpt_path):
        raise ValueError(f"Provided checkpoint path does not exist: {ckpt_path}")

    if ckpt_path.endswith(".safetensors"):
        map_device = device.type if device.type == "cuda" else "cpu"
        state_dict = load_file(ckpt_path, device=map_device)
    else:
        state_dict = torch.load(ckpt_path, map_location=device)
    return _normalize_unet_state_dict(state_dict)


def _disable_safety_checker(pipe: DiffusionPipeline) -> None:
    # Keep sampling deterministic with the previous scripts, which did not filter outputs.
    if hasattr(pipe, "safety_checker"):
        def _dummy_safety_checker(images, **kwargs):
            return images, [False] * len(images)
        pipe.safety_checker = _dummy_safety_checker


def _configure_scheduler(pipe: DiffusionPipeline, scheduler_mode: str, model_family: str) -> None:
    if scheduler_mode == "pipeline_default":
        return
    if scheduler_mode == "auto" and model_family == "sdxl":
        return
    # Previous SD script used DDIM explicitly.
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)


def load_pipeline(
    ckpt_path: str,
    pipeline_dir: str,
    device: torch.device,
    model_family: str,
    scheduler_mode: str,
) -> DiffusionPipeline:
    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = DiffusionPipeline.from_pretrained(pipeline_dir, torch_dtype=torch_dtype)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    _disable_safety_checker(pipe)
    _configure_scheduler(pipe, scheduler_mode=scheduler_mode, model_family=model_family)

    unet_state_dict = _load_unet_state_dict(ckpt_path, device)
    if unet_state_dict is None:
        print(f"Using base {pipeline_dir} weights.")
        return pipe

    missing, unexpected = pipe.unet.load_state_dict(unet_state_dict, strict=False)
    print(f"Loaded custom UNet '{len(unet_state_dict)}' keys from {ckpt_path}")
    print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    return pipe


def parse_celeb_subset(celeb_subset_arg: str | None, prompt_dir: str) -> list[str]:
    # If omitted, sample every prompt file in the prompt directory.
    if celeb_subset_arg is None:
        return sorted(
            os.path.splitext(filename)[0]
            for filename in os.listdir(prompt_dir)
            if filename.endswith(".txt")
        )

    if celeb_subset_arg.strip().startswith("["):
        parsed = ast.literal_eval(celeb_subset_arg)
        if not isinstance(parsed, (list, tuple)):
            raise ValueError(f"Expected list-like celeb subset, got: {celeb_subset_arg}")
        return [str(item) for item in parsed]

    return [celeb_subset_arg]


def load_prompt_cache(
    celeb_subset: Sequence[str],
    prompt_dir: str,
    num_prompts: int,
    start_idx: int | None,
):
    prompt_cache = {}
    total_iterations = 0

    for celeb in celeb_subset:
        prompt_path = os.path.join(prompt_dir, f"{celeb}.txt")
        with open(prompt_path, "r") as handle:
            prompts = handle.read().splitlines()

        if num_prompts:
            prompts = prompts[:num_prompts]
        if start_idx is not None and start_idx > 0:
            original_len = len(prompts)
            prompts = prompts[start_idx:]
            print(f"{celeb}: starting from prompt index {start_idx} (kept {len(prompts)} of {original_len})")

        prompt_cache[celeb] = prompts
        total_iterations += len(prompts)

    return prompt_cache, total_iterations


def generate_image(
    prompt: str,
    generator: torch.Generator,
    output_path: str,
    height: int,
    width: int,
    steps: int,
    guidance_scale: float,
    pipe: DiffusionPipeline,
    step_bar: tqdm | None = None,
) -> bool:
    if os.path.exists(output_path):
        return False

    callback = None
    if step_bar is not None:
        step_bar.reset()

        def _callback(step: int, _timestep: int, _latents):
            step_bar.update(1)

        callback = _callback

    images = pipe(
        prompt=prompt,
        generator=generator,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        callback=callback,
        callback_steps=1,
    ).images
    images[0].save(output_path)
    return True


def sample_celeb_images(args: Namespace) -> None:
    device = resolve_device(args.device)
    pipeline_dir = resolve_pipeline_dir(args.model_family, args.pipeline_dir)
    resolution = resolve_resolution(args.model_family, args.resolution)

    celeb_subset = parse_celeb_subset(args.celeb_subset, args.prompt_dir)
    if not celeb_subset:
        raise ValueError(f"No celeb prompt files found in: {args.prompt_dir}")

    pipe = load_pipeline(
        ckpt_path=args.ckpt,
        pipeline_dir=pipeline_dir,
        device=device,
        model_family=args.model_family,
        scheduler_mode=args.scheduler,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    prompt_cache, total_prompts = load_prompt_cache(
        celeb_subset=celeb_subset,
        prompt_dir=args.prompt_dir,
        num_prompts=args.num_prompts,
        start_idx=args.start_idx,
    )
    total_iterations = total_prompts * args.n_samples_per_prompt

    print(f"Model family: {args.model_family}")
    print(f"Pipeline dir: {pipeline_dir}")
    print(f"Resolution: {resolution}")
    print(f"Saving generated images to {args.output_dir}")

    overall_bar = tqdm(total=total_iterations, desc="Images", unit="image", position=0)
    for celeb in celeb_subset:
        prompts = prompt_cache.get(celeb, [])
        print(f"Generating images for {celeb} ({len(prompts)} prompts)")

        for seed in range(args.n_samples_per_prompt):
            seed_everything(seed)
            generator = torch.Generator(device=device).manual_seed(seed)

            for i, prompt in enumerate(prompts):
                img_save_path = os.path.join(args.output_dir, f"{celeb}_prompt{i+1}_seed{seed}.jpg")
                step_bar = tqdm(total=args.steps, desc="Sampling Steps", leave=False, position=1)
                step_bar.set_postfix(
                    {"prompt": f"'{prompt[:40]}...'" if len(prompt) > 40 else prompt[:40]}
                )
                generate_image(
                    prompt=prompt,
                    generator=generator,
                    output_path=img_save_path,
                    height=resolution,
                    width=resolution,
                    steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    pipe=pipe,
                    step_bar=step_bar,
                )
                step_bar.close()
                overall_bar.set_postfix(
                    {"celeb": f"'{celeb}'", "prompt": f"'{i+1}/{len(prompts)}'", "seed": f"'{seed}'"}
                )
                overall_bar.update(1)
    overall_bar.close()


def main() -> None:
    args = parse_cli_args()
    sample_celeb_images(args)


if __name__ == "__main__":
    main()
