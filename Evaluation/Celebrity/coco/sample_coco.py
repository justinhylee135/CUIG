from __future__ import annotations

import argparse
import os

import pandas as pd
import torch
from diffusers import DDIMScheduler, DiffusionPipeline
from safetensors.torch import load_file
from tqdm.auto import tqdm


DEFAULT_SD_PIPELINE_DIR = "CompVis/stable-diffusion-v1-4"
DEFAULT_SDXL_PIPELINE_DIR = "stabilityai/stable-diffusion-xl-base-1.0"


def resolve_device(device_arg: str) -> torch.device:
    if torch.cuda.is_available():
        torch.cuda.set_device(device_arg)
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_pipeline_dir(model_family: str, pipeline_dir: str | None) -> str:
    if pipeline_dir:
        return pipeline_dir
    return DEFAULT_SDXL_PIPELINE_DIR if model_family == "sdxl" else DEFAULT_SD_PIPELINE_DIR


def resolve_resolution(model_family: str, image_size: int | None) -> int:
    if image_size is not None:
        return image_size
    return 1024 if model_family == "sdxl" else 512


def _normalize_unet_state_dict(unet_state_dict: dict | None) -> dict | None:
    if unet_state_dict is None:
        return None
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
    if hasattr(pipe, "safety_checker"):
        def _dummy_safety_checker(images, **kwargs):
            return images, [False] * len(images)
        pipe.safety_checker = _dummy_safety_checker


def _configure_scheduler(pipe: DiffusionPipeline, scheduler_mode: str, model_family: str) -> None:
    if scheduler_mode == "pipeline_default":
        return
    if scheduler_mode == "auto" and model_family == "sdxl":
        return
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)


def load_pipeline(
    ckpt_path: str,
    pipeline_dir: str,
    device: torch.device,
    model_family: str,
    scheduler_mode: str = "auto",
) -> DiffusionPipeline:
    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = DiffusionPipeline.from_pretrained(pipeline_dir, torch_dtype=torch_dtype)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    _disable_safety_checker(pipe)
    _configure_scheduler(pipe, scheduler_mode=scheduler_mode, model_family=model_family)

    unet_state_dict = _load_unet_state_dict(ckpt_path, device)
    if unet_state_dict is None:
        print(f"Using base pipeline weights from '{pipeline_dir}'")
        return pipe

    missing, unexpected = pipe.unet.load_state_dict(unet_state_dict, strict=False)
    print(f"Loaded custom UNet '{len(unet_state_dict)}' keys from {ckpt_path}")
    print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    return pipe


def _get_prompt_from_row(row: pd.Series) -> str:
    if "prompt" in row.index and not pd.isna(row["prompt"]):
        return str(row["prompt"])
    if "text" in row.index and not pd.isna(row["text"]):
        return str(row["text"])
    raise ValueError("Prompt row must contain 'prompt' or 'text' column.")


def _get_seed_from_row(row: pd.Series) -> int:
    for key in ("evaluation_seed", "seed"):
        if key in row.index and not pd.isna(row[key]):
            return int(row[key])
    return 42


def _get_case_number_from_row(row: pd.Series, fallback_idx: int) -> int:
    for key in ("case_number", "image_id"):
        if key in row.index and not pd.isna(row[key]):
            return int(row[key])
    return int(fallback_idx)


def _load_prompts_dataframe(prompts_path: str, start_idx: int | None, num_prompts: int | None) -> pd.DataFrame:
    df = pd.read_csv(prompts_path)

    if start_idx is not None and start_idx > 0:
        total_prompts = len(df)
        df = df.iloc[start_idx:]
        print(f"Starting from row '{start_idx}' (kept {len(df)} of {total_prompts} prompts)")

    if num_prompts is not None:
        total_prompts = len(df)
        if num_prompts < total_prompts:
            df = df.head(num_prompts)
            print(f"Limiting to first '{num_prompts}' prompts out of '{total_prompts}' available")
        elif num_prompts > total_prompts:
            print(f"Warning: Requested '{num_prompts}' prompts but only '{total_prompts}' available; processing all")

    return df


def generate_images(
    model_name,
    pipeline_dir,
    prompts_path,
    output_dir,
    device="cuda:0",
    guidance_scale=7.5,
    image_size=512,
    ddim_steps=100,
    num_samples=10,
    from_case=0,
    start_idx=None,
    num_prompts=None,
    model_family="sd",
    scheduler_mode="auto",
):
    """
    Generate images for COCO prompts CSV using either SD or SDXL through DiffusionPipeline.

    Expected CSV headers (at least one from each group):
    - Prompt: 'prompt' or 'text'
    - Case id: 'case_number' or 'image_id'
    - Seed (optional): 'evaluation_seed' or 'seed'
    """
    torch_device = resolve_device(device)
    pipe = load_pipeline(
        ckpt_path=model_name,
        pipeline_dir=pipeline_dir,
        device=torch_device,
        model_family=model_family,
        scheduler_mode=scheduler_mode,
    )

    df = _load_prompts_dataframe(
        prompts_path=prompts_path,
        start_idx=start_idx,
        num_prompts=num_prompts,
    )

    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving generated images to '{output_dir}'")
    print(f"Model family: '{model_family}' | Resolution: '{image_size}x{image_size}'")

    prompt_bar = tqdm(df.iterrows(), total=len(df), desc="Prompts")
    for row_idx, row in prompt_bar:
        prompt = _get_prompt_from_row(row)
        seed = _get_seed_from_row(row)
        case_number = _get_case_number_from_row(row, fallback_idx=row_idx)
        prompt_bar.set_postfix({"prompt": f"'{prompt[:50]}'"})

        if case_number < from_case:
            continue

        expected_paths = [os.path.join(output_dir, f"{case_number}_{idx}.png") for idx in range(num_samples)]
        if expected_paths and all(os.path.exists(path) for path in expected_paths):
            prompt_bar.set_postfix({"prompt": f"'{prompt[:50]}'", "status": "exists"})
            continue

        sample_bar = tqdm(range(num_samples), desc="Samples", leave=False)
        for i in sample_bar:
            sample_bar.set_postfix({"case": case_number, "sample": i})
            output_path = os.path.join(output_dir, f"{case_number}_{i}.png")
            if os.path.exists(output_path):
                continue

            generator = torch.Generator(device=torch_device).manual_seed(int(seed) + i)
            images = pipe(
                prompt=prompt,
                height=image_size,
                width=image_size,
                num_inference_steps=ddim_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images
            images[0].save(output_path)
        sample_bar.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generateImages",
        description="Generate COCO images using Diffusers (supports SD and SDXL)",
    )
    parser.add_argument("--model_family", choices=["sd", "sdxl"], default="sd")
    parser.add_argument("--model_name", help="Custom UNet checkpoint path or 'base'", type=str, required=True)
    parser.add_argument(
        "--pipeline_dir",
        help="Path/name of diffusers pipeline (defaults depend on --model_family)",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--prompts_path",
        help="Path to CSV file with prompts",
        type=str,
        required=False,
        default="ms_coco.csv",
    )
    parser.add_argument("--output_dir", help="Folder where to save images", type=str, required=True)
    parser.add_argument("--device", help="CUDA device to run on", type=str, required=False, default="cuda:0")
    parser.add_argument("--guidance_scale", help="CFG guidance scale", type=float, required=False, default=7.5)
    parser.add_argument(
        "--resolution",
        "--image_size",
        dest="image_size",
        help="Image resolution (height/width). Default depends on --model_family",
        type=int,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["auto", "ddim", "pipeline_default"],
        default="auto",
        help="Scheduler mode. 'auto' matches legacy behavior (DDIM for SD, pipeline default for SDXL).",
    )
    parser.add_argument("--from_case", help="Continue generating from case_number", type=int, default=0)
    parser.add_argument("--start_idx", help="Start generating from this row index", type=int, default=None)
    parser.add_argument("--num_samples", help="Number of samples per prompt", type=int, default=1)
    parser.add_argument("--num_prompts", help="Limit number of prompts to process", type=int, default=None)
    parser.add_argument("--ddim_steps", help="Inference denoising steps", type=int, default=50)
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()

    pipeline_dir = resolve_pipeline_dir(args.model_family, args.pipeline_dir)
    image_size = resolve_resolution(args.model_family, args.image_size)

    generate_images(
        model_name=args.model_name,
        pipeline_dir=pipeline_dir,
        prompts_path=args.prompts_path,
        output_dir=args.output_dir,
        device=args.device,
        guidance_scale=args.guidance_scale,
        image_size=image_size,
        ddim_steps=args.ddim_steps,
        num_samples=args.num_samples,
        from_case=args.from_case,
        start_idx=args.start_idx,
        num_prompts=args.num_prompts,
        model_family=args.model_family,
        scheduler_mode=args.scheduler,
    )
