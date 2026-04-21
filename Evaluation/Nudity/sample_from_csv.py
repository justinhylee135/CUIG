from __future__ import annotations

import argparse
import os

import pandas as pd
import torch
from diffusers import DDIMScheduler, DiffusionPipeline, LMSDiscreteScheduler
from safetensors.torch import load_file
from tqdm.auto import tqdm


DEFAULT_SD_PIPELINE_DIR = "CompVis/stable-diffusion-v1-4"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generateImages",
        description="Generate images from a prompts CSV for the nudity benchmark.",
    )
    parser.add_argument("--model_name", help="Custom UNet checkpoint path or 'base'", type=str, required=True)
    parser.add_argument(
        "--pipeline_dir",
        help="Path/name of diffusers pipeline",
        type=str,
        required=False,
        default=DEFAULT_SD_PIPELINE_DIR,
    )
    parser.add_argument(
        "--prompts_path",
        help="Path to csv file with prompts",
        type=str,
        required=False,
        default="nudity_benchmark.csv",
    )
    parser.add_argument("--output_dir", help="Folder where to save images", type=str, required=True)
    parser.add_argument("--device", help="CUDA device to run on", type=str, required=False, default="cuda:0")
    parser.add_argument("--guidance_scale", help="CFG guidance scale", type=float, required=False, default=7.5)
    parser.add_argument(
        "--image_size",
        "--resolution",
        dest="image_size",
        help="Image resolution (height/width)",
        type=int,
        required=False,
        default=512,
    )
    parser.add_argument("--from_case", help="Continue generating from case_number", type=int, default=0)
    parser.add_argument("--num_samples", help="Number of samples per prompt", type=int, default=1)
    parser.add_argument("--num_prompts", help="Limit prompts to process", type=int, default=None)
    parser.add_argument("--ddim_steps", help="Inference denoising steps", type=int, default=50)
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["lms", "ddim", "pipeline_default"],
        default="lms",
        help="Scheduler mode. Default matches the old script's LMS scheduler behavior.",
    )
    return parser


def parse_cli_args() -> argparse.Namespace:
    return build_arg_parser().parse_args()


def parse_args() -> argparse.Namespace:
    # Backward-compatible alias.
    return parse_cli_args()


def resolve_device(device_arg: str) -> torch.device:
    if torch.cuda.is_available():
        torch.cuda.set_device(device_arg)
        return torch.device("cuda")
    return torch.device("cpu")


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


def _load_unet_state_dict(model_name: str, device: torch.device) -> dict | None:
    if not model_name or model_name.upper() in {"SD", "BASE", "DEFAULT"}:
        return None
    if not os.path.exists(model_name):
        raise FileNotFoundError(f"Custom model checkpoint not found: {model_name}")
    if model_name.endswith(".safetensors"):
        map_device = device.type if device.type == "cuda" else "cpu"
        state_dict = load_file(model_name, device=map_device)
    else:
        state_dict = torch.load(model_name, map_location=device)
    return _normalize_unet_state_dict(state_dict)


def _disable_safety_checker(pipe: DiffusionPipeline) -> None:
    if hasattr(pipe, "safety_checker"):
        def _dummy_safety_checker(images, **kwargs):
            return images, [False] * len(images)
        pipe.safety_checker = _dummy_safety_checker


def _configure_scheduler(pipe: DiffusionPipeline, scheduler_mode: str) -> None:
    if scheduler_mode == "pipeline_default":
        return
    if scheduler_mode == "ddim":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    else:
        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)


def load_pipeline(
    model_name: str,
    pipeline_dir: str,
    device: torch.device,
    scheduler_mode: str = "lms",
) -> DiffusionPipeline:
    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = DiffusionPipeline.from_pretrained(pipeline_dir, torch_dtype=torch_dtype)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    _disable_safety_checker(pipe)
    _configure_scheduler(pipe, scheduler_mode)

    unet_state_dict = _load_unet_state_dict(model_name, device)
    if unet_state_dict is None:
        print(f"Using base pipeline weights from '{pipeline_dir}'")
        return pipe

    missing, unexpected = pipe.unet.load_state_dict(unet_state_dict, strict=False)
    print(f"Loaded custom UNet '{len(unet_state_dict)}' keys from {model_name}")
    print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    return pipe


def _get_prompt_from_row(row: pd.Series) -> str:
    if "prompt" in row.index and not pd.isna(row["prompt"]):
        return str(row["prompt"])
    if "text" in row.index and not pd.isna(row["text"]):
        return str(row["text"])
    raise ValueError("Prompt row must contain 'prompt' or 'text' column.")


def _get_case_number_from_row(row: pd.Series, fallback_idx: int) -> int:
    for key in ("case_number", "image_id"):
        if key in row.index and not pd.isna(row[key]):
            return int(row[key])
    return int(fallback_idx)


def _get_seed_from_row(row: pd.Series) -> int:
    for key in ("evaluation_seed", "seed"):
        if key in row.index and not pd.isna(row[key]):
            return int(row[key])
    return 42


def load_prompts_dataframe(prompts_path: str, num_prompts: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(prompts_path)
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
    prompts_path,
    output_dir,
    device="cuda:0",
    guidance_scale=7.5,
    image_size=512,
    ddim_steps=100,
    num_samples=10,
    from_case=0,
    num_prompts=None,
    pipeline_dir=DEFAULT_SD_PIPELINE_DIR,
    scheduler="lms",
):
    torch_device = resolve_device(device)
    pipe = load_pipeline(model_name, pipeline_dir, torch_device, scheduler_mode=scheduler)
    df = load_prompts_dataframe(prompts_path, num_prompts=num_prompts)

    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving generated images to '{output_dir}'")

    prompt_bar = tqdm(df.iterrows(), total=len(df), desc="Prompts")
    for row_idx, row in prompt_bar:
        prompt = _get_prompt_from_row(row)
        prompt_bar.set_postfix({"prompt": f"'{prompt[:50]}'"})
        seed = _get_seed_from_row(row)
        case_number = _get_case_number_from_row(row, fallback_idx=row_idx)
        if case_number < from_case:
            continue

        expected_paths = [os.path.join(output_dir, f"{case_number}_{idx}.png") for idx in range(num_samples)]
        if expected_paths and all(os.path.exists(path) for path in expected_paths):
            prompt_bar.set_postfix({"prompt": f"'{prompt[:50]}'", "status": "exists"})
            continue

        for sample_idx in tqdm(range(num_samples), desc="Samples", leave=False):
            output_path = os.path.join(output_dir, f"{case_number}_{sample_idx}.png")
            if os.path.exists(output_path):
                continue
            generator = torch.Generator(device=torch_device).manual_seed(int(seed) + sample_idx)
            image = pipe(
                prompt=prompt,
                height=image_size,
                width=image_size,
                num_inference_steps=ddim_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images[0]
            image.save(output_path)

    return {
        "output_dir": output_dir,
        "num_prompts": len(df),
        "num_samples": num_samples,
    }


def generate_images_from_args(args: argparse.Namespace) -> dict:
    return generate_images(
        model_name=args.model_name,
        prompts_path=args.prompts_path,
        output_dir=args.output_dir,
        device=args.device,
        guidance_scale=args.guidance_scale,
        image_size=args.image_size,
        ddim_steps=args.ddim_steps,
        num_samples=args.num_samples,
        from_case=args.from_case,
        num_prompts=args.num_prompts,
        pipeline_dir=args.pipeline_dir,
        scheduler=args.scheduler,
    )


def main() -> None:
    args = parse_cli_args()
    generate_images_from_args(args)


if __name__ == "__main__":
    main()
