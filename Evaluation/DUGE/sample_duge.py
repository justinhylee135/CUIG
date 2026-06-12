from __future__ import annotations

import argparse
import ast
import csv
import os
import re
from pathlib import Path


DEFAULT_PIPELINE_DIR = "stabilityai/stable-diffusion-2-1-base"
BASE_MODEL_NAMES = {"base", "default", "sd", "sd2", "sd2.1"}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample images for DUGE object prompts.")
    parser.add_argument("--model_name", "--ckpt", dest="model_name", required=True, help="UNet checkpoint path or 'base'.")
    parser.add_argument("--pipeline_dir", default=DEFAULT_PIPELINE_DIR, help="Base Diffusers pipeline directory.")
    parser.add_argument("--prompts_path", default="prompts.csv", help="CSV with concept,prompt columns.")
    parser.add_argument("--output_dir", required=True, help="Directory where generated images are written.")
    parser.add_argument("--device", default="cuda:0", help="Device such as cuda:0 or cpu.")
    parser.add_argument("--resolution", type=int, default=512, help="Generated image height and width.")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Diffusion denoising steps.")
    parser.add_argument("--guidance_scale", type=float, default=8.0, help="Classifier-free guidance scale.")
    parser.add_argument("--num_samples", type=int, default=8, help="Images to sample per prompt.")
    parser.add_argument("--seed", type=int, default=1, help="Base random seed.")
    parser.add_argument(
        "--target_concepts",
        default=None,
        help="Optional concept subset as a Python list or comma-separated string.",
    )
    return parser


def parse_cli_args() -> argparse.Namespace:
    return build_arg_parser().parse_args()


def slugify(text: str) -> str:
    text = text.strip().lower().replace("_", "-")
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def parse_concept_list(raw_value: str | None) -> list[str] | None:
    if raw_value is None:
        return None

    value = raw_value.strip()
    if value.startswith("["):
        parsed = ast.literal_eval(value)
        if not isinstance(parsed, (list, tuple)):
            raise ValueError("--target_concepts must be a list-like literal.")
        return [slugify(str(item)) for item in parsed]

    return [slugify(item) for item in value.split(",") if item.strip()]


def load_prompt_rows(prompts_path: str, target_concepts: list[str] | None = None) -> list[dict[str, str]]:
    if not os.path.exists(prompts_path):
        raise FileNotFoundError(f"Prompt CSV not found: {prompts_path}")

    with open(prompts_path, newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"Prompt CSV has no rows: {prompts_path}")

    normalized_rows = []
    for row_index, row in enumerate(rows):
        prompt = (row.get("prompt") or row.get("text") or "").strip()
        if not prompt:
            raise ValueError(f"Prompt CSV row {row_index} is missing a prompt/text value.")

        concept = slugify(row.get("concept") or prompt)
        if target_concepts is not None and concept not in target_concepts:
            continue

        normalized_rows.append({"concept": concept, "prompt": prompt})

    if not normalized_rows:
        raise ValueError("No prompt rows matched the requested concept subset.")
    return normalized_rows


def resolve_device(device_arg: str):
    import torch

    if device_arg == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")

    torch.cuda.set_device(device_arg)
    return torch.device("cuda")


def _load_unet_state_dict(model_name: str, device):
    import torch
    from safetensors.torch import load_file

    if model_name.lower() in BASE_MODEL_NAMES:
        return None
    if not os.path.exists(model_name):
        raise FileNotFoundError(f"UNet checkpoint not found: {model_name}")

    if model_name.endswith(".safetensors"):
        map_device = device.type if device.type == "cuda" else "cpu"
        state_dict = load_file(model_name, device=map_device)
    else:
        state_dict = torch.load(model_name, map_location=device)

    if isinstance(state_dict, dict) and "unet" in state_dict and isinstance(state_dict["unet"], dict):
        state_dict = state_dict["unet"]

    for key in list(state_dict.keys()):
        if key.startswith("unet."):
            state_dict[key.replace("unet.", "", 1)] = state_dict.pop(key)

    return state_dict


def load_pipeline(model_name: str, pipeline_dir: str, device):
    import torch
    from diffusers import DiffusionPipeline

    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = DiffusionPipeline.from_pretrained(pipeline_dir, torch_dtype=torch_dtype)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    if hasattr(pipe, "safety_checker"):
        def _dummy_safety_checker(images, **kwargs):
            return images, [False] * len(images)

        pipe.safety_checker = _dummy_safety_checker

    unet_state_dict = _load_unet_state_dict(model_name, device)
    if unet_state_dict is None:
        print(f"Using base pipeline weights from '{pipeline_dir}'")
        return pipe

    missing, unexpected = pipe.unet.load_state_dict(unet_state_dict, strict=False)
    print(f"Loaded UNet checkpoint from '{model_name}'")
    print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    return pipe


def sample_duge_images(args: argparse.Namespace) -> dict[str, int | str]:
    import torch
    from tqdm.auto import tqdm

    target_concepts = parse_concept_list(args.target_concepts)
    prompt_rows = load_prompt_rows(args.prompts_path, target_concepts=target_concepts)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    pipe = load_pipeline(args.model_name, args.pipeline_dir, device)

    metadata_rows = []
    total = len(prompt_rows) * args.num_samples
    with tqdm(total=total, desc="DUGE sampling", unit="image") as progress:
        for prompt_index, row in enumerate(prompt_rows):
            concept = row["concept"]
            prompt = row["prompt"]
            concept_dir = output_dir / concept
            concept_dir.mkdir(parents=True, exist_ok=True)

            for sample_index in range(args.num_samples):
                filename = f"{prompt_index:04d}_{sample_index:02d}_{slugify(prompt)}.png"
                image_path = concept_dir / filename
                relative_path = image_path.relative_to(output_dir)

                if not image_path.exists():
                    generator = torch.Generator(device=device).manual_seed(args.seed + prompt_index * args.num_samples + sample_index)
                    image = pipe(
                        prompt=prompt,
                        height=args.resolution,
                        width=args.resolution,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        generator=generator,
                    ).images[0]
                    image.save(image_path)

                metadata_rows.append(
                    {
                        "relative_path": str(relative_path),
                        "concept": concept,
                        "prompt": prompt,
                        "seed": args.seed + prompt_index * args.num_samples + sample_index,
                        "sample_index": sample_index,
                    }
                )
                progress.update(1)

    metadata_path = output_dir / "metadata.csv"
    with open(metadata_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "concept", "prompt", "seed", "sample_index"])
        writer.writeheader()
        writer.writerows(metadata_rows)

    return {
        "output_dir": str(output_dir),
        "metadata_path": str(metadata_path),
        "num_prompts": len(prompt_rows),
        "num_images": len(metadata_rows),
    }


def main() -> None:
    result = sample_duge_images(parse_cli_args())
    print(result)


if __name__ == "__main__":
    main()
