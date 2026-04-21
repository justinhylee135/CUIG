# Standard Library
import argparse
import ast
import os
from typing import Sequence

# Third Party
import torch
from diffusers import StableDiffusionPipeline
from pytorch_lightning import seed_everything
from safetensors.torch import load_file
from tqdm import tqdm

# Local
try:
    # Package-style import (works when imported as a module)
    from .constants import STYLES_AVAILABLE, OBJECTS_AVAILABLE
except ImportError:
    # Script-style import (works when executed directly from this directory)
    from constants import STYLES_AVAILABLE, OBJECTS_AVAILABLE


def build_arg_parser() -> argparse.ArgumentParser:
    # Keep CLI construction separate so the parser can be reused by tests/tools.
    parser = argparse.ArgumentParser(
        prog="Sampler",
        description="Sample Images from Unlearned Model",
    )

    # Diffusion Pipeline parameters
    parser.add_argument("--unet_ckpt_path", help="Path to UNet ckpt", type=str, required=False)
    parser.add_argument("--pipeline_dir", help="Directory for Diffusers pipeline", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")

    # Generation parameters
    parser.add_argument("--output_dir", help="Folder to save images", type=str, required=True)
    parser.add_argument("--seed", type=int, nargs="+", default=[188, 288, 588, 688, 888])
    parser.add_argument(
        "--guidance_scale",
        help="Classifier Free Guidance strength",
        type=float,
        required=False,
        default=9.0,
    )
    parser.add_argument(
        "--num_inference_steps",
        help="Inference steps",
        type=int,
        required=False,
        default=100,
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="image height and width in pixel space",
    )

    # Concept Subset to Generate parameters
    parser.add_argument("--styles_subset", type=str, default=None)
    parser.add_argument("--objects_subset", type=str, default=None)
    return parser


def parse_cli_args() -> argparse.Namespace:
    # Thin wrapper to preserve the previous one-call CLI behavior.
    return build_arg_parser().parse_args()


def _parse_optional_list_arg(raw_value: str | None, default_values: Sequence[str]) -> list[str]:
    # If the user did not provide a subset, evaluate/sample the full benchmark list.
    if raw_value is None:
        return list(default_values)

    # CLI passes subset lists as Python literals (e.g. "['A', 'B']"), so parse them.
    parsed = ast.literal_eval(raw_value)
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(f"Expected a list-like literal, got: {raw_value}")

    # Normalize entries to strings for downstream path/prompt formatting.
    return [str(item) for item in parsed]


def resolve_sampling_subsets(styles_subset_arg: str | None, objects_subset_arg: str | None) -> tuple[list[str], list[str]]:
    # Resolve both style and object subsets once so callers can reuse the same logic.
    styles_subset = _parse_optional_list_arg(styles_subset_arg, STYLES_AVAILABLE)
    objects_subset = _parse_optional_list_arg(objects_subset_arg, OBJECTS_AVAILABLE)
    return styles_subset, objects_subset


def resolve_device(device_arg: str) -> torch.device:
    # Respect explicit GPU selection when CUDA is available.
    if torch.cuda.is_available():
        torch.cuda.set_device(device_arg)
        return torch.device("cuda")
    # CPU fallback keeps the helper usable on non-GPU systems.
    return torch.device("cpu")


def load_sampling_pipeline(pipeline_dir: str, device: torch.device) -> StableDiffusionPipeline:
    # Default to fp16 on CUDA to match the original script's memory/perf behavior.
    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipeline = StableDiffusionPipeline.from_pretrained(pipeline_dir, torch_dtype=torch_dtype).to(device)
    return pipeline


def load_unet_state_dict(unet_ckpt_path: str | None, device: torch.device):
    # Optional checkpoint override: if none is supplied, keep the base pipeline UNet.
    if unet_ckpt_path is None:
        return None

    # Match load device to runtime device to avoid accidental GPU/CPU mismatches.
    map_device = "cuda" if device.type == "cuda" else "cpu"
    if unet_ckpt_path.endswith(".safetensors"):
        return load_file(unet_ckpt_path, device=map_device)
    return torch.load(unet_ckpt_path, map_location=map_device)


def normalize_unet_state_dict(unet_state_dict: dict | None) -> dict | None:
    # Support several checkpoint formats used across training scripts.
    if unet_state_dict is None:
        return None

    # Some checkpoints prefix parameters with "unet."; strip for diffusers UNet loading.
    keys_list = list(unet_state_dict.keys())
    for key in keys_list:
        if key.startswith("unet."):
            unet_state_dict[key.replace("unet.", "")] = unet_state_dict.pop(key)

    # Accelerate-style checkpoints may wrap the module state under "unet".
    if "unet" in unet_state_dict:
        unet_state_dict = unet_state_dict["unet"]

    return unet_state_dict


def apply_unet_checkpoint(pipe: StableDiffusionPipeline, unet_state_dict: dict | None, unet_ckpt_path: str | None) -> None:
    # No-op if the caller is evaluating the base pipeline directly.
    if unet_state_dict is None:
        return

    # Non-strict load keeps compatibility with small naming/shape differences across runs.
    missing, unexpected = pipe.unet.load_state_dict(unet_state_dict, strict=False)
    print(f"Loaded UNet from {unet_ckpt_path}")
    print(f"Loaded keys: {len(unet_state_dict)}")
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")


def disable_safety_checker(pipe: StableDiffusionPipeline) -> None:
    # Disable the default checker so benchmark images are not silently filtered.
    def dummy_safety_checker(images, **kwargs):
        return images, [False]

    pipe.safety_checker = dummy_safety_checker


def generate_unlearncanvas_images(
    pipe: StableDiffusionPipeline,
    output_dir: str,
    styles_subset: Sequence[str],
    objects_subset: Sequence[str],
    seeds: Sequence[int],
    resolution: int = 512,
    num_inference_steps: int = 100,
    guidance_scale: float = 9.0,
) -> None:
    # Create output directory to save results
    print(f"Saving generated images to '{output_dir}'")
    os.makedirs(output_dir, exist_ok=True)

    # Count every (seed, style, object) combination for a deterministic progress bar length.
    total_iterations = len(seeds) * len(styles_subset) * len(objects_subset)
    with tqdm(total=total_iterations, desc="Generating images", unit="image") as pbar:
        # Iterate through each specified seed
        for seed in seeds:
            # pytorch_lightning helper seeds PyTorch / NumPy / Python RNGs together.
            seed_everything(seed)
            for style in styles_subset:
                for obj in objects_subset:
                    pbar.set_postfix({"Style": f"'{style}'", "Object": f"'{obj}'", "Seed": f"'{seed}'"})
                    output_path = os.path.join(output_dir, f"{style}_{obj}_seed{seed}.jpg")

                    # Skip existing files to support resume/restart without regenerating everything.
                    if os.path.exists(output_path):
                        print(f"Image already exists! Skipping: '{output_path}'")
                        pbar.update(1)
                        continue

                    # UnlearnCanvas benchmark prompts follow a fixed "object in style" template.
                    prompt = f"A {obj} image in {style} style"
                    image = pipe(
                        prompt=prompt,
                        width=resolution,
                        height=resolution,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                    ).images[0]

                    # Save one image per (style, object, seed) tuple.
                    image.save(output_path)
                    pbar.update(1)


def sample_unlearncanvas_images(
    pipeline_dir: str,
    output_dir: str,
    device: str = "cuda:0",
    unet_ckpt_path: str | None = None,
    seeds: Sequence[int] = (188, 288, 588, 688, 888),
    guidance_scale: float = 9.0,
    num_inference_steps: int = 100,
    resolution: int = 512,
    styles_subset: Sequence[str] | None = None,
    objects_subset: Sequence[str] | None = None,
) -> dict:
    # This is the main reusable entry point for programmatic sampling.
    torch_device = resolve_device(device)

    # Default to all benchmark concepts if no subsets were supplied.
    styles_subset = list(STYLES_AVAILABLE if styles_subset is None else styles_subset)
    objects_subset = list(OBJECTS_AVAILABLE if objects_subset is None else objects_subset)

    # Build and optionally patch the diffusion pipeline.
    pipe = load_sampling_pipeline(pipeline_dir, torch_device)
    unet_state_dict = normalize_unet_state_dict(load_unet_state_dict(unet_ckpt_path, torch_device))
    apply_unet_checkpoint(pipe, unet_state_dict, unet_ckpt_path)
    disable_safety_checker(pipe)

    # Generate benchmark images to disk.
    generate_unlearncanvas_images(
        pipe=pipe,
        output_dir=output_dir,
        styles_subset=styles_subset,
        objects_subset=objects_subset,
        seeds=seeds,
        resolution=resolution,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )

    # Return light metadata so callers can log what was sampled without re-scanning the folder.
    return {
        "output_dir": output_dir,
        "num_styles": len(styles_subset),
        "num_objects": len(objects_subset),
        "num_seeds": len(list(seeds)),
    }


def run_sampling_from_args(args: argparse.Namespace) -> dict:
    # CLI adapter: parse subset strings, then dispatch to the reusable API.
    styles_subset, objects_subset = resolve_sampling_subsets(args.styles_subset, args.objects_subset)
    return sample_unlearncanvas_images(
        pipeline_dir=args.pipeline_dir,
        output_dir=args.output_dir,
        device=args.device,
        unet_ckpt_path=args.unet_ckpt_path,
        seeds=args.seed,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        resolution=args.resolution,
        styles_subset=styles_subset,
        objects_subset=objects_subset,
    )


def main() -> None:
    # CLI entry point kept intentionally small for easier reuse and testing.
    args = parse_cli_args()
    run_sampling_from_args(args)


if __name__ == "__main__":
    main()
