import argparse
import math
from pathlib import Path
from typing import List

import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate class-specific image datasets using a Stable Diffusion pipeline."
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Space-separated list of labels to render (e.g. --labels Picasso Monet).",
    )
    parser.add_argument(
        "--labels_file",
        type=str,
        default=None,
        help="Optional path to a newline-separated txt file containing labels.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        help="Diffusers model repo or local path.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Directory where label folders and images will be written.",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=500,
        help="Images to generate per label.",
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        default="a photo of {label}",
        help="Template applied to labels that are not in drawing_labels.",
    )
    parser.add_argument(
        "--drawing_template",
        type=str,
        default="a drawing of {label}",
        help="Template applied to labels listed in drawing_labels.",
    )
    parser.add_argument(
        "--drawing_labels",
        nargs="*",
        default=("Picasso", "Van Gogh"),
        help="Labels that should use drawing_template instead of prompt_template.",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
        help="Optional negative prompt applied to every generation.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale passed to the pipeline.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=30,
        help="Number of sampling steps per image.",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        choices=["float32", "bfloat16", "float16"],
        default="bfloat16",
        help="Datatype for loading the diffusion pipeline.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device to run generation on.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed used for reproducible generation.",
    )
    parser.add_argument(
        "--no_safetensors",
        action="store_true",
        help="Force loading legacy .bin weights instead of safetensors.",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Only load weights that already exist locally.",
    )
    parser.add_argument(
        "--disable_safety_checker",
        action="store_true",
        help="Disable the built-in safety checker for the pipeline.",
    )
    parser.add_argument(
        "--image_format",
        type=str,
        default="png",
        choices=["png", "jpg", "jpeg", "webp"],
        help="Output image format.",
    )
    return parser.parse_args()


def get_labels(args: argparse.Namespace) -> List[str]:
    labels: List[str] = []
    if args.labels:
        labels.extend(args.labels)
    if args.labels_file:
        labels_path = Path(args.labels_file).expanduser()
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_path}")
        with labels_path.open("r", encoding="utf-8") as handle:
            labels.extend([line.strip() for line in handle if line.strip()])
    if not labels:
        raise ValueError("Provide at least one label via --labels or --labels_file.")
    # Preserve order while removing duplicates
    seen = set()
    unique_labels = []
    for label in labels:
        if label not in seen:
            unique_labels.append(label)
            seen.add(label)
    return unique_labels


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    dtype_str = dtype_str.lower()
    if dtype_str == "float32":
        return torch.float32
    if dtype_str == "bfloat16":
        return torch.bfloat16
    if dtype_str == "float16":
        return torch.float16
    raise ValueError(f"Unsupported torch dtype: {dtype_str}")


def build_prompt(label: str, args: argparse.Namespace) -> str:
    template = args.prompt_template
    if label in args.drawing_labels:
        template = args.drawing_template
    return template.format(label=label)


def main() -> None:
    args = parse_args()
    labels = get_labels(args)
    torch_dtype = get_torch_dtype(args.torch_dtype)

    device = torch.device(args.device)
    if device.type.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is not available, but device '{args.device}' was requested.")

    pipeline_kwargs = dict(
        torch_dtype=torch_dtype,
        use_safetensors=not args.no_safetensors,
        local_files_only=args.local_files_only,
        safety_checker=None
    )
    if args.disable_safety_checker:
        pipeline_kwargs["safety_checker"] = None

    pipeline = AutoPipelineForText2Image.from_pretrained(
        args.model_path,
        **pipeline_kwargs,
    ).to(device)
    pipeline.set_progress_bar_config(disable=True)

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator(device=device).manual_seed(args.seed)

    for label in labels:
        label_dir = output_root
        label_dir.mkdir(parents=True, exist_ok=True)
        prompt = build_prompt(label, args)
        desc = f"Generating images for {label}"

        for idx in tqdm(
            range(args.num_images),
            desc=f"Prompt: '{prompt}'",
            dynamic_ncols=True,
        ):
            # Derive a deterministic per-sample seed
            sample_seed = args.seed + idx + int(math.fabs(hash(label)) % (10**6))
            generator = generator.manual_seed(sample_seed)

            output = pipeline(
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
            )
            image: Image.Image = output.images[0]
            image_path = label_dir / f"{label}_{idx:04d}.{args.image_format}"
            image.save(image_path)


if __name__ == "__main__":
    main()
