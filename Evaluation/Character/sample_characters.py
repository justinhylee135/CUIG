from __future__ import annotations

import argparse
import json
import os
from typing import Sequence

import torch
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from PIL import Image
from safetensors.torch import load_file
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer


BASE_MODEL = "CompVis/stable-diffusion-v1-4"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate character-specific samples using the SixCD template. "
            "All occurrences of '{name}' in the template are replaced with provided characters."
        )
    )
    parser.add_argument("--model_name", required=True, help="UNet checkpoint path or 'SD' for base v1-4 weights.")
    parser.add_argument("--output_dir", required=True, help="Directory where generated images will be stored.")
    parser.add_argument("--template_path", default="SixCD_Template.txt", help="Path to the template text file.")
    parser.add_argument("--characters", type=str, default=None, help="Single character or JSON list string.")
    parser.add_argument("--character_file", help="Optional text file with one character name per line.")
    parser.add_argument("--device", default="cuda:0", help="Torch device string (e.g. cuda:0 or cpu).")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Classifier-free guidance scale.")
    parser.add_argument("--image_size", type=int, default=512, help="Generated image size (square).")
    parser.add_argument("--ddim_steps", type=int, default=50, help="Number of denoising steps.")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples per prompt.")
    return parser


def parse_cli_args() -> argparse.Namespace:
    return build_arg_parser().parse_args()


def parse_args() -> argparse.Namespace:
    # Backward-compatible alias for older scripts importing parse_args().
    return parse_cli_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if "cuda" in device_arg and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_arg)


def load_template(path: str) -> list[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Template file not found: {path}")

    prompts = []
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith('"') and line.endswith('"') and len(line) > 1:
                line = line[1:-1]
            prompts.append(line)

    if not prompts:
        raise ValueError(f"No prompts found in template: {path}")
    print(f"Loaded '{len(prompts)}' prompts from template: '{path}'")
    return prompts


def _parse_character_list_arg(raw_value: str | None) -> list[str]:
    if raw_value is None:
        return []
    if "[" in raw_value:
        print(f"Character Subset before parsing: {raw_value}")
        parsed = json.loads(raw_value)
        if not isinstance(parsed, list):
            parsed = [parsed]
        parsed = [str(x) for x in parsed]
        print(f"Character Subset after parsing: {parsed}")
        return parsed
    return [raw_value]


def _dedupe_character_names(names: Sequence[str]) -> list[str]:
    unique_names = []
    seen = set()
    for name in names:
        key = str(name).lower()
        if key in seen:
            continue
        seen.add(key)
        unique_names.append(str(name))
    return unique_names


def collect_characters_from_sources(characters_arg: str | None, character_file: str | None) -> list[str]:
    names = _parse_character_list_arg(characters_arg)
    if character_file:
        with open(character_file, "r", encoding="utf-8") as handle:
            for raw in handle:
                name = raw.strip()
                if name:
                    names.append(name)
    names = _dedupe_character_names(names)
    if not names:
        raise ValueError("No characters provided. Use --characters and/or --character_file.")
    return names


def collect_characters(args: argparse.Namespace) -> list[str]:
    return collect_characters_from_sources(args.characters, args.character_file)


def build_prompt_records(characters: Sequence[str], template_lines: Sequence[str]) -> list[tuple[str, int, str]]:
    records: list[tuple[str, int, str]] = []
    for character in characters:
        character_text = str(character).replace("_", " ")
        for prompt_idx, template in enumerate(template_lines):
            records.append((character_text, prompt_idx, template.replace("{name}", character_text)))
    return records


def _normalize_unet_state_dict(state_dict: dict | None) -> dict | None:
    if state_dict is None:
        return None
    if "unet" in state_dict and isinstance(state_dict["unet"], dict):
        state_dict = state_dict["unet"]
    keys = list(state_dict.keys())
    for key in keys:
        if key.startswith("unet."):
            state_dict[key.replace("unet.", "")] = state_dict.pop(key)
    return state_dict


def load_unet_weights(unet: UNet2DConditionModel, model_name: str, device: torch.device) -> None:
    if model_name.upper() == "SD":
        print("Using base Stable Diffusion v1-4 weights.")
        return
    if not os.path.exists(model_name):
        raise FileNotFoundError(f"Custom model checkpoint not found: {model_name}")

    if model_name.endswith(".safetensors"):
        map_device = device.type if device.type == "cuda" else "cpu"
        state_dict = load_file(model_name, device=map_device)
    else:
        state_dict = torch.load(model_name, map_location=device)

    state_dict = _normalize_unet_state_dict(state_dict)
    missing, unexpected = unet.load_state_dict(state_dict, strict=False)
    print(f"Loaded '{len(state_dict)}' UNet keys from '{model_name}'")
    print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")


def _build_ddim_scheduler() -> DDIMScheduler:
    return DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )


def load_sd_components(model_name: str, device: torch.device):
    vae = AutoencoderKL.from_pretrained(BASE_MODEL, subfolder="vae").to(device)
    tokenizer = CLIPTokenizer.from_pretrained(BASE_MODEL, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    unet = UNet2DConditionModel.from_pretrained(BASE_MODEL, subfolder="unet").to(device)
    load_unet_weights(unet, model_name, device)
    scheduler = _build_ddim_scheduler()
    return vae, tokenizer, text_encoder, unet, scheduler


def _save_decoded_latents_as_image(vae, latents, output_path: str) -> None:
    latents = latents / 0.18215
    with torch.no_grad():
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1).detach().cpu()
    image = image.permute(0, 2, 3, 1).numpy()[0]
    image = (image * 255).round().astype("uint8")
    Image.fromarray(image).save(output_path)


def _generate_one_image_manual(
    prompt: str,
    output_path: str,
    sample_seed: int,
    image_size: int,
    ddim_steps: int,
    guidance_scale: float,
    device: torch.device,
    vae,
    tokenizer,
    text_encoder,
    unet,
    scheduler: DDIMScheduler,
) -> None:
    generator = torch.Generator(device=device).manual_seed(sample_seed)
    text_input = tokenizer(
        [prompt],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = text_encoder(text_input.input_ids.to(device))[0]

    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer([""], padding="max_length", max_length=max_length, return_tensors="pt")
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    latents = torch.randn(
        (1, unet.in_channels, image_size // 8, image_size // 8),
        generator=generator,
        device=device,
    )
    scheduler.set_timesteps(ddim_steps)
    latents = latents * scheduler.init_noise_sigma

    step_bar = tqdm(total=ddim_steps, desc="DDIM Steps", leave=False)
    step_bar.set_postfix({"prompt": prompt[:40]})
    for t in scheduler.timesteps:
        latent_input = torch.cat([latents] * 2)
        latent_input = scheduler.scale_model_input(latent_input, t)
        with torch.no_grad():
            noise_pred = unet(latent_input, t, encoder_hidden_states=text_embeddings).sample
        noise_uncond, noise_text = noise_pred.chunk(2)
        guided_noise = noise_uncond + guidance_scale * (noise_text - noise_uncond)
        latents = scheduler.step(guided_noise, t, latents).prev_sample
        step_bar.update(1)
    step_bar.close()

    _save_decoded_latents_as_image(vae, latents, output_path)


def sample_character_prompts_manual(
    prompt_records: Sequence[tuple[str, int, str]],
    output_dir: str,
    model_name: str,
    device: str | torch.device = "cuda:0",
    guidance_scale: float = 7.5,
    image_size: int = 512,
    ddim_steps: int = 50,
    num_samples: int = 10,
) -> dict:
    torch_device = device if isinstance(device, torch.device) else resolve_device(device)
    vae, tokenizer, text_encoder, unet, scheduler = load_sd_components(model_name, torch_device)

    os.makedirs(output_dir, exist_ok=True)
    total_generated = 0
    total_images = len(prompt_records) * num_samples
    image_bar = tqdm(total=total_images, desc="Images")

    for character, prompt_idx, prompt in prompt_records:
        image_bar.set_postfix({"character": character, "prompt": prompt[:40]})
        output_paths = [
            os.path.join(output_dir, f"{character.replace(' ', '_')}_{prompt_idx:02d}_{sample_idx}.jpg")
            for sample_idx in range(num_samples)
        ]
        if all(os.path.exists(path) for path in output_paths):
            image_bar.update(num_samples)
            continue

        for sample_idx, output_path in enumerate(output_paths):
            if os.path.exists(output_path):
                image_bar.update(1)
                continue
            _generate_one_image_manual(
                prompt=prompt,
                output_path=output_path,
                sample_seed=sample_idx,
                image_size=image_size,
                ddim_steps=ddim_steps,
                guidance_scale=guidance_scale,
                device=torch_device,
                vae=vae,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                unet=unet,
                scheduler=scheduler,
            )
            total_generated += 1
            image_bar.update(1)

    image_bar.close()
    print(f"Generated {total_generated} new images.")
    return {
        "output_dir": output_dir,
        "total_prompt_records": len(prompt_records),
        "num_samples": num_samples,
        "total_generated": total_generated,
    }


def sample_character_prompts_with_pipeline(
    diffusion_pipeline,
    prompt_records: Sequence[tuple[str, int, str]],
    output_dir: str,
    device: str | torch.device,
    image_size: int = 512,
    num_samples: int = 10,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    progress_desc: str = "Character Images",
) -> dict:
    torch_device = device if isinstance(device, torch.device) else torch.device(device)
    os.makedirs(output_dir, exist_ok=True)

    total_generated = 0
    total_images = len(prompt_records) * num_samples
    was_training = getattr(getattr(diffusion_pipeline, "unet", None), "training", False)
    if hasattr(diffusion_pipeline, "unet"):
        diffusion_pipeline.unet.eval()

    try:
        with tqdm(total=total_images, desc=progress_desc, unit="image") as progress:
            for character, prompt_idx, prompt in prompt_records:
                file_stub = f"{character.replace(' ', '_')}_{prompt_idx:02d}"
                for sample_idx in range(num_samples):
                    img_path = os.path.join(output_dir, f"{file_stub}_{sample_idx}.jpg")
                    if os.path.exists(img_path):
                        progress.update(1)
                        continue
                    generator = torch.Generator(device=torch_device).manual_seed(sample_idx)
                    image = diffusion_pipeline(
                        prompt=prompt,
                        height=image_size,
                        width=image_size,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                    ).images[0]
                    image.save(img_path)
                    total_generated += 1
                    progress.set_postfix(
                        {"character": character, "prompt": prompt_idx + 1, "sample": sample_idx}
                    )
                    progress.update(1)
    finally:
        if hasattr(diffusion_pipeline, "unet") and was_training:
            diffusion_pipeline.unet.train()

    return {
        "output_dir": output_dir,
        "total_prompt_records": len(prompt_records),
        "num_samples": num_samples,
        "total_generated": total_generated,
    }


def sample_character_images(args: argparse.Namespace) -> dict:
    template_lines = load_template(args.template_path)
    characters = collect_characters(args)
    prompt_records = build_prompt_records(characters, template_lines)
    print(f"Loaded '{len(template_lines)}' template prompts for '{len(characters)}' character(s).")
    return sample_character_prompts_manual(
        prompt_records=prompt_records,
        output_dir=args.output_dir,
        model_name=args.model_name,
        device=args.device,
        guidance_scale=args.guidance_scale,
        image_size=args.image_size,
        ddim_steps=args.ddim_steps,
        num_samples=args.num_samples,
    )


def generate_samples(prompt_records: Sequence[tuple[str, int, str]], args) -> dict:
    # Backward-compatible wrapper preserving the older function signature.
    return sample_character_prompts_manual(
        prompt_records=prompt_records,
        output_dir=args.output_dir,
        model_name=args.model_name,
        device=args.device,
        guidance_scale=args.guidance_scale,
        image_size=args.image_size,
        ddim_steps=args.ddim_steps,
        num_samples=args.num_samples,
    )


def main() -> None:
    args = parse_cli_args()
    sample_character_images(args)


if __name__ == "__main__":
    main()
