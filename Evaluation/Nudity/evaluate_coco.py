from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image

try:
    from scipy import linalg
except ImportError as exc:  # pragma: no cover
    raise ImportError("scipy is required for FID computation. Please install scipy.") from exc

from torch.utils.data import DataLoader, Dataset
from torchvision.models import Inception_V3_Weights, inception_v3
from torchvision.models.feature_extraction import create_feature_extractor
from transformers import CLIPModel, CLIPProcessor


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate MS-COCO FID and CLIP scores for generated images."
    )
    parser.add_argument(
        "--input_dir",
        default="./evaluation_folder/coco/SD/",
        help="Directory containing generated COCO images.",
    )
    parser.add_argument(
        "--prompts_csv",
        default="ms_coco.csv",
        help="CSV file with prompts/case numbers used to generate the images.",
    )
    parser.add_argument(
        "--fid_stats_path",
        default="ms_coco.npz",
        help="Path to the COCO FID statistics npz file (mu, sigma).",
    )
    parser.add_argument(
        "--output_dir",
        default="./metrics",
        help="Directory where evaluation summaries will be stored.",
    )
    parser.add_argument("--clip_batch_size", type=int, default=16, help="Batch size for CLIP score dataloader.")
    parser.add_argument("--clip_workers", type=int, default=4, help="Number of workers for CLIP score dataloader.")
    parser.add_argument("--fid_batch_size", type=int, default=32, help="Batch size for Inception features.")
    parser.add_argument("--fid_workers", type=int, default=4, help="Number of workers for FID dataloader.")
    parser.add_argument("--device", default=None, help="Device to run models on (e.g. cuda:0 or cpu).")
    return parser


def parse_cli_args() -> argparse.Namespace:
    return build_arg_parser().parse_args()


def parse_args() -> argparse.Namespace:
    # Backward-compatible alias.
    return parse_cli_args()


def resolve_device(requested_device: str | None = None) -> torch.device:
    if requested_device:
        return torch.device(requested_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_fid_stats(stats_path: str) -> dict[str, np.ndarray]:
    stats = np.load(stats_path)
    required_keys = {"mu", "sigma"}
    missing = required_keys.difference(stats.files)
    if missing:
        raise ValueError(f"FID stats file must contain {required_keys}, but missing {missing}")
    return {key: stats[key] for key in required_keys}


def list_image_paths(input_dir: str) -> list[str]:
    valid_ext = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    matched = []
    for root, _, files in os.walk(input_dir):
        for filename in files:
            if "_" not in filename:
                continue
            ext = os.path.splitext(filename)[1].lower()
            if ext in valid_ext:
                matched.append(os.path.join(root, filename))
    return sorted(matched)


def build_prompt_lookup(df: pd.DataFrame) -> dict[str, str]:
    id_field = "image_id" if "image_id" in df.columns else "case_number"
    text_field = "text" if "text" in df.columns else "prompt"
    if text_field not in df.columns:
        raise ValueError("Prompts CSV must contain a 'text' or 'prompt' column.")

    lookup = {}
    for _, row in df.iterrows():
        value = row[id_field]
        if pd.isna(value):
            continue
        try:
            key = str(int(value))
        except (ValueError, TypeError):
            key = str(value).strip()
        lookup[key] = str(row[text_field])
    return lookup


def build_image_prompt_pairs(
    image_paths: list[str],
    prompt_lookup: dict[str, str],
) -> tuple[list[str], dict[str, str], set[str]]:
    captions_mapping = {}
    matched_image_paths = []
    missing_prompts = set()

    for img_path in image_paths:
        image_id = os.path.basename(img_path).split("_", 1)[0]
        prompt = prompt_lookup.get(image_id)
        if prompt is None:
            missing_prompts.add(image_id)
            continue
        matched_image_paths.append(img_path)
        captions_mapping[img_path] = prompt

    return matched_image_paths, captions_mapping, missing_prompts


class ImagePathDataset(Dataset):
    def __init__(self, image_paths: list[str], transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        with Image.open(path) as img:
            image = img.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


class ClipDataset(Dataset):
    def __init__(self, image_paths: list[str], captions_mapping: dict[str, str]):
        self.items = [(path, captions_mapping[path]) for path in image_paths if path in captions_mapping]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def clip_collate_fn(batch):
    paths, texts = zip(*batch)
    images = []
    for path in paths:
        with Image.open(path) as img:
            images.append(img.convert("RGB"))
    return images, list(texts)


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6) -> float:
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def calculate_fid(
    image_paths: list[str],
    fid_stats: dict[str, np.ndarray],
    batch_size: int = 32,
    num_workers: int = 4,
    device: str | None = None,
) -> float:
    if not image_paths:
        raise ValueError("No images found to evaluate FID.")

    torch_device = resolve_device(device)
    weights = Inception_V3_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()
    model = inception_v3(weights=weights)
    feature_extractor = create_feature_extractor(model, {"avgpool": "features"})
    feature_extractor.to(torch_device)
    feature_extractor.eval()

    dataset = ImagePathDataset(image_paths, transform=preprocess)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch_device.type == "cuda",
    )

    activations = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(torch_device)
            outputs = feature_extractor(batch)["features"]
            activations.append(outputs.squeeze(-1).squeeze(-1).cpu().numpy())

    activations = np.concatenate(activations, axis=0)
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return frechet_distance(mu, sigma, fid_stats["mu"], fid_stats["sigma"])


def calculate_clip_score(
    image_paths: list[str],
    captions_mapping: dict[str, str],
    batch_size: int = 16,
    dataloader_workers: int = 4,
    device: str | None = None,
) -> float:
    dataset = ClipDataset(image_paths, captions_mapping)
    if len(dataset) == 0:
        raise ValueError("No image/text pairs found for CLIP score calculation.")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=dataloader_workers,
        collate_fn=clip_collate_fn,
    )

    torch_device = resolve_device(device)
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(torch_device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model.eval()

    scores = []
    with torch.no_grad():
        for images, texts in dataloader:
            inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(torch_device)
            outputs = model(**inputs)
            scores.append(torch.diagonal(outputs.logits_per_image, 0).cpu())

    return float(torch.cat(scores).mean().item())


def save_results_excel(output_dir: str, fid: float, clip_score: float | None) -> str:
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "results.xlsx")
    pd.DataFrame({"Metric": ["FID", "CLIP"], "Value": [fid, clip_score]}).to_excel(results_path, index=False)
    return results_path


def evaluate_coco_metrics(
    input_dir: str,
    prompts_csv: str,
    fid_stats_path: str,
    output_dir: str,
    clip_batch_size: int = 16,
    clip_workers: int = 4,
    fid_batch_size: int = 32,
    fid_workers: int = 4,
    device: str | None = None,
) -> dict[str, Any]:
    all_image_paths = list_image_paths(input_dir)
    print(f"Found '{len(all_image_paths)}' images in '{input_dir}'.")
    if not all_image_paths:
        raise ValueError(f"No images found in '{input_dir}'.")

    fid_stats = load_fid_stats(fid_stats_path)
    fid = calculate_fid(all_image_paths, fid_stats, fid_batch_size, fid_workers, device)
    print(f"FID: {fid}")

    prompt_lookup = build_prompt_lookup(pd.read_csv(prompts_csv))
    matched_image_paths, captions_mapping, missing_prompts = build_image_prompt_pairs(all_image_paths, prompt_lookup)

    if missing_prompts:
        missing_subset = ", ".join(sorted(missing_prompts)[:5])
        suffix = f" (e.g. {missing_subset})" if missing_subset else ""
        print(f"Warning: missing prompts for {len(missing_prompts)} image_ids{suffix}")

    clip_score = None
    if not matched_image_paths:
        print("Warning: no matching prompts found; skipping CLIP score.")
    else:
        clip_score = calculate_clip_score(
            matched_image_paths,
            captions_mapping,
            batch_size=clip_batch_size,
            dataloader_workers=clip_workers,
            device=device,
        )
        print(f"CLIP score: {clip_score}")

    results_path = save_results_excel(output_dir, fid, clip_score)
    print(f"Saved metrics summary to '{results_path}'")
    return {
        "fid": fid,
        "clip": clip_score,
        "num_images": len(all_image_paths),
        "num_clip_pairs": len(matched_image_paths),
        "missing_prompt_ids": sorted(missing_prompts),
        "results_path": results_path,
    }


def run_evaluation_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return evaluate_coco_metrics(
        input_dir=args.input_dir,
        prompts_csv=args.prompts_csv,
        fid_stats_path=args.fid_stats_path,
        output_dir=args.output_dir,
        clip_batch_size=args.clip_batch_size,
        clip_workers=args.clip_workers,
        fid_batch_size=args.fid_batch_size,
        fid_workers=args.fid_workers,
        device=args.device,
    )


def main() -> None:
    args = parse_cli_args()
    run_evaluation_from_args(args)


if __name__ == "__main__":
    main()
