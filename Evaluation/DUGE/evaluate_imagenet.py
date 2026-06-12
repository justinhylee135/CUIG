from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path


IMAGENET_LABELS = {
    "traffic-light": {920},
    "kite": {21},
    "broccoli": {937},
    "umbrella": {879},
    "orange": {950},
    "apple": {948},
    "backpack": {414},
    "goldfish": {1},
    "cassette-player": {482},
    "bowtie": {457},
    "banana": {954},
    "teddy-bear": {850},
    "broom": {462},
    "computer-mouse": {673},
    "pizza": {963},
    "computer-keyboard": {508},
    "pillow": {721},
    "french-horn": {566},
    "sunglasses": {836, 837},
    "harp": {594},
    "tench": {0},
    "parachute": {701},
    "golf-ball": {574},
    "chainsaw": {491},
    "english-springer": {217},
    "gas-pump": {571},
    "garbage-truck": {569},
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate DUGE object images with ImageNet ResNet-50.")
    parser.add_argument("--input_dir", required=True, help="Directory containing generated images.")
    parser.add_argument("--output_dir", required=True, help="Directory where metrics are written.")
    parser.add_argument("--metadata_path", default=None, help="Optional metadata CSV from sample_duge.py.")
    parser.add_argument(
        "--target_concepts",
        default=None,
        help="Optional concept subset as a Python list or comma-separated string.",
    )
    parser.add_argument("--device", default="cuda:0", help="Device such as cuda:0 or cpu.")
    parser.add_argument("--batch_size", type=int, default=32, help="ImageNet classifier batch size.")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader worker count.")
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


def resolve_device(device_arg: str):
    import torch

    if device_arg == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")

    torch.cuda.set_device(device_arg)
    return torch.device("cuda")


def load_metadata(input_dir: Path, metadata_path: str | None) -> dict[str, dict[str, str]]:
    path = Path(metadata_path) if metadata_path else input_dir / "metadata.csv"
    if not path.exists():
        return {}

    metadata = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            relative_path = row.get("relative_path")
            concept = row.get("concept")
            if relative_path and concept:
                metadata[relative_path] = row
    return metadata


def list_image_records(input_dir: Path, metadata: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    valid_ext = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    records = []

    for path in sorted(input_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in valid_ext:
            continue

        relative_path = str(path.relative_to(input_dir))
        metadata_row = metadata.get(relative_path, {})
        concept = metadata_row.get("concept")
        if concept is None:
            concept = path.parent.name if path.parent != input_dir else path.stem

        records.append(
            {
                "path": str(path),
                "relative_path": relative_path,
                "concept": slugify(concept),
                "prompt": metadata_row.get("prompt", ""),
            }
        )
    return records


class DUGEImageDataset:
    def __init__(self, records: list[dict[str, str]], transform):
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        from PIL import Image

        record = self.records[index]
        with Image.open(record["path"]) as image:
            image = image.convert("RGB")
        return self.transform(image), index


def evaluate_imagenet(args: argparse.Namespace) -> dict:
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input image directory not found: {input_dir}")

    target_concepts = parse_concept_list(args.target_concepts)
    metadata = load_metadata(input_dir, args.metadata_path)
    records = list_image_records(input_dir, metadata)
    if target_concepts is not None:
        records = [record for record in records if record["concept"] in target_concepts]
    records = [record for record in records if record["concept"] in IMAGENET_LABELS]

    if not records:
        raise ValueError("No evaluable images found. Check --input_dir, --metadata_path, and --target_concepts.")

    import torch
    from torch.utils.data import DataLoader
    from torchvision.models import ResNet50_Weights, resnet50

    device = resolve_device(args.device)
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights).to(device)
    model.eval()
    categories = weights.meta["categories"]

    dataset = DUGEImageDataset(records, weights.transforms())
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    per_image_rows = []
    correct_by_concept = defaultdict(int)
    total_by_concept = defaultdict(int)

    with torch.no_grad():
        for images, indices in dataloader:
            images = images.to(device)
            logits = model(images)
            predictions = torch.argmax(logits, dim=1).cpu().tolist()

            for pred_idx, record_index in zip(predictions, indices.tolist()):
                record = records[record_index]
                concept = record["concept"]
                is_correct = int(pred_idx in IMAGENET_LABELS[concept])
                total_by_concept[concept] += 1
                correct_by_concept[concept] += is_correct
                per_image_rows.append(
                    {
                        "relative_path": record["relative_path"],
                        "concept": concept,
                        "prompt": record["prompt"],
                        "predicted_index": pred_idx,
                        "predicted_label": categories[pred_idx],
                        "correct": is_correct,
                    }
                )

    summary = {}
    for concept in sorted(total_by_concept):
        total = total_by_concept[concept]
        correct = correct_by_concept[concept]
        summary[concept] = {
            "correct": correct,
            "total": total,
            "accuracy": correct / total if total else None,
            "unlearning_accuracy": 1.0 - (correct / total) if total else None,
        }

    overall_total = sum(total_by_concept.values())
    overall_correct = sum(correct_by_concept.values())
    result = {
        "input_dir": str(input_dir),
        "overall": {
            "correct": overall_correct,
            "total": overall_total,
            "accuracy": overall_correct / overall_total,
            "unlearning_accuracy": 1.0 - (overall_correct / overall_total),
        },
        "by_concept": summary,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "summary.json", "w") as handle:
        json.dump(result, handle, indent=2)
    with open(output_dir / "per_image.csv", "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["relative_path", "concept", "prompt", "predicted_index", "predicted_label", "correct"],
        )
        writer.writeheader()
        writer.writerows(per_image_rows)

    print(json.dumps(result["overall"], indent=2))
    return result


def main() -> None:
    evaluate_imagenet(parse_cli_args())


if __name__ == "__main__":
    main()
