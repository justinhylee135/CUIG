# Standard Library
import json
import os
from pathlib import Path

# Third Party
import torch

# Local
from Evaluation.Character import evaluate_character as character_eval
from Evaluation.Character import sample_characters as character_sample


def _parse_character_names(target_concepts) -> list[str]:
    if target_concepts is None:
        raw_entries = []
    elif isinstance(target_concepts, str):
        raw_value = target_concepts.strip()
        if raw_value.startswith("["):
            try:
                parsed = json.loads(raw_value)
            except json.JSONDecodeError:
                parsed = [target_concepts]
            raw_entries = parsed if isinstance(parsed, list) else [target_concepts]
        else:
            raw_entries = [target_concepts]
    else:
        raw_entries = list(target_concepts)

    characters: list[str] = []
    seen = set()
    for entry in raw_entries:
        if entry is None:
            continue
        name = str(entry)
        if "+" in name:
            # Training configs may encode prefixes like "character+Batman".
            name = name.split("+")[-1]
        formatted = name.replace("_", " ").strip()
        if not formatted:
            continue
        key = formatted.lower()
        if key not in seen:
            seen.add(key)
            characters.append(formatted)
    return characters


def _write_summary_json(metrics_dir: Path, results: dict) -> None:
    avg_ua_percent = results["avg_ua"] * 100
    summary = {
        "unlearn_accuracy_avg": round(avg_ua_percent, 2),
        "total_images": results.get("total_images", 0),
        "missing_characters": results.get("missing_characters", []),
    }
    for character in results.get("unlearn", []):
        summary[f"unlearn_{character}"] = round(results["ua"][character] * 100, 2)

    summary_path = metrics_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def sample_and_evaluate_ua_character(
    diffusion_pipeline,
    iteration,
    model_save_path,
    target_concepts,
    device,
    eval_classifier_dir,
    eval_prompt_dir,
):
    del eval_classifier_dir  # Character eval assets are read from eval_prompt_dir for this pipeline.

    characters = _parse_character_names(target_concepts)
    if not characters:
        raise ValueError("Character simultaneous evaluation requires at least one entry in target_concepts.")

    eval_assets_dir = Path(eval_prompt_dir)
    template_path = eval_assets_dir / "SixCD_Template.txt"
    classifier_path = eval_assets_dir / "resnet50_copyright_101_71.pt"
    labels_path = eval_assets_dir / "labels.csv"

    output_dir = Path(model_save_path).parent / f"logs/log_{iteration}"
    img_dir = output_dir / "images"
    metrics_dir = output_dir / "metrics"
    img_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Simultaneous-Character] Saving sampled images to: {img_dir}")

    template_prompts = character_sample.load_template(str(template_path))
    prompt_records = character_sample.build_prompt_records(characters, template_prompts)
    if not prompt_records:
        print("[Simultaneous-Character] No prompt records were generated; skipping sampling.")
    else:
        pipe_device = device if isinstance(device, torch.device) else torch.device(device)
        character_sample.sample_character_prompts_with_pipeline(
            diffusion_pipeline=diffusion_pipeline,
            prompt_records=prompt_records,
            output_dir=str(img_dir),
            device=pipe_device,
            image_size=512,
            num_samples=10,
            num_inference_steps=50,
            guidance_scale=7.5,
            progress_desc="Character Images",
        )

    eval_device = device if isinstance(device, torch.device) else torch.device(device)
    eval_output = character_eval.evaluate_character_metrics(
        input_dir=img_dir,
        output_dir=metrics_dir,
        unlearn=characters,
        retain=[],
        classifier_path=classifier_path,
        labels_path=labels_path,
        batch_size=64,
        num_workers=min(4, os.cpu_count() or 1),
        image_size=224,
        device=str(eval_device),
        extensions=character_eval.SUPPORTED_EXTENSIONS,
        results_json_name="character_results.json",
        write_excel=False,
    )
    results = eval_output["results"]

    for character in results.get("unlearn", []):
        print(f"[Simultaneous-Character] UA {character}: {results['ua'][character] * 100:.2f}%")
    if results.get("avg_ua", 0) > 0:
        print(f"[Simultaneous-Character] Average UA: {results['avg_ua'] * 100:.2f}%")

    _write_summary_json(metrics_dir, results)
    print(f"[Simultaneous-Character] Metrics saved to: {metrics_dir / 'character_results.json'}")
    return round(results["avg_ua"] * 100, 2)
