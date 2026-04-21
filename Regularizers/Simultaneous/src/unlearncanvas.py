# Standard Library
import json
import os
from pathlib import Path

# Local
from Evaluation.UnlearnCanvas.constants import (
    OBJECTS_AVAILABLE,
    STYLES_AVAILABLE,
    XLSX_ALL_RETAIN_OBJECTS,
    XLSX_ALL_RETAIN_STYLES,
)
from Evaluation.UnlearnCanvas.evaluate import evaluate_unlearncanvas
from Evaluation.UnlearnCanvas.sample import generate_unlearncanvas_images


def _append_unique(items, values):
    # Preserve first-seen ordering while avoiding duplicates in subset lists.
    for value in values:
        if value not in items:
            items.append(value)


def _normalize_target_concept(prompt: str) -> str:
    # Training code passes target prompts like "An image of X Style"; normalize into
    # the benchmark naming convention used by UnlearnCanvas constants/files.
    concept = prompt.replace("An image of ", "")
    concept = concept.replace(" Style", "")
    concept = concept.replace(" ", "_")
    return concept


def _resolve_concept_name(raw_concept: str) -> tuple[str, str]:
    # First try exact matches against canonical benchmark concept names.
    if raw_concept in STYLES_AVAILABLE:
        return "style", raw_concept
    if raw_concept in OBJECTS_AVAILABLE:
        return "object", raw_concept

    # Fall back to substring matching to support slightly different prompt spellings.
    normalized = raw_concept.lower()
    for theme in STYLES_AVAILABLE:
        if normalized in theme.lower():
            return "style", theme
    for obj in OBJECTS_AVAILABLE:
        if normalized in obj.lower():
            return "object", obj

    raise ValueError(f"Concept '{raw_concept}' not found in available themes or classes.")


def _parse_target_concepts(target_concepts):
    # Split target concepts by domain because evaluation summary inputs differ for
    # style-unlearning vs object-unlearning runs.
    unlearn_styles = []
    unlearn_objects = []
    unlearn_subset = []

    for prompt in target_concepts:
        concept_type, concept_name = _resolve_concept_name(_normalize_target_concept(prompt))
        if concept_type == "style":
            _append_unique(unlearn_styles, [concept_name])
        else:
            _append_unique(unlearn_objects, [concept_name])
        _append_unique(unlearn_subset, [concept_name])

    return unlearn_styles, unlearn_objects, unlearn_subset


def _build_sampling_subsets(concept_type: str, unlearn_styles: list[str], unlearn_objects: list[str]):
    # Sampling includes the unlearn targets plus fixed retain sets so UA/IRA/CRA can
    # be computed from a single image generation pass.
    styles_subset = list(unlearn_styles)
    objects_subset = list(unlearn_objects)

    if concept_type == "style":
        # Style unlearning: retain styles are in-domain, retain objects are cross-domain.
        _append_unique(objects_subset, XLSX_ALL_RETAIN_OBJECTS)
        _append_unique(styles_subset, XLSX_ALL_RETAIN_STYLES)
    elif concept_type == "object":
        # Object unlearning mirrors the style case with domains swapped.
        _append_unique(styles_subset, XLSX_ALL_RETAIN_STYLES)
        _append_unique(objects_subset, XLSX_ALL_RETAIN_OBJECTS)
    elif concept_type == "interleave":
        # Interleave runs can include targets from both domains, but we still sample
        # against the standard retain sets for consistent monitoring.
        _append_unique(styles_subset, XLSX_ALL_RETAIN_STYLES)
        _append_unique(objects_subset, XLSX_ALL_RETAIN_OBJECTS)
    else:
        raise ValueError(f"Unsupported UnlearnCanvas concept_type: '{concept_type}'")

    return styles_subset, objects_subset


def _build_eval_summary_args(
    concept_type: str,
    unlearn_styles: list[str],
    unlearn_objects: list[str],
):
    # The reusable evaluator computes UA/IRA/CRA only for single-domain unlearning.
    if concept_type == "style" and unlearn_styles and not unlearn_objects:
        return {
            "unlearn": unlearn_styles,
            "retain": list(XLSX_ALL_RETAIN_STYLES),
            "cross_retain": list(XLSX_ALL_RETAIN_OBJECTS),
            "is_style_unlearn": True,
        }
    if concept_type == "object" and unlearn_objects and not unlearn_styles:
        return {
            "unlearn": unlearn_objects,
            "retain": list(XLSX_ALL_RETAIN_OBJECTS),
            "cross_retain": list(XLSX_ALL_RETAIN_STYLES),
            "is_style_unlearn": False,
        }
    return {
        "unlearn": None,
        "retain": None,
        "cross_retain": None,
        "is_style_unlearn": None,
    }


def _compute_unlearn_accuracy_summary(unlearn_subset, results_style, results_object):
    # Simultaneous regularizer early stopping only needs UA. We recompute it from the
    # saved task results so the wrapper works for style, object, and mixed runs.
    unlearn_accuracies = []
    for concept in unlearn_subset:
        if concept in results_style["acc"]:
            accuracy = 100.0 - results_style["acc"][concept]
        elif concept in results_object["acc"]:
            accuracy = 100.0 - results_object["acc"][concept]
        else:
            raise ValueError(f"Concept '{concept}' not found in results_style or results_object dictionaries")
        unlearn_accuracies.append((concept, accuracy))
        print(f"Unlearn accuracy for {concept}: {accuracy:.2f}%")

    unlearn_accuracy_avg = sum(acc for _, acc in unlearn_accuracies) / len(unlearn_accuracies)
    print(f"Mean unlearn accuracy: {unlearn_accuracy_avg:.2f}%")
    return unlearn_accuracies, round(unlearn_accuracy_avg, 2)


def _write_summary_json(metrics_dir, unlearn_accuracy_avg, unlearn_accuracies, evaluation_summary=None):
    # Preserve the legacy fields used by the simultaneous regularizer, while retaining richer metrics when present.
    summary = {} if evaluation_summary is None else dict(evaluation_summary)
    summary["unlearn_accuracy_avg"] = round(unlearn_accuracy_avg, 2)
    for concept, accuracy in unlearn_accuracies:
        summary[f"unlearn_{concept}"] = round(accuracy, 2)

    with open(os.path.join(metrics_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=4)


def sample_and_evaluate_ua_unlearncanvas(
    diffusion_pipeline,
    concept_type,
    iteration,
    model_save_path,
    target_concepts,
    device,
    eval_classifier_dir,
):
    # Parse prompt strings into canonical UnlearnCanvas concept names and domains.
    unlearn_styles, unlearn_objects, unlearn_subset = _parse_target_concepts(target_concepts)
    print(f"Unlearn Subset: {unlearn_subset}")

    # Build the exact concept subsets to sample (targets + retain controls).
    styles_subset, objects_subset = _build_sampling_subsets(concept_type, unlearn_styles, unlearn_objects)
    print("Sampling:")
    print(f"\tClasses subset: {objects_subset}")
    print(f"\tThemes subset: {styles_subset}")

    # Keep the original simultaneous-eval directory layout so downstream scripts and
    # logs remain compatible across checkpoints/iterations.
    output_dir = os.path.join(Path(model_save_path).parent, f"logs/log_{iteration}")
    img_dir = os.path.join(output_dir, "images")
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    # Sampling should not leave the training UNet in eval mode if the caller resumes training.
    was_training = diffusion_pipeline.unet.training
    diffusion_pipeline.unet.eval()
    try:
        # Reuse the shared UnlearnCanvas sampler (same prompt template / seeds / CFG).
        generate_unlearncanvas_images(
            pipe=diffusion_pipeline,
            output_dir=img_dir,
            styles_subset=styles_subset,
            objects_subset=objects_subset,
            seeds=(188, 288, 588, 688, 888),
            resolution=512,
            num_inference_steps=100,
            guidance_scale=9.0,
        )
    finally:
        if was_training:
            diffusion_pipeline.unet.train()

    # Reuse the shared evaluator. Summary metrics are only computed when the run is
    # single-domain (pure style or pure object), but task JSON outputs are always produced.
    eval_summary_args = _build_eval_summary_args(concept_type, unlearn_styles, unlearn_objects)
    eval_output = evaluate_unlearncanvas(
        input_dir=img_dir,
        output_dir=metrics_dir,
        eval_classifier_dir=eval_classifier_dir,
        device=device,
        seed_list=(188, 288, 588, 688, 888),
        unlearn=eval_summary_args["unlearn"],
        retain=eval_summary_args["retain"],
        cross_retain=eval_summary_args["cross_retain"],
        styles_available_subset=styles_subset,
        objects_available_subset=objects_subset,
        is_style_unlearn=eval_summary_args["is_style_unlearn"],
    )

    # Keep the historical return value and summary fields used by early stopping code.
    unlearn_accuracies, unlearn_accuracy_avg = _compute_unlearn_accuracy_summary(
        unlearn_subset=unlearn_subset,
        results_style=eval_output["results_style"],
        results_object=eval_output["results_object"],
    )
    _write_summary_json(
        metrics_dir=metrics_dir,
        unlearn_accuracy_avg=unlearn_accuracy_avg,
        unlearn_accuracies=unlearn_accuracies,
        evaluation_summary=eval_output.get("summary"),
    )
    print(f"Unlearn accuracy (average): {unlearn_accuracy_avg:.2f}%")
    return unlearn_accuracy_avg
