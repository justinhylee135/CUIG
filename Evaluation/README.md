# Evaluation

This directory contains sampling and metric code for CUIG benchmarks. Each evaluation suite follows the same rough workflow:

1. Generate images from a base or unlearned model.
2. Run the benchmark-specific evaluator on those images.
3. Save metrics under an experiment output directory.

Most end-to-end runs are launched through `BashScripts/`; use the scripts here directly when debugging a benchmark, adding a new evaluator, or rerunning metrics.

## Layout

```text
Evaluation/
|-- UnlearnCanvas/   # Style/object sampling and classifier evaluation
|-- Celebrity/       # Celebrity prompt generation, sampling, face-ID evaluation, and COCO retention
|-- Character/       # Copyrighted character sampling and classifier evaluation
|-- Nudity/          # Nudity benchmark sampling, detection, and COCO retention
`-- README.md
```

Detailed benchmark instructions live in the sub-folder READMEs:

- [UnlearnCanvas](UnlearnCanvas/README.md)
- [Celebrity](Celebrity/README.md)
- [Character](Character/README.md)
- [Nudity](Nudity/README.md)

## UnlearnCanvas

[`UnlearnCanvas/`](UnlearnCanvas/README.md) evaluates object and style unlearning.

Main entrypoints:

- `sample.py`: samples images from a Diffusers pipeline and optional UNet checkpoint.
- `evaluate.py`: computes UnlearnCanvas classifier metrics using the object/style classifiers.
- `constants.py`: defines benchmark concept lists and report labels.

Typical inputs:

- `--pipeline_dir`: base generator checkpoint, usually `Checkpoints/Generators/UnlearnCanvas`.
- `--unet_ckpt_path`: optional unlearned `delta.bin` or merged checkpoint.
- `--eval_classifier_dir`: classifier folder, usually `Checkpoints/Classifiers/UnlearnCanvas`.

See [`UnlearnCanvas/README.md`](UnlearnCanvas/README.md) for sampling commands, evaluation commands, metric definitions, and output files.

## Celebrity

[`Celebrity/`](Celebrity/README.md) evaluates celebrity identity unlearning and general retention.

Main entrypoints:

- `generate_celeb_prompts.py`: creates prompts for celebrity names.
- `sample_celeb.py`: samples celebrity images from Stable Diffusion checkpoints.
- `evaluate_celeb.py`: evaluates generated images with the celebrity face classifier.
- `coco/sample_coco.py` and `coco/evaluate_coco.py`: sample and score COCO-style retention images.
- `celeb_eval_env.yaml`: environment for the celebrity face-ID evaluator.

See [`Celebrity/README.md`](Celebrity/README.md) for the external `celeb-detection-oss` setup, celebrity sampling/evaluation commands, and COCO retention checks.

## Character and Nudity

[`Character/`](Character/README.md) and [`Nudity/`](Nudity/README.md) contain additional benchmark-specific samplers and evaluators.

Character:

- `sample_characters.py`
- `evaluate_character.py`
- `labels.csv`

Nudity:

- `sample_from_csv.py`
- `evaluate_nudenet.py`
- `evaluate_coco.py`
- `nudity_benchmark.csv`

See [`Character/README.md`](Character/README.md) for character checkpoint setup and evaluation commands. See [`Nudity/README.md`](Nudity/README.md) for NudeNet setup, nudity benchmark evaluation, and COCO retention commands.

## Output Conventions

Experiment scripts usually write evaluation artifacts under `$OUTPUT_ROOT`:

```text
$OUTPUT_ROOT/.../Results/<target>/images/
$OUTPUT_ROOT/.../Results/<target>/metrics/
```

Celebrity COCO retention checks commonly use:

```text
$OUTPUT_ROOT/.../Results/<target>/coco/images/
$OUTPUT_ROOT/.../Results/<target>/coco/metrics/
```

## Adding a New Evaluation

Add new benchmark code as a subdirectory under `Evaluation/` and keep the interface script-friendly:

- provide a sampler when the benchmark needs generated images
- provide an evaluator that accepts `--input_dir` and `--output_dir`
- keep benchmark assets, labels, and small CSVs next to the evaluator
- store large downloaded models or classifiers under `Checkpoints/`
- add a short README if setup requires external assets or a separate environment
