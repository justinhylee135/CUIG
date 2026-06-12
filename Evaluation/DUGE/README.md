# DUGE Evaluation

To support the concurrent work, we incorporate the evaluation framework introduced in the paper [*Continual Unlearning for Foundational Text-to-Image Models without Generalization Erosion*](https://arxiv.org/pdf/2503.13769). The implementation was adapted and refactored from the authors' official [repository](https://github.com/IAB-IITJ/Continual-Unlearning). We ask that users of this benchmark setup appropriately credit the original paper and repository.

## Contents

- `sample_duge.py`: samples DUGE object prompts from a Diffusers pipeline and optional UNet checkpoint.
- `evaluate_imagenet.py`: evaluates generated object images with ImageNet ResNet-50.
- `evaluate_fid_kid.py`: computes FID and KID between generated and reference image folders.
- `prompts.csv`: default DUGE prompt list with `concept,prompt` columns.

## Setup

Start from the DUGE evaluation directory:

```bash
cd "$REPO_ROOT/Evaluation/DUGE"
conda activate cuig
```

`evaluate_fid_kid.py` uses `cleanfid`. Install it in the active environment if it
is not already available:

```bash
pip install clean-fid
```

## Sample Images

Sample the default prompt set from the base Stable Diffusion 2.1 pipeline:

```bash
python sample_duge.py \
  --model_name "base" \
  --pipeline_dir "stabilityai/stable-diffusion-2-1-base" \
  --prompts_path "prompts.csv" \
  --output_dir "images/base"
```

To sample from an unlearned checkpoint, pass the checkpoint path through
`--model_name` or the alias `--ckpt`:

```bash
python sample_duge.py \
  --ckpt "$OUTPUT_ROOT/path/to/Models/apple/delta.bin" \
  --pipeline_dir "stabilityai/stable-diffusion-2-1-base" \
  --prompts_path "prompts.csv" \
  --target_concepts '["apple", "broccoli", "traffic-light"]' \
  --output_dir "images/apple_unlearned"
```

Images are saved under one folder per concept:

```text
<output_dir>/<concept>/<prompt_index>_<sample_index>_<prompt>.png
<output_dir>/metadata.csv
```

The metadata file stores the prompt, concept, relative image path, and seed for
each generated image.

## Evaluate ImageNet Accuracy

Run the ImageNet evaluator on a sampled image folder:

```bash
python evaluate_imagenet.py \
  --input_dir "images/apple_unlearned" \
  --output_dir "metrics/apple_unlearned" \
  --target_concepts '["apple", "broccoli", "traffic-light"]'
```

The evaluator uses the sampler metadata when available. If metadata is missing,
it infers each image's concept from the parent folder name.

ImageNet accuracy is the fraction of generated images classified as the expected
object. Unlearning accuracy is reported as `1 - accuracy`.

## Evaluate FID and KID

Compare generated images against a reference folder:

```bash
python evaluate_fid_kid.py \
  --input_dir "images/apple_unlearned" \
  --reference_dir "images/base" \
  --output_dir "metrics/apple_unlearned_fid"
```

Both input folders must contain generated image files. For concept-specific FID,
point `--input_dir` and `--reference_dir` to matching concept subfolders.

## Outputs

`evaluate_imagenet.py` writes:

- `summary.json`: overall and per-concept ImageNet accuracy and unlearning accuracy.
- `per_image.csv`: one row per evaluated image with predicted ImageNet class and correctness.

`evaluate_fid_kid.py` writes:

- `fid_kid.json`: FID and KID scores for the compared folders.

## Notes

- Concept names are normalized to lowercase hyphenated form, so `traffic light`,
  `traffic_light`, and `traffic-light` are treated consistently.
- `evaluate_imagenet.py` scores only concepts with ImageNet class mappings.
- Use smaller `--num_samples` values for quick smoke tests.
- Full experiment launchers can wrap these scripts from `BashScripts/` when DUGE
  is added to a larger run pipeline.
