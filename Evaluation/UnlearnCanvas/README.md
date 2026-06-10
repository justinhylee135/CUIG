# UnlearnCanvas Evaluation

This folder contains the sampling and classifier-based evaluation utilities for the UnlearnCanvas benchmark. UnlearnCanvas evaluates whether a model forgets target artistic styles or object concepts while retaining performance on related in-domain concepts and cross-domain concepts.

The style/object evaluation setup uses resources from [UnlearnCanvas](https://github.com/OPTML-Group/UnlearnCanvas). Please credit the UnlearnCanvas authors when using this evaluation pipeline or checkpoint.

## Contents

- `sample.py`: samples benchmark images from a Diffusers pipeline, optionally with an unlearned UNet checkpoint.
- `evaluate.py`: evaluates generated images with the UnlearnCanvas style and object classifiers.
- `constants.py`: defines the supported style/object names and the paper-style reporting splits.

## Checkpoints

Before running this evaluation, prepare the UnlearnCanvas generator and classifier checkpoints described in [`../../Checkpoints/README.md`](../../Checkpoints/README.md).

By default, the examples below assume:

```bash
GENERATOR_DIR="$REPO_ROOT/Checkpoints/Generators/UnlearnCanvas"
CLASSIFIER_DIR="$REPO_ROOT/Checkpoints/Classifiers/UnlearnCanvas"
```

The classifier directory should contain the style and object classifier checkpoints used by `evaluate.py`.

## Sampling

Run `sample.py` to generate images for a style/object grid. Concept subsets are passed as Python-style list strings.

```bash
cd "$REPO_ROOT/Evaluation/UnlearnCanvas"

python sample.py \
  --pipeline_dir "$GENERATOR_DIR" \
  --output_dir "images/Abstractionism" \
  --styles_subset '["Abstractionism"]' \
  --objects_subset '["Architectures", "Butterfly", "Flame", "Flowers"]'
```

To sample from an unlearned model, pass the trained UNet checkpoint:

```bash
python sample.py \
  --pipeline_dir "$GENERATOR_DIR" \
  --unet_ckpt_path "$OUTPUT_ROOT/path/to/Models/checkpoint.pt" \
  --output_dir "images/Abstractionism_unlearned" \
  --styles_subset '["Abstractionism"]' \
  --objects_subset '["Architectures", "Butterfly", "Flame", "Flowers"]'
```

If `--styles_subset` or `--objects_subset` is omitted, sampling uses the full benchmark list from `constants.py`. The default seeds are `188 288 588 688 888`, and images are saved as:

```text
<style>_<object>_seed<seed>.jpg
```

## Evaluation

Run `evaluate.py` on a directory of sampled images. The evaluator always writes detailed style and object classifier results. If you also provide `--unlearn`, `--retain`, and `--cross_retain`, it writes summary metrics for the run.

Style-unlearning example:

```bash
python evaluate.py \
  --input_dir "images/Abstractionism_unlearned" \
  --output_dir "metrics/Abstractionism_unlearned" \
  --eval_classifier_dir "$CLASSIFIER_DIR" \
  --unlearn '["Abstractionism"]' \
  --retain '["Blossom_Season", "Rust", "Crayon", "Fauvism"]' \
  --cross_retain '["Architectures", "Butterfly", "Flame", "Flowers"]'
```

Object-unlearning example:

```bash
python evaluate.py \
  --input_dir "images/Bears_unlearned" \
  --output_dir "metrics/Bears_unlearned" \
  --eval_classifier_dir "$CLASSIFIER_DIR" \
  --unlearn '["Bears"]' \
  --retain '["Architectures", "Butterfly", "Flame", "Flowers"]' \
  --cross_retain '["Blossom_Season", "Rust", "Crayon", "Fauvism"]'
```

When using custom seeds during sampling, pass the same seed list during evaluation:

```bash
python evaluate.py \
  --input_dir "images/custom_seed_run" \
  --output_dir "metrics/custom_seed_run" \
  --eval_classifier_dir "$CLASSIFIER_DIR" \
  --seed 188 288 588 688 888
```

## Metrics

The summary metrics are reported as percentages:

- `UA`: unlearning accuracy, computed as `100 - target concept classifier accuracy`.
- `IRA`: in-domain retain accuracy for concepts from the same domain as the unlearned concept.
- `CRA`: cross-domain retain accuracy for concepts from the other domain.

For style unlearning, styles are the in-domain concepts and objects are cross-domain concepts. For object unlearning, objects are in-domain concepts and styles are cross-domain concepts.

## Outputs

`evaluate.py` writes:

- `style_results.json`: style classifier predictions, accuracies, losses, and per-image predictions.
- `object_results.json`: object classifier predictions, accuracies, losses, and per-image predictions.
- `summary.json`: written only when `--unlearn`, `--retain`, and `--cross_retain` are provided.
- `<run>_table.xlsx`: paper-style Excel report written with the summary metrics.

Missing sampled images are skipped with a warning, so interrupted sampling runs can still be inspected. For final reporting, make sure the sampled image directory contains every expected `<style>_<object>_seed<seed>.jpg` file.

## Notes

- Use exact concept names from `constants.py`; names are case-sensitive and use underscores.
- Provide all three summary arguments together: `--unlearn`, `--retain`, and `--cross_retain`.
- For full experiment pipelines, use the scripts under `BashScripts/`.
