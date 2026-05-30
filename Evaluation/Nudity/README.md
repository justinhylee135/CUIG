# Nudity Evaluation

This directory contains utilities to sample images from nudity-related prompts, evaluate them with NudeNet, and run COCO-style retention metrics.

## Setup

Start from the nudity evaluation directory:

```bash
cd "$REPO_ROOT/Evaluation/Nudity"
conda activate cuig
```

## Sample Nudity Benchmark Images

Sample images from the nudity benchmark prompts:

```bash
python sample_from_csv.py \
    --model_name "SD" \
    --output_dir "images" \
    --num_prompts 200 \
    --prompts_path "nudity_benchmark.csv"
```

`--model_name` can be `"SD"` for the base model or a checkpoint path for an unlearned model.

## Evaluate Nudity Accuracy

Run the NudeNet evaluator on the sampled images:

```bash
python evaluate_nudenet.py \
    --input_dir "images" \
    --output_dir "metrics"
```

The default prompt file is `nudity_benchmark.csv`; pass `--prompts_path` if you evaluate a different CSV.

## Evaluate COCO Retention

COCO-style retention evaluation reports FID and CLIP score on generated MS-COCO prompts.

Resource URLs used by the evaluator:

```text
COCO_CAPTIONS_URL = https://github.com/boomb0om/text2image-benchmark/releases/download/v0.0.1/MS-COCO_val2014_30k_captions.csv
COCO_FID_STATS_URL = https://github.com/boomb0om/text2image-benchmark/releases/download/v0.0.1/MS-COCO_val2014_fid_stats.npz
```

Sample COCO images:

```bash
python sample_from_csv.py \
    --model_name "SD" \
    --output_dir "coco_images" \
    --num_prompts 200 \
    --prompts_path "ms_coco.csv"
```

Evaluate FID and CLIP score:

```bash
python evaluate_coco.py \
    --input_dir "coco_images" \
    --output_dir "coco_metrics"
```

## Notes

- Adjust output paths and prompt CSVs for your experiment.
- Use a smaller `--num_prompts` value for quick smoke tests.
- Ensure all required dependencies are installed before running the scripts.
