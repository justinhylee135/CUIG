# Celebrity Evaluation

The celebrity evaluation setup uses resources from [GIPHY's Open-Source Celebrity Detection Deep Learning Model](https://github.com/Giphy/celeb-detection-oss).

This directory contains utilities to sample celebrity images and evaluate identity unlearning. The workflow is:

1. Generate or reuse prompts for each celebrity.
2. Sample images from a base or unlearned model.
3. Evaluate generated images with the celebrity face-recognition classifier.
4. Optionally evaluate COCO-style retention.

## Setup

Start from the celebrity evaluation directory:

```bash
cd "$REPO_ROOT/Evaluation/Celebrity"
conda activate cuig
```

Download the external GIPHY celebrity detector code and resources:

```bash
git clone https://github.com/Giphy/celeb-detection-oss
curl -L https://s3.amazonaws.com/giphy-public/models/celeb-detection/resources.tar.gz -o resources.tar.gz
tar -xzf resources.tar.gz
rm resources.tar.gz
```

Create `$REPO_ROOT/Evaluation/Celebrity/celeb-detection-oss/.env` and point it to the extracted resources:

```bash
APP_DATA_DIR=$REPO_ROOT/Evaluation/Celebrity/resources
APP_RECOGNITION_WEIGHTS_FILE=$REPO_ROOT/Evaluation/Celebrity/resources/face_recognition/best_model_states.pkl
```

Replace the detector example evaluator with the CUIG wrapper:

```bash
cp "$REPO_ROOT/Evaluation/Celebrity/evaluate_celeb.py" \
   "$REPO_ROOT/Evaluation/Celebrity/celeb-detection-oss/examples/evaluate_celeb.py"
```

Create and install the celebrity evaluation environment:

```bash
conda env create -f celeb_eval_env.yaml

cd "$REPO_ROOT/Evaluation/Celebrity/celeb-detection-oss"
conda activate celeb_eval
python setup.py install
```

Before running `setup.py`, comment out line 37 in `celeb-detection-oss/setup.py`; the requirements are already installed through `celeb_eval_env.yaml`.

## Sample Images

Generate prompts for a celebrity:

```bash
cd "$REPO_ROOT/Evaluation/Celebrity"
conda activate cuig

python generate_celeb_prompts.py \
    --prompt "Natalie Portman" \
    --output_path "prompts/Natalie_Portman.txt"
```

Sample images from standard Stable Diffusion or an unlearned checkpoint:

```bash
python sample_celeb.py \
    --model_family "sd" \
    --ckpt "SD" \
    --output_dir "images" \
    --celeb_subset '["Natalie_Portman"]'
```

`--ckpt` can be `"SD"` for the base model or a checkpoint path for an unlearned model.

## Evaluate Celebrity Accuracy

Run the celebrity evaluator:

```bash
cd "$REPO_ROOT/Evaluation/Celebrity/celeb-detection-oss"
conda activate celeb_eval

python examples/evaluate_celeb.py \
    --input_dir "../images" \
    --output_dir "../metrics" \
    --unlearn '[]' \
    --retain '["Natalie_Portman"]'
```

If setup is correct, evaluating base-model Natalie Portman samples should produce an RA score around 85-95%.

For experiment outputs, use the same command shape with your result folders:

```bash
python examples/evaluate_celeb.py \
    --input_dir "$OUTPUT_ROOT/.../Results/Natalie_Portman/images" \
    --output_dir "$OUTPUT_ROOT/.../Results/Natalie_Portman/metrics" \
    --unlearn '[]' \
    --retain '["Natalie_Portman"]'
```

## Evaluate COCO Retention

COCO-style retention evaluation uses the scripts under `Evaluation/Celebrity/coco`.

```bash
conda activate cuig
cd "$REPO_ROOT/Evaluation/Celebrity/coco"
```

`--model_name` can be `"base"` or the checkpoint path of an unlearned model. For final evaluation, set `--num_prompts 5000`; a smaller value is useful for smoke tests.

```bash
python sample_coco.py \
    --model_name "base" \
    --output_dir "images" \
    --num_prompts 50
```

```bash
python evaluate_coco.py \
    --input_dir "images" \
    --output_dir "metrics"
```

If COCO evaluation is working, base-model samples should produce an FID score around 347 and a CLIP score around 26.
