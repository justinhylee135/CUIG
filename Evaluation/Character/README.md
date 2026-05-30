# Character Evaluation

This directory contains utilities to sample images and evaluate copyrighted character unlearning.

## Setup

Start from the character evaluation directory:

```bash
cd "$REPO_ROOT/Evaluation/Character"
conda activate cuig
```

Download the character classifier checkpoint:

```bash
wget https://raw.githubusercontent.com/Artanisax/Six-CD/main/ResNet/ckpts/resnet50_copyright_101_71.pt
```

## Sample Images

Sample images for the character set you want to evaluate:

```bash
python sample_characters.py \
    --model_name "SD" \
    --characters '["Iron_Man", "Homer_Simpson", "Batman", "Naruto_Uzumaki", "Avatar_Aang", "Hulk", "Captain_America", "Bart_Simpson", "Superman", "Sasuke_Uchiha", "Zuko", "Thor"]' \
    --output_dir "images"
```

`--model_name` can be `"SD"` for the base model or a checkpoint path for an unlearned model.

## Evaluate Using Classifier

Run the character classifier on the sampled images:

```bash
python evaluate_character.py \
    --input_dir "images" \
    --output_dir "metrics" \
    --unlearn '["Iron_Man", "Homer_Simpson", "Batman", "Naruto_Uzumaki", "Avatar_Aang", "Hulk"]' \
    --retain '["Captain_America", "Bart_Simpson", "Superman", "Sasuke_Uchiha", "Zuko", "Thor"]'
```

## Notes

- Adjust the paths and character lists for your experiment.
- Use `--output_dir` to keep metrics separate for each run.
- Ensure all required dependencies are installed before running the scripts.
