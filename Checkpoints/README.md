# Checkpoints

This directory stores local model assets used by CUIG experiments. Checkpoint files are intentionally not tracked by git; download them locally before running the UnlearnCanvas object/style workflows.

## Expected Layout

```text
Checkpoints/
|-- Classifiers/
|   `-- UnlearnCanvas/
|       |-- object_classifier.pth
|       `-- style_classifier.pth
`-- Generators/
    `-- UnlearnCanvas/
        |-- model_index.json
        |-- scheduler/
        |-- text_encoder/
        |-- tokenizer/
        |-- unet/
        `-- vae/
```

The UnlearnCanvas generator is used as the base Diffusers pipeline for object/style experiments. The UnlearnCanvas classifiers are used for object/style evaluation.

You can extend this layout by adding other image generation model checkpoints under `Generators/` and other image classifiers for evaluation under `Classifiers/`.

## Download UnlearnCanvas Classifiers

From the repository root:

```bash
cd "$REPO_ROOT/Checkpoints"
mkdir -p Classifiers
cd Classifiers

gdown --folder https://drive.google.com/drive/folders/1AoazlvDgWgc3bAyHDpqlafqltmn4vm61
mv cls_model UnlearnCanvas
cd UnlearnCanvas

rm -f style60.pth
mv style50.pth style_classifier.pth
mv style50_cls.pth object_classifier.pth
```

After setup, the classifier directory should be:

```text
Checkpoints/Classifiers/UnlearnCanvas/
|-- object_classifier.pth
`-- style_classifier.pth
```

## Download UnlearnCanvas Generator

From the repository root:

```bash
cd "$REPO_ROOT/Checkpoints"
mkdir -p Generators
cd Generators

gdown --folder https://drive.google.com/drive/folders/18x40pLBcfNFyxBWZBGncTjqJTs_75SLx
mv style50 UnlearnCanvas
```

After setup, the generator should live at:

```text
Checkpoints/Generators/UnlearnCanvas/
```

## Configure Scripts

If you use `BashScripts/`, these paths are the defaults:

```bash
export CUIG_UNLEARNCANVAS_GENERATOR_DIR="${CUIG_REPO_ROOT}/Checkpoints/Generators/UnlearnCanvas"
export CUIG_UNLEARNCANVAS_CLASSIFIER_DIR="${CUIG_REPO_ROOT}/Checkpoints/Classifiers/UnlearnCanvas"
```

Set them in `BashScripts/config.sh` only if your checkpoints live somewhere else.

## Notes

- Install `gdown` if it is not available: `pip install gdown`.
- Keep downloaded checkpoints out of git.
- Celebrity experiments use their own evaluation setup under `Evaluation/Celebrity` and default to `CompVis/stable-diffusion-v1-4` unless `CUIG_CELEBRITY_BASE_MODEL_DIR` is overridden.
