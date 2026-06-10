# ESD-SDXL

This directory contains CUIG's SDXL implementation of ESD (Erasing Stable Diffusion). It keeps the SDXL-specific ESD training objective from the copied implementation, while using CUIG's current method layout, checkpoint convention, and reusable regularizers.

ESD reference: [rohitgandikota/erasing](https://github.com/rohitgandikota/erasing)

## Layout

```text
ESD-SDXL/
|-- train_esd_sdxl.py   # Main training entrypoint
`-- src/
    |-- args.py         # CLI arguments
    |-- model.py        # SDXL model loading, checkpoint loading, trainable parameter selection
    |-- sdxl_pipeline.py # SDXL pipeline call override for partial denoising
    |-- training.py     # Conditioning cache and ESD-SDXL training-step helpers
    `-- utils.py        # Output setup, prompt handling, regularizer wiring, summaries
```

## Direct Training

```bash
cd "$REPO_ROOT/UnlearningMethods/ESD-SDXL"

python train_esd_sdxl.py \
  --base_model_dir "stabilityai/stable-diffusion-xl-base-1.0" \
  --concept "Megan Fox" \
  --concept_type "celeb" \
  --train_method "esd-x-strict" \
  --output_dir "$OUTPUT_ROOT/esd_sdxl/Models/Megan_Fox" \
  --iterations 200 
```

The trained ESD parameter dictionary is written to:

```text
<output_dir>/delta.bin
```

Alternatively `--save_path` is accepted. If `--save_path` points to a file, that exact file is used. If it points to a directory, the final checkpoint is saved as `delta.bin` in that directory. When both `--output_dir` and `--save_path` are supplied, `--output_dir` takes precedence.

## Concept Inputs

Supported concept types:

- `style`
- `object`
- `celeb`
- `celebrity` as an alias for `celeb`

`--concept` can be a single concept, a Python-style list string, or a text file of complete prompts. For generated concept prompts, ESD-SDXL uses:

- style: `<concept> Style`
- object: `An image of <concept>`
- celeb: `<concept>`

Use `--erase_from` or `--erase_from_prompts` when the ESD target should be computed from a different sampling prompt than the concept prompt.

## Trainable Parameters

Use `--train_method` to select the ESD parameter subset:

- `esd-x`: cross-attention Linear/Conv parameters
- `esd-u`: non-cross-attention Linear/Conv parameters
- `esd-all`: all Linear/Conv parameters
- `esd-x-strict`: cross-attention key/value parameters

## Sequential Runs

Sequential or continual runs load the previous checkpoint into both the frozen teacher UNet and the trainable ESD UNet:

```bash
--unet_ckpt "$OUTPUT_ROOT/esd_sdxl/Models/previous/delta.bin"
```

## Regularizers

ESD-SDXL uses regularizers from [`../../Regularizers`](../../Regularizers/README.md).

Weight regularization:

```bash
--l1sp_weight 50
--l2sp_weight 25000
```

SelFT:

```bash
--with_selft \
--selft_loss ESD \
--selft_topk 0.01
```

Gradient projection:

```bash
--with_gradient_projection \
--auxiliary_prompts_path path/to/auxiliary_prompts.txt \
--gradient_projection_num_prompts 200
```

For SDXL, the projection setup encodes auxiliary prompts through the SDXL pipeline so the projector matches the UNet conditioning dimension.

Simultaneous early stopping/evaluation:

```bash
--eval_interval 10 \
--eval_classifier_dir path/to/classifier \
--patience 200 \
--stop_threshold 99
```

## Outputs

Common outputs under `--output_dir` include:

- `delta.bin`: final ESD trainable-parameter state dictionary
- `mask_dict.pt`: default SelFT mask cache when SelFT is enabled
- `grad_dict.pt`: default SelFT gradient cache when SelFT is enabled
- `logs/`: simultaneous evaluation logs when enabled by the regularizer

If the resolved final checkpoint path ends with `.safetensors`, the checkpoint is saved with `safetensors`; otherwise it is saved with `torch.save`.
