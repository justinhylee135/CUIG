# SculpMem

This directory contains CUIG's SculpMem training implementation. SculpMem is a
minimal extension of ConAbl: it keeps the same anchor-based unlearning objective,
current CUIG regularizer integration, checkpoint format, and experiment-script
interfaces, while adding optional dynamic SelFT-style attention masking during
training.

SculpMem Reference: [coulsonlee/Sculpting-Memory-ICCV-2025](https://github.com/coulsonlee/Sculpting-Memory-ICCV-2025/tree/main)

This implementation started as a direct copy of CUIG's ConAbl code. I then
implemented the key ideas from SculpMem using the Sculpting Memory repository as
the reference, while keeping the code compatible with my current conda
environment, CUIG evaluation workflow, and existing CUIG regularizers.

## Layout

```text
SculpMem/
|-- train_sculpmem.py    # Main training entrypoint
|-- src/
|   |-- args.py          # CLI arguments
|   |-- data.py          # Anchor prompt/image dataset setup
|   |-- dynamic_masking.py # SculpMem dynamic attention mask hooks
|   |-- model.py         # Diffusion model loading and trainable parameter setup
|   `-- utils.py         # Training setup, logging, regularizer wiring, checkpointing
|-- anchor_prompts/      # Default anchor prompt files
`-- anchor_datasets/     # Generated anchor image datasets
```

## When to Use It

Use SculpMem when you want ConAbl's target-to-anchor unlearning behavior with a
moving subset of trainable attention weights. The dynamic mask can be used alone
or together with CUIG regularizers such as Weight, Projection, and
Simultaneous early stopping.

Most full experiments should be launched through [`../../BashScripts`](../../BashScripts/README.md),
because those scripts handle concept ordering, anchors, output paths, sampling,
and evaluation.

## Direct Training

A minimal direct run looks like:

```bash
cd "$REPO_ROOT/UnlearningMethods/SculpMem"

accelerate launch \
  --config_file "$REPO_ROOT/Configs/Accelerator/single_gpu.yaml" \
  train_sculpmem.py \
  --base_model_dir "$BASE_MODEL_DIR" \
  --concept_type "object" \
  --anchor_target_concepts "horse+bear" \
  --output_dir "$OUTPUT_ROOT/example/Models/Bears" \
  --anchor_dataset_dirs "$REPO_ROOT/UnlearningMethods/SculpMem/anchor_datasets/object/Horses" \
  --anchor_prompt_paths "$REPO_ROOT/UnlearningMethods/SculpMem/anchor_prompts/object/Horses.txt" \
  --iterations 2000 \
  --num_anchor_images 200 \
  --num_anchor_prompts 200 \
  --scale_lr \
  --hflip \
  --noaug \
  --enable_xformers_memory_efficient_attention
```

The trained checkpoint is written to:

```text
<output_dir>/delta.bin
```

Sequential runs pass the previous step's checkpoint back into training:

```bash
--unet_ckpt "$OUTPUT_ROOT/example/Models/previous_target/delta.bin"
```

## Dynamic Masking

Enable SculpMem's dynamic attention mask with:

```bash
--selft_dynamic \
--selft_topk 0.10 \
--selft_dynamic_warmup_steps 50 \
--selft_dynamic_mask_update_interval 100 \
--selft_dynamic_initial_turnover_fraction 0.20
```

During warmup, gradient hooks accumulate importance scores on trainable attention
projections. After warmup, SculpMem initializes a binary mask that keeps the top
`--selft_topk` fraction active. Later updates rotate a fraction of active weights
out and inactive high-gradient weights in.

For `xattn` and `kv-xattn`, hooks are restricted to cross-attention modules. For
`full`, hooks can cover trainable self-attention and cross-attention projections.

## Concept Inputs

SculpMem supports the same concept types as ConAbl:

- `style`
- `object`
- `celeb`
- `character`
- `nudity`
- `inappropriate_content`

The main target-to-anchor mapping is provided with:

```bash
--anchor_target_concepts "anchor+target"
```

For multi-concept or more structured runs, use:

```bash
--concept_configs path/to/concepts.json
```

## Regularizers

SculpMem can be combined with regularizers from [`../../Regularizers`](../../Regularizers/README.md).

Weight regularization:

```bash
--l1sp_weight 50
--l2sp_weight 25000
```

Gradient projection:

```bash
--with_gradient_projection \
--auxiliary_prompts_path path/to/prompts.json \
--gradient_projection_num_prompts 200
```

Static SelFT:

```bash
--with_selft \
--selft_topk 0.10 \
--selft_loss ConAbl
```

Simultaneous early stopping/evaluation:

```bash
--eval_interval 200 \
--patience 1000 \
--stop_threshold 99 \
--eval_classifier_dir path/to/classifier
```

## Outputs

Common outputs under `--output_dir` include:

- `delta.bin`: final unlearned checkpoint
- `delta-<step>`: intermediate checkpoint exports when checkpointing is enabled
- `checkpoint-<step>/`: Accelerate training-state checkpoints
- `mask_dict.pt` and `grad_dict.pt`: default static SelFT cache files
- `logs/`: training logs and tracker outputs

## Related Docs

- [`../ConAbl/README.md`](../ConAbl/README.md): baseline method SculpMem extends
- [`../../BashScripts/README.md`](../../BashScripts/README.md): script-based experiment launchers
- [`../../Regularizers/README.md`](../../Regularizers/README.md): reusable regularizer implementations
- [`../../Evaluation/README.md`](../../Evaluation/README.md): benchmark sampling and evaluation
