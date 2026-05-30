# ConAbl

This directory contains CUIG's ConAbl training implementation. It is based on the Concept Ablation codebase, with CUIG-specific changes for continual unlearning workflows, regularizer integration, benchmark evaluation hooks, and script-oriented experiment pipelines.

Concept Ablation Reference: [nupurkmr9/concept-ablation](https://github.com/nupurkmr9/concept-ablation)

## Layout

```text
ConAbl/
|-- train_conabl.py      # Main training entrypoint
|-- src/
|   |-- args.py          # CLI arguments
|   |-- data.py          # Anchor prompt/image dataset setup
|   |-- model.py         # Diffusion model loading and trainable parameter setup
|   `-- utils.py         # Training setup, logging, regularizer wiring, checkpointing
|-- anchor_prompts/      # Default anchor prompt files
`-- anchor_datasets/     # Generated anchor image datasets
```

## When to Use It

ConAbl is used throughout CUIG for:

- independent unlearning: train one checkpoint per target concept
- sequential unlearning: continue from the previous unlearned checkpoint with `--unet_ckpt`
- simultaneous unlearning: train on a multi-concept configuration and optionally evaluate during training
- regularized unlearning: combine ConAbl with Weight, Projection, SelFT, or hybrid regularizers

Most full experiments should be launched through [`../../BashScripts`](../../BashScripts/README.md), because those scripts handle concept ordering, anchors, output paths, sampling, and evaluation.

## Direct Training

A minimal direct run looks like:

```bash
cd "$REPO_ROOT/UnlearningMethods/ConAbl"

accelerate launch \
  --config_file "$REPO_ROOT/Configs/Accelerator/single_gpu.yaml" \
  train_conabl.py \
  --base_model_dir "$BASE_MODEL_DIR" \
  --concept_type "object" \
  --anchor_target_concepts "horse+bear" \
  --output_dir "$OUTPUT_ROOT/example/Models/Bears" \
  --anchor_dataset_dirs "$REPO_ROOT/UnlearningMethods/ConAbl/anchor_datasets/object/Horses" \
  --anchor_prompt_paths "$REPO_ROOT/UnlearningMethods/ConAbl/anchor_prompts/object/Horses.txt" \
  --iterations 2000 \
  --num_anchor_images 200 \
  --num_anchor_prompts 200 \
  --scale_lr \
  --hflip \
  --noaug \
  --enable_xformers_memory_efficient_attention
```

The trained UNet checkpoint is written to:

```text
<output_dir>/delta.bin
```

Sequential runs pass the previous step's checkpoint back into training:

```bash
--unet_ckpt "$OUTPUT_ROOT/example/Models/previous_target/delta.bin"
```

## Concept Inputs

ConAbl supports the following concept types:

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

`--concept_configs` can override per-concept inputs such as anchor dataset directories and anchor prompt files.

## Anchor Prompts and Datasets

ConAbl uses anchor prompts and anchor images to build the training signal. Default prompt files are under:

```text
anchor_prompts/
```

Generated anchor images are stored under:

```text
anchor_datasets/
```

If an anchor dataset does not contain enough images, `train_conabl.py` generates the missing images using `--anchor_prompt_paths` and `--num_anchor_images`. Anchor dataset generation uses a lock so concurrent jobs targeting the same anchor directory do not both generate the same dataset; one job generates while the other waits and then reuses the completed images.

## Trainable Parameters

Use `--parameter_group` to choose which parameters are updated:

- `kv-xattn`: key/value cross-attention parameters
- `xattn`: cross-attention parameters
- `full`: full UNet
- `text-emb`: text embeddings

The default is `kv-xattn`, which is what most experiment scripts use unless they explicitly override it.

## Regularizers

ConAbl can be combined with regularizers from [`../../Regularizers`](../../Regularizers/README.md).

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

SelFT:

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

See [`../../Regularizers/README.md`](../../Regularizers/README.md) for implementation details and merge utilities.

## Outputs

Common outputs under `--output_dir` include:

- `delta.bin`: final unlearned UNet state dictionary
- `delta-<step>`: intermediate checkpoint exports when checkpointing is enabled
- `checkpoint-<step>/`: Accelerate training-state checkpoints when `--turn_on_checkpointing` is used
- `mask_dict.pt` and `grad_dict.pt`: default SelFT cache files when SelFT is enabled
- `logs/`: training logs and tracker outputs

Experiment scripts usually wrap this in a larger output layout such as:

```text
$OUTPUT_ROOT/.../Models/<target>/delta.bin
$OUTPUT_ROOT/.../Results/<target>/images/
$OUTPUT_ROOT/.../Results/<target>/metrics/
```

## Related Docs

- [`../../BashScripts/README.md`](../../BashScripts/README.md): script-based experiment launchers
- [`../../Regularizers/README.md`](../../Regularizers/README.md): reusable regularizer implementations
- [`../../Evaluation/README.md`](../../Evaluation/README.md): benchmark sampling and evaluation
- [`../../Configs/README.md`](../../Configs/README.md): Accelerate configuration files
