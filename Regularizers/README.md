# Regularizers

This directory contains reusable regularization utilities for CUIG unlearning methods. The goal is to keep the regularizers independent from any single method, so training code under `UnlearningMethods/` can mix and match them across independent, sequential, and simultaneous unlearning settings.

The current training integration is in `UnlearningMethods/ConAbl/`.

## Layout

```text
Regularizers/
|-- Weight/          # L1-SP and L2-SP weight-space regularization
|-- Projection/      # Gradient projection using auxiliary preservation prompts
|-- SelFT/           # Selective finetuning masks
|-- Merge/           # Post-hoc checkpoint merging utilities
|-- Simultaneous/    # Early stopping and lightweight evaluation helpers
`-- README.md
```

## Weight Regularizers

`Weight/` contains L1-SP and L2-SP penalties that keep trainable parameters close to their starting values.

Main files:

- `Weight/l1sp.py`: `calculate_l1sp_loss`
- `Weight/l2sp.py`: `calculate_l2sp_loss`

ConAbl enables these through:

```bash
--l1sp_weight 50
--l2sp_weight 25000
```

Set the weight to `0.0` to disable the corresponding penalty.

## Gradient Projection

`Projection/` contains utilities for projecting gradients away from directions associated with auxiliary preservation prompts. This is used to reduce forgetting of anchor or retained concepts during unlearning.

Main files:

- `Projection/projection.py`: public projection utilities
- `Projection/src/prompt_generation.py`: auxiliary prompt generation
- `Projection/src/auxiliary_embeddings.py`: auxiliary embedding construction
- `Projection/src/gradient_projection.py`: projector construction and gradient projection

Unlearning Methods enable this through:

```bash
--with_gradient_projection \
--auxiliary_prompts_path path/to/prompts.txt \
--gradient_projection_num_prompts 200
```

If auxiliary prompts are generated with an LLM-backed helper, configure any required credentials outside the repo through your local environment or private exports file.

## SelFT

`SelFT/` contains selective finetuning utilities. It scores trainable parameters, builds a global top-k mask, and applies gradient hooks so only selected parameters receive updates.

Main files:

- `SelFT/selft.py`: mask loading/building and gradient hook registration
- `SelFT/src/scoring.py`: parameter-importance scoring
- `SelFT/src/helpers.py`: top-k mask construction

Unlearning Methods enable this through:

```bash
--with_selft \
--selft_topk 0.10 \
--selft_loss ConAbl
```

Optional cache paths:

```bash
--selft_mask_dict_path path/to/mask_dict.pt
--selft_grad_dict_path path/to/grad_dict.pt
```

If these paths are not provided, ConAbl defaults to writing the mask and gradient dictionaries under the run output directory.

## Merge

`Merge/` contains post-hoc checkpoint merging methods for combining multiple independently trained unlearning checkpoints.

Main files:

- `Merge/merge.py`: CLI entrypoint
- `Merge/src/uniform.py`: uniform averaging
- `Merge/src/task_arithmetic.py`: task arithmetic
- `Merge/src/ties.py`: TIES merging
- `Merge/src/utils.py`: shared merge helpers

Example:

```bash
cd "$REPO_ROOT/Regularizers/Merge"

python merge.py \
  --base_model_dir "$BASE_MODEL_DIR" \
  --ckpt_paths '["/path/to/ckpt_1.pt", "/path/to/ckpt_2.pt"]' \
  --save_path "$OUTPUT_ROOT/merged/Models/merged.pt" \
  --merge_method ties \
  --ties_lambda 1.75 \
  --ties_topk 0.20 \
  --key_filter '["attn2"]'
```

`--merge_method` supports `uniform`, `task_arithmetic`, and `ties`.

## Simultaneous Helpers

`Simultaneous/` contains utilities for evaluation-driven simultaneous unlearning runs.

Main files:

- `Simultaneous/simultaneous.py`: dispatches benchmark-specific sampling/evaluation and checks early stopping
- `Simultaneous/src/unlearncanvas.py`: UnlearnCanvas sampling/evaluation helper
- `Simultaneous/src/character.py`: character sampling/evaluation helper
- `Simultaneous/src/celebrity.py`: celebrity sampling helper

Unlearning Methods use these when an experiment sets:

```bash
--eval_interval 200 \
--patience 1000 \
--stop_threshold 99 \
--eval_classifier_dir path/to/classifier
```

Note: Celebrity simultaneous runs currently sample evaluation images during training, while full celebrity identity evaluation is run separately because it uses a separate evaluation environment.

## Adding a Regularizer

Add reusable regularizers here rather than inside a specific method directory when the implementation can be shared across unlearning methods.

Suggested pattern:

- create a new subdirectory under `Regularizers/`
- expose a small public file, similar to `Projection/projection.py`
- keep implementation details in `src/`
- add training flags in the method-specific argument parser
- wire setup logic into the method-specific utilities, such as `UnlearningMethods/ConAbl/src/utils.py`
- keep large generated artifacts, masks, checkpoints, and prompt caches under the run output directory
