# CUIG Bash Scripts

This directory contains script-oriented entrypoints for CUIG unlearning, sampling, and evaluation workflows.

## One-Time Local Setup

Copy the tracked example config and fill in your local paths and SLURM settings:

```bash
cp BashScripts/config.example.sh BashScripts/config.sh
```

Edit `BashScripts/config.sh` with your own values:

```bash
export CUIG_REPO_ROOT="/path/to/CUIG"
export CUIG_OUTPUT_ROOT="/path/to/cuig_outputs"
export CUIG_SLURM_ACCOUNT="your_account"
export CUIG_SLURM_CLUSTER="your_cluster"
export CUIG_SLURM_PARTITION="your_partition"
```

`BashScripts/config.sh` is ignored by git. Do not commit credentials, private scratch paths, or tokens. If you need local secrets such as `OPENAI_API_KEY`, put them in a separate local file and set:

```bash
export CUIG_PRIVATE_EXPORTS="/path/to/private_exports.sh"
```

Details for setting up `private_exports.sh` is in the main README. 

## Running Scripts

Use `BashScripts/submit.sh` for scripts that are direct SLURM jobs with `#SBATCH` resource directives:

```bash
bash BashScripts/submit.sh BashScripts/Sequential/Object/Base/ConAbl.sh
```

Run submission/generator scripts directly with `bash`; they source `BashScripts/submit.sh` as the shared config loader, then create temporary SLURM job files and submit them with the configured account, cluster, and partition:

```bash
bash BashScripts/Independent/Object/Base/ConAbl.sh
```

Logs are written under `logs/` using paths derived from the script location or experiment folder.

## Directory Layout

Layer 1:
- `Independent`: unlearn one concept at a time for non-continual baselines
- `Sequential`: unlearn concepts sequentially from the previously unlearned model
- `Simultaneous`: unlearn cumulative concept sets from the base model

Layer 2:
- `Object`: UnlearnCanvas object experiments
- `Style`: UnlearnCanvas style experiments
- `Celebrity`: celebrity-domain experiments

Layer 3:
- `Base`: Unlearn without an additional continual-unlearning regularizer
- `Projection`, `SelFT`, `Weight`, `Hybrid`, `Merge`: regularized or merged variants

Layer 4:
- `ConAbl.sh`: ConAbl Unlearning Method

## Configuration Files

- `config.example.sh`: tracked template with placeholder values
- `config.sh`: ignored local configuration filled in by each user
- `submit.sh`: shared config loader when sourced, and helper for direct SLURM scripts when executed
