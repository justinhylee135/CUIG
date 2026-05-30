# Configs

This directory stores reusable configuration files for running CUIG experiments.

## Layout

```text
Configs/
`-- Accelerator/
    `-- single_gpu.yaml
```

## Accelerate Configs

`Accelerator/single_gpu.yaml` is the default Hugging Face Accelerate config used by the training scripts. It runs one process on one GPU with no distributed training:

```bash
accelerate launch \
    --config_file "$REPO_ROOT/Configs/Accelerator/single_gpu.yaml" \
    train_conabl.py \
    ...
```

Most scripts set:

```bash
accelerate_config="${REPO_ROOT}/Configs/Accelerator/single_gpu.yaml"
```

## Adding New Configs

Add new Accelerate configs under `Configs/Accelerator/` when you need a different runtime setup, such as multi-GPU, CPU debugging, mixed precision, or a cluster-specific launch configuration.

For example:

```text
Configs/Accelerator/
|-- single_gpu.yaml
|-- multi_gpu.yaml
`-- cpu_debug.yaml
```

Then point a script to the new config:

```bash
accelerate_config="${REPO_ROOT}/Configs/Accelerator/multi_gpu.yaml"
```

## Notes

- Keep machine-specific secrets and private paths out of tracked config files.
- Use `BashScripts/config.sh` for local paths and SLURM settings.
- Prefer adding reusable runtime configs here instead of copying long `accelerate launch` options into every script.
