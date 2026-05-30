#!/bin/bash

# Copy this file to BashScripts/config.sh and fill in local values.
# BashScripts/config.sh is intentionally gitignored.

export CUIG_REPO_ROOT="/path/to/CUIG"
export CUIG_OUTPUT_ROOT="/path/to/cuig_outputs"

# SLURM settings for your cluster.
export CUIG_SLURM_ACCOUNT="your_account"
export CUIG_SLURM_CLUSTER="your_cluster"
export CUIG_SLURM_PARTITION="your_partition"

# Optional: local script that exports credentials or activates site-specific setup.
# Leave empty if you do not need one. Never commit real credentials.
export CUIG_PRIVATE_EXPORTS=""

# Optional model/checkpoint overrides.
export CUIG_UNLEARNCANVAS_GENERATOR_DIR="${CUIG_REPO_ROOT}/Checkpoints/Generators/UnlearnCanvas"
export CUIG_UNLEARNCANVAS_CLASSIFIER_DIR="${CUIG_REPO_ROOT}/Checkpoints/Classifiers/UnlearnCanvas"
export CUIG_CELEBRITY_BASE_MODEL_DIR="CompVis/stable-diffusion-v1-4"
