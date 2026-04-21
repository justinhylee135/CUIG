#!/bin/bash
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --job-name=Sequential-Style-Weight-L2SP-25000-ConAbl
#SBATCH --time=15:00:00
#SBATCH --cluster=ascend
#SBATCH --partition=nextgen
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/Sequential/Style/Weight/L2SP/25000/ConAbl_%j.out
#SBATCH --error=logs/Sequential/Style/Weight/L2SP/25000/ConAbl_%j.err

# Script settings
source ~/.bashrc
set -euo pipefail

echo "Sequential Style Unlearning with Weight L2SP 25000 ConAbl starting..."

# User must set
REPO_ROOT="${REPO_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs}"

# Run (your own) script that activates the conda env and sets env variables
source "${REPO_ROOT}/private_exports.sh"

# Unlearning Method Agnostic
base_model_dir="${REPO_ROOT}/Checkpoints/Generators/UnlearnCanvas"
eval_classifier_dir="${REPO_ROOT}/Checkpoints/Classifiers/UnlearnCanvas"

# Select Unlearning Method and Evaluation Method
train_dir="${REPO_ROOT}/UnlearningMethods/ConAbl"
eval_dir="${REPO_ROOT}/Evaluation/UnlearnCanvas"

# ConAbl Specific
accelerate_config="${REPO_ROOT}/Configs/Accelerator/single_gpu.yaml"
anchor_dataset_dir="${REPO_ROOT}/UnlearningMethods/ConAbl/anchor_datasets/style/Laion"
anchor_prompt_path="${REPO_ROOT}/UnlearningMethods/ConAbl/anchor_prompts/style/Laion.txt"
models_root="${OUTPUT_ROOT}/Sequential/Style/Weight/L2SP/25000/ConAbl/Models"
results_root="${OUTPUT_ROOT}/Sequential/Style/Weight/L2SP/25000/ConAbl/Results"

array_to_json() {
    local json="["
    local item

    for item in "$@"; do
        json+="\"${item}\","
    done

    printf '%s' "${json%,}]"
}

# Define the list of styles to unlearn sequentially
unlearn_styles=("Abstractionism" "Byzantine" "Cartoon" "Cold_Warm" "Ukiyoe" "Van_Gogh" "Neon_Lines" "Picasso" "On_Fire" "Magic_Cube" "Winter" "Vibrant_Flow")

# Define held-out styles and objects to measure retention performance
retain_styles=("Blossom_Season" "Rust" "Crayon" "Fauvism" "Superstring" "Red_Blue_Ink" "Gorgeous_Love" "French" "Joy" "Greenfield" "Expressionism" "Impressionism")
retain_objects=("Architectures" "Butterfly" "Flame" "Flowers" "Horses" "Human" "Sea" "Trees")

# Initialize the base checkpoint. Leave empty on the first iteration to use the base model UNet.
base_unet_ckpt=""
retain_styles_json="$(array_to_json "${retain_styles[@]}")"
retain_objects_json="$(array_to_json "${retain_objects[@]}")"

# Keep track of the unlearned styles
unlearned=()  

# Iterate through each style to unlearn
for style in "${unlearn_styles[@]}"; do
    echo "Unlearning style: ${style}"
    
    # Define the new checkpoint path for the resulting UNet after unlearning style
    output_dir="${models_root}/thru${style}"
    result_dir="${results_root}/thru${style}"
    sample_output_dir="${result_dir}/images"
    metrics_output_dir="${result_dir}/metrics"
    
    # TRAIN: Unlearn the current style using the current base checkpoint
    cd "${train_dir}"
    train_args=(
        --anchor_target_concepts "${style}"
        --concept_type "style"
        --output_dir "${output_dir}"
        --base_model_dir "${base_model_dir}"
        --iterations 1000
        --num_anchor_images 200
        --num_anchor_prompts 200
        --anchor_dataset_dirs "${anchor_dataset_dir}"
        --anchor_prompt_paths "${anchor_prompt_path}"
        --scale_lr
        --hflip
        --noaug
        --enable_xformers_memory_efficient_attention
        --l2sp_weight 25000
    )

    if [[ -n "${base_unet_ckpt}" ]]; then
        train_args+=(--unet_ckpt "${base_unet_ckpt}")
    fi

    accelerate launch \
        --config_file "${accelerate_config}" \
        train_conabl.py \
        "${train_args[@]}"

    # Update the base checkpoint for the next iteration
    base_unet_ckpt="${output_dir}/delta.bin"

    # Append the current style to the list of unlearned styles
    unlearned+=("${style}")
    unlearned_json="$(array_to_json "${unlearned[@]}")"
    
    # Build the styles subset JSON for sampling.
    inner_unlearned=${unlearned_json:1:-1}
    styles_subset_json="[${inner_unlearned}, ${retain_styles_json:1}"
    
    echo "Styles Unlearned so far: ${unlearned_json}"
    echo "Styles subset for sampling: ${styles_subset_json}"
    
    # SAMPLE: Generate images from the current unlearned model
    cd "${eval_dir}"
    python sample.py \
        --unet_ckpt_path "${output_dir}/delta.bin" \
        --output_dir "${sample_output_dir}" \
        --styles_subset "${styles_subset_json}" \
        --objects_subset "${retain_objects_json}" \
        --pipeline_dir "${base_model_dir}"
    
    # EVALUATE: Run evaluation on the generated images.
    cd "${eval_dir}"
    python evaluate.py \
        --input_dir "${sample_output_dir}" \
        --output_dir "${metrics_output_dir}" \
        --eval_classifier_dir "${eval_classifier_dir}" \
        --unlearn "${unlearned_json}" \
        --retain "${retain_styles_json}" \
        --cross_retain "${retain_objects_json}"
    
    echo "Completed Unlearning and Sampling for Style: ${style}"
done

echo "Sequential Style Unlearning with Weight L2SP 25000 ConAbl finished!"
