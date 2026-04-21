#!/bin/bash
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --job-name=Sequential-Object-Base-ConAbl
#SBATCH --time=20:00:00
#SBATCH --cluster=ascend
#SBATCH --partition=nextgen
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/Sequential/Object/Base/ConAbl_%j.out
#SBATCH --error=logs/Sequential/Object/Base/ConAbl_%j.err

# Script settings
source ~/.bashrc
set -euo pipefail

echo "Sequential Object Unlearning with Base ConAbl starting..."

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
anchor_datasets_root="${REPO_ROOT}/UnlearningMethods/ConAbl/anchor_datasets/object"
anchor_prompts_root="${REPO_ROOT}/UnlearningMethods/ConAbl/anchor_prompts/object"
models_root="${OUTPUT_ROOT}/Sequential/Object/Base/ConAbl/Models"
results_root="${OUTPUT_ROOT}/Sequential/Object/Base/ConAbl/Results"
iterations=2000

array_to_json() {
    local json="["
    local item

    for item in "$@"; do
        json+="\"${item}\","
    done

    printf '%s' "${json%,}]"
}

# Active sequential object set for this script.
unlearn_objects=("Bears" "Birds" "Cats" "Dogs" "Fishes" "Frogs" "Jellyfish" "Rabbits" "Sandwiches" "Statues" "Towers" "Waterfalls")

# Held-out controls for retention evaluation.
retain_objects=("Architectures" "Butterfly" "Flame" "Flowers" "Horses" "Human" "Sea" "Trees")
retain_styles=("Blossom_Season" "Rust" "Crayon" "Fauvism" "Superstring" "Red_Blue_Ink" "Gorgeous_Love" "French" "Joy" "Greenfield" "Expressionism" "Impressionism")

# Each object uses a semantically related anchor concept for preservation data generation.
declare -A object_anchor=(
    ["Bears"]="Horses"
    ["Birds"]="Butterfly"
    ["Cats"]="Horses"
    ["Dogs"]="Horses"
    ["Fishes"]="Butterfly"
    ["Frogs"]="Butterfly"
    ["Jellyfish"]="Flowers"
    ["Rabbits"]="Horses"
    ["Sandwiches"]="Flowers"
    ["Statues"]="Trees"
    ["Towers"]="Trees"
    ["Waterfalls"]="Trees"
)
declare -A object_name_map=(
    ["Bears"]="bear"
    ["Birds"]="bird"
    ["Cats"]="cat"
    ["Dogs"]="dog"
    ["Fishes"]="fish"
    ["Frogs"]="frog"
    ["Jellyfish"]="jellyfish"
    ["Rabbits"]="rabbit"
    ["Sandwiches"]="sandwich"
    ["Statues"]="statue"
    ["Towers"]="tower"
    ["Waterfalls"]="waterfall"
)
declare -A anchor_name_map=(
    ["Butterfly"]="butterfly"
    ["Flowers"]="flower"
    ["Horses"]="horse"
    ["Trees"]="tree"
)

retain_objects_json="$(array_to_json "${retain_objects[@]}")"
retain_styles_json="$(array_to_json "${retain_styles[@]}")"

# Initialize the base checkpoint. Leave empty on the first iteration to use the base model UNet.
base_unet_ckpt=""
unlearned=()

for object in "${unlearn_objects[@]}"; do
    echo "Unlearning object: ${object}"

    anchor="${object_anchor[$object]:-}"
    if [[ -z "${anchor}" ]]; then
        echo "No anchor mapping configured for object: ${object}" >&2
        exit 1
    fi

    object_name="${object_name_map[$object]:-}"
    anchor_name="${anchor_name_map[$anchor]:-}"
    if [[ -z "${object_name}" || -z "${anchor_name}" ]]; then
        echo "Missing object/anchor name mapping for object: ${object} (anchor: ${anchor})" >&2
        exit 1
    fi

    anchor_dataset_dir="${anchor_datasets_root}/${anchor}"
    anchor_prompt_path="${anchor_prompts_root}/${anchor}.txt"
    output_dir="${models_root}/thru${object}"
    result_dir="${results_root}/thru${object}"
    sample_output_dir="${result_dir}/images"
    metrics_output_dir="${result_dir}/metrics"

    # TRAIN: Unlearn the current object using the current base checkpoint.
    cd "${train_dir}"
    train_args=(
        --anchor_target_concepts "${anchor_name}+${object_name}"
        --concept_type "object"
        --output_dir "${output_dir}"
        --base_model_dir "${base_model_dir}"
        --iterations "${iterations}"
        --num_anchor_images 200
        --num_anchor_prompts 200
        --anchor_dataset_dirs "${anchor_dataset_dir}"
        --anchor_prompt_paths "${anchor_prompt_path}"
        --scale_lr
        --hflip
        --noaug
        --enable_xformers_memory_efficient_attention
    )

    if [[ -n "${base_unet_ckpt}" ]]; then
        train_args+=(--unet_ckpt "${base_unet_ckpt}")
    fi

    accelerate launch \
        --config_file "${accelerate_config}" \
        train_conabl.py \
        "${train_args[@]}"

    # Update the base checkpoint for the next iteration.
    base_unet_ckpt="${output_dir}/delta.bin"

    # Append the current object to the cumulative unlearned set.
    unlearned+=("${object}")
    unlearned_json="$(array_to_json "${unlearned[@]}")"
    inner_unlearned=${unlearned_json:1:-1}
    objects_subset_json="[${inner_unlearned}, ${retain_objects_json:1}"

    echo "Objects Unlearned so far: ${unlearned_json}"
    echo "Objects subset for sampling: ${objects_subset_json}"

    # SAMPLE: Generate images from the current sequentially unlearned model.
    cd "${eval_dir}"
    python sample.py \
        --unet_ckpt_path "${output_dir}/delta.bin" \
        --output_dir "${sample_output_dir}" \
        --objects_subset "${objects_subset_json}" \
        --styles_subset "${retain_styles_json}" \
        --pipeline_dir "${base_model_dir}"

    # EVALUATE: Run evaluation on the generated images.
    python evaluate.py \
        --input_dir "${sample_output_dir}" \
        --output_dir "${metrics_output_dir}" \
        --eval_classifier_dir "${eval_classifier_dir}" \
        --unlearn "${unlearned_json}" \
        --retain "${retain_objects_json}" \
        --cross_retain "${retain_styles_json}"

    echo "Completed Unlearning and Sampling for Object: ${object}"
done

echo "Sequential Object Unlearning with Base ConAbl finished!"
