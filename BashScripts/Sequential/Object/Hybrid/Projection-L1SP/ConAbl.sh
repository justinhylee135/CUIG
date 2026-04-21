#!/bin/bash
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --job-name=Sequential-Object-Hybrid-Projection_L1SP-200_0.75-ConAbl
#SBATCH --time=20:00:00
#SBATCH --cluster=ascend
#SBATCH --partition=nextgen
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/Sequential/Object/Hybrid/Projection_L1SP/200_0.75/ConAbl_%j.out
#SBATCH --error=logs/Sequential/Object/Hybrid/Projection_L1SP/200_0.75/ConAbl_%j.err

# Script settings
source ~/.bashrc
set -euo pipefail

echo "Sequential Object Unlearning with Hybrid Projection_L1SP 200_0.75 ConAbl starting..."

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
models_root="${OUTPUT_ROOT}/Sequential/Object/Hybrid/Projection_L1SP/200_0.75/ConAbl/Models"
results_root="${OUTPUT_ROOT}/Sequential/Object/Hybrid/Projection_L1SP/200_0.75/ConAbl/Results"
iterations=2000

array_to_json() {
    local json="["
    local item

    for item in "$@"; do
        json+="\"${item}\","
    done

    printf '%s' "${json%,}]"
}

# Define the list of objects to unlearn sequentially
unlearn_objects=("Bears" "Birds" "Cats" "Dogs" "Fishes" "Frogs" "Jellyfish" "Rabbits" "Sandwiches" "Statues" "Towers" "Waterfalls")

# Define held-out objects and styles to measure retention performance
retain_objects=("Architectures" "Butterfly" "Flame" "Flowers" "Horses" "Human" "Sea" "Trees")
retain_styles=("Blossom_Season" "Rust" "Crayon" "Fauvism" "Superstring" "Red_Blue_Ink" "Gorgeous_Love" "French" "Joy" "Greenfield" "Expressionism" "Impressionism")

# Map each target object to the anchor concept and singular text form used by ConAbl.
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

# Initialize the base checkpoint. Leave empty on the first iteration to use the base model UNet.
base_unet_ckpt=""
retain_objects_json="$(array_to_json "${retain_objects[@]}")"
retain_styles_json="$(array_to_json "${retain_styles[@]}")"

# Keep track of the unlearned objects
unlearned=()

# Iterate through each object to unlearn
for object in "${unlearn_objects[@]}"; do
    echo "Unlearning object: ${object}"

    anchor="${object_anchor[$object]:-}"
    object_name="${object_name_map[$object]:-}"
    anchor_name="${anchor_name_map[$anchor]:-}"
    if [[ -z "${anchor}" || -z "${object_name}" || -z "${anchor_name}" ]]; then
        echo "Missing mapping for object: ${object}" >&2
        exit 1
    fi

    # Define the new checkpoint path for the resulting UNet after unlearning object
    anchor_dataset_dir="${anchor_datasets_root}/${anchor}"
    anchor_prompt_path="${anchor_prompts_root}/${anchor}.txt"
    output_dir="${models_root}/thru${object}"
    result_dir="${results_root}/thru${object}"
    sample_output_dir="${result_dir}/images"
    metrics_output_dir="${result_dir}/metrics"

    # TRAIN: Unlearn the current object using the current base checkpoint
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
        --with_gradient_projection
        --auxiliary_prompts_path "${output_dir}/prompts.txt"
        --previously_unlearned "$(printf '%s\n' "${unlearned[@]}" | jq -R . | jq -s .)" \
        --gradient_projection_num_prompts 200
        --l1sp_weight 0.75
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

    # Append the current object to the list of unlearned objects
    unlearned+=("${object}")
    unlearned_json="$(array_to_json "${unlearned[@]}")"
    inner_unlearned=${unlearned_json:1:-1}
    objects_subset_json="[${inner_unlearned}, ${retain_objects_json:1}"

    echo "Objects Unlearned so far: ${unlearned_json}"
    echo "Objects subset for sampling: ${objects_subset_json}"

    # SAMPLE: Generate images from the current unlearned model
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

echo "Sequential Object Unlearning with Hybrid Projection_L1SP 200_0.75 ConAbl finished!"
