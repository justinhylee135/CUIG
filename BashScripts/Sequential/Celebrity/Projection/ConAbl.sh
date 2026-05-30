#!/bin/bash
#SBATCH --job-name=Sequential-Celebrity-Projection-200-ConAbl
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1

# Script Settings
module --ignore_cache load cuda/12.4.1
source ~/.bashrc
set -euo pipefail

echo "Sequential Celebrity Unlearning with Projection 200 ConAbl starting..."

# Shared configuration
_cuig_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "${_cuig_script_dir}/BashScripts/submit.sh" && "${_cuig_script_dir}" != "/" ]]; do
    _cuig_script_dir="$(dirname "${_cuig_script_dir}")"
done
source "${_cuig_script_dir}/BashScripts/submit.sh"
unset _cuig_script_dir

# Run (your own) script that activates the conda env and sets env variables
if [[ -n "${CUIG_PRIVATE_EXPORTS:-}" && -f "${CUIG_PRIVATE_EXPORTS}" ]]; then
    source "${CUIG_PRIVATE_EXPORTS}"
fi

# Unlearning Method Agnostic
base_model_dir="${CUIG_CELEBRITY_BASE_MODEL_DIR}"

# Select Unlearning Method and Evaluation Method
train_dir="${REPO_ROOT}/UnlearningMethods/ConAbl"
eval_dir="${REPO_ROOT}/Evaluation/Celebrity"

# ConAbl Specific
accelerate_config="${REPO_ROOT}/Configs/Accelerator/single_gpu.yaml"
anchor_datasets_root="${REPO_ROOT}/UnlearningMethods/ConAbl/anchor_datasets/celebrity"
anchor_prompts_root="${REPO_ROOT}/UnlearningMethods/ConAbl/anchor_prompts/celebrity"
models_root="${OUTPUT_ROOT}/Sequential/Celebrity/Projection/200/ConAbl/Models"
results_root="${OUTPUT_ROOT}/Sequential/Celebrity/Projection/200/ConAbl/Results"
prompt_dir="${eval_dir}/Prompts"
prompt_locks_dir="${prompt_dir}/.locks"
iterations=2000
num_prompts=50
num_seeds=1

# Set to false to skip COCO retention checks after each sequential checkpoint.
submit_coco=false
coco_num_prompts=5000

array_to_json() {
    local json="["
    local item

    for item in "$@"; do
        json+="\"${item}\","
    done

    printf '%s' "${json%,}]"
}

# Active sequential celebrity set for this script.
unlearn_celebrities=("Neil_Degrasse_Tyson" "Benicio_Del_Toro" "Aziz_Ansari" "Oprah_Winfrey" "Betty_White" "Megan_Fox")

# Held-out celebrities to measure retention performance.
retain_celebrities=("Morgan_Freeman" "Keanu_Reeves" "George_Takei" "Aretha_Franklin" "Maya_Angelou" "Natalie_Portman")

retain_celebrities_json="$(array_to_json "${retain_celebrities[@]}")"

mkdir -p "${REPO_ROOT}/logs/Sequential/Celebrity/Projection/200/ConAbl" "${prompt_dir}" "${prompt_locks_dir}"

# Initialize the base checkpoint. Leave empty on the first iteration to use the base model UNet.
base_unet_ckpt=""
unlearned=()

for celeb in "${unlearn_celebrities[@]}"; do
    echo "Unlearning celebrity: ${celeb}"

    anchor="man"
    if [[ "${celeb}" == "Oprah_Winfrey" || "${celeb}" == "Betty_White" || "${celeb}" == "Megan_Fox" ]]; then
        anchor="woman"
    fi

    celeb_name="${celeb//_/ }"
    anchor_dataset_dir="${anchor_datasets_root}/${anchor}"
    anchor_prompt_path="${anchor_prompts_root}/${anchor}_prompts.txt"
    output_dir="${models_root}/thru${celeb}"
    result_dir="${results_root}/thru${celeb}"
    sample_output_dir="${result_dir}/images"
    metrics_output_dir="${result_dir}/metrics"
    coco_images_dir="${result_dir}/coco/images"
    coco_metrics_dir="${result_dir}/coco/metrics"

    # TRAIN: Unlearn the current celebrity using the current base checkpoint.
    conda activate cuig
    cd "${train_dir}"
    train_args=(
        --anchor_target_concepts "${anchor}+${celeb_name}"
        --concept_type "celeb"
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
        --previously_unlearned "$(printf '%s\n' "${unlearned[@]}" | jq -R . | jq -s .)"
        --gradient_projection_num_prompts 200
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

    # Append the current celebrity to the cumulative unlearned set.
    unlearned+=("${celeb}")
    unlearned_json="$(array_to_json "${unlearned[@]}")"
    celebrities_subset_json="$(array_to_json "${unlearned[@]}" "${retain_celebrities[@]}")"
    sample_celebrities=("${unlearned[@]}" "${retain_celebrities[@]}")

    echo "Celebrities unlearned so far: ${unlearned_json}"
    echo "Celebrity subset for sampling: ${celebrities_subset_json}"

    # PROMPTS: Ensure prompts exist for all celebrities sampled at this checkpoint.
    cd "${eval_dir}"
    mkdir -p "${sample_output_dir}" "${metrics_output_dir}" "${prompt_dir}" "${prompt_locks_dir}"
    for prompt_celeb in "${sample_celebrities[@]}"; do
        prompt_output_path="${prompt_dir}/${prompt_celeb}.txt"
        prompt_lock_path="${prompt_locks_dir}/${prompt_celeb}.lock"
        prompt_text="${prompt_celeb//_/ }"

        (
            flock -x 200
            python generate_celeb_prompts.py \
                --prompt "${prompt_text}" \
                --num_prompts "${num_prompts}" \
                --output_path "${prompt_output_path}"
        ) 200>"${prompt_lock_path}"
    done

    # SAMPLE: Generate images from the current sequentially unlearned model.
    cd "${eval_dir}"
    python sample_celeb.py \
        --model_family sd \
        --ckpt "${output_dir}/delta.bin" \
        --pipeline_dir "${base_model_dir}" \
        --prompt_dir "${prompt_dir}" \
        --output_dir "${sample_output_dir}" \
        --celeb_subset "${celebrities_subset_json}" \
        --num_prompts "${num_prompts}" \
        --n_samples_per_prompt "${num_seeds}"

    # EVALUATE: Run celebrity classifier evaluation on generated images.
    conda activate celeb_eval
    cd "${eval_dir}/celeb-detection-oss"
    python examples/evaluate_celeb.py \
        --input_dir "${sample_output_dir}" \
        --output_dir "${metrics_output_dir}" \
        --unlearn "${unlearned_json}" \
        --retain "${retain_celebrities_json}" \
        --num_prompts "${num_prompts}" \
        --num_seeds "${num_seeds}"

    if [[ "${submit_coco}" == true ]]; then
        echo "Running COCO sample and evaluation for sequential celebrity checkpoint through: ${celeb}"

        conda activate cuig
        mkdir -p "${coco_images_dir}" "${coco_metrics_dir}"

        cd "${eval_dir}/coco"
        python sample_coco.py \
            --model_family sd \
            --model_name "${output_dir}/delta.bin" \
            --pipeline_dir "${base_model_dir}" \
            --output_dir "${coco_images_dir}" \
            --num_prompts "${coco_num_prompts}"

        python evaluate_coco.py \
            --input_dir "${coco_images_dir}" \
            --output_dir "${coco_metrics_dir}"
    else
        echo "Skipping COCO evaluation because submit_coco is set to false."
    fi

    echo "Completed Sequential Celebrity Unlearning with Projection 200 ConAbl through celebrity: ${celeb}"
done

echo "Sequential Celebrity Unlearning with Projection 200 ConAbl finished!"
