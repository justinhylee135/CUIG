#!/bin/bash
#SBATCH --job-name=Sequential-Celebrity-Weight-L2SP-100-ESD-SDXL
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1

module --ignore_cache load cuda/12.4.1
source ~/.bashrc
set -euo pipefail

echo "Sequential Celebrity Unlearning with Weight L2SP 100 ESD-SDXL starting..."

# Shared configuration
_cuig_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "${_cuig_script_dir}/BashScripts/submit.sh" && "${_cuig_script_dir}" != "/" ]]; do
    _cuig_script_dir="$(dirname "${_cuig_script_dir}")"
done
source "${_cuig_script_dir}/BashScripts/submit.sh"
unset _cuig_script_dir

# Run the local script that activates the conda env and sets private environment variables.
if [[ -n "${CUIG_PRIVATE_EXPORTS:-}" && -f "${CUIG_PRIVATE_EXPORTS}" ]]; then
    source "${CUIG_PRIVATE_EXPORTS}"
fi

base_model_dir="stabilityai/stable-diffusion-xl-base-1.0"
train_dir="${REPO_ROOT}/UnlearningMethods/ESD-SDXL"
eval_dir="${REPO_ROOT}/Evaluation/Celebrity"
models_root="${OUTPUT_ROOT}/Sequential/Celebrity/Weight/L2SP/100/ESD-SDXL/Models"
results_root="${OUTPUT_ROOT}/Sequential/Celebrity/Weight/L2SP/100/ESD-SDXL/Results"
prompt_dir="${eval_dir}/Prompts"
prompt_locks_dir="${prompt_dir}/.locks"
iterations=200
num_prompts=50
num_seeds=1
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

unlearn_celebrities=("Neil_Degrasse_Tyson" "Benicio_Del_Toro" "Aziz_Ansari" "Oprah_Winfrey" "Betty_White" "Megan_Fox")
retain_celebrities=("Morgan_Freeman" "Keanu_Reeves" "George_Takei" "Aretha_Franklin" "Maya_Angelou" "Natalie_Portman")
retain_celebrities_json="$(array_to_json "${retain_celebrities[@]}")"

mkdir -p "${REPO_ROOT}/logs/Sequential/Celebrity/Weight/L2SP/100/ESD-SDXL" "${prompt_dir}" "${prompt_locks_dir}"

base_unet_ckpt=""
unlearned=()

for celeb in "${unlearn_celebrities[@]}"; do
    echo "Unlearning celebrity: ${celeb}"

    celeb_name="${celeb//_/ }"
    output_dir="${models_root}/thru${celeb}"
    result_dir="${results_root}/thru${celeb}"
    sample_output_dir="${result_dir}/images"
    metrics_output_dir="${result_dir}/metrics"
    coco_images_dir="${result_dir}/coco/images"
    coco_metrics_dir="${result_dir}/coco/metrics"

    conda activate cuig
    cd "${train_dir}"
    train_args=(
        --concept "${celeb_name}"
        --concept_type "celeb"
        --train_method "esd-x-strict"
        --lr "2e-4"
        --iterations "${iterations}"
        --output_dir "${output_dir}"
        --base_model_dir "${base_model_dir}"
        --l2sp_weight 100
    )
    if [[ -n "${base_unet_ckpt}" ]]; then
        train_args+=(--unet_ckpt "${base_unet_ckpt}")
    fi
    python train_esd_sdxl.py "${train_args[@]}"

    base_unet_ckpt="${output_dir}/delta.bin"
    unlearned+=("${celeb}")
    unlearned_json="$(array_to_json "${unlearned[@]}")"
    celebrities_subset_json="$(array_to_json "${unlearned[@]}" "${retain_celebrities[@]}")"
    sample_celebrities=("${unlearned[@]}" "${retain_celebrities[@]}")

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

    python sample_celeb.py \
        --model_family sdxl \
        --ckpt "${output_dir}/delta.bin" \
        --pipeline_dir "${base_model_dir}" \
        --resolution 1024 \
        --prompt_dir "${prompt_dir}" \
        --output_dir "${sample_output_dir}" \
        --celeb_subset "${celebrities_subset_json}" \
        --num_prompts "${num_prompts}" \
        --n_samples_per_prompt "${num_seeds}"

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
        conda activate cuig
        mkdir -p "${coco_images_dir}" "${coco_metrics_dir}"
        cd "${eval_dir}/coco"
        python sample_coco.py \
            --model_family sdxl \
            --model_name "${output_dir}/delta.bin" \
            --pipeline_dir "${base_model_dir}" \
            --resolution 1024 \
            --output_dir "${coco_images_dir}" \
            --num_prompts "${coco_num_prompts}"
        python evaluate_coco.py \
            --input_dir "${coco_images_dir}" \
            --output_dir "${coco_metrics_dir}"
    else
        echo "Skipping COCO evaluation because submit_coco is set to false."
    fi

    echo "Completed Sequential Celebrity Unlearning with Weight L2SP 100 ESD-SDXL through celebrity: ${celeb}"
done

echo "Sequential Celebrity Unlearning with Weight L2SP 100 ESD-SDXL finished!"
