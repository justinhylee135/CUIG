#!/bin/bash

set -euo pipefail

echo "Independent Celebrity Merge TIES with ESD-SDXL starting..."

# Shared configuration
_cuig_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "${_cuig_script_dir}/BashScripts/submit.sh" && "${_cuig_script_dir}" != "/" ]]; do
    _cuig_script_dir="$(dirname "${_cuig_script_dir}")"
done
source "${_cuig_script_dir}/BashScripts/submit.sh"
unset _cuig_script_dir

# Shared paths
base_model_dir="stabilityai/stable-diffusion-xl-base-1.0"
merge_dir="${REPO_ROOT}/Regularizers/Merge"
eval_dir="${REPO_ROOT}/Evaluation/Celebrity"
source_models_root="${OUTPUT_ROOT}/Independent/Celebrity/Base/ESD-SDXL/Models"
models_root="${OUTPUT_ROOT}/Independent/Celebrity/Merge/TIES/ESD-SDXL/Models"
results_root="${OUTPUT_ROOT}/Independent/Celebrity/Merge/TIES/ESD-SDXL/Results"
logs_root="${REPO_ROOT}/logs/Independent/Celebrity/Merge/TIES/ESD-SDXL"
prompt_dir="${eval_dir}/Prompts"
prompt_locks_dir="${prompt_dir}/.locks"
num_prompts=50
num_seeds=1

# Set to true to run COCO retention checks for every merged checkpoint.
submit_coco=false
coco_num_prompts=5000

# Independent checkpoints to merge. Each configured endpoint runs a cumulative
# merge from Neil_Degrasse_Tyson through that endpoint.
merge_celebrities=("Neil_Degrasse_Tyson" "Benicio_Del_Toro" "Aziz_Ansari" "Oprah_Winfrey" "Betty_White" "Megan_Fox")

# Held-out celebrities to measure retention performance.
retain_celebrities=("Morgan_Freeman" "Keanu_Reeves" "George_Takei" "Aretha_Franklin" "Maya_Angelou" "Natalie_Portman")

# Per-endpoint TIES hyperparameters selected from the gridsearch.
declare -A celebrity_ties_lambda=(
    ["Benicio_Del_Toro"]="1.25"
    ["Aziz_Ansari"]="1.25"
    ["Oprah_Winfrey"]="1.25"
    ["Betty_White"]="1.25"
    ["Megan_Fox"]="1.25"
)
declare -A celebrity_ties_top_k=(
    ["Benicio_Del_Toro"]="0.20"
    ["Aziz_Ansari"]="0.20"
    ["Oprah_Winfrey"]="0.80"
    ["Betty_White"]="0.60"
    ["Megan_Fox"]="0.60"
)
ties_merge_func="mean"
merge_device="cpu"
key_filter_json='["attn2.to_k", "attn2.to_v"]'

array_to_json() {
    local json="["
    local item

    for item in "$@"; do
        json+="\"${item}\","
    done

    printf '%s' "${json%,}]"
}

array_to_shell_words() {
    local words=""
    local item

    for item in "$@"; do
        words+="\"${item}\" "
    done

    printf '%s' "${words% }"
}

format_decimal() {
    printf "%.2f" "$1"
}

retain_celebrities_json="$(array_to_json "${retain_celebrities[@]}")"

mkdir -p "${prompt_dir}" "${prompt_locks_dir}"

cumulative_celebrities=()

for celeb in "${merge_celebrities[@]}"; do
    cumulative_celebrities+=("${celeb}")

    if [[ -z "${celebrity_ties_lambda[$celeb]+x}" ]]; then
        echo "Skipping thru${celeb}; no celebrity-specific TIES hyperparameters configured."
        continue
    fi

    merge_celebrities_json="$(array_to_json "${cumulative_celebrities[@]}")"
    celebrities_subset_json="$(array_to_json "${cumulative_celebrities[@]}" "${retain_celebrities[@]}")"
    sample_celebrities_shell_words="$(array_to_shell_words "${cumulative_celebrities[@]}" "${retain_celebrities[@]}")"

    checkpoint_paths=()
    for checkpoint_celeb in "${cumulative_celebrities[@]}"; do
        checkpoint_paths+=("${source_models_root}/${checkpoint_celeb}/delta.bin")
    done
    checkpoints_json="$(array_to_json "${checkpoint_paths[@]}")"

    ties_lambda="${celebrity_ties_lambda[$celeb]}"
    ties_top_k="${celebrity_ties_top_k[$celeb]}"
    lambda_name="$(format_decimal "${ties_lambda}")"
    topk_name="$(format_decimal "${ties_top_k}")"
    experiment_tag="lambda${lambda_name}_topk${topk_name}"
    job_logs_root="${logs_root}/${experiment_tag}"
    merged_ckpt_path="${models_root}/${experiment_tag}/thru${celeb}.pth"
    result_dir="${results_root}/${experiment_tag}/thru${celeb}"
    sample_output_dir="${result_dir}/images"
    metrics_output_dir="${result_dir}/metrics"
    coco_images_dir="${result_dir}/coco/images"
    coco_metrics_dir="${result_dir}/coco/metrics"
    job_file="$(mktemp "/tmp/ties_independent_celebrity_esd_sdxl_thru${celeb}_${experiment_tag}_XXXXXX.sh")"

    mkdir -p "${job_logs_root}"

    if [[ -d "${metrics_output_dir}" ]]; then
        echo "Skipping thru${celeb}; metrics already exist at ${metrics_output_dir}"
        rm "${job_file}"
        continue
    fi

    echo "Submitting Merge TIES for ESD-SDXL thru${celeb}: lambda=${lambda_name}, topk=${topk_name}"

    cat > "${job_file}" <<EOF
#!/bin/bash
#SBATCH --account=${CUIG_SLURM_ACCOUNT}
#SBATCH --job-name=Independent-Celebrity-Merge-TIES-ESD-SDXL-thru${celeb}-${experiment_tag}
#SBATCH --time=02:00:00
#SBATCH --cluster=${CUIG_SLURM_CLUSTER}
#SBATCH --partition=${CUIG_SLURM_PARTITION}
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --ntasks-per-node=1
#SBATCH --output=${job_logs_root}/%j.out
#SBATCH --error=${job_logs_root}/%j.err

module --ignore_cache load cuda/12.4.1
source ~/.bashrc
set -euo pipefail

echo "Independent Celebrity TIES merge for ESD-SDXL starting for thru${celeb} with ${experiment_tag}..."

REPO_ROOT="${REPO_ROOT}"
base_model_dir="${base_model_dir}"
merge_dir="${merge_dir}"
eval_dir="${eval_dir}"
merged_ckpt_path="${merged_ckpt_path}"
sample_output_dir="${sample_output_dir}"
metrics_output_dir="${metrics_output_dir}"
prompt_dir="${prompt_dir}"
prompt_locks_dir="${prompt_locks_dir}"
coco_images_dir="${coco_images_dir}"
coco_metrics_dir="${coco_metrics_dir}"
num_prompts=${num_prompts}
num_seeds=${num_seeds}
coco_num_prompts=${coco_num_prompts}
submit_coco=${submit_coco}
merge_celebrities_json='${merge_celebrities_json}'
retain_celebrities_json='${retain_celebrities_json}'
celebrities_subset_json='${celebrities_subset_json}'
sample_celebrities=(${sample_celebrities_shell_words})
checkpoints_json='${checkpoints_json}'
key_filter_json='${key_filter_json}'

if [[ -n "${CUIG_PRIVATE_EXPORTS:-}" && -f "${CUIG_PRIVATE_EXPORTS}" ]]; then
    source "${CUIG_PRIVATE_EXPORTS}"
fi

# MERGE: Build one merged checkpoint from the independent ESD-SDXL deltas.
conda activate cuig
cd "\${merge_dir}"
python merge.py \
    --merge_method "ties" \
    --base_model_dir "\${base_model_dir}" \
    --ckpt_paths "\${checkpoints_json}" \
    --save_path "\${merged_ckpt_path}" \
    --ties_lambda "${ties_lambda}" \
    --ties_topk "${ties_top_k}" \
    --ties_merge_func "${ties_merge_func}" \
    --device "${merge_device}" \
    --key_filter "\${key_filter_json}"

# PROMPTS: Ensure prompts exist for all celebrities sampled in this job.
cd "\${eval_dir}"
mkdir -p "\${sample_output_dir}" "\${metrics_output_dir}" "\${prompt_dir}" "\${prompt_locks_dir}"
for prompt_celeb in "\${sample_celebrities[@]}"; do
    prompt_output_path="\${prompt_dir}/\${prompt_celeb}.txt"
    prompt_lock_path="\${prompt_locks_dir}/\${prompt_celeb}.lock"
    prompt_text="\${prompt_celeb//_/ }"

    (
        flock -x 200
        python generate_celeb_prompts.py \
            --prompt "\${prompt_text}" \
            --num_prompts "\${num_prompts}" \
            --output_path "\${prompt_output_path}"
    ) 200>"\${prompt_lock_path}"
done

# SAMPLE: Generate images from the merged SDXL model.
cd "\${eval_dir}"
python sample_celeb.py \
    --model_family sdxl \
    --ckpt "\${merged_ckpt_path}" \
    --pipeline_dir "\${base_model_dir}" \
    --resolution 1024 \
    --prompt_dir "\${prompt_dir}" \
    --output_dir "\${sample_output_dir}" \
    --celeb_subset "\${celebrities_subset_json}" \
    --num_prompts "\${num_prompts}" \
    --n_samples_per_prompt "\${num_seeds}"

# EVALUATE: Run celebrity classifier evaluation on generated images.
conda activate celeb_eval
cd "\${eval_dir}/celeb-detection-oss"
python examples/evaluate_celeb.py \
    --input_dir "\${sample_output_dir}" \
    --output_dir "\${metrics_output_dir}" \
    --unlearn "\${merge_celebrities_json}" \
    --retain "\${retain_celebrities_json}" \
    --num_prompts "\${num_prompts}" \
    --num_seeds "\${num_seeds}"

if [[ "\${submit_coco}" == true ]]; then
    echo "Running COCO sample and evaluation for merged celebrity checkpoint: thru${celeb} ${experiment_tag}"

    conda activate cuig
    mkdir -p "\${coco_images_dir}" "\${coco_metrics_dir}"

    cd "\${eval_dir}/coco"
    python sample_coco.py \
        --model_family sdxl \
        --model_name "\${merged_ckpt_path}" \
        --pipeline_dir "\${base_model_dir}" \
        --resolution 1024 \
        --output_dir "\${coco_images_dir}" \
        --num_prompts "\${coco_num_prompts}"

    python evaluate_coco.py \
        --input_dir "\${coco_images_dir}" \
        --output_dir "\${coco_metrics_dir}"
else
    echo "Skipping COCO evaluation because submit_coco is set to false."
fi

echo "Completed ESD-SDXL TIES merge for thru${celeb} with ${experiment_tag}"
EOF

    sbatch "${job_file}"
    rm "${job_file}"
done

echo "Independent Celebrity Merge TIES with ESD-SDXL submission finished!"
