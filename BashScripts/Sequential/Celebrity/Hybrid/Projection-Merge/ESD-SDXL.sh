#!/bin/bash

set -euo pipefail

echo "Independent Celebrity Hybrid Projection-Merge with ESD-SDXL submission starting..."

# Shared configuration
_cuig_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "${_cuig_script_dir}/BashScripts/submit.sh" && "${_cuig_script_dir}" != "/" ]]; do
    _cuig_script_dir="$(dirname "${_cuig_script_dir}")"
done
source "${_cuig_script_dir}/BashScripts/submit.sh"
unset _cuig_script_dir

# Shared paths
base_model_dir="stabilityai/stable-diffusion-xl-base-1.0"
train_dir="${REPO_ROOT}/UnlearningMethods/ESD-SDXL"
merge_dir="${REPO_ROOT}/Regularizers/Merge"
eval_dir="${REPO_ROOT}/Evaluation/Celebrity"
run_root="${OUTPUT_ROOT}/Independent/Celebrity/Hybrid/Projection_Merge/200_TIES/ESD-SDXL"
source_models_root="${OUTPUT_ROOT}/Independent/Celebrity/Projection/200/ESD-SDXL"
models_root="${run_root}/Models"
results_root="${run_root}/Results"
logs_root="${REPO_ROOT}/logs/Independent/Celebrity/Hybrid/Projection_Merge/200_TIES/ESD-SDXL"
prompt_dir="${eval_dir}/Prompts"
prompt_locks_dir="${prompt_dir}/.locks"
iterations=200
num_prompts=50
num_seeds=1

# Set to true to run COCO retention checks for every merged checkpoint.
submit_coco=false
coco_num_prompts=5000

# Projection settings
gradient_projection_num_prompts=200

# TIES merge settings. Each configured endpoint runs a cumulative merge from
# Neil_Degrasse_Tyson through that endpoint.
declare -A celebrity_ties_lambda=(
    ["Benicio_Del_Toro"]="1.25"
    ["Aziz_Ansari"]="1.25"
    ["Oprah_Winfrey"]="1.25"
    ["Betty_White"]="1.25"
    ["Megan_Fox"]="1.25"
)
declare -A celebrity_ties_top_k=(
    ["Benicio_Del_Toro"]="0.20"
    ["Aziz_Ansari"]="0.60"
    ["Oprah_Winfrey"]="0.80"
    ["Betty_White"]="0.80"
    ["Megan_Fox"]="0.40"
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

submit_job() {
    local job_file="$1"
    local dependency="${2:-}"
    local job_id

    if [[ -n "${dependency}" ]]; then
        job_id="$(sbatch --parsable --dependency="afterok:${dependency}" "${job_file}")"
    else
        job_id="$(sbatch --parsable "${job_file}")"
    fi

    # Slurm can return job_id;cluster with --parsable. Dependencies only need the id.
    printf '%s' "${job_id%%;*}"
}

join_by_colon() {
    local IFS=":"
    printf '%s' "$*"
}

# Active celebrity set for projected independent training and cumulative merge.
unlearn_celebrities=("Neil_Degrasse_Tyson" "Benicio_Del_Toro" "Aziz_Ansari" "Oprah_Winfrey" "Betty_White" "Megan_Fox")

# Held-out celebrities to measure retention performance.
retain_celebrities=("Morgan_Freeman" "Keanu_Reeves" "George_Takei" "Aretha_Franklin" "Maya_Angelou" "Natalie_Portman")
retain_celebrities_json="$(array_to_json "${retain_celebrities[@]}")"

mkdir -p "${logs_root}/independent" "${logs_root}/merge" "${source_models_root}" "${models_root}" "${prompt_dir}" "${prompt_locks_dir}"

# Phase 1: submit one independent ESD-SDXL projection job per celebrity.
declare -A independent_job_ids=()
previously_unlearned=()

for celeb in "${unlearn_celebrities[@]}"; do
    celeb_name="${celeb//_/ }"
    output_dir="${source_models_root}/${celeb}"
    checkpoint_path="${output_dir}/delta.bin"
    previously_unlearned_json="$(array_to_json "${previously_unlearned[@]}")"

    if [[ -f "${checkpoint_path}" ]]; then
        echo "Independent Projection checkpoint already exists for ${celeb}: ${checkpoint_path}"
        independent_job_ids["${celeb}"]=""
        previously_unlearned+=("${celeb}")
        continue
    fi

    job_file="$(mktemp "/tmp/esd_sdxl_projection_independent_${celeb}_XXXXXX.sh")"
    cat > "${job_file}" <<EOF
#!/bin/bash
#SBATCH --account=${CUIG_SLURM_ACCOUNT}
#SBATCH --job-name=Independent-Celebrity-Projection-200-ESD-SDXL-${celeb}
#SBATCH --time=04:00:00
#SBATCH --cluster=${CUIG_SLURM_CLUSTER}
#SBATCH --partition=${CUIG_SLURM_PARTITION}
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --output=${logs_root}/independent/${celeb}_%j.out
#SBATCH --error=${logs_root}/independent/${celeb}_%j.err

module --ignore_cache load cuda/12.4.1
source ~/.bashrc
set -euo pipefail

echo "Independent Celebrity Projection ESD-SDXL training starting for celebrity: ${celeb}"

REPO_ROOT="${REPO_ROOT}"
if [[ -n "${CUIG_PRIVATE_EXPORTS:-}" && -f "${CUIG_PRIVATE_EXPORTS}" ]]; then
    source "${CUIG_PRIVATE_EXPORTS}"
fi

base_model_dir="${base_model_dir}"
train_dir="${train_dir}"
output_dir="${output_dir}"
previously_unlearned_json='${previously_unlearned_json}'

conda activate cuig
cd "\${train_dir}"
python train_esd_sdxl.py \
    --concept "${celeb_name}" \
    --concept_type "celeb" \
    --train_method "esd-x-strict" \
    --lr "2e-4" \
    --iterations "${iterations}" \
    --output_dir "\${output_dir}" \
    --base_model_dir "\${base_model_dir}" \
    --with_gradient_projection \
    --auxiliary_prompts_path "\${output_dir}/prompts.txt" \
    --previously_unlearned "\${previously_unlearned_json}" \
    --gradient_projection_num_prompts "${gradient_projection_num_prompts}"

echo "Independent Celebrity Projection ESD-SDXL training completed for celebrity: ${celeb}"
EOF

    job_id="$(submit_job "${job_file}")"
    rm "${job_file}"
    independent_job_ids["${celeb}"]="${job_id}"
    previously_unlearned+=("${celeb}")
    echo "Submitted independent projection job for ${celeb}: ${job_id}"
done

# Phase 2: submit one merge/evaluation job per configured cumulative endpoint.
cumulative_celebrities=()
checkpoint_paths=()

for celeb in "${unlearn_celebrities[@]}"; do
    cumulative_celebrities+=("${celeb}")
    checkpoint_paths+=("${source_models_root}/${celeb}/delta.bin")

    if [[ -z "${celebrity_ties_lambda[$celeb]+x}" ]]; then
        echo "Skipping thru${celeb}; no celebrity-specific TIES hyperparameters configured."
        continue
    fi

    merge_celebrities_json="$(array_to_json "${cumulative_celebrities[@]}")"
    celebrities_subset_json="$(array_to_json "${cumulative_celebrities[@]}" "${retain_celebrities[@]}")"
    checkpoints_json="$(array_to_json "${checkpoint_paths[@]}")"
    sample_celebrities_shell_words="$(array_to_shell_words "${cumulative_celebrities[@]}" "${retain_celebrities[@]}")"

    ties_lambda="${celebrity_ties_lambda[$celeb]}"
    ties_top_k="${celebrity_ties_top_k[$celeb]}"
    lambda_name="$(format_decimal "${ties_lambda}")"
    topk_name="$(format_decimal "${ties_top_k}")"
    experiment_tag="lambda${lambda_name}_topk${topk_name}"
    merged_ckpt_path="${models_root}/${experiment_tag}/thru${celeb}.pth"
    result_dir="${results_root}/${experiment_tag}/thru${celeb}"
    sample_output_dir="${result_dir}/images"
    metrics_output_dir="${result_dir}/metrics"
    coco_images_dir="${result_dir}/coco/images"
    coco_metrics_dir="${result_dir}/coco/metrics"

    if [[ -d "${metrics_output_dir}" ]]; then
        echo "Skipping thru${celeb}; metrics already exist at ${metrics_output_dir}"
        continue
    fi

    dependency_job_ids=()
    for dependency_celeb in "${cumulative_celebrities[@]}"; do
        dependency_job_id="${independent_job_ids[$dependency_celeb]:-}"
        if [[ -n "${dependency_job_id}" ]]; then
            dependency_job_ids+=("${dependency_job_id}")
        fi
    done
    dependency="$(join_by_colon "${dependency_job_ids[@]}")"

    job_file="$(mktemp "/tmp/esd_sdxl_projection_merge_ties_thru${celeb}_${experiment_tag}_XXXXXX.sh")"
    cat > "${job_file}" <<EOF
#!/bin/bash
#SBATCH --account=${CUIG_SLURM_ACCOUNT}
#SBATCH --job-name=Independent-Celebrity-Projection-Merge-ESD-SDXL-thru${celeb}-${experiment_tag}
#SBATCH --time=02:00:00
#SBATCH --cluster=${CUIG_SLURM_CLUSTER}
#SBATCH --partition=${CUIG_SLURM_PARTITION}
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --ntasks-per-node=1
#SBATCH --output=${logs_root}/merge/thru${celeb}_${experiment_tag}_%j.out
#SBATCH --error=${logs_root}/merge/thru${celeb}_${experiment_tag}_%j.err

module --ignore_cache load cuda/12.4.1
source ~/.bashrc
set -euo pipefail

echo "Independent Celebrity Projection-Merge ESD-SDXL starting for thru${celeb} with ${experiment_tag}"

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

# MERGE: Build one merged checkpoint from the projected independent ESD-SDXL deltas.
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

# PROMPTS: Ensure prompts exist for all celebrities sampled in this merge.
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

# SAMPLE: Generate images from the merged SDXL checkpoint.
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
    echo "Running COCO sample and evaluation for projected merged checkpoint: thru${celeb} ${experiment_tag}"

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

echo "Completed projected ESD-SDXL TIES merge for thru${celeb} with ${experiment_tag}"
EOF

    merge_job_id="$(submit_job "${job_file}" "${dependency}")"
    rm "${job_file}"

    if [[ -n "${dependency}" ]]; then
        echo "Submitted merge job for thru${celeb}: ${merge_job_id} afterok:${dependency}"
    else
        echo "Submitted merge job for thru${celeb}: ${merge_job_id} with no pending checkpoint dependencies"
    fi
done

echo "Independent Celebrity Hybrid Projection-Merge with ESD-SDXL submission finished!"
