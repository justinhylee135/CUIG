#!/bin/bash

set -euo pipefail

echo "Independent Celebrity Unlearning with Base ConAbl starting..."

# Shared configuration
_cuig_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "${_cuig_script_dir}/BashScripts/submit.sh" && "${_cuig_script_dir}" != "/" ]]; do
    _cuig_script_dir="$(dirname "${_cuig_script_dir}")"
done
source "${_cuig_script_dir}/BashScripts/submit.sh"
unset _cuig_script_dir

# Unlearning Method Agnostic
base_model_dir="${CUIG_CELEBRITY_BASE_MODEL_DIR}"

# Select Unlearning Method and Evaluation Method
train_dir="${REPO_ROOT}/UnlearningMethods/ConAbl"
eval_dir="${REPO_ROOT}/Evaluation/Celebrity"

# ConAbl Specific
accelerate_config="${REPO_ROOT}/Configs/Accelerator/single_gpu.yaml"
anchor_datasets_root="${REPO_ROOT}/UnlearningMethods/ConAbl/anchor_datasets/celebrity"
anchor_prompts_root="${REPO_ROOT}/UnlearningMethods/ConAbl/anchor_prompts/celebrity"
models_root="${OUTPUT_ROOT}/Independent/Celebrity/Base/ConAbl/Models"
results_root="${OUTPUT_ROOT}/Independent/Celebrity/Base/ConAbl/Results"
logs_root="${REPO_ROOT}/logs/Independent/Celebrity/Base"
prompt_dir="${eval_dir}/Prompts"
prompt_locks_dir="${prompt_dir}/.locks"
iterations=2000
num_prompts=50
num_seeds=1

# If submit_celeb=false and submit_coco=true, COCO jobs are submitted for existing ConAbl checkpoints.
submit_celeb=true
submit_coco=true
coco_num_prompts=5000

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

# Define the list of celebrities to unlearn independently
unlearn_celebrities=("Neil_Degrasse_Tyson" "Benicio_Del_Toro" "Aziz_Ansari" "Oprah_Winfrey" "Betty_White" "Megan_Fox")

# Define held-out celebrities to measure retention performance
retain_celebrities=("Morgan_Freeman" "Keanu_Reeves" "George_Takei" "Aretha_Franklin" "Maya_Angelou" "Natalie_Portman")

retain_celebrities_json="$(array_to_json "${retain_celebrities[@]}")"

mkdir -p "${logs_root}/ConAbl" "${prompt_dir}" "${prompt_locks_dir}"

# Submit one independent job per celebrity
if [[ "${submit_celeb}" == true ]]; then
for celeb in "${unlearn_celebrities[@]}"; do
    echo "Submitting celebrity: ${celeb}"

    anchor="man"
    if [[ "${celeb}" == "Oprah_Winfrey" || "${celeb}" == "Betty_White" || "${celeb}" == "Megan_Fox" ]]; then
        anchor="woman"
    fi

    celeb_name="${celeb//_/ }"
    anchor_dataset_dir="${anchor_datasets_root}/${anchor}"
    anchor_prompt_path="${anchor_prompts_root}/${anchor}_prompts.txt"
    output_dir="${models_root}/${celeb}"
    result_dir="${results_root}/${celeb}"
    sample_output_dir="${result_dir}/images"
    metrics_output_dir="${result_dir}/metrics"
    coco_images_dir="${result_dir}/coco/images"
    coco_metrics_dir="${result_dir}/coco/metrics"
    unlearn_json="$(array_to_json "${celeb}")"
    celebrities_subset_json="$(array_to_json "${celeb}" "${retain_celebrities[@]}")"
    sample_celebrities_shell_words="$(array_to_shell_words "${celeb}" "${retain_celebrities[@]}")"
    job_file="$(mktemp "/tmp/conabl_independent_celebrity_${celeb}_XXXXXX.sh")"

    cat > "${job_file}" <<EOF
#!/bin/bash
#SBATCH --account=${CUIG_SLURM_ACCOUNT}
#SBATCH --job-name=Independent-Celebrity-Base-ConAbl-${celeb}
#SBATCH --time=04:00:00
#SBATCH --cluster=${CUIG_SLURM_CLUSTER}
#SBATCH --partition=${CUIG_SLURM_PARTITION}
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --output=${logs_root}/ConAbl/${celeb}_%j.out
#SBATCH --error=${logs_root}/ConAbl/${celeb}_%j.err

# Script Settings
module --ignore_cache load cuda/12.4.1
source ~/.bashrc
set -euo pipefail

echo "Independent Celebrity Unlearning with Base ConAbl starting for celebrity: ${celeb}..."

# User must set
REPO_ROOT="${REPO_ROOT}"
OUTPUT_ROOT="${OUTPUT_ROOT}"

# Run (your own) script that activates the conda env and sets env variables
if [[ -n "${CUIG_PRIVATE_EXPORTS:-}" && -f "${CUIG_PRIVATE_EXPORTS}" ]]; then
    source "${CUIG_PRIVATE_EXPORTS}"
fi

# Unlearning Method Agnostic
base_model_dir="${base_model_dir}"

# Select Unlearning Method and Evaluation Method
train_dir="${train_dir}"
eval_dir="${eval_dir}"

# ConAbl Specific
accelerate_config="${accelerate_config}"
anchor_dataset_dir="${anchor_dataset_dir}"
anchor_prompt_path="${anchor_prompt_path}"
output_dir="${output_dir}"
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
unlearn_json='${unlearn_json}'
retain_celebrities_json='${retain_celebrities_json}'
celebrities_subset_json='${celebrities_subset_json}'
sample_celebrities=(${sample_celebrities_shell_words})

# TRAIN: Unlearn the current celebrity from the base model
cd "\${train_dir}"
train_args=(
    --anchor_target_concepts "${anchor}+${celeb_name}"
    --concept_type "celeb"
    --output_dir "\${output_dir}"
    --base_model_dir "\${base_model_dir}"
    --iterations ${iterations}
    --num_anchor_images 200
    --num_anchor_prompts 200
    --anchor_dataset_dirs "\${anchor_dataset_dir}"
    --anchor_prompt_paths "\${anchor_prompt_path}"
    --scale_lr
    --hflip
    --noaug
    --enable_xformers_memory_efficient_attention
)

accelerate launch \
    --config_file "\${accelerate_config}" \
    train_conabl.py \
    "\${train_args[@]}"

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

# SAMPLE: Generate images from the independently unlearned model
cd "\${eval_dir}"
python sample_celeb.py \
    --model_family sd \
    --ckpt "\${output_dir}/delta.bin" \
    --pipeline_dir "\${base_model_dir}" \
    --prompt_dir "\${prompt_dir}" \
    --output_dir "\${sample_output_dir}" \
    --celeb_subset "\${celebrities_subset_json}" \
    --num_prompts "\${num_prompts}" \
    --n_samples_per_prompt "\${num_seeds}"

# EVALUATE: Run celebrity classifier evaluation on generated images
conda activate celeb_eval
cd "\${eval_dir}/celeb-detection-oss"
python examples/evaluate_celeb.py \
    --input_dir "\${sample_output_dir}" \
    --output_dir "\${metrics_output_dir}" \
    --unlearn "\${unlearn_json}" \
    --retain "\${retain_celebrities_json}" \
    --num_prompts "\${num_prompts}" \
    --num_seeds "\${num_seeds}"

if [[ "\${submit_coco}" == true ]]; then
    echo "Running COCO sample and evaluation for celebrity checkpoint: ${celeb}"

    conda activate cuig
    mkdir -p "\${coco_images_dir}" "\${coco_metrics_dir}"

    cd "\${eval_dir}/coco"
    python sample_coco.py \
        --model_family sd \
        --model_name "\${output_dir}/delta.bin" \
        --pipeline_dir "\${base_model_dir}" \
        --output_dir "\${coco_images_dir}" \
        --num_prompts "\${coco_num_prompts}"

    python evaluate_coco.py \
        --input_dir "\${coco_images_dir}" \
        --output_dir "\${coco_metrics_dir}"
else
    echo "Skipping COCO evaluation because submit_coco is set to false."
fi

echo "Completed Independent Celebrity Unlearning with Base ConAbl for celebrity: ${celeb}"
EOF

    sbatch "${job_file}"
    rm "${job_file}"
done
else
    echo "Skipping celebrity evaluation because submit_celeb is set to false."
fi

if [[ "${submit_celeb}" != true && "${submit_coco}" == true ]]; then
    for celeb in "${unlearn_celebrities[@]}"; do
        echo "Submitting COCO evaluation for existing celebrity checkpoint: ${celeb}"

        output_dir="${models_root}/${celeb}"
        coco_images_dir="${results_root}/coco/${celeb}/images"
        coco_metrics_dir="${results_root}/coco/${celeb}/metrics"
        coco_job_file="$(mktemp "/tmp/conabl_independent_celebrity_coco_${celeb}_XXXXXX.sh")"

        cat > "${coco_job_file}" <<EOF
#!/bin/bash
#SBATCH --account=${CUIG_SLURM_ACCOUNT}
#SBATCH --job-name=Independent-Celebrity-Base-ConAbl-coco-${celeb}
#SBATCH --time=03:00:00
#SBATCH --cluster=${CUIG_SLURM_CLUSTER}
#SBATCH --partition=${CUIG_SLURM_PARTITION}
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --output=${logs_root}/ConAbl/coco_${celeb}_%j.out
#SBATCH --error=${logs_root}/ConAbl/coco_${celeb}_%j.err

# Script Settings
module --ignore_cache load cuda/12.4.1
source ~/.bashrc
set -euo pipefail

echo "Independent Celebrity Base ConAbl COCO evaluation starting for celebrity checkpoint: ${celeb}..."

# User must set
REPO_ROOT="${REPO_ROOT}"
OUTPUT_ROOT="${OUTPUT_ROOT}"

# Run (your own) script that activates the conda env and sets env variables
if [[ -n "${CUIG_PRIVATE_EXPORTS:-}" && -f "${CUIG_PRIVATE_EXPORTS}" ]]; then
    source "${CUIG_PRIVATE_EXPORTS}"
fi

# Evaluation settings
eval_dir="${eval_dir}"
base_model_dir="${base_model_dir}"
output_dir="${output_dir}"
coco_images_dir="${coco_images_dir}"
coco_metrics_dir="${coco_metrics_dir}"
coco_num_prompts=${coco_num_prompts}

conda activate cuig
mkdir -p "\${coco_images_dir}" "\${coco_metrics_dir}"

cd "\${eval_dir}/coco"
python sample_coco.py \
    --model_family sd \
    --model_name "\${output_dir}/delta.bin" \
    --pipeline_dir "\${base_model_dir}" \
    --output_dir "\${coco_images_dir}" \
    --num_prompts "\${coco_num_prompts}"

python evaluate_coco.py \
    --input_dir "\${coco_images_dir}" \
    --output_dir "\${coco_metrics_dir}"

echo "Completed Independent Celebrity Base ConAbl COCO evaluation for celebrity checkpoint: ${celeb}."
EOF

        sbatch "${coco_job_file}"
        rm "${coco_job_file}"
    done
elif [[ "${submit_coco}" != true ]]; then
    echo "Skipping COCO evaluation because submit_coco is set to false."
fi

echo "Independent Celebrity Unlearning with Base ConAbl finished!"
