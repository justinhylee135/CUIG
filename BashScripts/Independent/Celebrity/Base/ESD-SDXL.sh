#!/bin/bash

set -euo pipefail

echo "Independent Celebrity Unlearning with Base ESD-SDXL starting..."

# Shared configuration
_cuig_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "${_cuig_script_dir}/BashScripts/submit.sh" && "${_cuig_script_dir}" != "/" ]]; do
    _cuig_script_dir="$(dirname "${_cuig_script_dir}")"
done
source "${_cuig_script_dir}/BashScripts/submit.sh"
unset _cuig_script_dir

# Unlearning Method Agnostic
base_model_dir="stabilityai/stable-diffusion-xl-base-1.0"

# Select Unlearning Method and Evaluation Method
train_dir="${REPO_ROOT}/UnlearningMethods/ESD-SDXL"
eval_dir="${REPO_ROOT}/Evaluation/Celebrity"

# ESD-SDXL Specific
models_root="${OUTPUT_ROOT}/Independent/Celebrity/Base/ESD-SDXL/Models"
results_root="${OUTPUT_ROOT}/Independent/Celebrity/Base/ESD-SDXL/Results"
logs_root="${REPO_ROOT}/logs/Independent/Celebrity/Base/ESD-SDXL"
prompt_dir="${eval_dir}/Prompts"
prompt_locks_dir="${prompt_dir}/.locks"
iterations=200
num_prompts=50
num_seeds=1

# If submit_celeb=false and submit_coco=true, COCO jobs are submitted for existing ESD-SDXL checkpoints.
submit_celeb=true
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

array_to_shell_words() {
    local words=""
    local item

    for item in "$@"; do
        words+="\"${item}\" "
    done

    printf '%s' "${words% }"
}

# Define the list of celebrities to unlearn independently.
unlearn_celebrities=("Neil_Degrasse_Tyson" "Benicio_Del_Toro" "Aziz_Ansari" "Oprah_Winfrey" "Betty_White" "Megan_Fox")

# Define held-out celebrities to measure retention performance.
retain_celebrities=("Morgan_Freeman" "Keanu_Reeves" "George_Takei" "Aretha_Franklin" "Maya_Angelou" "Natalie_Portman")

retain_celebrities_json="$(array_to_json "${retain_celebrities[@]}")"

mkdir -p "${logs_root}" "${prompt_dir}" "${prompt_locks_dir}"

# Submit one independent job per celebrity.
if [[ "${submit_celeb}" == true ]]; then
for celeb in "${unlearn_celebrities[@]}"; do
    echo "Submitting celebrity: ${celeb}"

    celeb_name="${celeb//_/ }"
    output_dir="${models_root}/${celeb}"
    result_dir="${results_root}/${celeb}"
    sample_output_dir="${result_dir}/images"
    metrics_output_dir="${result_dir}/metrics"
    coco_images_dir="${result_dir}/coco/images"
    coco_metrics_dir="${result_dir}/coco/metrics"
    unlearn_json="$(array_to_json "${celeb}")"
    celebrities_subset_json="$(array_to_json "${celeb}" "${retain_celebrities[@]}")"
    sample_celebrities_shell_words="$(array_to_shell_words "${celeb}" "${retain_celebrities[@]}")"
    job_file="$(mktemp "/tmp/esd_sdxl_independent_celebrity_${celeb}_XXXXXX.sh")"

    cat > "${job_file}" <<EOF
#!/bin/bash
#SBATCH --account=${CUIG_SLURM_ACCOUNT}
#SBATCH --job-name=Independent-Celebrity-Base-ESD-SDXL-${celeb}
#SBATCH --time=04:00:00
#SBATCH --cluster=${CUIG_SLURM_CLUSTER}
#SBATCH --partition=${CUIG_SLURM_PARTITION}
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --output=${logs_root}/${celeb}_%j.out
#SBATCH --error=${logs_root}/${celeb}_%j.err

# Script Settings
module --ignore_cache load cuda/12.4.1
source ~/.bashrc
set -euo pipefail

echo "Independent Celebrity Unlearning with Base ESD-SDXL starting for celebrity: ${celeb}..."

REPO_ROOT="${REPO_ROOT}"
OUTPUT_ROOT="${OUTPUT_ROOT}"
if [[ -n "${CUIG_PRIVATE_EXPORTS:-}" && -f "${CUIG_PRIVATE_EXPORTS}" ]]; then
    source "${CUIG_PRIVATE_EXPORTS}"
fi

base_model_dir="${base_model_dir}"
train_dir="${train_dir}"
eval_dir="${eval_dir}"
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

# TRAIN: Unlearn the current celebrity from the base SDXL model.
conda activate cuig
cd "\${train_dir}"
python train_esd_sdxl.py \
    --concept "${celeb_name}" \
    --concept_type "celeb" \
    --train_method "esd-x-strict" \
    --lr "2e-4" \
    --iterations ${iterations} \
    --output_dir "\${output_dir}" \
    --base_model_dir "\${base_model_dir}"

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

# SAMPLE: Generate images from the independently unlearned SDXL model.
cd "\${eval_dir}"
python sample_celeb.py \
    --model_family sdxl \
    --ckpt "\${output_dir}/delta.bin" \
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
        --model_family sdxl \
        --model_name "\${output_dir}/delta.bin" \
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

echo "Completed Independent Celebrity Unlearning with Base ESD-SDXL for celebrity: ${celeb}"
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
        coco_job_file="$(mktemp "/tmp/esd_sdxl_independent_celebrity_coco_${celeb}_XXXXXX.sh")"

        cat > "${coco_job_file}" <<EOF
#!/bin/bash
#SBATCH --account=${CUIG_SLURM_ACCOUNT}
#SBATCH --job-name=Independent-Celebrity-Base-ESD-SDXL-coco-${celeb}
#SBATCH --time=03:00:00
#SBATCH --cluster=${CUIG_SLURM_CLUSTER}
#SBATCH --partition=${CUIG_SLURM_PARTITION}
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --output=${logs_root}/coco_${celeb}_%j.out
#SBATCH --error=${logs_root}/coco_${celeb}_%j.err

module --ignore_cache load cuda/12.4.1
source ~/.bashrc
set -euo pipefail

REPO_ROOT="${REPO_ROOT}"
if [[ -n "${CUIG_PRIVATE_EXPORTS:-}" && -f "${CUIG_PRIVATE_EXPORTS}" ]]; then
    source "${CUIG_PRIVATE_EXPORTS}"
fi

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
    --model_family sdxl \
    --model_name "\${output_dir}/delta.bin" \
    --pipeline_dir "\${base_model_dir}" \
    --resolution 1024 \
    --output_dir "\${coco_images_dir}" \
    --num_prompts "\${coco_num_prompts}"

python evaluate_coco.py \
    --input_dir "\${coco_images_dir}" \
    --output_dir "\${coco_metrics_dir}"
EOF

        sbatch "${coco_job_file}"
        rm "${coco_job_file}"
    done
elif [[ "${submit_coco}" != true ]]; then
    echo "Skipping COCO evaluation because submit_coco is set to false."
fi

echo "Independent Celebrity Unlearning with Base ESD-SDXL finished!"
