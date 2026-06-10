#!/bin/bash

set -euo pipefail

echo "Independent Celebrity Base Stable Diffusion XL evaluation starting..."

# Shared configuration
_cuig_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "${_cuig_script_dir}/BashScripts/submit.sh" && "${_cuig_script_dir}" != "/" ]]; do
    _cuig_script_dir="$(dirname "${_cuig_script_dir}")"
done
source "${_cuig_script_dir}/BashScripts/submit.sh"
unset _cuig_script_dir

# Evaluation settings
eval_dir="${REPO_ROOT}/Evaluation/Celebrity"
results_root="${OUTPUT_ROOT}/Independent/Celebrity/Base/StableDiffusionXL/Results"
logs_root="${REPO_ROOT}/logs/Independent/Celebrity/Base/StableDiffusionXL"
prompt_dir="${eval_dir}/Prompts"
prompt_locks_dir="${prompt_dir}/.locks"
pipeline_dir="stabilityai/stable-diffusion-xl-base-1.0"
num_prompts=50
num_seeds=1

# If you want to run just FID and CLIP scores, set submit_celeb=false and submit_coco=true.
submit_celeb=true

# Set to true to also submit a job for baseline FID and CLIP scores.
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

# Define the list of celebrities to evaluate independently as unlearned targets.
unlearn_celebrities=("Neil_Degrasse_Tyson" "Benicio_Del_Toro" "Aziz_Ansari" "Oprah_Winfrey" "Betty_White" "Megan_Fox")

# Define held-out celebrities to measure retention performance.
retain_celebrities=("Morgan_Freeman" "Keanu_Reeves" "George_Takei" "Aretha_Franklin" "Maya_Angelou" "Natalie_Portman")

all_celebrities=("${unlearn_celebrities[@]}" "${retain_celebrities[@]}")

mkdir -p "${logs_root}" "${prompt_dir}" "${prompt_locks_dir}"

# Submit one independent SDXL baseline evaluation job per celebrity.
# This measures classifier accuracy on a non-unlearned model so RA can be normalized.
if [[ "${submit_celeb}" == true ]]; then
for celeb in "${all_celebrities[@]}"; do
    echo "Submitting celebrity: ${celeb}"

    result_dir="${results_root}/${celeb}"
    sample_output_dir="${result_dir}/images"
    metrics_output_dir="${result_dir}/metrics"
    unlearn_json="[]"
    retain_json="$(array_to_json "${celeb}")"
    celebrities_subset_json="$(array_to_json "${celeb}")"
    job_file="$(mktemp "/tmp/sdxl_independent_celebrity_${celeb}_XXXXXX.sh")"

    cat > "${job_file}" <<EOF
#!/bin/bash
#SBATCH --account=${CUIG_SLURM_ACCOUNT}
#SBATCH --job-name=Independent-Celebrity-Base-SDXL-${celeb}
#SBATCH --time=01:00:00
#SBATCH --cluster=${CUIG_SLURM_CLUSTER}
#SBATCH --partition=${CUIG_SLURM_PARTITION}
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --output=${logs_root}/StableDiffusionXL_${celeb}_%j.out
#SBATCH --error=${logs_root}/StableDiffusionXL_${celeb}_%j.err

# Script Settings
module --ignore_cache load cuda/12.4.1
source ~/.bashrc
set -euo pipefail

echo "Independent Celebrity Base Stable Diffusion XL evaluation starting for celebrity: ${celeb}..."

# User must set
REPO_ROOT="${REPO_ROOT}"
OUTPUT_ROOT="${OUTPUT_ROOT}"

# Run the local script that activates the conda env and sets private environment variables.
if [[ -n "${CUIG_PRIVATE_EXPORTS:-}" && -f "${CUIG_PRIVATE_EXPORTS}" ]]; then
    source "${CUIG_PRIVATE_EXPORTS}"
fi

# Evaluation settings
eval_dir="${eval_dir}"
results_root="${results_root}"
prompt_dir="${prompt_dir}"
prompt_locks_dir="${prompt_locks_dir}"
pipeline_dir="${pipeline_dir}"
sample_output_dir="${sample_output_dir}"
metrics_output_dir="${metrics_output_dir}"
num_prompts=${num_prompts}
num_seeds=${num_seeds}
unlearn_json='${unlearn_json}'
retain_json='${retain_json}'
celebrities_subset_json='${celebrities_subset_json}'
celeb="${celeb}"

mkdir -p "\${sample_output_dir}" "\${metrics_output_dir}" "\${prompt_dir}" "\${prompt_locks_dir}"

# PROMPTS: Generate prompts only for the celebrity submitted in this job.
cd "\${eval_dir}"
prompt_output_path="\${prompt_dir}/\${celeb}.txt"
prompt_lock_path="\${prompt_locks_dir}/\${celeb}.lock"
prompt_text="\${celeb//_/ }"

(
    flock -x 200
    python generate_celeb_prompts.py \
        --prompt "\${prompt_text}" \
        --num_prompts "\${num_prompts}" \
        --output_path "\${prompt_output_path}"
) 200>"\${prompt_lock_path}"

# SAMPLE: Generate images from the base SDXL model.
conda activate cuig
cd "\${eval_dir}"
python sample_celeb.py \
    --model_family sdxl \
    --ckpt base \
    --pipeline_dir "\${pipeline_dir}" \
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
    --retain "\${retain_json}" \
    --num_prompts "\${num_prompts}" \
    --num_seeds "\${num_seeds}"

echo "Completed Independent Celebrity Base Stable Diffusion XL evaluation for celebrity: ${celeb}"
EOF

    sbatch "${job_file}"
    rm "${job_file}"
done
else
    echo "Skipping celebrity evaluation because submit_celeb is set to false."
fi

if [[ "${submit_coco}" == true ]]; then
    echo "Submitting COCO evaluation..."

    coco_result_dir="${results_root}/coco"
    coco_images_dir="${coco_result_dir}/images"
    coco_metrics_dir="${coco_result_dir}/metrics"
    coco_job_file="$(mktemp "/tmp/sdxl_coco_XXXXXX.sh")"

    cat > "${coco_job_file}" <<EOF
#!/bin/bash
#SBATCH --account=${CUIG_SLURM_ACCOUNT}
#SBATCH --job-name=Independent-Celebrity-Base-SDXL-coco
#SBATCH --time=15:00:00
#SBATCH --cluster=${CUIG_SLURM_CLUSTER}
#SBATCH --partition=${CUIG_SLURM_PARTITION}
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --output=${logs_root}/StableDiffusionXL_coco_%j.out
#SBATCH --error=${logs_root}/StableDiffusionXL_coco_%j.err

# Script Settings
module --ignore_cache load cuda/12.4.1
source ~/.bashrc
set -euo pipefail

echo "Independent Celebrity Base Stable Diffusion XL COCO evaluation starting..."

# User must set
REPO_ROOT="${REPO_ROOT}"
OUTPUT_ROOT="${OUTPUT_ROOT}"

# Run the local script that activates the conda env and sets private environment variables.
if [[ -n "${CUIG_PRIVATE_EXPORTS:-}" && -f "${CUIG_PRIVATE_EXPORTS}" ]]; then
    source "${CUIG_PRIVATE_EXPORTS}"
fi

# Evaluation settings
eval_dir="${eval_dir}"
pipeline_dir="${pipeline_dir}"
coco_images_dir="${coco_images_dir}"
coco_metrics_dir="${coco_metrics_dir}"
coco_num_prompts=${coco_num_prompts}

conda activate cuig
mkdir -p "\${coco_images_dir}" "\${coco_metrics_dir}"

cd "\${eval_dir}/coco"
python sample_coco.py \
    --model_family sdxl \
    --model_name "base" \
    --pipeline_dir "\${pipeline_dir}" \
    --resolution 1024 \
    --output_dir "\${coco_images_dir}" \
    --num_prompts "\${coco_num_prompts}"

python evaluate_coco.py \
    --input_dir "\${coco_images_dir}" \
    --output_dir "\${coco_metrics_dir}"

echo "Completed Independent Celebrity Base Stable Diffusion XL COCO evaluation."
EOF

    sbatch "${coco_job_file}"
    rm "${coco_job_file}"
else
    echo "Skipping COCO evaluation because submit_coco is set to false."
fi

echo "Independent Celebrity Base Stable Diffusion XL evaluation submitted!"
