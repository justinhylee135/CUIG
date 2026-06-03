#!/bin/bash

set -euo pipefail

echo "Simultaneous Celebrity Unlearning with Base ConAbl starting..."

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
eval_classifier_dir="${eval_dir}/celeb-detection-oss"

# ConAbl Specific
accelerate_config="${REPO_ROOT}/Configs/Accelerator/single_gpu.yaml"
anchor_datasets_root="${REPO_ROOT}/UnlearningMethods/ConAbl/anchor_datasets/celebrity"
anchor_prompts_root="${REPO_ROOT}/UnlearningMethods/ConAbl/anchor_prompts/celebrity"
models_root="${OUTPUT_ROOT}/Simultaneous/Celebrity/Base/ConAbl/Models"
results_root="${OUTPUT_ROOT}/Simultaneous/Celebrity/Base/ConAbl/Results"
logs_root="${REPO_ROOT}/logs/Simultaneous/Celebrity/Base"
prompt_dir="${eval_dir}/Prompts"
prompt_locks_dir="${prompt_dir}/.locks"
eval_interval=100
patience=20000
num_prompts=50
num_seeds=1

# Set to false to skip COCO retention checks for each simultaneous checkpoint.
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

# Define the list of celebrities to unlearn simultaneously.
unlearn_celebrities=("Neil_Degrasse_Tyson" "Benicio_Del_Toro" "Aziz_Ansari" "Oprah_Winfrey" "Betty_White" "Megan_Fox")

# Per-cumulative-endpoint training budgets and SLURM wall times.
declare -A celebrity_iterations=(
    ["Neil_Degrasse_Tyson"]="3000"
    ["Benicio_Del_Toro"]="4000"
    ["Aziz_Ansari"]="4000"
    ["Oprah_Winfrey"]="6000"
    ["Betty_White"]="8000"
    ["Megan_Fox"]="12000"
)
declare -A celebrity_time_limit=(
    ["Neil_Degrasse_Tyson"]="06:00:00"
    ["Benicio_Del_Toro"]="11:00:00"
    ["Aziz_Ansari"]="12:00:00"
    ["Oprah_Winfrey"]="13:00:00"
    ["Betty_White"]="40:00:00"
    ["Megan_Fox"]="45:00:00"
)

# Define held-out celebrities to measure retention performance.
retain_celebrities=("Morgan_Freeman" "Keanu_Reeves" "George_Takei" "Aretha_Franklin" "Maya_Angelou" "Natalie_Portman")

retain_celebrities_json="$(array_to_json "${retain_celebrities[@]}")"

mkdir -p "${logs_root}/ConAbl" "${prompt_dir}" "${prompt_locks_dir}"

# Keep track of the cumulative celebrities to unlearn.
unlearned=()
unlearned_targets=()
unlearned_anchor_dataset_dirs=()
unlearned_anchor_prompt_paths=()

# Submit one simultaneous job per cumulative celebrity prefix.
for celeb in "${unlearn_celebrities[@]}"; do
    anchor="man"
    if [[ "${celeb}" == "Oprah_Winfrey" || "${celeb}" == "Betty_White" || "${celeb}" == "Megan_Fox" ]]; then
        anchor="woman"
    fi

    celeb_name="${celeb//_/ }"
    unlearned+=("${celeb}")
    unlearned_targets+=("${anchor}+${celeb_name}")
    unlearned_anchor_dataset_dirs+=("${anchor_datasets_root}/${anchor}")
    unlearned_anchor_prompt_paths+=("${anchor_prompts_root}/${anchor}_prompts.txt")

    unlearned_json="$(array_to_json "${unlearned[@]}")"
    unlearned_target_json="$(array_to_json "${unlearned_targets[@]}")"
    unlearned_anchor_dataset_dirs_json="$(array_to_json "${unlearned_anchor_dataset_dirs[@]}")"
    unlearned_anchor_prompt_paths_json="$(array_to_json "${unlearned_anchor_prompt_paths[@]}")"
    celebrities_subset_json="$(array_to_json "${unlearned[@]}" "${retain_celebrities[@]}")"
    sample_celebrities_shell_words="$(array_to_shell_words "${unlearned[@]}" "${retain_celebrities[@]}")"
    job_iterations="${celebrity_iterations[$celeb]:-}"
    job_time_limit="${celebrity_time_limit[$celeb]:-}"
    if [[ -z "${job_iterations}" || -z "${job_time_limit}" ]]; then
        echo "Missing iterations or time-limit mapping for celebrity: ${celeb}" >&2
        exit 1
    fi

    echo "Submitting cumulative celebrity set through: ${celeb}"
    echo "Celebrities unlearned so far: ${unlearned_json}"
    echo "Celebrity subset for sampling: ${celebrities_subset_json}"
    echo "Training budget: ${job_iterations} iterations, wall time: ${job_time_limit}"

    output_dir="${models_root}/thru${celeb}"
    result_dir="${results_root}/thru${celeb}"
    sample_output_dir="${result_dir}/images"
    metrics_output_dir="${result_dir}/metrics"
    coco_images_dir="${result_dir}/coco/images"
    coco_metrics_dir="${result_dir}/coco/metrics"
    job_file="$(mktemp "/tmp/conabl_simultaneous_celebrity_${celeb}_XXXXXX.sh")"

    cat > "${job_file}" <<EOF
#!/bin/bash
#SBATCH --account=${CUIG_SLURM_ACCOUNT}
#SBATCH --job-name=Simultaneous-Celebrity-Base-ConAbl-thru${celeb}
#SBATCH --time=${job_time_limit}
#SBATCH --cluster=${CUIG_SLURM_CLUSTER}
#SBATCH --partition=${CUIG_SLURM_PARTITION}
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --output=${logs_root}/ConAbl/thru${celeb}_%j.out
#SBATCH --error=${logs_root}/ConAbl/thru${celeb}_%j.err

# Script Settings
module --ignore_cache load cuda/12.4.1
source ~/.bashrc
set -euo pipefail

echo "Simultaneous Celebrity Unlearning with Base ConAbl starting through celebrity: ${celeb}..."

# User must set
REPO_ROOT="${REPO_ROOT}"
OUTPUT_ROOT="${OUTPUT_ROOT}"

# Run (your own) script that activates the conda env and sets env variables
if [[ -n "${CUIG_PRIVATE_EXPORTS:-}" && -f "${CUIG_PRIVATE_EXPORTS}" ]]; then
    source "${CUIG_PRIVATE_EXPORTS}"
fi

# Unlearning Method Agnostic
base_model_dir="${base_model_dir}"
eval_classifier_dir="${eval_classifier_dir}"

# Select Unlearning Method and Evaluation Method
train_dir="${train_dir}"
eval_dir="${eval_dir}"

# ConAbl Specific
accelerate_config="${accelerate_config}"
output_dir="${output_dir}"
sample_output_dir="${sample_output_dir}"
metrics_output_dir="${metrics_output_dir}"
prompt_dir="${prompt_dir}"
prompt_locks_dir="${prompt_locks_dir}"
coco_images_dir="${coco_images_dir}"
coco_metrics_dir="${coco_metrics_dir}"
num_prompts=${num_prompts}
num_seeds=${num_seeds}
iterations=${job_iterations}
eval_interval=${eval_interval}
patience=${patience}
coco_num_prompts=${coco_num_prompts}
submit_coco=${submit_coco}
unlearned_json='${unlearned_json}'
unlearned_target_json='${unlearned_target_json}'
unlearned_anchor_dataset_dirs_json='${unlearned_anchor_dataset_dirs_json}'
unlearned_anchor_prompt_paths_json='${unlearned_anchor_prompt_paths_json}'
retain_celebrities_json='${retain_celebrities_json}'
celebrities_subset_json='${celebrities_subset_json}'
sample_celebrities=(${sample_celebrities_shell_words})

# PROMPTS: Ensure prompts exist before training-time sampling runs.
conda activate cuig
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

# TRAIN: Unlearn the cumulative celebrity set from the base model.
conda activate cuig
cd "\${train_dir}"
train_args=(
    --anchor_target_concepts "\${unlearned_target_json}"
    --concept_type "celeb"
    --output_dir "\${output_dir}"
    --base_model_dir "\${base_model_dir}"
    --iterations "\${iterations}"
    --num_anchor_images 200
    --num_anchor_prompts 200
    --anchor_dataset_dirs "\${unlearned_anchor_dataset_dirs_json}"
    --anchor_prompt_paths "\${unlearned_anchor_prompt_paths_json}"
    --scale_lr
    --hflip
    --noaug
    --enable_xformers_memory_efficient_attention
    --eval_interval "\${eval_interval}"
    --patience "\${patience}"
    --eval_classifier_dir "\${eval_classifier_dir}"
    --eval_prompt_dir "\${prompt_dir}"
    --overwrite_existing_ckpt
)

accelerate launch \
    --config_file "\${accelerate_config}" \
    train_conabl.py \
    "\${train_args[@]}"

# EVALUATE: Run celebrity classifier evaluation for each sampled training checkpoint.
conda activate celeb_eval
cd "\${eval_dir}/celeb-detection-oss"
for log_num in \$(seq "\${eval_interval}" "\${eval_interval}" "\${iterations}"); do
    log_images_dir="\${output_dir}/logs/log_\${log_num}/images"
    log_metrics_dir="\${output_dir}/logs/log_\${log_num}/metrics"

    if [[ ! -d "\${log_images_dir}" ]]; then
        echo "Skipping celebrity evaluation for missing image directory: \${log_images_dir}"
        continue
    fi

    echo "Evaluating celebrity training log: \${log_num}"
    python examples/evaluate_celeb.py \
        --input_dir "\${log_images_dir}" \
        --output_dir "\${log_metrics_dir}" \
        --unlearn "\${unlearned_json}" \
        --retain "\${retain_celebrities_json}" \
        --num_prompts "\${num_prompts}" \
        --num_seeds "\${num_seeds}"
done

# Select the first sampled training checkpoint whose celebrity evaluation clears the UA threshold.
if ! selected_checkpoint_info="\$(OUTPUT_DIR="\${output_dir}" EVAL_INTERVAL="\${eval_interval}" ITERATIONS="\${iterations}" python - <<'PY'
import json
import os
import sys

output_dir = os.environ["OUTPUT_DIR"]
eval_interval = int(os.environ["EVAL_INTERVAL"])
iterations = int(os.environ["ITERATIONS"])
threshold = 0.9900

for log_num in range(eval_interval, iterations + 1, eval_interval):
    log_dir = os.path.join(output_dir, "logs", f"log_{log_num}")
    metrics_dir = os.path.join(log_dir, "metrics")
    results_path = None
    for filename in ("results.json", "resu.lts.json"):
        candidate = os.path.join(metrics_dir, filename)
        if os.path.isfile(candidate):
            results_path = candidate
            break
    if results_path is None:
        continue

    with open(results_path, "r") as handle:
        avg_ua = float(json.load(handle).get("avg_ua", 0.0))

    if avg_ua > threshold:
        ckpt_path = os.path.join(log_dir, f"{log_num}.ckpt")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Selected metrics file {results_path} but checkpoint is missing: {ckpt_path}")
        print(f"{log_num} {ckpt_path}")
        sys.exit(0)

sys.exit(1)
PY
)"; then
    echo "No sampled training checkpoint under \${output_dir}/logs has avg_ua > 0.9900." >&2
    exit 1
fi
read -r selected_log_num selected_ckpt <<< "\${selected_checkpoint_info}"
sample_output_dir="\${sample_output_dir}-log_\${selected_log_num}"
metrics_output_dir="\${metrics_output_dir}-log_\${selected_log_num}"
coco_images_dir="\${coco_images_dir}-log_\${selected_log_num}"
coco_metrics_dir="\${coco_metrics_dir}-log_\${selected_log_num}"
echo "Selected sampled training checkpoint for downstream sampling: \${selected_ckpt}"
echo "Writing downstream outputs with suffix: -log_\${selected_log_num}"

# SAMPLE: Generate images from the current simultaneously unlearned model.
conda activate cuig
cd "\${eval_dir}"
python sample_celeb.py \
    --model_family sd \
    --ckpt "\${selected_ckpt}" \
    --pipeline_dir "\${base_model_dir}" \
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
    --unlearn "\${unlearned_json}" \
    --retain "\${retain_celebrities_json}" \
    --num_prompts "\${num_prompts}" \
    --num_seeds "\${num_seeds}"

if [[ "\${submit_coco}" == true ]]; then
    echo "Running COCO sample and evaluation for simultaneous celebrity checkpoint through: ${celeb}"

    conda activate cuig
    mkdir -p "\${coco_images_dir}" "\${coco_metrics_dir}"

    cd "\${eval_dir}/coco"
    python sample_coco.py \
        --model_family sd \
        --model_name "\${selected_ckpt}" \
        --pipeline_dir "\${base_model_dir}" \
        --output_dir "\${coco_images_dir}" \
        --num_prompts "\${coco_num_prompts}"

    python evaluate_coco.py \
        --input_dir "\${coco_images_dir}" \
        --output_dir "\${coco_metrics_dir}"
else
    echo "Skipping COCO evaluation because submit_coco is set to false."
fi

echo "Completed Simultaneous Celebrity Unlearning with Base ConAbl through celebrity: ${celeb}"
EOF

    sbatch "${job_file}"
    rm "${job_file}"
done

echo "Simultaneous Celebrity Unlearning with Base ConAbl submission finished!"
