#!/bin/bash

set -euo pipefail

echo "Independent Style Unlearning with Base SculpMem starting..."

# Shared configuration
_cuig_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "${_cuig_script_dir}/BashScripts/submit.sh" && "${_cuig_script_dir}" != "/" ]]; do
    _cuig_script_dir="$(dirname "${_cuig_script_dir}")"
done
source "${_cuig_script_dir}/BashScripts/submit.sh"
unset _cuig_script_dir

# Unlearning Method Agnostic
base_model_dir="${CUIG_UNLEARNCANVAS_GENERATOR_DIR}"
eval_classifier_dir="${CUIG_UNLEARNCANVAS_CLASSIFIER_DIR}"

# Select Unlearning Method and Evaluation Method
train_dir="${REPO_ROOT}/UnlearningMethods/SculpMem"
eval_dir="${REPO_ROOT}/Evaluation/UnlearnCanvas"

# SculpMem Specific
accelerate_config="${REPO_ROOT}/Configs/Accelerator/single_gpu.yaml"
anchor_dataset_dir="${REPO_ROOT}/UnlearningMethods/SculpMem/anchor_datasets/style/Laion"
anchor_prompt_path="${REPO_ROOT}/UnlearningMethods/SculpMem/anchor_prompts/style/Laion.txt"
models_root="${OUTPUT_ROOT}/Independent/Style/Base/SculpMem/Models"
results_root="${OUTPUT_ROOT}/Independent/Style/Base/SculpMem/Results"
logs_root="${REPO_ROOT}/logs/Independent/Style/Base"

array_to_json() {
    local json="["
    local item

    for item in "$@"; do
        json+="\"${item}\","
    done

    printf '%s' "${json%,}]"
}

# Define the list of styles to unlearn independently
unlearn_styles=("Abstractionism" "Byzantine" "Cartoon" "Cold_Warm" "Ukiyoe" "Van_Gogh" "Neon_Lines" "Picasso" "On_Fire" "Magic_Cube" "Winter" "Vibrant_Flow")

# Define held-out styles and objects to measure retention performance
retain_styles=("Blossom_Season" "Rust" "Crayon" "Fauvism" "Superstring" "Red_Blue_Ink" "Gorgeous_Love" "French" "Joy" "Greenfield" "Expressionism" "Impressionism")
retain_objects=("Architectures" "Butterfly" "Flame" "Flowers" "Horses" "Human" "Sea" "Trees")

retain_styles_json="$(array_to_json "${retain_styles[@]}")"
retain_objects_json="$(array_to_json "${retain_objects[@]}")"

mkdir -p "${logs_root}/SculpMem"

# Submit one independent job per style
for style in "${unlearn_styles[@]}"; do
    echo "Submitting style: ${style}"

    output_dir="${models_root}/${style}"
    result_dir="${results_root}/${style}"
    sample_output_dir="${result_dir}/images"
    metrics_output_dir="${result_dir}/metrics"
    unlearn_json="$(array_to_json "${style}")"
    styles_subset_json="$(array_to_json "${style}" "${retain_styles[@]}")"
    job_file="$(mktemp "/tmp/sculpmem_independent_style_${style}_XXXXXX.sh")"

    cat > "${job_file}" <<EOF
#!/bin/bash
#SBATCH --account=${CUIG_SLURM_ACCOUNT}
#SBATCH --job-name=Independent-Style-Base-SculpMem-${style}
#SBATCH --time=02:00:00
#SBATCH --cluster=${CUIG_SLURM_CLUSTER}
#SBATCH --partition=${CUIG_SLURM_PARTITION}
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --output=${logs_root}/SculpMem/${style}_%j.out
#SBATCH --error=${logs_root}/SculpMem/${style}_%j.err

# Script Settings
source ~/.bashrc
set -euo pipefail

echo "Independent Style Unlearning with Base SculpMem starting for style: ${style}..."

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

# SculpMem Specific
accelerate_config="${accelerate_config}"
anchor_dataset_dir="${anchor_dataset_dir}"
anchor_prompt_path="${anchor_prompt_path}"
output_dir="${output_dir}"
result_dir="${result_dir}"
sample_output_dir="${sample_output_dir}"
metrics_output_dir="${metrics_output_dir}"
unlearn_json='${unlearn_json}'
retain_styles_json='${retain_styles_json}'
retain_objects_json='${retain_objects_json}'
styles_subset_json='${styles_subset_json}'

# TRAIN: Unlearn the current style from the base model
cd "\${train_dir}"
train_args=(
    --anchor_target_concepts "${style}"
    --concept_type "style"
    --output_dir "\${output_dir}"
    --base_model_dir "\${base_model_dir}"
    --iterations 1000
    --num_anchor_images 200
    --num_anchor_prompts 200
    --anchor_dataset_dirs "\${anchor_dataset_dir}"
    --anchor_prompt_paths "\${anchor_prompt_path}"
    --scale_lr
    --hflip
    --noaug
    --enable_xformers_memory_efficient_attention
        --selft_dynamic
        --selft_topk 0.50
)

accelerate launch \
    --config_file "\${accelerate_config}" \
    train_sculpmem.py \
    "\${train_args[@]}"

# SAMPLE: Generate images from the independently unlearned model
cd "\${eval_dir}"
python sample.py \
    --unet_ckpt_path "\${output_dir}/delta.bin" \
    --output_dir "\${sample_output_dir}" \
    --styles_subset "\${styles_subset_json}" \
    --objects_subset "\${retain_objects_json}" \
    --pipeline_dir "\${base_model_dir}"

# EVALUATE: Run evaluation on the generated images
cd "\${eval_dir}"
python evaluate.py \
    --input_dir "\${sample_output_dir}" \
    --output_dir "\${metrics_output_dir}" \
    --eval_classifier_dir "\${eval_classifier_dir}" \
    --unlearn "\${unlearn_json}" \
    --retain "\${retain_styles_json}" \
    --cross_retain "\${retain_objects_json}"

echo "Completed Unlearning and Sampling for Style: ${style}"
EOF

    sbatch "${job_file}"
    rm "${job_file}"
done

echo "Independent Style Unlearning with Base SculpMem finished!"
