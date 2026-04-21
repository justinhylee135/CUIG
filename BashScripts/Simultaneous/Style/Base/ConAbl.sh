#!/bin/bash

set -euo pipefail

echo "Simultaneous Style Unlearning with Base ConAbl starting..."

# User must set
REPO_ROOT="${REPO_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs}"

# Unlearning Method Agnostic
base_model_dir="${REPO_ROOT}/Checkpoints/Generators/UnlearnCanvas"
eval_classifier_dir="${REPO_ROOT}/Checkpoints/Classifiers/UnlearnCanvas"

# Select Unlearning Method and Evaluation Method
train_dir="${REPO_ROOT}/UnlearningMethods/ConAbl"
eval_dir="${REPO_ROOT}/Evaluation/UnlearnCanvas"

# ConAbl Specific
accelerate_config="${REPO_ROOT}/Configs/Accelerator/single_gpu.yaml"
anchor_dataset_dir="${REPO_ROOT}/UnlearningMethods/ConAbl/anchor_datasets/style/Laion"
anchor_prompt_path="${REPO_ROOT}/UnlearningMethods/ConAbl/anchor_prompts/style/Laion.txt"
models_root="${OUTPUT_ROOT}/Simultaneous/Style/Base/ConAbl/Models"
results_root="${OUTPUT_ROOT}/Simultaneous/Style/Base/ConAbl/Results"
logs_root="${REPO_ROOT}/logs/Simultaneous/Style/Base"

array_to_json() {
    local json="["
    local item

    for item in "$@"; do
        json+="\"${item}\","
    done

    printf '%s' "${json%,}]"
}

# Define the list of styles to unlearn simultaneously
unlearn_styles=("Abstractionism" "Byzantine" "Cartoon" "Cold_Warm" "Ukiyoe" "Van_Gogh" "Neon_Lines" "Picasso" "On_Fire" "Magic_Cube" "Winter" "Vibrant_Flow")

# Define held-out styles and objects to measure retention performance
retain_styles=("Blossom_Season" "Rust" "Crayon" "Fauvism" "Superstring" "Red_Blue_Ink" "Gorgeous_Love" "French" "Joy" "Greenfield" "Expressionism" "Impressionism")
retain_objects=("Architectures" "Butterfly" "Flame" "Flowers" "Horses" "Human" "Sea" "Trees")

retain_styles_json="$(array_to_json "${retain_styles[@]}")"
retain_objects_json="$(array_to_json "${retain_objects[@]}")"

# Keep track of the cumulative styles to unlearn
unlearned=()

# Submit one simultaneous job per cumulative style prefix
for style in "${unlearn_styles[@]}"; do
    unlearned+=("${style}")
    unlearned_json="$(array_to_json "${unlearned[@]}")"
    inner_unlearned=${unlearned_json:1:-1}
    styles_subset_json="[${inner_unlearned}, ${retain_styles_json:1}"

    echo "Submitting cumulative style set through: ${style}"
    echo "Styles Unlearned so far: ${unlearned_json}"
    echo "Styles subset for sampling: ${styles_subset_json}"

    output_dir="${models_root}/thru${style}"
    result_dir="${results_root}/thru${style}"
    sample_output_dir="${result_dir}/images"
    metrics_output_dir="${result_dir}/metrics"
    job_file="$(mktemp "/tmp/conabl_simultaneous_style_${style}_XXXXXX.sh")"

    cat > "${job_file}" <<EOF
#!/bin/bash
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --job-name=Simultaneous-Style-Base-ConAbl-thru${style}
#SBATCH --time=15:00:00
#SBATCH --cluster=ascend
#SBATCH --partition=nextgen
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --output=${logs_root}/ConAbl_thru${style}_%j.out
#SBATCH --error=${logs_root}/ConAbl_thru${style}_%j.err

# Script settings
source ~/.bashrc
set -euo pipefail

echo "Simultaneous Style Unlearning with Base ConAbl starting through style: ${style}..."

# User must set
REPO_ROOT="${REPO_ROOT}"
OUTPUT_ROOT="${OUTPUT_ROOT}"

# Run (your own) script that activates the conda env and sets env variables
source "\${REPO_ROOT}/private_exports.sh"

# Unlearning Method Agnostic
base_model_dir="${base_model_dir}"
eval_classifier_dir="${eval_classifier_dir}"

# Select Unlearning Method and Evaluation Method
train_dir="${train_dir}"
eval_dir="${eval_dir}"

# ConAbl Specific
accelerate_config="${accelerate_config}"
anchor_dataset_dir="${anchor_dataset_dir}"
anchor_prompt_path="${anchor_prompt_path}"
output_dir="${output_dir}"
result_dir="${result_dir}"
sample_output_dir="${sample_output_dir}"
metrics_output_dir="${metrics_output_dir}"
unlearned_json='${unlearned_json}'
retain_styles_json='${retain_styles_json}'
retain_objects_json='${retain_objects_json}'
styles_subset_json='${styles_subset_json}'

# TRAIN: Unlearn the cumulative style set from the base model
cd "\${train_dir}"
train_args=(
    --anchor_target_concepts "\${unlearned_json}"
    --concept_type "style"
    --output_dir "\${output_dir}"
    --base_model_dir "\${base_model_dir}"
    --iterations 4000
    --num_anchor_images 200
    --num_anchor_prompts 200
    --anchor_dataset_dirs "\${anchor_dataset_dir}"
    --anchor_prompt_paths "\${anchor_prompt_path}"
    --scale_lr
    --hflip
    --noaug
    --enable_xformers_memory_efficient_attention
    --eval_interval 100
    --patience 1000
    --eval_classifier_dir "\${eval_classifier_dir}"
    --overwrite_existing_ckpt
)

accelerate launch \
    --config_file "\${accelerate_config}" \
    train_conabl.py \
    "\${train_args[@]}"

# SAMPLE: Generate images from the current simultaneously unlearned model
cd "\${eval_dir}"
python sample.py \
    --unet_ckpt_path "\${output_dir}/delta.bin" \
    --output_dir "\${sample_output_dir}" \
    --styles_subset "\${styles_subset_json}" \
    --objects_subset "\${retain_objects_json}" \
    --pipeline_dir "\${base_model_dir}"

# EVALUATE: Run evaluation on the generated images.
cd "\${eval_dir}"
python evaluate.py \
    --input_dir "\${sample_output_dir}" \
    --output_dir "\${metrics_output_dir}" \
    --eval_classifier_dir "\${eval_classifier_dir}" \
    --unlearn "\${unlearned_json}" \
    --retain "\${retain_styles_json}" \
    --cross_retain "\${retain_objects_json}"

echo "Completed Unlearning and Sampling through Style: ${style}"
EOF

    sbatch "${job_file}"
    rm "${job_file}"
done

echo "Simultaneous Style Unlearning with Base ConAbl submission finished!"
