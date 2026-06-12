#!/bin/bash

set -euo pipefail

echo "Independent Style Merge TIES with SculpMem starting..."

# Shared configuration
_cuig_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "${_cuig_script_dir}/BashScripts/submit.sh" && "${_cuig_script_dir}" != "/" ]]; do
    _cuig_script_dir="$(dirname "${_cuig_script_dir}")"
done
source "${_cuig_script_dir}/BashScripts/submit.sh"
unset _cuig_script_dir

# Shared paths
base_model_dir="${CUIG_UNLEARNCANVAS_GENERATOR_DIR}"
eval_classifier_dir="${CUIG_UNLEARNCANVAS_CLASSIFIER_DIR}"
merge_dir="${REPO_ROOT}/Regularizers/Merge"
eval_dir="${REPO_ROOT}/Evaluation/UnlearnCanvas"
source_models_root="${OUTPUT_ROOT}/Independent/Style/Base/SculpMem/Models"
models_root="${OUTPUT_ROOT}/Independent/Style/Merge/TIES/SculpMem/Models"
results_root="${OUTPUT_ROOT}/Independent/Style/Merge/TIES/SculpMem/Results"
logs_root="${REPO_ROOT}/logs/Independent/Style/Merge/TIES"

# Independent checkpoints to merge
merge_styles=("Abstractionism" "Byzantine" "Cartoon" "Cold_Warm" "Ukiyoe" "Van_Gogh" "Neon_Lines" "Picasso" "On_Fire" "Magic_Cube" "Winter" "Vibrant_Flow")

# Held-out controls for sampling/evaluation
retain_styles=("Blossom_Season" "Rust" "Crayon" "Fauvism" "Superstring" "Red_Blue_Ink" "Gorgeous_Love" "French" "Joy" "Greenfield" "Expressionism" "Impressionism")
retain_objects=("Architectures" "Butterfly" "Flame" "Flowers" "Horses" "Human" "Sea" "Trees")

# Per-style TIES hyperparameters (best settings we found while maintaing UA as close to 99% as possible) for each cumulative merge endpoint.
# In our original experiments we did a hyperparameter sweep with lambda: [1.25, 1.75, 2.25, 2.75, 3.25] and topk: [0.20, 0.40, 0.60, 0.80]
declare -A style_ties_lambda=(
    ["Byzantine"]="1.25"
    ["Cartoon"]="1.25"
    ["Cold_Warm"]="1.75"
    ["Ukiyoe"]="2.25"
    ["Van_Gogh"]="2.25"
    ["Neon_Lines"]="2.25"
    ["Picasso"]="2.25"
    ["On_Fire"]="2.75"
    ["Magic_Cube"]="2.75"
    ["Winter"]="3.25"
    ["Vibrant_Flow"]="3.25"
)
declare -A style_ties_top_k=(
    ["Byzantine"]="0.80"
    ["Cartoon"]="0.40"
    ["Cold_Warm"]="0.40"
    ["Ukiyoe"]="0.40"
    ["Van_Gogh"]="0.40"
    ["Neon_Lines"]="0.40"
    ["Picasso"]="0.40"
    ["On_Fire"]="0.60"
    ["Magic_Cube"]="0.60"
    ["Winter"]="0.20"
    ["Vibrant_Flow"]="0.20"
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

format_decimal() {
    printf "%.2f" "$1"
}

retain_styles_json="$(array_to_json "${retain_styles[@]}")"
retain_objects_json="$(array_to_json "${retain_objects[@]}")"

cumulative_styles=()

for style in "${merge_styles[@]}"; do
    cumulative_styles+=("${style}")

    if [[ -z "${style_ties_lambda[$style]+x}" ]]; then
        echo "Skipping thru${style}; no style-specific TIES hyperparameters configured."
        continue
    fi

    ties_lambda="${style_ties_lambda[$style]}"
    ties_top_k="${style_ties_top_k[$style]}"
    lambda_name="$(format_decimal "${ties_lambda}")"
    topk_name="$(format_decimal "${ties_top_k}")"
    experiment_tag="lambda${lambda_name}_topk${topk_name}"
    job_logs_root="${logs_root}/${experiment_tag}"
    merged_ckpt_path="${models_root}/${experiment_tag}/thru${style}.pth"
    result_dir="${results_root}/${experiment_tag}/thru${style}"
    sample_output_dir="${result_dir}/images"
    metrics_output_dir="${result_dir}/metrics"
    job_file="$(mktemp "/tmp/ties_independent_style_thru${style}_${experiment_tag}_XXXXXX.sh")"
    merge_styles_json="$(array_to_json "${cumulative_styles[@]}")"
    styles_subset_json="$(array_to_json "${cumulative_styles[@]}" "${retain_styles[@]}")"

    checkpoint_paths=()
    for checkpoint_style in "${cumulative_styles[@]}"; do
        checkpoint_paths+=("${source_models_root}/${checkpoint_style}/delta.bin")
    done
    checkpoints_json="$(array_to_json "${checkpoint_paths[@]}")"

    mkdir -p "${job_logs_root}/SculpMem"

    if [[ -d "${metrics_output_dir}" ]]; then
        echo "Skipping thru${style}; metrics already exist at ${metrics_output_dir}"
        rm "${job_file}"
        continue
    fi

    echo "Submitting Merge TIES for SculpMem thru${style}: lambda=${lambda_name}, topk=${topk_name}"

    cat > "${job_file}" <<EOF
#!/bin/bash
#SBATCH --account=${CUIG_SLURM_ACCOUNT}
#SBATCH --job-name=Independent-Style-Merge-TIES-SculpMem-thru${style}-${experiment_tag}
#SBATCH --time=01:00:00
#SBATCH --cluster=${CUIG_SLURM_CLUSTER}
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --ntasks-per-node=1
#SBATCH --output=${job_logs_root}/SculpMem/%j.out
#SBATCH --error=${job_logs_root}/SculpMem/%j.err

source ~/.bashrc
set -euo pipefail

echo "Independent Style TIES merge starting for thru${style} with ${experiment_tag}..."

REPO_ROOT="${REPO_ROOT}"
OUTPUT_ROOT="${OUTPUT_ROOT}"
base_model_dir="${base_model_dir}"
eval_classifier_dir="${eval_classifier_dir}"
merge_dir="${merge_dir}"
eval_dir="${eval_dir}"
merged_ckpt_path="${merged_ckpt_path}"
sample_output_dir="${sample_output_dir}"
metrics_output_dir="${metrics_output_dir}"
merge_styles_json='${merge_styles_json}'
retain_styles_json='${retain_styles_json}'
retain_objects_json='${retain_objects_json}'
styles_subset_json='${styles_subset_json}'
checkpoints_json='${checkpoints_json}'
key_filter_json='${key_filter_json}'

if [[ -n "${CUIG_PRIVATE_EXPORTS:-}" && -f "${CUIG_PRIVATE_EXPORTS}" ]]; then
    source "${CUIG_PRIVATE_EXPORTS}"
fi

# MERGE: Build one merged checkpoint from the independent SculpMem deltas.
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

# SAMPLE: Generate images from the merged model.
cd "\${eval_dir}"
python sample.py \
    --unet_ckpt_path "\${merged_ckpt_path}" \
    --output_dir "\${sample_output_dir}" \
    --styles_subset "\${styles_subset_json}" \
    --objects_subset "\${retain_objects_json}" \
    --pipeline_dir "\${base_model_dir}"

# EVALUATE: Score the merged model on the sampled images.
python evaluate.py \
    --input_dir "\${sample_output_dir}" \
    --output_dir "\${metrics_output_dir}" \
    --eval_classifier_dir "\${eval_classifier_dir}" \
    --unlearn "\${merge_styles_json}" \
    --retain "\${retain_styles_json}" \
    --cross_retain "\${retain_objects_json}"

echo "Completed TIES merge for thru${style} with ${experiment_tag}"
EOF

    sbatch "${job_file}"
    rm "${job_file}"
done

echo "Independent Style TIES merge submission finished!"
