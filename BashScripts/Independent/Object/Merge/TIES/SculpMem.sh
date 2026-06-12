#!/bin/bash

set -euo pipefail

echo "Independent Object Merge TIES with SculpMem starting..."

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
source_models_root="${OUTPUT_ROOT}/Independent/Object/Base/SculpMem/Models"
models_root="${OUTPUT_ROOT}/Independent/Object/Merge/TIES/SculpMem/Models"
results_root="${OUTPUT_ROOT}/Independent/Object/Merge/TIES/SculpMem/Results"
logs_root="${REPO_ROOT}/logs/Independent/Object/Merge/TIES"

# Independent checkpoints to merge
merge_objects=("Bears" "Birds" "Cats" "Dogs" "Fishes" "Frogs" "Jellyfish" "Rabbits" "Sandwiches" "Statues" "Towers" "Waterfalls")

# Held-out controls for sampling/evaluation
retain_objects=("Architectures" "Butterfly" "Flame" "Flowers" "Horses" "Human" "Sea" "Trees")
retain_styles=("Blossom_Season" "Rust" "Crayon" "Fauvism" "Superstring" "Red_Blue_Ink" "Gorgeous_Love" "French" "Joy" "Greenfield" "Expressionism" "Impressionism")

# Per-style TIES hyperparameters (best settings we found while maintaing UA as close to 99% as possible) for each cumulative merge endpoint.
# In our original experiments we did a hyperparameter sweep with lambda: [1.25, 1.75, 2.25, 2.75] and topk: [0.20, 0.40, 0.60, 0.80]
declare -A object_ties_lambda=(
    ["Birds"]="1.25"
    ["Cats"]="1.25"
    ["Dogs"]="1.25"
    ["Fishes"]="1.25"
    ["Frogs"]="1.25"
    ["Jellyfish"]="1.75"
    ["Rabbits"]="1.75"
    ["Sandwiches"]="1.75"
    ["Statues"]="2.25"
    ["Towers"]="2.25"
    ["Waterfalls"]="2.75"
)
declare -A object_ties_top_k=(
    ["Birds"]="0.80"
    ["Cats"]="0.80"
    ["Dogs"]="0.80"
    ["Fishes"]="0.20"
    ["Frogs"]="0.40"
    ["Jellyfish"]="0.60"
    ["Rabbits"]="0.60"
    ["Sandwiches"]="0.40"
    ["Statues"]="0.60"
    ["Towers"]="0.40"
    ["Waterfalls"]="0.80"
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

retain_objects_json="$(array_to_json "${retain_objects[@]}")"
retain_styles_json="$(array_to_json "${retain_styles[@]}")"

cumulative_objects=()

for object in "${merge_objects[@]}"; do
    cumulative_objects+=("${object}")

    if [[ -z "${object_ties_lambda[$object]+x}" ]]; then
        echo "Skipping thru${object}; no object-specific TIES hyperparameters configured."
        continue
    fi

    # Build the checkpoint list and evaluation subsets for the current cumulative object prefix.
    merge_objects_json="$(array_to_json "${cumulative_objects[@]}")"
    objects_subset_json="$(array_to_json "${cumulative_objects[@]}" "${retain_objects[@]}")"
    checkpoint_paths=()
    for checkpoint_object in "${cumulative_objects[@]}"; do
        checkpoint_paths+=("${source_models_root}/${checkpoint_object}/delta.bin")
    done
    checkpoints_json="$(array_to_json "${checkpoint_paths[@]}")"

    ties_lambda="${object_ties_lambda[$object]}"
    ties_top_k="${object_ties_top_k[$object]}"
    lambda_name="$(format_decimal "${ties_lambda}")"
    topk_name="$(format_decimal "${ties_top_k}")"
    experiment_tag="lambda${lambda_name}_topk${topk_name}"
    job_logs_root="${logs_root}/${experiment_tag}"
    merged_ckpt_path="${models_root}/${experiment_tag}/thru${object}.pth"
    result_dir="${results_root}/${experiment_tag}/thru${object}"
    sample_output_dir="${result_dir}/images"
    metrics_output_dir="${result_dir}/metrics"
    job_file="$(mktemp "/tmp/ties_independent_object_thru${object}_${experiment_tag}_XXXXXX.sh")"

    mkdir -p "${job_logs_root}/SculpMem"

    if [[ -d "${metrics_output_dir}" ]]; then
        echo "Skipping thru${object}; metrics already exist at ${metrics_output_dir}"
        rm "${job_file}"
        continue
    fi

    echo "Submitting Merge TIES for SculpMem thru${object}: lambda=${lambda_name}, topk=${topk_name}"

    cat > "${job_file}" <<EOF
#!/bin/bash
#SBATCH --account=${CUIG_SLURM_ACCOUNT}
#SBATCH --job-name=Independent-Object-Merge-TIES-SculpMem-thru${object}-${experiment_tag}
#SBATCH --time=01:30:00
#SBATCH --cluster=${CUIG_SLURM_CLUSTER}
#SBATCH --partition=${CUIG_SLURM_PARTITION}
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --ntasks-per-node=1
#SBATCH --output=${job_logs_root}/SculpMem/%j.out
#SBATCH --error=${job_logs_root}/SculpMem/%j.err

source ~/.bashrc
set -euo pipefail

echo "Independent Object TIES merge starting for thru${object} with ${experiment_tag}..."

REPO_ROOT="${REPO_ROOT}"
OUTPUT_ROOT="${OUTPUT_ROOT}"
base_model_dir="${base_model_dir}"
eval_classifier_dir="${eval_classifier_dir}"
merge_dir="${merge_dir}"
eval_dir="${eval_dir}"
merged_ckpt_path="${merged_ckpt_path}"
sample_output_dir="${sample_output_dir}"
metrics_output_dir="${metrics_output_dir}"
merge_objects_json='${merge_objects_json}'
retain_objects_json='${retain_objects_json}'
retain_styles_json='${retain_styles_json}'
objects_subset_json='${objects_subset_json}'
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
    --objects_subset "\${objects_subset_json}" \
    --styles_subset "\${retain_styles_json}" \
    --pipeline_dir "\${base_model_dir}"

# EVALUATE: Score the merged model on the sampled images.
python evaluate.py \
    --input_dir "\${sample_output_dir}" \
    --output_dir "\${metrics_output_dir}" \
    --eval_classifier_dir "\${eval_classifier_dir}" \
    --unlearn "\${merge_objects_json}" \
    --retain "\${retain_objects_json}" \
    --cross_retain "\${retain_styles_json}"

echo "Completed TIES merge for thru${object} with ${experiment_tag}"
EOF

    sbatch "${job_file}"
    rm "${job_file}"
done

echo "Independent Object Merge TIES submission finished!"
