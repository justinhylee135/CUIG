#!/bin/bash

set -euo pipefail

echo "Simultaneous Object Unlearning with Base ConAbl starting..."

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
train_dir="${REPO_ROOT}/UnlearningMethods/ConAbl"
eval_dir="${REPO_ROOT}/Evaluation/UnlearnCanvas"

# ConAbl Specific
accelerate_config="${REPO_ROOT}/Configs/Accelerator/single_gpu.yaml"
anchor_datasets_root="${REPO_ROOT}/UnlearningMethods/ConAbl/anchor_datasets/object"
anchor_prompts_root="${REPO_ROOT}/UnlearningMethods/ConAbl/anchor_prompts/object"
models_root="${OUTPUT_ROOT}/Simultaneous/Object/Base/ConAbl/Models"
results_root="${OUTPUT_ROOT}/Simultaneous/Object/Base/ConAbl/Results"
logs_root="${REPO_ROOT}/logs/Simultaneous/Object/Base"
iterations=4000
eval_interval=100
patience=1000

array_to_json() {
    local json="["
    local item

    for item in "$@"; do
        json+="\"${item}\","
    done

    printf '%s' "${json%,}]"
}

# Define the list of objects to unlearn simultaneously
unlearn_objects=("Bears" "Birds" "Cats" "Dogs" "Fishes" "Frogs" "Jellyfish" "Rabbits" "Sandwiches" "Statues" "Towers" "Waterfalls")

# Define held-out objects and styles to measure retention performance
retain_objects=("Architectures" "Butterfly" "Flame" "Flowers" "Horses" "Human" "Sea" "Trees")
retain_styles=("Blossom_Season" "Rust" "Crayon" "Fauvism" "Superstring" "Red_Blue_Ink" "Gorgeous_Love" "French" "Joy" "Greenfield" "Expressionism" "Impressionism")

# Map each target object to the anchor concept and singular text form used by ConAbl.
declare -A object_anchor=(
    ["Bears"]="Horses"
    ["Birds"]="Butterfly"
    ["Cats"]="Horses"
    ["Dogs"]="Horses"
    ["Fishes"]="Butterfly"
    ["Frogs"]="Butterfly"
    ["Jellyfish"]="Flowers"
    ["Rabbits"]="Horses"
    ["Sandwiches"]="Flowers"
    ["Statues"]="Trees"
    ["Towers"]="Trees"
    ["Waterfalls"]="Trees"
)
declare -A object_name_map=(
    ["Bears"]="bear"
    ["Birds"]="bird"
    ["Cats"]="cat"
    ["Dogs"]="dog"
    ["Fishes"]="fish"
    ["Frogs"]="frog"
    ["Jellyfish"]="jellyfish"
    ["Rabbits"]="rabbit"
    ["Sandwiches"]="sandwich"
    ["Statues"]="statue"
    ["Towers"]="tower"
    ["Waterfalls"]="waterfall"
)
declare -A anchor_name_map=(
    ["Butterfly"]="butterfly"
    ["Flowers"]="flower"
    ["Horses"]="horse"
    ["Trees"]="tree"
)

retain_objects_json="$(array_to_json "${retain_objects[@]}")"
retain_styles_json="$(array_to_json "${retain_styles[@]}")"

mkdir -p "${logs_root}/ConAbl"

# Keep track of the cumulative objects to unlearn
unlearned=()
unlearned_targets=()
unlearned_anchor_dataset_dirs=()
unlearned_anchor_prompt_paths=()

# Submit one simultaneous job per cumulative object prefix
for object in "${unlearn_objects[@]}"; do
    anchor="${object_anchor[$object]:-}"
    object_name="${object_name_map[$object]:-}"
    anchor_name="${anchor_name_map[$anchor]:-}"
    if [[ -z "${anchor}" || -z "${object_name}" || -z "${anchor_name}" ]]; then
        echo "Missing mapping for object: ${object}" >&2
        exit 1
    fi

    unlearned+=("${object}")
    unlearned_targets+=("${anchor_name}+${object_name}")
    unlearned_anchor_dataset_dirs+=("${anchor_datasets_root}/${anchor}")
    unlearned_anchor_prompt_paths+=("${anchor_prompts_root}/${anchor}.txt")
    unlearned_json="$(array_to_json "${unlearned[@]}")"
    unlearned_target_json="$(array_to_json "${unlearned_targets[@]}")"
    unlearned_anchor_dataset_dirs_json="$(array_to_json "${unlearned_anchor_dataset_dirs[@]}")"
    unlearned_anchor_prompt_paths_json="$(array_to_json "${unlearned_anchor_prompt_paths[@]}")"
    inner_unlearned=${unlearned_json:1:-1}
    objects_subset_json="[${inner_unlearned}, ${retain_objects_json:1}"

    echo "Submitting cumulative object set through: ${object}"
    echo "Objects Unlearned so far: ${unlearned_json}"
    echo "Objects subset for sampling: ${objects_subset_json}"

    output_dir="${models_root}/thru${object}"
    result_dir="${results_root}/thru${object}"
    sample_output_dir="${result_dir}/images"
    metrics_output_dir="${result_dir}/metrics"
    job_file="$(mktemp "/tmp/conabl_simultaneous_object_${object}_XXXXXX.sh")"

    cat > "${job_file}" <<EOF
#!/bin/bash
#SBATCH --account=${CUIG_SLURM_ACCOUNT}
#SBATCH --job-name=Simultaneous-Object-Base-ConAbl-thru${object}
#SBATCH --time=30:00:00
#SBATCH --cluster=${CUIG_SLURM_CLUSTER}
#SBATCH --partition=${CUIG_SLURM_PARTITION}
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --output=${logs_root}/ConAbl/thru${object}_%j.out
#SBATCH --error=${logs_root}/ConAbl/thru${object}_%j.err

# Script settings
source ~/.bashrc
set -euo pipefail

echo "Simultaneous Object Unlearning with Base ConAbl starting through object: ${object}..."

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
iterations=${iterations}
eval_interval=${eval_interval}
patience=${patience}
unlearned_json='${unlearned_json}'
unlearned_target_json='${unlearned_target_json}'
unlearned_anchor_dataset_dirs_json='${unlearned_anchor_dataset_dirs_json}'
unlearned_anchor_prompt_paths_json='${unlearned_anchor_prompt_paths_json}'
retain_objects_json='${retain_objects_json}'
retain_styles_json='${retain_styles_json}'
objects_subset_json='${objects_subset_json}'

# TRAIN: Unlearn the cumulative object set from the base model
cd "\${train_dir}"
train_args=(
    --anchor_target_concepts "\${unlearned_target_json}"
    --concept_type "object"
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
    --overwrite_existing_ckpt
)

accelerate launch \
    --config_file "\${accelerate_config}" \
    train_conabl.py \
    "\${train_args[@]}"

# Select the first sampled training checkpoint whose UnlearnCanvas UA clears the UA threshold.
if ! selected_checkpoint_info="\$(OUTPUT_DIR="\${output_dir}" EVAL_INTERVAL="\${eval_interval}" ITERATIONS="\${iterations}" python - <<'PY'
import json
import os
import sys

output_dir = os.environ["OUTPUT_DIR"]
eval_interval = int(os.environ["EVAL_INTERVAL"])
iterations = int(os.environ["ITERATIONS"])
threshold = 0.9900

for log_num in range(eval_interval, iterations + 1, eval_interval):
    summary_path = os.path.join(output_dir, "logs", f"log_{log_num}", "metrics", "summary.json")
    if not os.path.isfile(summary_path):
        continue

    with open(summary_path, "r", encoding="utf-8") as handle:
        raw_ua = float(json.load(handle).get("UA", 0.0))
    ua = raw_ua / 100.0 if raw_ua > 1.0 else raw_ua

    if ua > threshold:
        ckpt_path = os.path.join(output_dir, "delta.bin")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Selected metrics file {summary_path} but checkpoint is missing: {ckpt_path}")
        print(f"{log_num} {ckpt_path}")
        sys.exit(0)

sys.exit(1)
PY
)"; then
    echo "No sampled training checkpoint under \${output_dir}/logs has UA > 0.9900." >&2
    exit 1
fi
read -r selected_log_num selected_ckpt <<< "\${selected_checkpoint_info}"
sample_output_dir="\${sample_output_dir}-log_\${selected_log_num}"
metrics_output_dir="\${metrics_output_dir}-log_\${selected_log_num}"
echo "Selected sampled training checkpoint for downstream sampling: \${selected_ckpt}"
echo "Writing downstream outputs with suffix: -log_\${selected_log_num}"

# SAMPLE: Generate images from the current simultaneously unlearned model
cd "\${eval_dir}"
python sample.py \
    --unet_ckpt_path "\${selected_ckpt}" \
    --output_dir "\${sample_output_dir}" \
    --objects_subset "\${objects_subset_json}" \
    --styles_subset "\${retain_styles_json}" \
    --pipeline_dir "\${base_model_dir}"

# EVALUATE: Run evaluation on the generated images.
python evaluate.py \
    --input_dir "\${sample_output_dir}" \
    --output_dir "\${metrics_output_dir}" \
    --eval_classifier_dir "\${eval_classifier_dir}" \
    --unlearn "\${unlearned_json}" \
    --retain "\${retain_objects_json}" \
    --cross_retain "\${retain_styles_json}"

echo "Completed Unlearning and Sampling through Object: ${object}"
EOF

    sbatch "${job_file}"
    rm "${job_file}"
done

echo "Simultaneous Object Unlearning with Base ConAbl submission finished!"
