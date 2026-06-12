#!/bin/bash

set -euo pipefail

echo "Independent Object Unlearning with Base SculpMem starting..."

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
anchor_datasets_root="${REPO_ROOT}/UnlearningMethods/SculpMem/anchor_datasets/object"
anchor_prompts_root="${REPO_ROOT}/UnlearningMethods/SculpMem/anchor_prompts/object"
models_root="${OUTPUT_ROOT}/Independent/Object/Base/SculpMem/Models"
results_root="${OUTPUT_ROOT}/Independent/Object/Base/SculpMem/Results"
logs_root="${REPO_ROOT}/logs/Independent/Object/Base"
iterations=2000

array_to_json() {
    local json="["
    local item

    for item in "$@"; do
        json+="\"${item}\","
    done

    printf '%s' "${json%,}]"
}

# Define the list of objects to unlearn independently
unlearn_objects=("Bears" "Birds" "Cats" "Dogs" "Fishes" "Frogs" "Jellyfish" "Rabbits" "Sandwiches" "Statues" "Towers" "Waterfalls")


# Define held-out objects and styles to measure retention performance
retain_objects=("Architectures" "Butterfly" "Flame" "Flowers" "Horses" "Human" "Sea" "Trees")
retain_styles=("Blossom_Season" "Rust" "Crayon" "Fauvism" "Superstring" "Red_Blue_Ink" "Gorgeous_Love" "French" "Joy" "Greenfield" "Expressionism" "Impressionism")

# Map each target object to the anchor concept and singular text form used by SculpMem.
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

mkdir -p "${logs_root}/SculpMem"

# Submit one independent job per object
for object in "${unlearn_objects[@]}"; do
    echo "Submitting object: ${object}"

    anchor="${object_anchor[$object]:-}"
    object_name="${object_name_map[$object]:-}"
    anchor_name="${anchor_name_map[$anchor]:-}"
    if [[ -z "${anchor}" || -z "${object_name}" || -z "${anchor_name}" ]]; then
        echo "Missing mapping for object: ${object}" >&2
        exit 1
    fi

    # Define the new checkpoint path for the resulting UNet after unlearning object
    anchor_dataset_dir="${anchor_datasets_root}/${anchor}"
    anchor_prompt_path="${anchor_prompts_root}/${anchor}.txt"
    output_dir="${models_root}/${object}"
    result_dir="${results_root}/${object}"
    sample_output_dir="${result_dir}/images"
    metrics_output_dir="${result_dir}/metrics"
    unlearn_json="$(array_to_json "${object}")"
    objects_subset_json="$(array_to_json "${object}" "${retain_objects[@]}")"
    job_file="$(mktemp "/tmp/sculpmem_independent_object_${object}_XXXXXX.sh")"

    cat > "${job_file}" <<EOF
#!/bin/bash
#SBATCH --account=${CUIG_SLURM_ACCOUNT}
#SBATCH --job-name=Independent-Object-Base-SculpMem-${object}
#SBATCH --time=02:00:00
#SBATCH --cluster=${CUIG_SLURM_CLUSTER}
#SBATCH --partition=${CUIG_SLURM_PARTITION}
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --output=${logs_root}/SculpMem/${object}_%j.out
#SBATCH --error=${logs_root}/SculpMem/${object}_%j.err

# Script Settings
source ~/.bashrc
set -euo pipefail

echo "Independent Object Unlearning with Base SculpMem starting for object: ${object}..."

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
sample_output_dir="${sample_output_dir}"
metrics_output_dir="${metrics_output_dir}"
unlearn_json='${unlearn_json}'
retain_objects_json='${retain_objects_json}'
retain_styles_json='${retain_styles_json}'
objects_subset_json='${objects_subset_json}'

# TRAIN: Unlearn the current object from the base model
cd "\${train_dir}"
train_args=(
    --anchor_target_concepts "${anchor_name}+${object_name}"
    --concept_type "object"
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
    --objects_subset "\${objects_subset_json}" \
    --styles_subset "\${retain_styles_json}" \
    --pipeline_dir "\${base_model_dir}"

# EVALUATE: Run evaluation on the generated images
python evaluate.py \
    --input_dir "\${sample_output_dir}" \
    --output_dir "\${metrics_output_dir}" \
    --eval_classifier_dir "\${eval_classifier_dir}" \
    --unlearn "\${unlearn_json}" \
    --retain "\${retain_objects_json}" \
    --cross_retain "\${retain_styles_json}"

echo "Completed Unlearning and Sampling for Object: ${object}"
EOF

    sbatch "${job_file}"
    rm "${job_file}"
done

echo "Independent Object Unlearning with Base SculpMem finished!"
