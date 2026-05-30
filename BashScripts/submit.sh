#!/bin/bash

cuig_load_config() {
    if [[ -n "${CUIG_CONFIG_LOADED:-}" ]]; then
        return 0
    fi
    CUIG_CONFIG_LOADED=1

    local bashscripts_dir
    bashscripts_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    if [[ -z "${CUIG_REPO_ROOT:-}" ]]; then
        CUIG_REPO_ROOT="$(cd "${bashscripts_dir}/.." && pwd)"
    fi

    local config_path="${CUIG_BASHSCRIPTS_CONFIG:-${CUIG_REPO_ROOT}/BashScripts/config.sh}"
    if [[ -f "${config_path}" ]]; then
        source "${config_path}"
    fi

    local missing=()
    local var
    for var in \
        CUIG_REPO_ROOT \
        CUIG_OUTPUT_ROOT \
        CUIG_SLURM_ACCOUNT \
        CUIG_SLURM_CLUSTER \
        CUIG_SLURM_PARTITION
    do
        if [[ -z "${!var:-}" ]]; then
            missing+=("${var}")
        fi
    done

    if [[ "${#missing[@]}" -gt 0 ]]; then
        printf 'Missing CUIG BashScripts configuration: %s\n' "${missing[*]}" >&2
        printf 'Copy BashScripts/config.example.sh to BashScripts/config.sh and fill in your local values.\n' >&2
        return 1
    fi

    export CUIG_REPO_ROOT
    export CUIG_OUTPUT_ROOT
    export CUIG_SLURM_ACCOUNT
    export CUIG_SLURM_CLUSTER
    export CUIG_SLURM_PARTITION

    export REPO_ROOT="${CUIG_REPO_ROOT}"
    export OUTPUT_ROOT="${CUIG_OUTPUT_ROOT}"

    export CUIG_PRIVATE_EXPORTS="${CUIG_PRIVATE_EXPORTS:-}"
    export CUIG_UNLEARNCANVAS_GENERATOR_DIR="${CUIG_UNLEARNCANVAS_GENERATOR_DIR:-${REPO_ROOT}/Checkpoints/Generators/UnlearnCanvas}"
    export CUIG_UNLEARNCANVAS_CLASSIFIER_DIR="${CUIG_UNLEARNCANVAS_CLASSIFIER_DIR:-${REPO_ROOT}/Checkpoints/Classifiers/UnlearnCanvas}"
    export CUIG_CELEBRITY_BASE_MODEL_DIR="${CUIG_CELEBRITY_BASE_MODEL_DIR:-CompVis/stable-diffusion-v1-4}"

    if [[ -n "${CUIG_PRIVATE_EXPORTS}" && -f "${CUIG_PRIVATE_EXPORTS}" ]]; then
        source "${CUIG_PRIVATE_EXPORTS}"
    fi
}

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    cuig_load_config
    return $?
fi

set -euo pipefail

usage() {
    cat <<EOF
Usage: $(basename "$0") BashScripts/path/to/script.sh [sbatch args...]

Submit a direct SLURM script using the local settings in BashScripts/config.sh.
Additional arguments are passed to sbatch before the script path.
EOF
}

if [[ "$#" -lt 1 ]]; then
    usage >&2
    exit 1
fi

submit_script="$1"
shift

cuig_load_config

if [[ ! -f "${submit_script}" ]]; then
    printf 'Script not found: %s\n' "${submit_script}" >&2
    exit 1
fi

script_abs="$(cd "$(dirname "${submit_script}")" && pwd)/$(basename "${submit_script}")"
script_rel="${script_abs#${REPO_ROOT}/BashScripts/}"
if [[ "${script_rel}" == "${script_abs}" ]]; then
    printf 'Script must live under %s/BashScripts: %s\n' "${REPO_ROOT}" "${script_abs}" >&2
    exit 1
fi

log_stem="${script_rel%.sh}"
log_dir="${REPO_ROOT}/logs/${log_stem}"
mkdir -p "${log_dir}"

sbatch \
    --account="${CUIG_SLURM_ACCOUNT}" \
    --cluster="${CUIG_SLURM_CLUSTER}" \
    --partition="${CUIG_SLURM_PARTITION}" \
    --output="${log_dir}/%j.out" \
    --error="${log_dir}/%j.err" \
    "$@" \
    "${script_abs}"
