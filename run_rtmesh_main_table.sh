#!/usr/bin/env bash
# Reproduce the 12 RT-Mesh entries in the main table on one GPU.
# One model/evaluation per split produces P1, P2, P3, and ALL metrics.
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

BASE_CONFIG="${BASE_CONFIG:-${ROOT_DIR}/config.yaml}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/home/rtx3090/miniconda3/envs/m4human/bin/torchrun}"
S1_CHECKPOINT="${S1_CHECKPOINT:-${ROOT_DIR}/experiments/exp_20260918_200225/best_model_epoch19.pth}"
TRAIN_S1="${TRAIN_S1:-0}"

CONFIG_DIR="${ROOT_DIR}/configs/main_table"
LOG_DIR="${ROOT_DIR}/logs/main_table"
RUN_INDEX="${ROOT_DIR}/experiments/rtmesh_main_table_runs.tsv"

mkdir -p "${CONFIG_DIR}" "${LOG_DIR}" "${ROOT_DIR}/experiments"

if [[ ! -f "${BASE_CONFIG}" ]]; then
    echo "Base config not found: ${BASE_CONFIG}" >&2
    exit 1
fi
if [[ ! -x "${TORCHRUN_BIN}" ]]; then
    echo "torchrun not found or not executable: ${TORCHRUN_BIN}" >&2
    exit 1
fi
if ! grep -q '"vertex_mean_1_35"' "${ROOT_DIR}/sources/evaluation_module_pc_multigpu.py"; then
    echo "Official action grouping is missing from the evaluator." >&2
    echo "Expected P1=1-35, P2=36-40, P3=41-50." >&2
    exit 1
fi

make_config() {
    local split="$1"
    local test_mode="$2"
    local checkpoint="$3"
    local output="$4"

    cp "${BASE_CONFIG}" "${output}"
    sed -i -E "s|^  split:.*$|  split: '${split}'  # s1=random, s2=cross-subject, s3=cross-action|" "${output}"
    sed -i -E "s|^  scale:.*$|  scale: 'p1'  # full dataset|" "${output}"
    sed -i -E "s|^  resume_path:.*$|  resume_path: ''|" "${output}"
    sed -i -E "s|^  test_mode:.*$|  test_mode: ${test_mode}|" "${output}"
    sed -i -E "s|^  test_model_path:.*$|  test_model_path: '${checkpoint}'|" "${output}"
    sed -i -E "s|^  plot_gif:.*$|  plot_gif: false|" "${output}"
    sed -i -E "s|^  plot_static:.*$|  plot_static: false|" "${output}"
}

latest_experiment_after() {
    local marker="$1"
    find "${ROOT_DIR}/experiments" -mindepth 2 -maxdepth 2 \
        -type f -name train_test.log -newer "${marker}" \
        -printf '%T@ %h\n' | sort -nr | head -n 1 | cut -d' ' -f2-
}

run_job() {
    local label="$1"
    local config_path="$2"
    local marker
    local status
    local experiment

    marker="$(mktemp /tmp/m4human-main-table-marker.XXXXXX)"
    echo
    echo "[$(date '+%F %T')] Starting ${label}"
    echo "Config: ${config_path}"

    set +e
    env \
        MPLBACKEND=Agg \
        MPLCONFIGDIR="/tmp/m4human-mpl-${label}" \
        CUDA_VISIBLE_DEVICES=0 \
        "${TORCHRUN_BIN}" \
        --standalone \
        --nproc_per_node=1 \
        "${ROOT_DIR}/main1_multigpu_clean.py" \
        --config "${config_path}" \
        2>&1 | tee "${LOG_DIR}/${label}.log"
    status=${PIPESTATUS[0]}
    set -e

    experiment="$(latest_experiment_after "${marker}")"
    rm -f "${marker}"

    if [[ ${status} -ne 0 ]]; then
        echo "${label} failed with exit code ${status}." >&2
        exit "${status}"
    fi
    if [[ -z "${experiment}" || ! -f "${experiment}/results_best.csv" ]]; then
        echo "Cannot locate results_best.csv for ${label}." >&2
        exit 1
    fi

    printf '%s\t%s\n' "${label}" "${experiment}" >> "${RUN_INDEX}"
    echo "[$(date '+%F %T')] Finished ${label}: ${experiment}"
}

print_result() {
    local label="$1"
    local experiment="$2"
    awk -F, -v label="${label}" '
        $1 == "avg_vertex_error" {
            printf "%-8s P1=%8.4f  P2=%8.4f  P3=%8.4f  ALL=%8.4f mm\n", \
                   label, $(NF-3), $(NF-2), $(NF-1), $NF
        }
    ' "${experiment}/results_best.csv"
}

: > "${RUN_INDEX}"

if [[ "${TRAIN_S1}" == "1" ]]; then
    make_config s1 false "" "${CONFIG_DIR}/rtmesh_s1_train.yaml"
    run_job s1 "${CONFIG_DIR}/rtmesh_s1_train.yaml"
else
    if [[ ! -f "${S1_CHECKPOINT}" ]]; then
        echo "S1 checkpoint not found: ${S1_CHECKPOINT}" >&2
        echo "Set TRAIN_S1=1 to train S1 from scratch." >&2
        exit 1
    fi
    make_config s1 true "${S1_CHECKPOINT}" "${CONFIG_DIR}/rtmesh_s1_test.yaml"
    run_job s1 "${CONFIG_DIR}/rtmesh_s1_test.yaml"
fi

make_config s2 false "" "${CONFIG_DIR}/rtmesh_s2_train.yaml"
run_job s2 "${CONFIG_DIR}/rtmesh_s2_train.yaml"

make_config s3 false "" "${CONFIG_DIR}/rtmesh_s3_train.yaml"
run_job s3 "${CONFIG_DIR}/rtmesh_s3_train.yaml"

echo
echo "RT-Mesh main-table MVE summary"
echo "-----------------------------------------------"
while IFS=$'\t' read -r label experiment; do
    print_result "${label}" "${experiment}"
done < "${RUN_INDEX}"
echo "-----------------------------------------------"
echo "Run index: ${RUN_INDEX}"
