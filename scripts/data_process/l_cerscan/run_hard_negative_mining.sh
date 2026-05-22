#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "sam2" ]]; then
    echo "run_hard_negative_mining.sh must be run in the sam2 conda environment." >&2
    exit 1
fi

cd "$(dirname "$0")/../../.."

RUN_DIR="${1:-work_dir/mlc/smartccs/fb_pos_gate/neg1000}"
CURRENT_ANN_FILE="${2:-annofiles/multilabel_puretrain_neg1000.json}"
OUTPUT_ANN_FILE="${3:-annofiles/multilabel_puretrain_neg1750.json}"
NUM_CLUSTERS="${4:-50}"
SAMPLES_PER_CLUSTER="${5:-10}"

GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
TEST_PORT="${TEST_PORT:-12347}"
MINING_PORT="${MINING_PORT:-12349}"
VAL_JSON="${VAL_JSON:-annofiles/multilabel_puretrain.json}"
TOTAL_ANN_FILE="${TOTAL_ANN_FILE:-annofiles/multilabel_puretrain.json}"
FP_THRESHOLD="${FP_THRESHOLD:-0.5}"

CONFIG_FILE="${RUN_DIR}/config.py"
CKPT_FILE="${RUN_DIR}/checkpoints/best.pth"
PRED_RESULT="${RUN_DIR}/pred_result.pkl"
FP_FILE="${RUN_DIR}/filter_FP.json"
CLUSTER_VIS_FILE="${RUN_DIR}/clusters_${NUM_CLUSTERS}.jpg"

CUDA_VISIBLE_DEVICES="${GPU_LIST}" torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${TEST_PORT}" \
    tools/test_PatchNet.py \
    "${CONFIG_FILE}" \
    "${CKPT_FILE}" \
    "${RUN_DIR}" \
    --val_json "${VAL_JSON}" \
    --save_result

python scripts/data_process/l_cerscan/filter_FP.py \
    --pred-result "${PRED_RESULT}" \
    --output-file "${FP_FILE}" \
    --threshold "${FP_THRESHOLD}"

CUDA_VISIBLE_DEVICES="${GPU_LIST}" torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${MINING_PORT}" \
    scripts/data_process/l_cerscan/neg_mining.py \
    --candidate-ann-file "${FP_FILE}" \
    --total-ann-file "${TOTAL_ANN_FILE}" \
    --current-ann-file "${CURRENT_ANN_FILE}" \
    --output-file "${OUTPUT_ANN_FILE}" \
    --num-clusters "${NUM_CLUSTERS}" \
    --samples-per-cluster "${SAMPLES_PER_CLUSTER}" \
    --visualize \
    --vis-output-file "${CLUSTER_VIS_FILE}"
