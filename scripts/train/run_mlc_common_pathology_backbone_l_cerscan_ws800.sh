#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "sam2" ]]; then
    echo "run_mlc_common_pathology_backbone_l_cerscan_ws800.sh must be run in the sam2 conda environment." >&2
    exit 1
fi

cd "$(dirname "$0")/../.."

DATASET_NAME="l_cerscan_ws800"
DATASET_BASE_CONFIG="../../../configs/dataset/l_cerscan_ws800.py"
OUTPUT_ROOT="work_dir/mlc/common_pathology_backbone/${DATASET_NAME}"
TMP_CONFIG_DIR="work_dir/tmp/mlc_common_pathology_backbone_${DATASET_NAME}_configs"
GPU_LIST="0,1,2,3,4,5,6,7"
NPROC_PER_NODE="8"

mkdir -p "${TMP_CONFIG_DIR}"

classifier_config_file() {
    local classifier="$1"

    if [[ "${classifier}" == "query2label" ]]; then
        echo "../../../configs/model/query2label.py"
    elif [[ "${classifier}" == "ml_decoder" ]]; then
        echo "../../../configs/model/ml_decoder.py"
    elif [[ "${classifier}" == "mlc_nc" ]]; then
        echo "../../../configs/model/mlc_nc.py"
    elif [[ "${classifier}" == "wscer_mlc" ]]; then
        echo "../../../configs/model/wscernet.py"
    else
        echo "Unsupported classifier: ${classifier}" >&2
        exit 1
    fi
}

write_dataset_config() {
    local classifier="$1"
    local backbone="$2"
    local batch_size="$3"
    local input_size="$4"
    local config_file="${TMP_CONFIG_DIR}/dataset_${classifier}_${backbone}.py"

    cat > "${config_file}" <<EOF
_base_ = [
    '${DATASET_BASE_CONFIG}',
]

train_bs = ${batch_size}
val_bs = ${batch_size}
input_size = ${input_size}

train_datasets = _base_.train_datasets.copy()
train_datasets['pipeline'][1]['scale'] = (input_size, input_size)

val_datasets = _base_.val_datasets.copy()
val_datasets['pipeline'][1]['scale'] = (input_size, input_size)
EOF
}

write_model_config() {
    local classifier="$1"
    local backbone="$2"
    local config_file="${TMP_CONFIG_DIR}/model_${classifier}_${backbone}.py"
    local classifier_config
    classifier_config="$(classifier_config_file "${classifier}")"

    cat > "${config_file}" <<EOF
_base_ = [
    '${classifier_config}',
]

net_type = 'patch'

backbone_type = '${backbone}'
backbone_cfg = _base_.backbone_cfgdict[backbone_type]
backbone_cfg['frozen_backbone'] = True
backbone_cfg['use_peft'] = None

positive_thr = 0.5
eval_prime_score = 'multi-label/f1-score'
EOF
}

write_strategy_config() {
    local classifier="$1"
    local backbone="$2"
    local batch_size="$3"
    local input_size="$4"
    local config_file="${TMP_CONFIG_DIR}/strategy_${classifier}_${backbone}.py"

    cat > "${config_file}" <<EOF
_base_ = [
    '../../../configs/strategy_patch.py',
]

save_each_epoch = False
logger_name = 'mlc_common_pathology_backbone_${DATASET_NAME}_${classifier}_${backbone}_bs${batch_size}_in${input_size}'
EOF
}

run_train() {
    local classifier="$1"
    local backbone="$2"
    local batch_size="$3"
    local input_size="$4"
    local port="$5"
    local dataset_config="${TMP_CONFIG_DIR}/dataset_${classifier}_${backbone}.py"
    local model_config="${TMP_CONFIG_DIR}/model_${classifier}_${backbone}.py"
    local strategy_config="${TMP_CONFIG_DIR}/strategy_${classifier}_${backbone}.py"
    local save_dir="${OUTPUT_ROOT}/${classifier}/${backbone}"

    write_dataset_config "${classifier}" "${backbone}" "${batch_size}" "${input_size}"
    write_model_config "${classifier}" "${backbone}"
    write_strategy_config "${classifier}" "${backbone}" "${batch_size}" "${input_size}"
    mkdir -p "${save_dir}"

    echo "Start MLC training: dataset=${DATASET_NAME}, classifier=${classifier}, backbone=${backbone}, bs=${batch_size}, input_size=${input_size}, save_each_epoch=False, port=${port}, save_dir=${save_dir}"
    CUDA_VISIBLE_DEVICES="${GPU_LIST}" torchrun \
        --nproc_per_node="${NPROC_PER_NODE}" \
        --master_port="${port}" \
        main4PatchNet.py \
        "${dataset_config}" \
        "${model_config}" \
        "${strategy_config}" \
        --record_save_dir "${save_dir}"
}

run_train "query2label" "dinov2" "128" "224" "12720"
run_train "query2label" "virchow2" "64" "224" "12721"
run_train "query2label" "smartccs" "128" "224" "12722"
run_train "query2label" "fusionnet" "128" "1024" "12723"
run_train "ml_decoder" "dinov2" "128" "224" "12724"
run_train "ml_decoder" "virchow2" "64" "224" "12725"
run_train "ml_decoder" "smartccs" "128" "224" "12726"
run_train "ml_decoder" "fusionnet" "128" "1024" "12727"
run_train "mlc_nc" "dinov2" "128" "224" "12728"
run_train "mlc_nc" "virchow2" "64" "224" "12729"
run_train "mlc_nc" "smartccs" "128" "224" "12730"
run_train "mlc_nc" "fusionnet" "128" "1024" "12731"
run_train "wscer_mlc" "dinov2" "128" "224" "12732"
run_train "wscer_mlc" "virchow2" "64" "224" "12733"
run_train "wscer_mlc" "smartccs" "128" "224" "12734"
run_train "wscer_mlc" "fusionnet" "128" "1024" "12735"
