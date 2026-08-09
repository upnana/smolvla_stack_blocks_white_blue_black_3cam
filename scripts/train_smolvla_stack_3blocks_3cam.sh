#!/bin/bash
# =============================================================================
# SmolVLA 训练（3 相机新库 stack 3 blocks: white → blue → black）
# Usage:
#   conda activate lerobot
#   bash train_smolvla_stack_3blocks_3cam.sh              # 默认双卡
#   BACKGROUND=1 bash train_smolvla_stack_3blocks_3cam.sh # 后台
#   RESUME=1 bash train_smolvla_stack_3blocks_3cam.sh
#   FRESH=1 bash train_smolvla_stack_3blocks_3cam.sh
#   tail -f logs/smolvla_stack_3blocks_white_blue_black_3cam.log
#
# 与旧 2cam 库 / 旧 output 分开，勿混用。
# =============================================================================
set -euo pipefail

PROJECT_ROOT="/home/rxn/lerobot"
DATA_ROOT="${DATA_ROOT:-/home/rxn/datasets/stack_3blocks_white_blue_black_3cam}"
DATA_REPO="${DATA_REPO:-my_pick_place/stack_3blocks_white_blue_black_3cam}"
BASE_MODEL="${BASE_MODEL:-/home/rxn/.cache/modelscope/models/lerobot--smolvla_base/snapshots/master}"
VLM_MODEL="${VLM_MODEL:-/home/rxn/.cache/modelscope/models/HuggingFaceTB--SmolVLM2-500M-Video-Instruct/snapshots/master}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/train/smolvla_stack_3blocks_white_blue_black_3cam}"
CONDA_ENV="${CONDA_ENV:-lerobot}"
JOB_NAME="${JOB_NAME:-smolvla_stack_3blocks_white_blue_black_3cam}"

# front/wrist/side -> camera1/2/3；真三路，不要 empty_cameras
RENAME_MAP='{"observation.images.front":"observation.images.camera1","observation.images.wrist":"observation.images.camera2","observation.images.side":"observation.images.camera3"}'

NUM_GPUS="${NUM_GPUS:-2}"
BATCH_SIZE="${BATCH_SIZE:-8}"  # per-GPU; effective = BATCH_SIZE * NUM_GPUS
NUM_WORKERS="${NUM_WORKERS:-4}"
# ~79251 frames @ effective bs=16 → ~4953 steps/epoch；~20 epoch ≈ 100000
STEPS="${STEPS:-100000}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
LOG_FREQ="${LOG_FREQ:-200}"
TOLERANCE_S="${TOLERANCE_S:-0.05}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
MIN_FREE_MIB="${MIN_FREE_MIB:-8000}"
LOG_FILE="${LOG_FILE:-${PROJECT_ROOT}/logs/${JOB_NAME}.log}"
PID_FILE="${PID_FILE:-${PROJECT_ROOT}/logs/${JOB_NAME}.pid}"

if [ "${BACKGROUND:-0}" = "1" ] && [ -z "${TRAIN_SMOLVLA_STACK3_3CAM_BG:-}" ]; then
    mkdir -p "${PROJECT_ROOT}/logs"
    if [ -f "${PID_FILE}" ] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
        echo "Already running PID $(cat "${PID_FILE}")"
        echo "Log: ${LOG_FILE}"
        exit 1
    fi
    echo "Starting nohup background training..."
    echo "  log: ${LOG_FILE}"
    nohup env TRAIN_SMOLVLA_STACK3_3CAM_BG=1 \
        FRESH="${FRESH:-0}" RESUME="${RESUME:-0}" \
        STEPS="${STEPS}" NUM_GPUS="${NUM_GPUS}" BATCH_SIZE="${BATCH_SIZE}" \
        SAVE_FREQ="${SAVE_FREQ}" LOG_FREQ="${LOG_FREQ}" NUM_WORKERS="${NUM_WORKERS}" \
        TOLERANCE_S="${TOLERANCE_S}" \
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
        OUTPUT_DIR="${OUTPUT_DIR}" JOB_NAME="${JOB_NAME}" \
        DATA_ROOT="${DATA_ROOT}" DATA_REPO="${DATA_REPO}" \
        bash "$0" > "${LOG_FILE}" 2>&1 &
    echo $! > "${PID_FILE}"
    echo "  pid: $(cat "${PID_FILE}")"
    echo "  tail -f ${LOG_FILE}"
    exit 0
fi

echo "=============================="
echo "SmolVLA stack_3blocks 3cam Training"
echo "=============================="

source /home/rxn/miniconda3/etc/profile.d/conda.sh
if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
    echo "ERROR: conda env '${CONDA_ENV}' not found."
    exit 1
fi
conda activate "${CONDA_ENV}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export ACCELERATE_DISABLE_RICH=1
export ACCELERATE_USE_META=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES

for path in "${DATA_ROOT}" "${BASE_MODEL}"; do
    [ -d "${path}" ] || { echo "ERROR: not found: ${path}"; exit 1; }
done
[ -f "${DATA_ROOT}/meta/info.json" ] || {
    echo "ERROR: invalid dataset (missing meta/info.json): ${DATA_ROOT}"
    exit 1
}
[ -f "${VLM_MODEL}/config.json" ] || {
    echo "ERROR: VLM not found: ${VLM_MODEL}"
    exit 1
}

echo "Dataset:  ${DATA_ROOT} (${DATA_REPO})"
echo "Model:    ${BASE_MODEL}"
echo "VLM:      ${VLM_MODEL}"
echo "Output:   ${OUTPUT_DIR}"
echo "Cameras:  front/wrist/side -> camera1/2/3 (empty_cameras=0)"
echo "GPUs:     ${NUM_GPUS} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
echo "Batch:    ${BATCH_SIZE} per GPU (effective: $((BATCH_SIZE * NUM_GPUS)))"
echo "Steps:    ${STEPS}"
echo "Tolerance:${TOLERANCE_S}s"
echo ""

if command -v nvidia-smi >/dev/null 2>&1; then
    IFS=',' read -ra GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
    if [ "${NUM_GPUS}" -gt "${#GPU_IDS[@]}" ]; then
        echo "ERROR: NUM_GPUS=${NUM_GPUS} but CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} has ${#GPU_IDS[@]} GPU(s)"
        exit 1
    fi
    for ((i=0; i<NUM_GPUS; i++)); do
        GPU_IDX="${GPU_IDS[$i]}"
        FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${GPU_IDX}" | tr -d ' ')
        echo "  GPU ${GPU_IDX}: free ${FREE_MIB} MiB (need >= ${MIN_FREE_MIB} MiB)"
        if [ "${FREE_MIB}" -lt "${MIN_FREE_MIB}" ]; then
            echo "ERROR: GPU ${GPU_IDX} OOM headroom too low."
            exit 1
        fi
    done
    echo ""
fi

python - <<PY
import json
from pathlib import Path
info = json.loads(Path("${DATA_ROOT}/meta/info.json").read_text())
print(f"Episodes: {info['total_episodes']}, Frames: {info['total_frames']}")
bs = ${BATCH_SIZE} * ${NUM_GPUS}
print(f"Approx epochs at step ${STEPS}: ${STEPS} * {bs} / {info['total_frames']:.0f} = {${STEPS} * bs / info['total_frames']:.1f}")
cams = [k for k in info.get("features", {}) if "images" in k]
print("Image keys:", cams)
PY

python - <<PY
import json
from pathlib import Path
vlm = "${VLM_MODEL}"
prep_path = Path("${BASE_MODEL}") / "policy_preprocessor.json"
cfg = json.loads(prep_path.read_text())
for step in cfg.get("steps", []):
    if step.get("registry_name") == "tokenizer_processor":
        if step["config"].get("tokenizer_name") != vlm:
            step["config"]["tokenizer_name"] = vlm
            prep_path.write_text(json.dumps(cfg, indent=2) + "\n")
            print(f"Patched tokenizer_name -> {vlm}")
        break
PY

RESUME="${RESUME:-0}"
CONFIG_PATH=""
if [ "${FRESH:-0}" = "1" ] && [ -d "${OUTPUT_DIR}" ]; then
    echo "FRESH=1: removing ${OUTPUT_DIR}"
    rm -rf "${OUTPUT_DIR}"
elif [ "${RESUME}" = "1" ]; then
    CONFIG_PATH="${OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json"
    [ -f "${CONFIG_PATH}" ] || { echo "ERROR: missing ${CONFIG_PATH}"; exit 1; }
    echo "Resume from: ${CONFIG_PATH}"
elif [ -d "${OUTPUT_DIR}" ]; then
    echo "ERROR: ${OUTPUT_DIR} exists. Use FRESH=1 or RESUME=1"
    exit 1
fi
echo ""

ACCEL_ARGS=(--num_processes="${NUM_GPUS}" --num_machines=1 --mixed_precision=bf16 --dynamo_backend=no)
[ "${NUM_GPUS}" -gt 1 ] && ACCEL_ARGS+=(--multi_gpu --main_process_port="${MAIN_PROCESS_PORT:-29507}")

POLICY_ARGS=(
  --policy.device=cuda
  --policy.vlm_model_name="${VLM_MODEL}"
  --policy.load_vlm_weights=false
  --policy.empty_cameras=0
)
if [ "${RESUME}" != "1" ]; then
  POLICY_ARGS=(--policy.path="${BASE_MODEL}" "${POLICY_ARGS[@]}")
fi

accelerate launch "${ACCEL_ARGS[@]}" \
  "$(which lerobot-train)" \
  "${POLICY_ARGS[@]}" \
  --dataset.root="${DATA_ROOT}" \
  --dataset.repo_id="${DATA_REPO}" \
  --tolerance_s="${TOLERANCE_S}" \
  --rename_map="${RENAME_MAP}" \
  --batch_size="${BATCH_SIZE}" \
  --num_workers="${NUM_WORKERS}" \
  --steps="${STEPS}" \
  --save_freq="${SAVE_FREQ}" \
  --log_freq="${LOG_FREQ}" \
  --eval_freq=-1 \
  --seed=1000 \
  $([ "${RESUME}" = "1" ] && echo "--resume=true --config_path=${CONFIG_PATH}") \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${JOB_NAME}" \
  --wandb.enable=false \
  --policy.push_to_hub=false

echo "=============================="
echo "Training finished: ${OUTPUT_DIR}"
echo "Infer:"
echo "  bash infer_smolvla_stack_3blocks_3cam.sh offline"
echo "  bash infer_smolvla_stack_3blocks_3cam.sh robot"
echo "=============================="
