#!/bin/bash
# =============================================================================
# SmolVLA 推理（3 相机 stack 3 blocks: white → blue → black）
# Usage:
#   bash infer_smolvla_stack_3blocks_3cam.sh offline   # 数据集离线诊断
#   bash infer_smolvla_stack_3blocks_3cam.sh robot     # 真机推理并落盘
#   bash infer_smolvla_stack_3blocks_3cam.sh cameras
#   bash infer_smolvla_stack_3blocks_3cam.sh viz
#   bash infer_smolvla_stack_3blocks_3cam.sh list      # 列出可用 checkpoint
#
# 选不同 step / 约略 epoch（effective bs=16, frames≈79251 → ~4953 steps/epoch）:
#   STEP=50000  bash infer_smolvla_stack_3blocks_3cam.sh robot    # ~10 epoch
#   STEP=80000  bash infer_smolvla_stack_3blocks_3cam.sh robot    # ~16 epoch
#   STEP=100000 bash infer_smolvla_stack_3blocks_3cam.sh robot    # ~20 epoch (last)
#   EPOCH=10    bash infer_smolvla_stack_3blocks_3cam.sh robot
#   CHECKPOINT=.../checkpoints/080000/pretrained_model bash infer_smolvla_stack_3blocks_3cam.sh robot
#
#   EPISODE_TIME_S=180 NUM_EPISODES=1 STEP=100000 bash infer_smolvla_stack_3blocks_3cam.sh robot
#
# 与旧 2cam infer / 旧库分开，勿混用。
# =============================================================================
set -euo pipefail

MODE="${1:-offline}"

PROJECT_ROOT="/home/rxn/lerobot"
DATA_ROOT="${DATA_ROOT:-/home/rxn/datasets/stack_3blocks_white_blue_black_3cam}"
DATA_REPO="${DATA_REPO:-my_pick_place/stack_3blocks_white_blue_black_3cam}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/train/smolvla_stack_3blocks_white_blue_black_3cam}"
CKPT_ROOT="${OUTPUT_DIR}/checkpoints"
# 默认 last；可用 STEP=100000 / EPOCH=20 / CHECKPOINT=路径 覆盖
CHECKPOINT="${CHECKPOINT:-}"
CONDA_ENV="${CONDA_ENV:-lerobot}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
TOLERANCE_S="${TOLERANCE_S:-0.05}"
# 与训练一致：effective batch 16, frames≈79251
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-4953}"

# 禁止冒号：draccus/CLI 会把 "a: b" 拆成 dict
SINGLE_TASK="${SINGLE_TASK:-stack the blocks from bottom to top white then blue then black}"

FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6083854-if00}"
ROBOT_PORT="${ROBOT_PORT:-${FOLLOWER_PORT}}"
ROBOT_ID="${ROBOT_ID:-so101_follower}"
FRONT_CAM="${FRONT_CAM:-/dev/v4l/by-id/usb-UGREEN_Camera_2K_UGREEN_Camera_2K_SN0001-video-index0}"
WRIST_CAM="${WRIST_CAM:-/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0}"
SIDE_CAM="${SIDE_CAM:-/dev/v4l/by-id/usb-Generic_Web_Camera_20250708V1.000-video-index0}"
CAM_WIDTH="${CAM_WIDTH:-640}"
CAM_HEIGHT="${CAM_HEIGHT:-480}"
# Generic Web 硬件只认 800x480
SIDE_CAM_WIDTH="${SIDE_CAM_WIDTH:-800}"
SIDE_CAM_HEIGHT="${SIDE_CAM_HEIGHT:-480}"
CAM_WARMUP_S="${CAM_WARMUP_S:-5}"
WRIST_WARMUP_S="${WRIST_WARMUP_S:-10}"
SIDE_WARMUP_S="${SIDE_WARMUP_S:-8}"

# 与训练一致：front/wrist/side -> camera1/2/3
RENAME_MAP='{"observation.images.front":"observation.images.camera1","observation.images.wrist":"observation.images.camera2","observation.images.side":"observation.images.camera3"}'

FPS=30
NUM_EPISODES="${NUM_EPISODES:-1}"
EPISODE_TIME_S="${EPISODE_TIME_S:-150}"
RESET_TIME_S="${RESET_TIME_S:-15}"
MAX_RELATIVE_TARGET="${MAX_RELATIVE_TARGET:-10}"
PLAY_SOUNDS="${PLAY_SOUNDS:-true}"
DISPLAY_DATA="${DISPLAY_DATA:-true}"
MIN_FREE_MIB="${MIN_FREE_MIB:-6000}"
EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT}/outputs/eval/smolvla_stack_3blocks_white_blue_black_3cam}"
EVAL_REPO_ID="${EVAL_REPO_ID:-local/eval_smolvla_stack_3blocks_white_blue_black_3cam}"
# 每次 infer 导出独立 clip（永不覆盖旧文件）
CLIPS_DIR="${CLIPS_DIR:-${EVAL_ROOT}/clips}"
SAVE_CLIP="${SAVE_CLIP:-1}"

source /home/rxn/miniconda3/etc/profile.d/conda.sh
if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
    echo "ERROR: conda env '${CONDA_ENV}' not found."
    exit 1
fi
conda activate "${CONDA_ENV}"

export HF_HUB_OFFLINE=0
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES
export RERUN_IMAGE_ENTITIES="${RERUN_IMAGE_ENTITIES:-front,wrist,side}"

python -c "from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy" 2>/dev/null || {
    echo "ERROR: SmolVLA not installed. pip install -e \".[smolvla,feetech]\""
    exit 1
}

list_checkpoints() {
    echo "Available checkpoints under ${CKPT_ROOT}:"
    echo "  step       ~epoch   path"
    if [ ! -d "${CKPT_ROOT}" ]; then
        echo "  (none)"
        return
    fi
    for d in $(ls -1 "${CKPT_ROOT}" | grep -E '^[0-9]+$' | sort -n); do
        ep=$(python3 -c "print(f'{int(\"${d}\")/${STEPS_PER_EPOCH}:.1f}')")
        printf "  %-10s ~%-6s %s\n" "${d}" "${ep}" "${CKPT_ROOT}/${d}/pretrained_model"
    done
    if [ -L "${CKPT_ROOT}/last" ] || [ -d "${CKPT_ROOT}/last" ]; then
        tgt=$(readlink -f "${CKPT_ROOT}/last" 2>/dev/null || echo "${CKPT_ROOT}/last")
        echo "  last    -> $(basename "${tgt}")"
    fi
}

if [ "${MODE}" = "list" ]; then
    list_checkpoints
    exit 0
fi

# Resolve CHECKPOINT from STEP / EPOCH / default last
if [ -z "${CHECKPOINT}" ]; then
    if [ -n "${STEP:-}" ]; then
        STEP_PAD=$(printf "%06d" "${STEP}")
        if [ -d "${CKPT_ROOT}/${STEP}/pretrained_model" ]; then
            CHECKPOINT="${CKPT_ROOT}/${STEP}/pretrained_model"
        elif [ -d "${CKPT_ROOT}/${STEP_PAD}/pretrained_model" ]; then
            CHECKPOINT="${CKPT_ROOT}/${STEP_PAD}/pretrained_model"
        else
            echo "ERROR: no checkpoint for STEP=${STEP}"
            list_checkpoints
            exit 1
        fi
    elif [ -n "${EPOCH:-}" ]; then
        TARGET_STEP=$(python3 -c "print(int(round(float('${EPOCH}') * ${STEPS_PER_EPOCH})))")
        BEST=""
        BEST_DIST=999999999
        for d in $(ls -1 "${CKPT_ROOT}" | grep -E '^[0-9]+$' | sort -n); do
            dist=$(python3 -c "print(abs(int('${d}') - ${TARGET_STEP}))")
            if [ "${dist}" -lt "${BEST_DIST}" ]; then
                BEST_DIST="${dist}"
                BEST="${d}"
            fi
        done
        if [ -z "${BEST}" ]; then
            echo "ERROR: no numeric checkpoints found"
            exit 1
        fi
        CHECKPOINT="${CKPT_ROOT}/${BEST}/pretrained_model"
        echo "EPOCH=${EPOCH} -> nearest step ${BEST} (target ~${TARGET_STEP})"
    else
        CHECKPOINT="${CKPT_ROOT}/last/pretrained_model"
    fi
fi

# allow CHECKPOINT=.../last or .../last/pretrained_model
if [ -d "${CHECKPOINT}/pretrained_model" ] && [ ! -f "${CHECKPOINT}/config.json" ]; then
    CHECKPOINT="${CHECKPOINT}/pretrained_model"
fi
if [ ! -f "${CHECKPOINT}/config.json" ]; then
    echo "ERROR: checkpoint not found: ${CHECKPOINT}"
    list_checkpoints
    exit 1
fi

CKPT_DIR="$(dirname "${CHECKPOINT}")"
CHECKPOINT_STEP="$(basename "${CKPT_DIR}")"
if [ "${CHECKPOINT_STEP}" = "last" ] && [ -L "${CKPT_DIR}" ]; then
    CHECKPOINT_STEP="$(basename "$(readlink -f "${CKPT_DIR}")")"
fi

DATASET_TASK="${SINGLE_TASK}"

echo "=============================="
echo "SmolVLA stack_3blocks 3cam Inference (${MODE})"
echo "Checkpoint: ${CHECKPOINT} (step ${CHECKPOINT_STEP})"
echo "Dataset:    ${DATA_ROOT}"
echo "Eval out:   ${EVAL_ROOT}"
echo "Task:       ${DATASET_TASK}"
echo "Cameras:    front/wrist/side -> camera1/2/3"
echo "GPU:        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Duration:   ${NUM_EPISODES} ep × ${EPISODE_TIME_S}s (+ ${RESET_TIME_S}s reset)"
echo "=============================="

if [ "${MODE}" = "cameras" ]; then
    ls -la /dev/v4l/by-id/ 2>/dev/null || true
    echo "Defaults: FRONT_CAM=${FRONT_CAM}"
    echo "          WRIST_CAM=${WRIST_CAM}"
    echo "          SIDE_CAM =${SIDE_CAM} (${SIDE_CAM_WIDTH}x${SIDE_CAM_HEIGHT})"
    lerobot-find-cameras opencv || true
    exit 0
fi

if [ "${MODE}" = "viz" ]; then
    if [ -d "${EVAL_ROOT}" ] && [ -n "$(ls -A "${EVAL_ROOT}" 2>/dev/null)" ]; then
        LATEST_EVAL=$(ls -td "${EVAL_ROOT}"/*/ 2>/dev/null | head -1)
        LATEST_EVAL="${LATEST_EVAL%/}"
        lerobot-dataset-viz \
          --repo-id="${EVAL_REPO_ID}" \
          --root="${LATEST_EVAL}" \
          --episode-index=0 \
          --mode=local
    else
        lerobot-dataset-viz \
          --repo-id="${DATA_REPO}" \
          --root="${DATA_ROOT}" \
          --episode-index=0 \
          --mode=local
    fi
    exit 0
fi

if [ "${MODE}" = "offline" ]; then
    python - <<PY
import json
import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.processor.rename_processor import rename_stats

checkpoint = "${CHECKPOINT}"
data_root = "${DATA_ROOT}"
repo_id = "${DATA_REPO}"
rename_map = json.loads('''${RENAME_MAP}''')

cfg = PreTrainedConfig.from_pretrained(checkpoint)
cfg.pretrained_path = checkpoint
cfg.device = "cuda"

ds = LeRobotDataset(repo_id=repo_id, root=data_root, tolerance_s=${TOLERANCE_S})
policy = make_policy(cfg=cfg, ds_meta=ds.meta, rename_map=rename_map)
preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=cfg,
    pretrained_path=checkpoint,
    dataset_stats=rename_stats(ds.meta.stats, rename_map),
    preprocessor_overrides={
        "rename_observations_processor": {"rename_map": rename_map},
    },
)
policy.eval()

print(f"Dataset frames={len(ds)} episodes={ds.meta.total_episodes}")
img_keys = [k for k in ds.meta.features if "images" in k]
print("Image keys:", img_keys)
print("Running offline inference on 3cam stack_3blocks dataset...")
preds, maes = [], []
# spread frames across ~79k frames
candidates = [0, 1000, 5000, 15000, 30000, 50000, 70000, 79000]
for idx in candidates:
    if idx >= len(ds):
        continue
    policy.reset()
    sample = ds[idx]
    batch = preprocessor({k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else [v] for k, v in sample.items()})
    with torch.inference_mode():
        action = postprocessor(policy.select_action(batch))
    mae = (action.cpu().float() - sample["action"].float()).abs().mean().item()
    p = action.cpu().numpy().round(3).reshape(-1)
    preds.append(p)
    maes.append(mae)
    print(f"  frame {idx:6d}  pred={p}  gt={sample['action'].numpy().round(3)}  mae={mae:.4f}")

if len(preds) >= 2:
    std = np.array(preds).std(axis=0)
    print(f"\n[Diagnostic] pred std per joint: {std.round(2)}")
    print(f"[Diagnostic] avg MAE vs gt:      {float(np.mean(maes)):.3f}")
    if float(np.mean(maes)) > 5.0:
        print("[WARN] MAE high — check rename_map / camera / checkpoint")
    if float(np.array(preds).std()) < 0.05:
        print("[WARN] pred nearly constant — possible collapse")
print("Offline inference OK.")
PY

elif [ "${MODE}" = "robot" ]; then
    if [ ! -e "${ROBOT_PORT}" ]; then
        echo "ERROR: Robot port not found: ${ROBOT_PORT}"
        ls /dev/ttyACM* /dev/serial/by-id/ 2>/dev/null || true
        exit 1
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        PHYS_GPU="${CUDA_VISIBLE_DEVICES%%,*}"
        FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${PHYS_GPU}" | tr -d ' ')
        echo "GPU ${PHYS_GPU} free: ${FREE_MIB} MiB (need ~${MIN_FREE_MIB} MiB)"
        if [ "${FREE_MIB}" -lt "${MIN_FREE_MIB}" ]; then
            echo "ERROR: Not enough GPU memory on GPU ${PHYS_GPU}."
            nvidia-smi
            exit 1
        fi
    fi

    python -c "import scservo_sdk" 2>/dev/null || {
        echo "ERROR: pip install 'feetech-servo-sdk>=1.0.0,<2.0.0'"
        exit 1
    }

    echo "Robot port: ${ROBOT_PORT}"
    echo "Front cam:  ${FRONT_CAM} (${CAM_WIDTH}x${CAM_HEIGHT})"
    echo "Wrist cam:  ${WRIST_CAM} (${CAM_WIDTH}x${CAM_HEIGHT})"
    echo "Side cam:   ${SIDE_CAM} (${SIDE_CAM_WIDTH}x${SIDE_CAM_HEIGHT})"
    echo ""
    echo "Scene tip: white/blue/black flat in A4 workspace; gap >=5cm; no tray"
    echo "Success:   bottom white, middle blue, top black, stable"
    echo ""
    echo "Checking cameras..."
    python - <<PY
import sys, cv2

def check(name, target, width, height):
    try:
        target_int = int(target)
        src = target_int
    except ValueError:
        src = target
    cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
    ok = cap.isOpened()
    shape = None
    if ok:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        ret, frame = cap.read()
        ok = ok and ret and frame is not None
        shape = frame.shape if ok else None
    cap.release()
    print(f"  {name}: {'OK' if ok else 'FAIL'} {shape or ''}  ({target})")
    return ok

failed = False
failed |= not check("front", "${FRONT_CAM}", ${CAM_WIDTH}, ${CAM_HEIGHT})
failed |= not check("wrist", "${WRIST_CAM}", ${CAM_WIDTH}, ${CAM_HEIGHT})
failed |= not check("side", "${SIDE_CAM}", ${SIDE_CAM_WIDTH}, ${SIDE_CAM_HEIGHT})
if failed:
    sys.exit(1)
PY

    echo ""
    echo ">>> Starting SmolVLA 3cam stack_3blocks inference in 3s (Ctrl+C to abort)..."
    sleep 3

    # 唯一目录：ckpt + 时间戳 + pid，避免同秒启动互相覆盖
    RUN_TS="$(date +%Y%m%d_%H%M%S)"
    EVAL_RUN_ID="ckpt${CHECKPOINT_STEP}_${RUN_TS}_$$"
    EVAL_DATA_ROOT="${EVAL_ROOT}/${EVAL_RUN_ID}"
    if [ -e "${EVAL_DATA_ROOT}" ]; then
        EVAL_RUN_ID="${EVAL_RUN_ID}_$(date +%N)"
        EVAL_DATA_ROOT="${EVAL_ROOT}/${EVAL_RUN_ID}"
    fi
    # repo_id 必须是 org/name 两段，且 name 以 eval_ 开头（policy 推理时）
    EVAL_REPO_ID_RUN="local/eval_smolvla_stack3_3cam_${EVAL_RUN_ID}"
    echo "Eval save: ${EVAL_DATA_ROOT}"
    echo "Repo id:   ${EVAL_REPO_ID_RUN}"
    echo "Clips dir: ${CLIPS_DIR} (SAVE_CLIP=${SAVE_CLIP})"

    ROBOT_CAMERAS="{ front: {type: opencv, index_or_path: \"${FRONT_CAM}\", width: ${CAM_WIDTH}, height: ${CAM_HEIGHT}, fps: ${FPS}, fourcc: MJPG, warmup_s: ${CAM_WARMUP_S}}, wrist: {type: opencv, index_or_path: \"${WRIST_CAM}\", width: ${CAM_WIDTH}, height: ${CAM_HEIGHT}, fps: ${FPS}, fourcc: MJPG, warmup_s: ${WRIST_WARMUP_S}}, side: {type: opencv, index_or_path: \"${SIDE_CAM}\", width: ${SIDE_CAM_WIDTH}, height: ${SIDE_CAM_HEIGHT}, fps: ${FPS}, fourcc: MJPG, warmup_s: ${SIDE_WARMUP_S}} }"

    lerobot-record \
      --robot.type=so101_follower \
      --robot.port="${ROBOT_PORT}" \
      --robot.id="${ROBOT_ID}" \
      --robot.max_relative_target="${MAX_RELATIVE_TARGET}" \
      --robot.cameras="${ROBOT_CAMERAS}" \
      --display_data="${DISPLAY_DATA}" \
      --play_sounds="${PLAY_SOUNDS}" \
      --dataset.repo_id="${EVAL_REPO_ID_RUN}" \
      --dataset.root="${EVAL_DATA_ROOT}" \
      --dataset.single_task="${DATASET_TASK}" \
      --dataset.fps="${FPS}" \
      --dataset.num_episodes="${NUM_EPISODES}" \
      --dataset.episode_time_s="${EPISODE_TIME_S}" \
      --dataset.reset_time_s="${RESET_TIME_S}" \
      --dataset.push_to_hub=false \
      --dataset.rename_map="${RENAME_MAP}" \
      --policy.path="${CHECKPOINT}" \
      --policy.device=cuda \
      2>&1 | python -u -c "
import sys
phase = 'init'
shown = False
for raw in sys.stdin:
    line = raw.rstrip('\n')
    if 'Recording episode' in line:
        phase = 'record'
        shown = False
        print(); print('=' * 62)
        print('  SmolVLA 3cam stack_3blocks 推理中 (${EPISODE_TIME_S}s)')
        print('  task: white → blue → black')
        print('=' * 62); print(line, flush=True)
    elif 'Reset the environment' in line:
        phase = 'reset'
        shown = False
        print(); print('-' * 62)
        print('  RESET 复位 (${RESET_TIME_S}s)')
        print('-' * 62); print(line, flush=True)
    elif 'No policy or teleoperator' in line and phase == 'reset':
        if not shown:
            print('  (reset 阶段日志已隐藏)', flush=True)
            shown = True
    else:
        print(line, flush=True)
"

    # 导出独立 clip：每路相机 + 可选三联屏，文件名带 run id，永不覆盖
    if [ "${SAVE_CLIP}" = "1" ]; then
        mkdir -p "${CLIPS_DIR}"
        echo ""
        echo "Exporting clips (no overwrite) -> ${CLIPS_DIR}"
        SAVED_FRONT=""
        SAVED_WRIST=""
        SAVED_SIDE=""
        for cam in front wrist side; do
            SRC="${EVAL_DATA_ROOT}/videos/observation.images.${cam}/chunk-000/file-000.mp4"
            DST="${CLIPS_DIR}/${EVAL_RUN_ID}_${cam}.mp4"
            if [ -f "${SRC}" ]; then
                if [ -e "${DST}" ]; then
                    DST="${CLIPS_DIR}/${EVAL_RUN_ID}_${cam}_$(date +%N).mp4"
                fi
                cp -n "${SRC}" "${DST}"
                echo "  saved: ${DST}"
                case "${cam}" in
                    front) SAVED_FRONT="${DST}" ;;
                    wrist) SAVED_WRIST="${DST}" ;;
                    side)  SAVED_SIDE="${DST}" ;;
                esac
            else
                echo "  skip ${cam}: missing ${SRC}"
            fi
        done

        COMBO_MP4="${CLIPS_DIR}/${EVAL_RUN_ID}_combo.mp4"
        if [ -e "${COMBO_MP4}" ]; then
            COMBO_MP4="${CLIPS_DIR}/${EVAL_RUN_ID}_combo_$(date +%N).mp4"
        fi
        if command -v ffmpeg >/dev/null 2>&1 \
            && [ -n "${SAVED_FRONT}" ] && [ -n "${SAVED_WRIST}" ] && [ -n "${SAVED_SIDE}" ]; then
            # 三联横拼成一个文件，方便直接看
            ffmpeg -y -hide_banner -loglevel error \
              -i "${SAVED_FRONT}" -i "${SAVED_WRIST}" -i "${SAVED_SIDE}" \
              -filter_complex "[0:v]scale=640:480,setsar=1[v0];[1:v]scale=640:480,setsar=1[v1];[2:v]scale=640:480,setsar=1[v2];[v0][v1][v2]hstack=inputs=3" \
              -c:v libx264 -pix_fmt yuv420p -movflags +faststart \
              "${COMBO_MP4}" \
              && echo "  combo: ${COMBO_MP4}" \
              || echo "  WARN: ffmpeg combo failed (per-cam clips still kept)"
        fi
        echo "Clips kept under: ${CLIPS_DIR}"
        ls -lt "${CLIPS_DIR}"/"${EVAL_RUN_ID}"*.mp4 2>/dev/null || true
    fi

else
    echo "Usage: bash infer_smolvla_stack_3blocks_3cam.sh [offline|robot|cameras|viz|list]"
    echo "  STEP=100000 bash infer_smolvla_stack_3blocks_3cam.sh robot"
    echo "  EPOCH=20    bash infer_smolvla_stack_3blocks_3cam.sh robot"
    exit 1
fi

echo "=============================="
echo "Inference finished"
if [ "${MODE}" = "robot" ]; then
    echo "Eval dataset: ${EVAL_DATA_ROOT:-}"
    echo "Clips:        ${CLIPS_DIR:-}  (${EVAL_RUN_ID:-}*.mp4)"
fi
echo "=============================="
