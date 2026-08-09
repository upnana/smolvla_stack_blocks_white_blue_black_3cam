# Reproducing this experiment

## Environment

```bash
# Python 3.10 conda env used here was named `lerobot`
# LeRobot 0.4.3 editable from source with smolvla + feetech extras
pip install -e ".[smolvla,feetech]"
```

Base weights (this lab used ModelScope caches):

- `lerobot--smolvla_base`
- `HuggingFaceTB--SmolVLM2-500M-Video-Instruct`

## 1) Collect

```bash
bash scripts/get-data-stack_3blocks_3cam.sh cameras
bash scripts/get-data-stack_3blocks_3cam.sh preview
NUM_EPISODES=120 bash scripts/get-data-stack_3blocks_3cam.sh
```

Edit camera by-id paths and serial ports at the top of the script.

## 2) Train

```bash
# ensure DATA_ROOT / BASE_MODEL / VLM_MODEL exist
bash scripts/train_smolvla_stack_3blocks_3cam.sh
```

Critical flags already in script:

- `rename_map` front/wrist/side → camera1/2/3  
- `policy.empty_cameras=0`  
- dual GPU batch 8×2  

## 3) Eval

```bash
STEP=50000 bash scripts/infer_smolvla_stack_3blocks_3cam.sh robot
STEP=100000 EPISODE_TIME_S=180 NUM_EPISODES=3 bash scripts/infer_smolvla_stack_3blocks_3cam.sh robot
```

Clips land in `outputs/eval/.../clips/` with unique filenames.

## Success criteria (scoring)

Count **success** only if:

1. No human touch in workspace during the episode  
2. Final tower order white → blue → black  
3. Tower still standing at episode end  

Partial credit: stable 2-stack (white←blue) without completing black.
