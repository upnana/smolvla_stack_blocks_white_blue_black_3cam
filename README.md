# SmolVLA · Stack 3 Blocks (white → blue → black) · 3 Cameras

Experiment log for SO101 teleop data collection → SmolVLA training → real-robot eval.

**Repo:** [upnana/smolvla_stack_blocks_white_blue_black_3cam](https://github.com/upnana/smolvla_stack_blocks_white_blue_black_3cam)  
**Date range:** 2026-08-08 → 2026-08-09  
**Hardware:** SO101 follower + leader (Feetech)  
**Policy:** SmolVLA (`smolvla_base` + SmolVLM2-500M)  
**LeRobot:** `0.4.3` (editable install, conda env `lerobot`)

---

## TL;DR

| Stage | Result |
|-------|--------|
| Data | **120** episodes / **79 251** frames / 3 cams / new gripper |
| Train | Dual-GPU SmolVLA, **100 000** steps (~**40** epochs), finished 2026-08-09 06:36 |
| Eval (operator count) | **~4/5** full 3-stack success on one session; later tests **>50%** SR |
| Eval (video re-score) | Mid-episode full towers confirmed (e.g. 50k@~85s, 80k@~205s); earlier “0/4” was **wrong** (end-of-file scoring) |
| Latency | Action-chunk forward ≈ **143 ms**; queue pop ≈ **1.2 ms**; control @30 Hz with `n_action_steps=50` |
| Robustness | Grasps OK when white/blue positions are swapped / randomized |

Dataset (HF mirror): [upna/stack_3blocks_white_blue_black_3cam](https://hf-mirror.com/datasets/upna/stack_3blocks_white_blue_black_3cam)  
Local zip ≈ 693 MB: `stack_3blocks_white_blue_black_3cam.zip`

---

## Task definition

**Goal (bottom → top):** white → blue → black, tower stable.

**Language task string** (no colon — colon breaks LeRobot CLI / draccus into a dict):

```text
stack the blocks from bottom to top white then blue then black
```

**Scene protocol**
- Start: three blocks flat, randomized in ~A4 workspace, gap ≥5 cm, no tray
- Optional mix-in: already-stable 2-stack, only place black (finish demos)
- Success: correct order + stable before ending episode; tip → re-record (←)

---

## Hardware & cameras

| Role | Device | Path / notes |
|------|--------|----------------|
| Follower | SO101 | `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6083854-if00` · id `so101_follower` |
| Leader | SO101 | `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6084864-if00` · id `so101_leader` |
| Front cam | UGREEN 2K | 640×480 @30 · MJPG |
| Wrist cam | Sonix USB2.0 CAM1 | 640×480 @30 · MJPG |
| Side cam | Generic Web | **800×480** @30 · MJPG (640 fails on this sensor) |

**Important:** gripper was replaced before this 3cam dataset. Do **not** mix with the old 2-cam 299-ep library (`stack_3blocks_white_blue_black`).

Calibration backups (local machine): `calibration_backups/so101_*before_gripper_swap_*.json`

---

## Pipeline overview

```text
teleop collect (3cam)
        │
        ▼
LeRobot dataset  (front / wrist / side)
        │
        ├─ zip → Hugging Face (upna/…)
        │
        ▼
SmolVLA train   rename_map → camera1/2/3 , empty_cameras=0
        │
        ▼
checkpoints/010000 … 100000
        │
        ▼
lerobot-record + policy  (same 3 cams + rename_map)
        │
        ▼
eval dataset + clips/  (unique run id, no overwrite)
```

Scripts in this repo (`scripts/`):

| Script | Role |
|--------|------|
| `get-data-stack_3blocks_3cam.sh` | Teleop record / camera preview / viz |
| `train_smolvla_stack_3blocks_3cam.sh` | Accelerate multi-GPU train |
| `infer_smolvla_stack_3blocks_3cam.sh` | Offline diagnose / robot eval / clip export |

Paths inside scripts point at the author’s machine (`/home/rxn/...`); adapt `DATA_ROOT`, ports, and camera by-id paths for your setup.

---

## 1. Data collection

### Command

```bash
conda activate lerobot
bash get-data-stack_3blocks_3cam.sh              # default 120 eps
bash get-data-stack_3blocks_3cam.sh cameras
bash get-data-stack_3blocks_3cam.sh preview
bash get-data-stack_3blocks_3cam.sh viz
```

### Key defaults

| Param | Value |
|-------|--------|
| `DATA_ROOT` | `/home/rxn/datasets/stack_3blocks_white_blue_black_3cam` |
| `DATA_REPO` | `my_pick_place/stack_3blocks_white_blue_black_3cam` |
| `NUM_EPISODES` | 120 |
| `EPISODE_TIME_S` | 60 |
| `RESET_TIME_S` | 5 |
| `DISPLAY_DATA` | true (Rerun; `RERUN_IMAGE_ENTITIES=front,wrist,side`) |
| `FPS` | 30 |

### Dataset stats (final)

| Metric | Value |
|--------|--------|
| Episodes | **120** |
| Frames | **79 251** |
| FPS | 30 |
| Image keys | `observation.images.front`, `.wrist`, `.side` |
| Action / state | 6-DoF SO101 |

QC notes from collection: grip cycles typically ~3–5 per success episode; all three video streams present; stacks visible in spot checks.

### Upload

```bash
# Used hf-mirror (direct huggingface.co timed out in this network)
HF_ENDPOINT=https://hf-mirror.com
# Dataset card / zip: upna/stack_3blocks_white_blue_black_3cam
```

Local archive: `/home/rxn/datasets/stack_3blocks_white_blue_black_3cam.zip` (~693 MB)

---

## 2. Training (SmolVLA)

### Command

```bash
conda activate lerobot
# dual GPU default
bash train_smolvla_stack_3blocks_3cam.sh
# or background:
BACKGROUND=1 bash train_smolvla_stack_3blocks_3cam.sh
tail -f logs/smolvla_stack_3blocks_white_blue_black_3cam.log
```

### Config used

| Item | Value |
|------|--------|
| Base policy | ModelScope `lerobot--smolvla_base` |
| VLM | ModelScope `HuggingFaceTB--SmolVLM2-500M-Video-Instruct` |
| `empty_cameras` | **0** (true 3-cam) |
| `rename_map` | front→camera1, wrist→camera2, side→camera3 |
| GPUs | 2 (`CUDA_VISIBLE_DEVICES=0,1`) |
| Batch / GPU | 8 → **effective 16** |
| Steps | **100 000** |
| Save freq | 10 000 |
| Approx epochs | `100000 * 16 / 79251 ≈ 20.2` per “nominal”; log showed ~**40** epoch counter by end (LeRobot epch metric) |
| Output | `outputs/train/smolvla_stack_3blocks_white_blue_black_3cam` |
| Log | `logs/smolvla_stack_3blocks_white_blue_black_3cam.log` |

Finished **2026-08-09 ~06:36**. `checkpoints/last` → **100000**. Late train loss ≈ **0.017**.

Checkpoints kept: `010000` … `100000` (every 10k).

### Why not mix old 2cam data

1. New gripper → distribution shift vs old 299-ep set  
2. Camera set changed (2 → 3; side added)  
3. Separate `rename_map` / feature shapes

---

## 3. Inference / eval

### Command

```bash
bash infer_smolvla_stack_3blocks_3cam.sh list
bash infer_smolvla_stack_3blocks_3cam.sh offline
STEP=50000 bash infer_smolvla_stack_3blocks_3cam.sh robot
STEP=100000 EPISODE_TIME_S=180 bash infer_smolvla_stack_3blocks_3cam.sh robot
```

### Eval defaults

| Param | Value |
|-------|--------|
| Same 3 cams + rename_map as train | required |
| `max_relative_target` | 10 (default) |
| `EVAL_ROOT` | `outputs/eval/smolvla_stack_3blocks_white_blue_black_3cam` |
| Clip export | `SAVE_CLIP=1` → `…/clips/` unique names (no overwrite) |
| `repo_id` form | `local/eval_smolvla_stack3_3cam_<runid>` (must be `org/name`, name starts with `eval_`) |

Each robot run uses unique folder:

```text
ckpt{STEP}_{YYYYMMDD_HHMMSS}_{PID}/
```

plus flat clips:

```text
clips/ckpt…_front.mp4
clips/ckpt…_wrist.mp4
clips/ckpt…_side.mp4
clips/ckpt…_combo.mp4   # if ffmpeg available
```

---

## 4. Eval results (2026-08-09)

> **Correction:** an earlier draft claimed **0/4** full success by scoring only end-of-file state on long recordings. That was wrong. Operator ground truth + mid-episode video review show the policy **does** complete white→blue→black towers.

Detail write-up: [`docs/EVAL_ANALYSIS.md`](docs/EVAL_ANALYSIS.md) · numbers: [`eval/eval_runs_summary.json`](eval/eval_runs_summary.json)

### Success rate (3cam, this experiment)

| Source | SR |
|--------|----|
| Operator count (one session) | **4 / 5** episodes (~**80%**) full 3-stack |
| Later informal tests | **>50%** |
| Video re-score of saved long runs | Full towers visible mid-session (e.g. 50k @~85 s, 80k @~205 s) |

Long `lerobot-record` folders often contain **multiple attempts** inside one file; ending unstacked ≠ never succeeded.

### Demo videos

| File | What |
|------|------|
| [`media/3_blocks_stack_2cam.mp4`](media/3_blocks_stack_2cam.mp4) | Older **2cam** stack eval clip (~62 s) |
| [`media/3_blocks_stack_3cam.mp4`](media/3_blocks_stack_3cam.mp4) | **3cam** stack eval clip (~178 s) |

### Inference latency (offline, ckpt 50k, 3 cams, CUDA)

| Metric | Value |
|--------|-------|
| Action-chunk forward / first `select_action` | **~143 ms** |
| Queued `select_action` | **~1.2 ms** |
| `n_action_steps` | 50 → refill ~every **1.67 s** at 30 Hz control |

Realtime loop stays **30 Hz**; heavy compute is chunked.

### Capability (revised)

**Works**
- Reach / center / grasp under **randomized** white–blue–black layout (incl. white/blue swap)
- Stable **2-stack** (white←blue) and **full 3-stack** (white←blue←black)
- Retries after tip inside a session

**Still noisy**
- Gripper chatter on long recordings
- Occasional place miss on black → needs retry
- Hands-in-FOV make some saved folders unfair for autonomy scoring

**Checkpoint note:** mid ckpts (e.g. **50k**) and **80k** both show completed towers in video; do not assume only `last` works.

### Saved run index (local machine folders)

| Run | Ckpt | Duration | Notes |
|-----|------|----------|-------|
| `20260809_124809` | 100000 | 76 s | Short; 2-stack + approach black |
| `…125644` | 050000 | 91 s | Short; strong 2-stack |
| `…130518` | 050000 | 302 s | Mid-episode **full tower** (~85 s); later retries |
| `…132147` | 060000 | — | Incomplete / aborted |
| `…132457` | 080000 | 355 s | Mid-episode **full tower** (~205 s); multi-attempt session |

---

## 5. Next steps (recommended)

1. Score SR per **attempt** (end episode on success) so folders match operator counts.  
2. Keep collecting finish demos (black-on-stable-2stack) to raise stability above 50–80%.  
3. Optional ckpt sweep with fixed protocol: 50k / 80k / 100k × N clean trials.  
4. Keep clip export (`clips/`) on for every robot run.

---

## Related links

- GitHub (this log): https://github.com/upnana/smolvla_stack_blocks_white_blue_black_3cam  
- HF dataset (mirror): https://hf-mirror.com/datasets/upna/stack_3blocks_white_blue_black_3cam  
- LeRobot: https://github.com/huggingface/lerobot  

---

## License / notes

Experiment notes and scripts for reproducibility. Model weights and raw eval videos are large and stay on the lab machine unless uploaded separately. Do not commit secrets or private calibration JSON with motor offsets if you fork publicly without review.
