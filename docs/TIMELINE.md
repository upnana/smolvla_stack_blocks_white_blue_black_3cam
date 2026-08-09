# Experiment timeline

All times local (UTC+8), 2026.

## Aug 8 — hardware + data

1. **Gripper swap** on SO101 follower; leader/follower gripper misaligned → recalibrate with `lerobot-calibrate` (choose `c` on mismatch).
2. Decided **not** to resume into old 2-cam 299-ep dataset (new gripper + new cameras).
3. Created **3-cam** collection script `get-data-stack_3blocks_3cam.sh`.
4. Cameras: front UGREEN 640×480, wrist Sonix 640×480, side Generic Web **800×480**.
5. Fixed empty leftover dataset dir (`FileExistsError`).
6. Collected **120** episodes / **79 251** frames.
7. QC: grip cycles ~3–5/ep, videos present, stacks visible.
8. Zipped dataset (~693 MB).
9. Uploaded via **hf-mirror** as user `upna`:
   - https://hf-mirror.com/datasets/upna/stack_3blocks_white_blue_black_3cam
10. Started SmolVLA 3cam training (`train_smolvla_stack_3blocks_3cam.sh`), dual GPU, 100k steps.

## Aug 9 — train finish + eval

1. Training finished ~**06:36**, `last` → **100000**, late loss ~0.017.
2. Wrote `infer_smolvla_stack_3blocks_3cam.sh` (3cam rename_map, clip export).
3. First robot infer hit `ValueError: too many values to unpack` because `repo_id` had extra `/` — fixed to `local/eval_smolvla_stack3_3cam_<runid>`.
4. Eval runs at ckpts **50k / 60k(incomplete) / 80k / 100k**.
5. Video analysis: **0/4** full success; best near-miss at **50k** ~181s; 2-stack capability strong; black finish weak.
6. Experiment log pushed to this GitHub repo.

## Prior context (2cam, not mixed)

- Old dataset `stack_3blocks_white_blue_black`: ~299 eps, 2 cams, old gripper.
- SmolVLA trained to 300k on that set; Aug 8 eval also **0** clean 3-stack.
- Those weights / data are **out of scope** for this 3cam repo.
