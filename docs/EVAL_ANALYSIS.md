# Eval analysis — SmolVLA 3cam stack white→blue→black

**Date:** 2026-08-09  
**Eval root (local):** `outputs/eval/smolvla_stack_3blocks_white_blue_black_3cam`  
**Train ckpts:** `outputs/train/smolvla_stack_3blocks_white_blue_black_3cam/checkpoints/`  
**Method:** timeline frames (front/side/wrist) + parquet action/state stats

## Scorecard

| Run folder | Ckpt | Frames | Dur (s) | Grip enter_closed | Human | Best | Full 3? |
|------------|------|--------|---------|-------------------|-------|------|---------|
| `20260809_124809` | 100000 | 2274 | 75.8 | 25 | light | 2-stack W←B, approach black | No |
| `ckpt050000_20260809_125644_3310655` | 50000 | 2727 | 90.9 | 24 | none/light | 2-stack W←B, pre-grasp black | No |
| `ckpt050000_20260809_130518_3336464` | 50000 | 9045 | 301.5 | 93 | yes (reset) | **Near 3** @~181s | Near-miss |
| `ckpt060000_20260809_132147_3385771` | 60000 | 0 | — | — | — | incomplete | — |
| `ckpt080000_20260809_132457_3395144` | 80000 | 10650 | 355.0 | 98 | heavy | pick/approach | No |

## Correction (2026-08-09 evening)

Previous “0/4 full success” was **incorrect**. Causes:
1. Scored mostly **end-of-recording** state; long runs keep going after a success → end looks unstacked.
2. Sparse frame sampling missed mid-episode towers.
3. Operator ground truth: **5 episodes, 4 successes (~80%)** that day; later informal tests **>50%** SR.
4. Model still picks the right color when white/blue layout is randomized.

### Video-confirmed full towers (side cam)

| Run | ~time | Stack |
|-----|-------|-------|
| `ckpt050000_…130518` | ~85 s | white ← blue ← black |
| `ckpt080000_…132457` | ~205 s | white ← blue ← black |

Treat saved folders as **long sessions with multiple attempts**, not one binary trial each.

## Near-miss / retry detail (50k long, ~181s)

Also saw black held over W+B then tip/retry — consistent with multi-attempt sessions, not “never succeeds”.

## Action health

Across runs:
- Actions not collapsed (joint std large; gripper range ~0.3–60).
- `|Δstate| > 8` rare → `max_relative_target=10` not the main bottleneck in these logs.
- Gripper chatter high on long episodes (enter_closed ≈ 24–98) vs ~3 needed for a clean 3-stack.

## Capability summary (revised)

| Skill | Status |
|-------|--------|
| Reach / align | Strong |
| Grasp under position shuffle | Strong (incl. white/blue swap) |
| Place blue on white | Strong |
| Full 3-stack (W←B←K) | **Achievable**; operator SR ~4/5 one session, later >50% |
| Recovery after tip | Partial (retries inside long sessions) |
| Gripper chatter | Still high on long recordings |

## Latency (offline bench, ckpt 50k, CUDA, 3 cams)

| Metric | Value |
|--------|-------|
| `predict_action_chunk` / first `select_action` | **~143 ms** mean (≈140–147) |
| Subsequent queued `select_action` | **~1.2 ms** |
| `n_action_steps` | 50 |
| Control target | 30 Hz (33 ms budget); chunk refill every ~50 steps ≈ every **1.67 s** |

So wall-clock policy compute is chunky (~7 chunk-forwards/s max if saturated), but realtime loop stays at 30 Hz via action queue.

## Recommendations

1. Log SR per **attempt** (stop episode on success) so folders match operator counts.  
2. Keep clip export; optionally stamp success timestamps.  
3. More black-finish demos still help stability, but model is already past “never finishes”.
