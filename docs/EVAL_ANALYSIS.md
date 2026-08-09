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

**Clean full 3-stack success: 0 / 4** (excluding incomplete 60k).

## Near-miss detail (50k long, ~181s)

Front frame ~180.9s: gripper holds **black** centered over a stable **white + blue** stack (correct order).  
By ~185s: all three blocks flat again (failed place / tip / or immediate reset).  
Later: human hand in FOV; episode ends again near a 2-stack while approaching black.

## Action health

Across runs:
- Actions not collapsed (joint std large; gripper range ~0.3–60).
- `|Δstate| > 8` rare → `max_relative_target=10` not the main bottleneck in these logs.
- Gripper chatter high on long episodes (enter_closed ≈ 24–98) vs ~3 needed for a clean 3-stack.

## Capability summary

| Skill | Status |
|-------|--------|
| Reach / align | OK |
| Grasp white/blue | OK |
| Place blue on white | OK (common) |
| Grasp black after 2-stack | Often attempted |
| Place black on 2-stack | **Fail / unstable** |
| Order discipline early | Mixed (sometimes blue first) |
| Autonomy w/o human | OK on short; poor on long |

## Recommendations

1. Clean re-eval: `STEP=50000` and `STEP=100000`, no intervention, ≥180s, ≥3 trials.  
2. Targeted demos: black-on-stable-2stack (40–60).  
3. Prefer mid ckpt until clean sweep says otherwise.  
4. Keep unique clip export (`clips/`) for every robot run.
