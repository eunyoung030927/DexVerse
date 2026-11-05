# baselines/lerobot_dp — LeRobot Diffusion Policy (image) baseline on DexVerse teleop demos

Trains LeRobot 0.4.2's official Diffusion Policy per (task, hand) on the HF `Dexverse-CSI/dv-demos-v0` teleop sets and
evaluates it in the DexVerse Isaac Sim env, restoring each demo's initial state. Moved here from
`experiments/<server>/20261031_teleop_single/teleop_dp_image/` on 2026-11-05 so the baseline code is versioned with DexVerse
(the DexLatent / XL-VLA comparison lives next to it). Study notes, run files and results stay in that experiments folder;
models go to HF `Dexverse-CSI/dexverse-dp-image` (`v0/teleop_lerobot/<category>/<Task>/<robot_type>/{best,last}/`).

| file | role |
|---|---|
| `run_sets.sh` | one-GPU session: train → snapshot selection (pkl episodes 0–9) → final 50 rollouts → run files; same RID = resume/attach |
| `train_lerobot.py` | LeRobot `lerobot_train` with the DexVerse settings (batch 64, 300 epochs, snapshot every 50, fp32, seed from `experiments/seed_control.py`) |
| `image_resize.py`, `data_keys.py` | 480×640 → 240×320 resize in the data loader, needed-keys/worker patches (see the study README "주의") |
| `eval_lerobot.py` + `policy_server.py` | Isaac Sim rollouts (DexVerse env) with the LeRobot policy in its own env (multiprocessing connection) |
| `finalize_run.py` | EXPERIMENT_RULES run files: meta.json, config.yaml, settings.md, metrics.csv, README, INDEX.csv |
| `upload_dp_image.py` | HF upload of best/ + last/ + run files (token on stdin) |
| `dp_repo_readme.py` | (central PC) results table for the HF repo README from the central INDEX.csv files |

Run (inside a server container, after `sync.sh code`):

```bash
RID=20261104-014353 NUM_WORKERS=12 bash /workspace/codes/dexverse/DexVerse/baselines/lerobot_dp/run_sets.sh 0 GraspKettle:shadow_right BimanualLiftTray:leap_bimanual
echo <hf token> | /workspace/envs/dexverse/lerobot/bin/python baselines/lerobot_dp/upload_dp_image.py <run_dir> ...
```

Env vars: `RUNS` (run folder, default `/workspace/experiments/<server>/20261031_teleop_single/teleop_dp_image`), `EPOCHS`, `SNAPSHOT_EPOCHS`,
`NUM_WORKERS`, `MAX_STEPS`, `EXPERIMENTS_ROOT` (seed_control.py location, default `/workspace/experiments`).
Conventions (seed 42 once, best-only final evaluation, 240×320 loader resize) are in the study README and `EXPERIMENT_RULES.md`.
