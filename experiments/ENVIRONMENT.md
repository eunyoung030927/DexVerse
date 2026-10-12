# Environment

## Dev box (pipeline tests, GraspCup v0 -> Allegro, v1 replay check)

| item | value |
|---|---|
| GPU | 1x NVIDIA RTX 3090 24 GB, driver 580.95.05 (shared with processes outside the container) |
| CPU / RAM | Intel i9-10940X, 28 threads / 125 GB |
| OS | Linux 6.8 container, Ubuntu 22.04, headless (Vulkan via libEGL_nvidia ICD) |
| Python env | conda, Python 3.11 (`source /workspace/dexverse/dexverse_env.sh`) |
| Simulator | Isaac Sim 5.1.0 (pip), Isaac Lab v2.3.2 |
| Libraries | torch 2.7.0+cu128, numpy 1.26, dex_retargeting 0.4.6, trimesh 4.5.1, scipy 1.15 |
| DexVerse | fork eunyoung030927/DexVerse, branch `retarget/shadow-cross-hand` (upstream ycyao216/DexVerse + `develop` v1 tasks + teleop branch) |

The checkout's `source/dexverse` must be first on `PYTHONPATH` (the env's editable install points at the main checkout);
`scripts/retarget/*` and `run_episodes.sh` set it.

## Cluster: pro6000 (SLURM, OpenOnDemand)

Same node as the Bench2Dex retargeting (`ondemand/bench2dex/sysinfo.sbatch`, 2026-10-07): p1, 4x RTX PRO 6000 Blackwell
96 GB (sm_120), driver 580.82.07 (R580, the branch NVIDIA tested Isaac Sim 5.1 on), Ubuntu 24.04, 2x EPYC 9135 / 503 GB,
SLURM 23.11.4, apptainer 1.5.2; node `/tmp` nearly full, `/home` NAS. Container `nvcr.io/nvidia/isaac-sim:5.1.0` SIF
(shared with Bench2Dex), torch 2.7.0+cu128, Isaac Lab v2.3.2. Jobs: `ondemand/` (README there). DexVerse has not run
on the cluster yet.

## Simulation settings

Physics is DexVerse's and is not modified for retargeting: the task config is built exactly as `record_demos.py` builds
it (`dexverse.mimic.env_setup.build_env_cfg` with the target `robot_type`, cameras stripped), so hand actuators, objects
and table are the benchmark's. Retargeting writes joint position targets directly and steps physics `decimation` times
per frame (the physics of `env.step`). 60 Hz control (= recording rate). Success = the task's own `success` termination
term held 10 consecutive frames (the `record_demos` rule), evaluated on env 0 every frame.

## Capacity notes

- One SPIDER process with 1024 samples uses a few GB of GPU memory (Bench2Dex: ~5-6 GB; at most 3 per 24 GB GPU).
  A CUDA OOM corrupts PhysX state; stage 3 raises when object poses go non-finite.
- Measured so far: GraspCup v0 ep0 (280 frames) -> Allegro, 35.8 min on the 3090 (4 backtracks). v1 demos are
  311-1738 steps, so expect roughly 1-3 GPU-h per attempt; measure with `EPISODES=1` before sizing `PER_GPU`.
- The Isaac stage scripts can exit 0 after an exception (Kit shutdown): `run_target.py` / `run_queue.py` judge each
  stage by its output file, not the exit code.
