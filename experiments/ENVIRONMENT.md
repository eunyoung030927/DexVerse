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
(shared with Bench2Dex), torch 2.7.0+cu128, Isaac Lab v2.3.2. Jobs: `ondemand/` (README there). First DexVerse jobs
2026-10-13 (1649, 1651).

### Node limits (p1) and what a job may ask for

Hard caps of the node (`scontrol show node p1`, sysinfo job 1534, 2026-10-07). An `#SBATCH` request above a cap never
schedules; the sum of all jobs on the node cannot exceed it either.

| resource | p1 cap | `main_job.sbatch` default (2 GPUs) | whole node (4 GPUs) |
|---|---|---|---|
| GPUs (`--gres=gpu:pro6000:N`) | 4 x 96 GB | 2 | 4 |
| CPU threads (`--cpus-per-task`) | 64 (2 x EPYC 9135 16c/32t) | 48 | 64 |
| RAM (`--mem`) | 500000M (~488 GiB) | 320G | 440G |
| wall time (`--time`) | partition `gpu` limit not recorded; 3-00:00:00 accepted | 3 days, continuation job chains (MAX_CHAIN 10) | same |
| node `/tmp` | 7.0 TB, ~460 GB free (shared, apptainer cache + MPS pipe) | `cleanup` of the OCI cache in setup | same |
| `/home` NAS | 44 TB, 26 TB free (results, demos, checkout) | | |

Per-session footprint (Bench2Dex measurement, confirmed by job 1651: 34 sessions on 48 threads): ~1.4 CPU threads and
~6.6 GB RAM per SPIDER session, ~2 GB VRAM each under MPS at 1024 envs (66 GB used by 34 sessions). So the session count
is capped by CPU first: 48 threads -> ~34 sessions (`PER_GPU=16` x 2 GPUs, VRAM-gated), 64 threads -> ~45 sessions
(`PER_GPU=11` x 4 GPUs). A job that asks for 4 GPUs takes the whole node (nothing else, including Bench2Dex jobs, can
run on p1 meanwhile).

## Simulation settings

Physics is DexVerse's and is not modified for retargeting: the task config is built exactly as `record_demos.py` builds
it (`dexverse.mimic.env_setup.build_env_cfg` with the target `robot_type`, cameras stripped), so hand actuators, objects
and table are the benchmark's. Retargeting writes joint position targets directly and steps physics `decimation` times
per frame (the physics of `env.step`). 60 Hz control (= recording rate). Success = the task's own `success` termination
term held 10 consecutive frames (the `record_demos` rule), evaluated on env 0 every frame.

## Capacity notes

- One SPIDER process with 1024 samples uses a few GB of GPU memory (Bench2Dex: ~5-6 GB; at most 3 per 24 GB GPU).
  A CUDA OOM corrupts PhysX state; stage 3 raises when object poses go non-finite.
- Measured: GraspCup v0 ep0 (280 frames) -> Allegro, 35.8 min on the 3090 (4 backtracks). Cluster job 1651
  (34 sessions on 2 pro6000, GPU util 100 %): GraspBleach-v1 ep7 -> Allegro 28 min, -> Inspire 46 min (first attempt
  each). Throughput ~34 items/h at ~1 h/item -> single-hand 1,250 items ~1.5 days, bimanual 1,000 items ~3.5-4 days.
- The Isaac stage scripts can exit 0 after an exception (Kit shutdown): `run_target.py` / `run_queue.py` judge each
  stage by its output file, not the exit code.
