# DexVerse DP3 Baseline

This package adds a DexVerse imitation-learning baseline based on 3D Diffusion Policy (DP3). DP3 conditions on a table-cropped point cloud plus proprioception and predicts temporally chunked robot actions with a diffusion policy.

The implementation is self-contained under `dexverse.IL.dp3`. The original DP3 algorithm components are vendored in `dp3_core/` and adapted from [3D-Diffusion-Policy](https://github.com/YanjieZe/3D-Diffusion-Policy), so users do not need to install the upstream package.

## Layout

```text
source/dexverse/dexverse/IL/dp3/
|-- config.py          # Dataclass configs for dataset, policy, training, and eval
|-- dataset.py         # HDF5 dataset loader and windowed sequence sampler
|-- model.py           # DP3 policy, scheduler, EMA, and checkpoint helpers
|-- train.py           # Training loop
|-- runner.py          # Closed-loop inference runner
|-- online_eval.py     # Online simulator evaluation
`-- dp3_core/          # Vendored DP3 network, normalizer, EMA, PointNet, and utils

scripts/dp3/
|-- convert_demos_to_dp3.py      # create_demo_files_sequential.py HDF5 -> DP3 HDF5
|-- train.py                     # Training CLI
|-- eval_online.py               # Online evaluation CLI
`-- visualize_dp3_dataset.py     # Dataset inspection and point cloud visualization
```

## Inputs and Data Format

DP3 expects HDF5 datasets produced by `scripts/dp3/convert_demos_to_dp3.py`:

```text
dp3_dataset.hdf5
|-- attrs:
|   |-- task
|   |-- observation_preset       # the preset used to record demos (e.g. "pointcloud")
|   |-- pointcloud_term          # "camera_point_cloud_w" or "merged_point_cloud_w"
|   |-- num_points
|   |-- bbox_min                 # workspace AABB over valid points
|   |-- bbox_max
|   |-- source_h5
|   `-- num_episodes
`-- data/
    `-- episode_0/
        |-- obs/
        |   |-- point_cloud      # (T, num_points, 3), world frame
        |   `-- agent_pos        # (T, proprio_dim)
        `-- action               # (T, action_dim)
```

The point cloud is back-projected from the `third_person_camera` depth in DexVerse, cropped to the table workspace by `mdp.camera_point_cloud_w`, and then per-frame downsampled to `num_points` with farthest-point sampling (uses `pytorch3d` when available, falls back to a built-in iterative FPS otherwise).

`agent_pos` is the concatenation of every term in the env's `proprio` observation group (sorted by term name). Under the `pointcloud` observation preset there is a single term (`joint_pos`) and `flatten_history_dim=True` already folds the 3-frame proprio history into the trailing dim, so `proprio_dim` is `history_length * num_joints`.

The dataset loader fits a `LinearNormalizer` over point clouds, proprioception, and actions from the training split. `num_points`, `proprio_dim`, and `action_dim` are derived from the dataset at runtime, so changing any of them in the converter requires no policy-config changes.

## Setup

Run commands from the repository root with the IsaacLab/DexVerse Python environment activated:

```bash
python -m pip install -e "source/dexverse[dp3]"
# (equivalent: python -m pip install -e source/dexverse diffusers einops h5py omegaconf termcolor)
```

Optional packages:

```bash
# Faster farthest-point sampling, if compatible with your CUDA/PyTorch install.
python -m pip install pytorch3d

# Only needed for scripts/dp3/visualize_dp3_dataset.py.
python -m pip install matplotlib pillow
```

## End-to-End Usage

For a short test using 50 PushT v1 demonstrations and two training epochs, see
the shared [DP and DP3 walkthrough](../../../../../docs/baseline_training.md).
The example below uses the original GraspPan v0 task for a longer DP3 run.

### 0. Fetch assets and teleop demonstrations

The downloaders default to the public `dexverse/DexVerse_release` dataset;
no login is required:

```bash
python scripts/asset_tools/download_robot_agents.py --bundle shadow
python scripts/asset_tools/download_assets.py --core
python scripts/demo_tools/download_demos.py --task functional/Dexverse-GraspPan-v0
# (or --category functional for the whole category, or --all for everything)
```

Demos land under `source/dexverse/demonstrations/<category>/<task>/*.pkl`. These are
teleop trajectories (joint-target actions + per-episode initial scene state) and do
**not** ship vision data — that gets regenerated in step 1.

### 1. Replay demos headlessly and capture point clouds

`scripts/demo_tools/create_demo_files_sequential.py` is the canonical headless replayer: it restores
recorded scene states one episode at a time, captures the active
observation groups, and writes one HDF5 per task with the layout
`data/demo_<i>/obs/<group>/<term>`.

```bash
python scripts/demo_tools/create_demo_files_sequential.py \
    --task functional/Dexverse-GraspPan-v0 \
    --obs-groups pointcloud \
    --set-state --device cpu --seed 42 --headless \
    --output-dir outputs/demos_h5
```

`--obs-groups pointcloud` applies the `pointcloud` observation preset (single
front-view camera; enables `policy`, `proprio`, `goal`, `pointcloud`; 3-frame
`policy` / `proprio` history). For the 3-view variant, pass `--obs-groups 3view_pointcloud`
instead. `--enable_cameras` is auto-set by `create_demo_files_sequential.py` when a
camera-driven group/preset is selected.

The output file ends in `.pointcloud.seq.demo.h5` (the preset name is mixed into
the filename) and the root attribute `observation_preset` records which preset
was used, so downstream tools can spawn the env with the same obs space.

### 2. Convert to DP3 HDF5

```bash
python scripts/dp3/convert_demos_to_dp3.py \
    --input_h5 outputs/demos_h5/v0/functional/Dexverse-GraspPan-v0/Dexverse-GraspPan-v0.pointcloud.seq.demo.h5 \
    --out_file outputs/dp3_graspPan.hdf5 \
    --num_points 512
```

The converter auto-detects whether the pointcloud term is `camera_point_cloud_w`
(single-view) or `merged_point_cloud_w` (3-view) and stores it in the root
attribute `pointcloud_term`. By default, demos marked `success=False` are
dropped; pass `--include_failed` to keep them.

### 3. Inspect the converted dataset

```bash
python scripts/dp3/visualize_dp3_dataset.py --hdf5_file outputs/dp3_graspPan.hdf5 --mode info
python scripts/dp3/visualize_dp3_dataset.py --hdf5_file outputs/dp3_graspPan.hdf5 --episode 0 --frame 0
```

### 4. Train

```bash
python scripts/dp3/train.py \
    --hdf5_path outputs/dp3_graspPan.hdf5 \
    --output_dir runs/dp3_graspPan \
    --num_epochs 3000 \
    --batch_size 128 \
    --horizon 16 \
    --n_obs_steps 2 \
    --n_action_steps 8 \
    --device cuda
```

Training writes:

- `last.pt`: checkpoint from the latest epoch
- `best.pt`: checkpoint with the best nonzero validation loss
- `epoch_XXXX.pt`: periodic checkpoints controlled by `--save_every_epochs`
- `metrics.json`: config, per-epoch losses, and summary metrics

### 5. Evaluate online in simulation

```bash
python scripts/dp3/eval_online.py \
    --ckpt runs/dp3_graspPan/best.pt \
    --task Dexverse-GraspPan-v0 \
    --observation_preset pointcloud \
    --output_dir runs/dp3_graspPan_eval \
    --num_episodes 20 \
    --max_steps 500 \
    --replan_interval 8 \
    --device cpu --headless
```

This evaluation command uses CPU for physics and policy inference; rendering
still uses the GPU.

`--enable_cameras` is auto-set when the preset includes a camera-driven group
(`pointcloud` / `3view_pointcloud`). Pass `--no_auto_enable_cameras` if you need
to override that. For template tasks, pass the template JSON through `--json_path`.

> The preset used at eval time **must** match the preset that recorded the demos
> in step 1 (e.g. both `pointcloud`, or both `3view_pointcloud`). The obs preset
> controls history stacking on `policy` / `proprio`, so a mismatch silently
> changes `proprio_dim`.

## Programmatic API

Training can be called directly:

```python
from dexverse.IL.dp3.config import DP3DatasetConfig, DP3PolicyConfig, DP3TrainConfig
from dexverse.IL.dp3.train import train_main

cfg = DP3TrainConfig(
    dataset=DP3DatasetConfig(hdf5_path="outputs/dp3_graspPan.hdf5"),
    policy=DP3PolicyConfig(),
    output_dir="runs/dp3_graspPan",
    num_epochs=3000,
    batch_size=128,
)
summary = train_main(cfg)
```

For closed-loop inference from an existing environment:

```python
from dexverse.IL.dp3.runner import load_runner_from_checkpoint

runner, ckpt = load_runner_from_checkpoint(
    "runs/dp3_graspPan/best.pt",
    device="cuda",
    replan_interval=8,
)

obs, _ = env.reset()       # env constructed with observation_preset="pointcloud"
runner.reset()
action = runner.act(obs)   # reads obs["pointcloud"] and obs["proprio"]
```

`runner.act(obs)` expects an observation dict containing the `pointcloud` and
`proprio` groups as produced by the `pointcloud` / `3view_pointcloud` preset.

## Key Configuration

Dataset options are in `DP3DatasetConfig`:

- `hdf5_path`: converted DP3 HDF5 dataset
- `horizon`: full sequence length used by the diffusion model
- `n_obs_steps`: number of observation frames used for conditioning
- `n_action_steps`: number of predicted actions per policy call
- `val_ratio`: episode-level validation split

Policy options are in `DP3PolicyConfig`:

- `num_points`: point cloud size, filled from the dataset at runtime
- `proprio_dim` and `action_dim`: filled from the dataset at runtime
- `num_train_timesteps`: DDIM training noise steps
- `num_inference_steps`: DDIM sampling steps at inference
- `encoder_output_dim`: PointNet feature size
- `down_dims`, `condition_type`, and diffusion UNet settings

Evaluation options are in `DP3OnlineEvalConfig`, including checkpoint path,
task, episode count, maximum rollout length, device, `replan_interval`, and
`observation_preset`.
