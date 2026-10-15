# Train and run DP and DP3

This walkthrough uses the `develop` branch and all 50 curated demonstrations for
`Dexverse-PushT-v1`. It covers demonstration replay, dataset conversion, two-epoch
training runs, and short online evaluations. Two epochs are a pipeline check;
they are not enough to establish policy performance.

## 1. Prepare the environment and demonstrations

Complete the [Isaac Lab and DexVerse installation](../README.md#installation)
first. Run the commands below from that DexVerse checkout's root:

```bash
conda activate dexverse
python -m pip install -e "source/dexverse[dp3]"
```

The `dp3` extra supplies the optional packages needed for both baselines. Keep
the PyTorch and NumPy versions from the benchmark installation. Use the same
checkout for conversion, training, and evaluation; an editable installation
points Python at that checkout.

Download the assets and curated pickle for this task:

```bash
python scripts/asset_tools/download_robot_agents.py --bundle shadow
python scripts/asset_tools/download_assets.py --core
python scripts/demo_tools/download_demos.py --task Dexverse-PushT-v1
```

The downloaders default to the public `dexverse/DexVerse_release` dataset and
require no login. PushT needs the core and Shadow bundles; the full asset
download is unnecessary for this example. Its demonstrations are stored at
`source/dexverse/demonstrations/v1/non_prehensile/Dexverse-PushT-v1/demos.pkl`.

To collect your own trajectories, follow
[Record demonstrations](../README.md#record-demonstrations-record_demospy).
For a raw session, replace `--task Dexverse-PushT-v1` in the replay commands below
with `--file /path/to/session.pkl`. Use the H5 path printed by the converter in
the subsequent dataset command, and adjust the expected episode count.
`--task` selects the curated release pickle; it does not search local recordings.

## 2. Generate observation H5 files

The released pickles contain actions and recorded scene states. Replay them in
Isaac Sim to generate the observations used by each policy. Here `--set-state`
restores the recorded states each frame, and `--device cpu` selects CPU physics.
Isaac Sim still needs a supported NVIDIA GPU for startup and rendering.

Generate state observations for DP:

```bash
python scripts/demo_tools/create_demo_files_sequential.py \
  --task Dexverse-PushT-v1 \
  --obs-groups state \
  --set-state --device cpu --seed 42 --headless \
  --output-dir outputs/baseline_smoke/replay
```

Generate point clouds and proprioception for DP3:

```bash
python scripts/demo_tools/create_demo_files_sequential.py \
  --task Dexverse-PushT-v1 \
  --obs-groups pointcloud \
  --set-state --device cpu --seed 42 --headless \
  --output-dir outputs/baseline_smoke/replay
```

Run these sequentially. Cameras are enabled automatically for point-cloud
replay, which takes longer than state replay. Both commands convert all
episodes; a complete curated set contains 50. The resulting files are:

```text
outputs/baseline_smoke/replay/v1/non_prehensile/Dexverse-PushT-v1/
├── Dexverse-PushT-v1.state.seq.demo.h5
└── Dexverse-PushT-v1.pointcloud.seq.demo.h5
```

For a smaller conversion check, add `--select-episodes 0 1 2` and choose a
different `--output-dir`. Existing H5s are reused only when the inputs, settings,
episode selection, and completeness match. See
[demo conversion](demo_conversion.md) for output reuse and selection options.

## 3. Build the training datasets

Flatten the state observations for DP:

```bash
python scripts/diffusion/build_dataset.py \
  --in_file outputs/baseline_smoke/replay/v1/non_prehensile/Dexverse-PushT-v1/Dexverse-PushT-v1.state.seq.demo.h5 \
  --out_file outputs/baseline_smoke/dp_dataset.h5
```

Prepare DP3 point clouds with 512 points per frame:

```bash
python scripts/dp3/convert_demos_to_dp3.py \
  --input_h5 outputs/baseline_smoke/replay/v1/non_prehensile/Dexverse-PushT-v1/Dexverse-PushT-v1.pointcloud.seq.demo.h5 \
  --out_file outputs/baseline_smoke/dp3_dataset.h5 \
  --num_points 512
```

These are offline conversions and do not launch Isaac Sim. The DP3 converter
skips episodes marked `success=False` by default; use `--include_failed` if you
want to include those episodes from your own recordings. Unlike replay H5 reuse,
these dataset builders replace their `--out_file` when rerun.

Check that both curated datasets contain 50 episodes before starting training:

```bash
python - <<'PY'
import h5py

for name in ("dp", "dp3"):
    path = f"outputs/baseline_smoke/{name}_dataset.h5"
    with h5py.File(path, "r") as dataset:
        count = len(dataset["data"])
        print(f"{name}: {count} episodes")
        assert count == 50, (path, count)
PY
```

## 4. Train each baseline for two epochs

These commands use CUDA for neural-network training and do not start the
simulator. Run them sequentially. Each trainer reserves some episodes for
validation, so its training split contains fewer than 50 episodes.

DP:

```bash
python -u scripts/diffusion/train.py \
  --task Dexverse-PushT-v1 \
  --dataset_file outputs/baseline_smoke/dp_dataset.h5 \
  --output_dir outputs/baseline_smoke/dp_run \
  --epochs 2 --batch_size 64 --num_workers 0 \
  --save_every_epochs 2 --seed 42 --device cuda:0
```

DP3:

```bash
python -u scripts/dp3/train.py \
  --hdf5_path outputs/baseline_smoke/dp3_dataset.h5 \
  --output_dir outputs/baseline_smoke/dp3_run \
  --num_epochs 2 --batch_size 16 --num_workers 0 \
  --save_every_epochs 2 --val_every 1 --log_every 1 \
  --seed 42 --device cuda:0
```

Check for finite training and validation losses in the terminal and in each
run's `metrics.json`. Each run should also produce `best.pt` and `last.pt`.
Use a new output directory for a new experiment. Reduce `--batch_size` if GPU
memory is insufficient; DP3's default network is larger than the state-based DP
network. Increase `--epochs` (DP) or `--num_epochs` (DP3) for longer training.

## 5. Load checkpoints and run short evaluations

Run each learned policy in simulation for two episodes, capped at 100 steps
each. These commands use CPU for both physics and policy inference; camera
rendering still uses the GPU. Keep the task version and observation preset the
same as in conversion and training.

DP:

```bash
python scripts/diffusion/eval_online.py \
  --task Dexverse-PushT-v1 \
  --ckpt outputs/baseline_smoke/dp_run/best.pt \
  --observation_preset state \
  --output_dir outputs/baseline_smoke/dp_run/eval \
  --num_episodes 2 --max_steps 100 --device cpu --headless
```

DP3:

```bash
python scripts/dp3/eval_online.py \
  --task Dexverse-PushT-v1 \
  --ckpt outputs/baseline_smoke/dp3_run/best.pt \
  --observation_preset pointcloud --num_points 512 \
  --output_dir outputs/baseline_smoke/dp3_run/eval \
  --num_episodes 2 --max_steps 100 --device cpu --headless
```

Evaluation writes `metrics.json` under each specified evaluation directory,
including success rate, episode length, and inference latency. The check is
that checkpoints load, observation/action shapes match, and rollouts complete.
A low success rate after two training epochs and 100-step rollouts is not by
itself a pipeline error. Use longer training, more evaluation episodes, and a
task-appropriate step limit to measure policy quality.

For other tasks, select the matching versioned demonstrations and assets, then
update the task IDs and paths throughout. Observation groups can differ across
tasks and versions; see [demo conversion](demo_conversion.md) and
[baseline task versions](../README.md#baseline-task-versions) before reusing a
checkpoint or changing presets.
