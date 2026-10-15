# Download demonstrations and convert to H5

For commands that continue from H5 conversion through DP/DP3 dataset preparation,
training, and online evaluation, see [Train and run DP and DP3](baseline_training.md).

Run from the DexVerse checkout installed in your `dexverse` environment, using
NumPy 1.26 and the task's assets. Only load trusted pickles. Isaac Sim requires a
supported NVIDIA GPU for startup/rendering even when physics uses CPU.

## Download

The downloader defaults to the public `dexverse/DexVerse_release` dataset; no login is required:

```bash
conda activate dexverse
python scripts/demo_tools/download_demos.py --baseline
```

`--baseline` selects available v0 and v1 sets. Use `--version v0` or `--version v1`
to narrow selection, or `--task Dexverse-PushT-v1` for one set. Follow the dataset's
access requirements if authentication is requested. Files are checksum-verified;
different local files are not overwritten. Use `--dest PATH` for another location.

Downloads use `source/dexverse/demonstrations/v0|v1/<category>/<task>/demos.pkl`.
Consult the dataset card for the demonstration license.

## Convert by task and version

```bash
python scripts/demo_tools/create_demo_files_sequential.py \
  --task Dexverse-PushT-v1 --obs-groups state --device cpu \
  --set-state --output-dir outputs/h5
```

Change `-v1` to `-v0` to select the original task. Output goes under
`outputs/h5/v1/non_prehensile/Dexverse-PushT-v1/`; v0 uses a separate directory.
All trajectories are converted by default (50 in a complete set).

- Add `--select-episodes 0 1 2` for a small test, or `--dry-run` to inspect selection
  without launching the simulator.
- Use `--demos-root PATH` for a different download directory.
- Use `--obs-groups rgb` for image observations. Groups depend on the task:
  Push-T v0 needs `--obs-groups privileged policy proprio` for object/goal state.
- Task selection uses only the curated `demos.pkl`, not raw sessions or backups.
  Use `--file PATH` explicitly for a raw recording.
- Existing H5s are reused only when inputs, selection, settings and completeness
  match. Otherwise choose a new output directory; `--overwrite` deliberately
  replaces existing output.

CPU physics and set-state replay are defaults. Set-state restores recorded
states; it does not verify action-driven success or identical contact physics.
Run the scripts with `--help` for additional options.
