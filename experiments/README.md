# experiments/

Numbers, configs and notes of the DexVerse cross-hand retargeting (tracked in git), laid out like Bench2Dex's
`experiments/` (HyeonseokE/bench2dex_retarget). Heavy outputs (references, kinematic solutions, SPIDER trajectories,
logs, trajectory pickles) are in `$DEXVERSE_RETARGET_RUNS` (default `results/retarget/`, git-ignored; cluster
`~/dexverse/runs/retarget/`).

```
experiments/
  STATUS.md                     all (task, target hand) pairs: success / failed / in progress
  ENVIRONMENT.md                machines, software versions, simulation settings (DexVerse physics unchanged)
  benchmark_plan.md             scope (9 v1 tasks x 5 hands), dataset counts, pipeline, output format, compute estimate
  stage3_settings.md            current SPIDER / env / attempt settings and why (60 Hz port of the Bench2Dex "v2")
  v1_reproducibility.md         can the v1 environment reproduce the released v1 demos (state vs action replay)
  <task>/                       the 9 v1 tasks (Dexverse-GraspBleach-v1 ...)
    README.md                   task description, source demos, target hands
    <target>/                   the 5 target robot_types (floating_<hand>_right or floating_<hand>_bimanual)
      README.md                 the pair, where its data is, notes
      config.yaml               stage-3 / env / attempt settings in force (generated from the code)
      STATUS.md, results.csv    per-episode progress and verdicts
```

`scripts/retarget/init_experiments.py` creates the tree (keeps existing READMEs, rewrites config.yaml);
`scripts/retarget/status_report.py` writes the STATUS/results files (`run_queue.py` does both, every 2 min).
On the cluster `DEXVERSE_EXPERIMENTS` points them to `~/dexverse/runs/report/` so the git checkout stays clean;
copy that folder here to record cluster results. Each SPIDER attempt also stores its exact settings in
`<runs>/<task>/demos/epNNN/<target>/spider_aK.json`.
