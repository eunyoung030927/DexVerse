# Can the v1 environment reproduce the released v1 demos? (2026-10-12)

The cross-hand dataset retargets the **v1** demos (`dexverse/DexVerse_release` `demonstrations/v1/`), so the v1
environment in this branch (upstream `develop`, `baseline_v1/`, `BENCHMARK_REVISION = baseline-v1-2026-09-08`) must
match the one the demos were recorded in.

Method: the authors' sequential converter `scripts/demo_tools/create_demo_files_sequential.py` with the replay fix of
upstream `fix/replay-reproducibility` (b646dd9: `env.sim.reset()` before each episode, measured `replay_success`),
CPU physics, seed 42 (the demos' seed), first 5 episodes per task, two modes:
- **state restore** (`--set-state`): write the recorded scene state every step
- **action replay** (`--no-set-state`): restore the initial state, then only step the recorded actions

| task | state restore | action replay |
|---|---|---|
| GraspBleach-v1 | 5/5 | 1/5 |
| GraspPan-v1 | 5/5 | 1/5 |
| GraspKettle-v1 | 5/5 | 0/5 |
| FunctionalPourMug-v1 | 5/5 | 0/5 |
| FunctionalHammerStrike-v1 | 5/5 | 1/5 |
| BimanualLiftTray-v1 | 5/5 | 0/5 |
| BimanualLiftCarton-v1 | 5/5 | 0/5 |
| SlideUtilityKnife-v1 | 5/5 | 0/5 |
| OpenFlatFolder-v1 | 5/5 | 0/5 |

- **State restore 45/45**: scene, objects and success conditions match the demos. Retargeting reads the recorded states
  (stage 1), so the v1 demos can be used as they are.
- **Action replay 3/45**: open-loop replay does not reproduce the recorded runs, even with the authors' fix and CPU
  physics (the demos' `sim_device` is not recorded). Not needed for retargeting; it matters only if someone trains on the
  demos' actions and evaluates by replay. Same picture with `record_demos.py --replay_demos` on GPU physics
  (GraspBleach 1/33).
- Four v1 IDs are different tasks from their v0 namesakes: OpenFlatFolder-v1 = CloseFolderInsertShelf,
  SlideUtilityKnife-v1 = CutSeamUtilityKnife, BimanualLiftTray-v1 = RetrieveTrayFromRack,
  BimanualLiftCarton-v1 = CartonRegrasp (`dexverse/benchmark.py` `LEGACY_UPGRADE_TASKS`).

Data: `/workspace/dexverse/results/v1_replay_check/` (`run.sh`, `job.sh`, `summarize.py`, per-task HDF5 + logs).
