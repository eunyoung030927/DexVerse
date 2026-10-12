# Dexverse-FunctionalHammerStrike-v1: Shadow Hand -> Inspire Hand (`floating_inspire_right`)

Retargeting of the released Dexverse-FunctionalHammerStrike-v1 demos from `floating_shadow_right` onto `floating_inspire_right`.

| file | content |
|---|---|
| `config.yaml` | stage-3 settings in force (from the code defaults; each attempt also stores its own in `spider_aK.json`) |
| `STATUS.md`, `results.csv` | per-episode progress and verdicts (status_report.py) |

Data: `<runs>/Dexverse-FunctionalHammerStrike-v1/demos/epNNN/floating_inspire_right/` (kinematic.npz, spider_aK.json / _traj.pkl.gz, run.log, status.json); uploaded pickle: HF `<ns>/dexverse-functionalhammerstrike-v1-inspire-retargeting`.

## Notes

- Not supported by the retargeting yet (`dexverse/retarget/hands.py`); the queue skips it.
