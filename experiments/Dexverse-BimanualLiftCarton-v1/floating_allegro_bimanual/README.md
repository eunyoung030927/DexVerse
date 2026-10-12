# Dexverse-BimanualLiftCarton-v1: Shadow Hand -> Allegro Hand (`floating_allegro_bimanual`)

Retargeting of the released Dexverse-BimanualLiftCarton-v1 demos from `floating_shadow_bimanual` onto `floating_allegro_bimanual`.

| file | content |
|---|---|
| `config.yaml` | stage-3 settings in force (from the code defaults; each attempt also stores its own in `spider_aK.json`) |
| `STATUS.md`, `results.csv` | per-episode progress and verdicts (status_report.py) |

Data: `<runs>/Dexverse-BimanualLiftCarton-v1/demos/epNNN/floating_allegro_bimanual/` (kinematic.npz, spider_aK.json / _traj.pkl.gz, run.log, status.json); uploaded pickle: HF `<ns>/dexverse-bimanualliftcarton-v1-allegro-retargeting`.

## Notes

- Not supported by the retargeting yet (`dexverse/retarget/hands.py`); the queue skips it.
