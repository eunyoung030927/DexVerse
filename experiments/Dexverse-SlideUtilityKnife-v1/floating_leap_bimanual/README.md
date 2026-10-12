# Dexverse-SlideUtilityKnife-v1: Shadow Hand -> LEAP Hand (`floating_leap_bimanual`)

Retargeting of the released Dexverse-SlideUtilityKnife-v1 demos from `floating_shadow_bimanual` onto `floating_leap_bimanual`.

| file | content |
|---|---|
| `config.yaml` | stage-3 settings in force (from the code defaults; each attempt also stores its own in `spider_aK.json`) |
| `STATUS.md`, `results.csv` | per-episode progress and verdicts (status_report.py) |

Data: `<runs>/Dexverse-SlideUtilityKnife-v1/demos/epNNN/floating_leap_bimanual/` (kinematic.npz, spider_aK.json / _traj.pkl.gz, run.log, status.json); uploaded pickle: HF `<ns>/dexverse-slideutilityknife-v1-leap-retargeting`.

## Notes

- Not supported by the retargeting yet (`dexverse/retarget/hands.py`); the queue skips it.
