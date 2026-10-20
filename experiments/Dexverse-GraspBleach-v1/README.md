# Dexverse-GraspBleach-v1

Functional grasping: bleach bottle.

- Implementation: `dexverse.baseline_v1.config.functional.grasp_bleach_cfg.GraspBleachEnvFloatingDexHandRightCfg`
- Source demos: `floating_shadow_right`, `demonstrations/v1/functional/Dexverse-GraspBleach-v1/demos.pkl` (dexverse/DexVerse_release): 50 episodes, 311-933 steps (mean 588) at 60 Hz
- Targets: [floating_allegro_right](floating_allegro_right/), [floating_inspire_right](floating_inspire_right/), [floating_leap_right](floating_leap_right/), [floating_sharpa_right](floating_sharpa_right/), [floating_wuji_right](floating_wuji_right/)
- Data: `<runs>/Dexverse-GraspBleach-v1/demos/epNNN/<target>/` (runs = `$DEXVERSE_RETARGET_RUNS`, default `results/retarget/`)
