# Dexverse-GraspPan-v1

Functional grasping: pan (handle grasp) and place flat on a stove top.

- Implementation: `dexverse.baseline_v1.config.functional.grasp_pan_cfg.GraspPanEnvFloatingDexHandRightCfg`
- Source demos: `floating_shadow_right`, `demonstrations/v1/functional/Dexverse-GraspPan-v1/demos.pkl` (dexverse/DexVerse_release): 50 episodes, 234-860 steps (mean 494) at 60 Hz
- Targets: [floating_allegro_right](floating_allegro_right/), [floating_inspire_right](floating_inspire_right/), [floating_leap_right](floating_leap_right/), [floating_sharpa_right](floating_sharpa_right/), [floating_wuji_right](floating_wuji_right/)
- Data: `<runs>/Dexverse-GraspPan-v1/demos/epNNN/<target>/` (runs = `$DEXVERSE_RETARGET_RUNS`, default `results/retarget/`)
