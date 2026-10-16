# Dexverse-GraspKettle-v1

Functional grasping: kettle (handle grasp) and pour over a mug.

- Implementation: `dexverse.baseline_v1.config.functional.grasp_kettle_cfg.GraspKettleEnvFloatingDexHandRightCfg`
- Source demos: `floating_shadow_right`, `demonstrations/v1/functional/Dexverse-GraspKettle-v1/demos.pkl` (dexverse/DexVerse_release): 50 episodes, 244-787 steps (mean 445) at 60 Hz
- Targets: [floating_allegro_right](floating_allegro_right/), [floating_inspire_right](floating_inspire_right/), [floating_leap_right](floating_leap_right/), [floating_sharpa_right](floating_sharpa_right/), [floating_wuji_right](floating_wuji_right/)
- Data: `<runs>/Dexverse-GraspKettle-v1/demos/epNNN/<target>/` (runs = `$DEXVERSE_RETARGET_RUNS`, default `results/retarget/`)
