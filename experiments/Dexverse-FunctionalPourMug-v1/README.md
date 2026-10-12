# Dexverse-FunctionalPourMug-v1

Functional grasping: pour mug.

- Implementation: `dexverse.baseline_v1.config.functional.pour_mug_cfg.PourMugEnvFloatingDexHandRightCfg`
- Source demos: `floating_shadow_right`, `demonstrations/v1/functional/Dexverse-FunctionalPourMug-v1/demos.pkl` (dexverse/DexVerse_release): 50 episodes, 261-890 steps (mean 522) at 60 Hz
- Targets: [floating_allegro_right](floating_allegro_right/), [floating_inspire_right](floating_inspire_right/), [floating_leap_right](floating_leap_right/), [floating_sharpa_right](floating_sharpa_right/), [floating_wuji_right](floating_wuji_right/)
- Data: `<runs>/Dexverse-FunctionalPourMug-v1/demos/epNNN/<target>/` (runs = `$DEXVERSE_RETARGET_RUNS`, default `results/retarget/`)
