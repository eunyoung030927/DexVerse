# Dexverse-BimanualLiftCarton-v1

Bimanual carton hold by a per-episode **target face** into a goal box.

- Implementation: `dexverse.baseline_v1.config.bimanual.carton_regrasp_cfg.CartonRegraspEnvFloatingShadowBimanualCfg`
- Source demos: `floating_shadow_bimanual`, `demonstrations/v1/bimanual/Dexverse-BimanualLiftCarton-v1/demos.pkl` (dexverse/DexVerse_release): 50 episodes, 244-1299 steps (mean 700) at 60 Hz
- Targets: [floating_allegro_bimanual](floating_allegro_bimanual/), [floating_inspire_bimanual](floating_inspire_bimanual/), [floating_leap_bimanual](floating_leap_bimanual/), [floating_sharpa_bimanual](floating_sharpa_bimanual/), [floating_wuji_bimanual](floating_wuji_bimanual/)
- Data: `<runs>/Dexverse-BimanualLiftCarton-v1/demos/epNNN/<target>/` (runs = `$DEXVERSE_RETARGET_RUNS`, default `results/retarget/`)
