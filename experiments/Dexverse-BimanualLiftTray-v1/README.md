# Dexverse-BimanualLiftTray-v1

Bimanual: retrieve a food tray from a random level of a rack and set it level over a target on the tabletop -- without spilling the food.

- Implementation: `dexverse.baseline_v1.config.bimanual.retrieve_tray_from_rack_cfg.RetrieveTrayFromRackEnvFloatingShadowBimanualCfg`
- Source demos: `floating_shadow_bimanual`, `demonstrations/v1/bimanual/Dexverse-BimanualLiftTray-v1/demos.pkl` (dexverse/DexVerse_release): 50 episodes, 511-1214 steps (mean 766) at 60 Hz
- Targets: [floating_allegro_bimanual](floating_allegro_bimanual/), [floating_inspire_bimanual](floating_inspire_bimanual/), [floating_leap_bimanual](floating_leap_bimanual/), [floating_sharpa_bimanual](floating_sharpa_bimanual/), [floating_wuji_bimanual](floating_wuji_bimanual/)
- Data: `<runs>/Dexverse-BimanualLiftTray-v1/demos/epNNN/<target>/` (runs = `$DEXVERSE_RETARGET_RUNS`, default `results/retarget/`)
