# Dexverse-OpenFlatFolder-v1

Bimanual: close an open folder, rotate it upright, and insert it in a slot.

- Implementation: `dexverse.baseline_v1.config.articulation.close_folder_insert_shelf_cfg.CloseFolderInsertShelfEnvFloatingDexHandRightCfg`
- Source demos: `floating_shadow_bimanual`, `demonstrations/v1/articulation/Dexverse-OpenFlatFolder-v1/demos.pkl` (dexverse/DexVerse_release): 50 episodes, 677-1042 steps (mean 824) at 60 Hz
- Targets: [floating_allegro_bimanual](floating_allegro_bimanual/), [floating_inspire_bimanual](floating_inspire_bimanual/), [floating_leap_bimanual](floating_leap_bimanual/), [floating_sharpa_bimanual](floating_sharpa_bimanual/), [floating_wuji_bimanual](floating_wuji_bimanual/)
- Data: `<runs>/Dexverse-OpenFlatFolder-v1/demos/epNNN/<target>/` (runs = `$DEXVERSE_RETARGET_RUNS`, default `results/retarget/`)
