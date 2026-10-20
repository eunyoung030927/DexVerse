# Dexverse-SlideUtilityKnife-v1

Two-phase utility knife: slide the blade out, then cut a taped seam.

- Implementation: `dexverse.baseline_v1.config.articulation.cut_seam_utility_knife_cfg.CutSeamUtilityKnifeEnvFloatingDexHandRightCfg`
- Source demos: `floating_shadow_bimanual`, `demonstrations/v1/articulation/Dexverse-SlideUtilityKnife-v1/demos.pkl` (dexverse/DexVerse_release): 50 episodes, 638-1189 steps (mean 816) at 60 Hz
- Targets: [floating_allegro_bimanual](floating_allegro_bimanual/), [floating_inspire_bimanual](floating_inspire_bimanual/), [floating_leap_bimanual](floating_leap_bimanual/), [floating_sharpa_bimanual](floating_sharpa_bimanual/), [floating_wuji_bimanual](floating_wuji_bimanual/)
- Data: `<runs>/Dexverse-SlideUtilityKnife-v1/demos/epNNN/<target>/` (runs = `$DEXVERSE_RETARGET_RUNS`, default `results/retarget/`)
