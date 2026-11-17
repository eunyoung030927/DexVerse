# Cross-hand finger latent codec (codec v2 / v3) and image-DP transfer experiments

Code and write-ups for the hand-invariant finger latent experiments on Dexverse tasks (GraspKettle, FunctionalHammerStrike).
A per-hand MLP VAE (E2*, z = 32) encodes finger joints; the palm pose (pos 3 + rot6d 6) stays outside the latent, giving a
41-D state/action for a pooled multi-hand image Diffusion Policy (LeRobot 0.4.2).

## Codecs
- **E2\*** (baseline, `arm_train.py`): per-hand encoder/decoder, reconstruction + cross-hand fingertip alignment + KL.
- **codec v2** (`cv2_train.py`, design `docs/codec_v2/DESIGN.md`, results `docs/codec_v2/RESULTS_CV2.md`): warm start from E2*,
  adds cycle consistency, paired-scene alignment (synergy or functional closure pairs) and a Fisher term. Errors are divided by
  the total z variance. Downstream candidate: `cyc05clf05f002`.
- **codec v3** (`cv3_train.py`, `cv3alt_*.py`, design `docs/codec_v3/DESIGN.md`, results `docs/codec_v3/RESULTS_CV3.md`):
  alignment measured in whitened z (W-MSE), plus VICReg / Barlow Twins / CORAL / decoder null-space variants, and new paired data
  (taxonomy, play-IK, Sobol, human). Downstream candidate: `wb_cl_l`.

## Policy experiments
- **cv2dp** (`cv2dp_*`, `docs/cv2dp/RESULTS_CV2DP.md`): train shadow + wuji, test on sharpa (policy-unseen). E2* 0/50 vs v2 15/50.
- **cv3dp** (`cv3dp_*`, `docs/codec_v3/cv3dp/PREREG.md`): train shadow + wuji + allegro, test on sharpa (5-finger) and leap
  (4-finger); arms E2* / v3 (Z) and EE_common palm (D). Results pending.
- **cv3hm** (`cv3hm_*`, `docs/codec_v3/cv3hm/PREREG.md`): same design on FunctionalHammerStrike. Results pending.

`code/` is a flat copy of the `xhand_latent` package files these experiments use (with their imports). Checkpoints, datasets,
demo pickles, logs and videos are not included. Host names in launch scripts (`work1`, `work2`, `<VAST_HOST>`) are
placeholders for the original machines.
