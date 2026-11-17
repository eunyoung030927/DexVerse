# RESULTS_CV3: codec v3 (every-direction alignment + taxonomy pairs), 261116

**Winning Phase-A loss: none passes the gates.** Within the new loss family the best trade-off is **W-MSE + Barlow Twins at λ/5**
(`wb`, `cv3_train --norm white --lam-cyc 0.01 --lam-pair 0.01 --lam-bt 0.01`, run `wb_cl_l`). It removes the v2 collapse
(PC1 0.24, PR 9.3) and keeps cross-hand replay closest to E2* (31/40), but it does NOT align better than v2 and the probe does
not drop (GEO-H3 0.90 vs v2 0.84). It is the loss used for Phase B. **No variant qualifies for the DP stage** (pre-registered rule).

Design, pre-registered gates and amendments A1–A5: `artifacts/codec_v3/DESIGN.md`. Gate table: `measure/gates.md`.
Code (new files only): `xhand_latent/cv3_train.py, cv3_measure.py, cv3_report.py, cv3_taxonomy.py, cv3_coverage.py,
cv3_xreplay.py, cv3_xreplay_run.sh, cv3_xreplay_summary.py, cv3_nullspace.py, cv3_floor.py, cv3_viz.py, cv3_launch.sh,
cv3_queue.sh, cv3_measure_runs.sh, cv3_status.sh, cv3_replay.sh` (cv3_replay.sh = cv2_replay.sh with the output root moved).

## Interpretation (8 lines)
1. **Whitened (every-direction) alignment losses do not align better than v2; they mainly cost reconstruction.** At λ = 0.05
   (`w_cl`) joint RMSE is 2.1× E2* and pad residual 8.6 mm, yet whitened GEO same/rand only moves 0.87/0.76 → 0.84/0.74
   (v2: 0.81/0.65). λ/5 (`wb_cl_l`) keeps RMSE 1.33× / 7.3 mm with the same alignment. VICReg (λ 0.05) is harmful (RMSE 3.3×).
2. **The collapse is fixed, the hand signal is not.** Whitening (and the null term) keep PC1 0.14–0.34 and PR 6–15 (v2 0.85 / 1.4),
   but the standardized probe stays 0.87–1.00 (GEO-H3) and QDA / whitened kNN stay ≥ 0.95.
3. **Where the hand lives: the decoder null space.** Decoders use only 19–21 of 32 z directions (rank of dq/dz). The LDA-1 hand
   direction has 46–60 % of its energy there in E2*/v2 (random 40–43 %) and 77–90 % in the whitened variants: the stronger the
   alignment pressure on the functional directions, the more the residual hand code moves into directions no decoder reads.
   In v2, 62–65 % of the within-hand variance sat there (the inflated PC1). A null-space penalty (`--lam-null`, A4) removes
   this (v2n: null energy 0.47 ≈ random, within-hand null variance 0.10) but the probe stays 0.81–0.87: the hand moves back
   into functional directions, where the 4- vs 5-finger and per-hand kinematic support really differ.
4. **The registered W1 threshold is below its own floor.** GEO pairs are matched across hands only to 16 mm (H3) / 25 mm (ALL5)
   fingertip residual. Same-hand pairs at that residual give whitened same/rand 0.54–0.58 / 0.59–0.63 = what a perfectly
   invariant codec would score. v2 already closes ~66 % (H3) / ~35 % (ALL5) of the E2*-to-floor gap; no v3 variant closes more.
5. **Taxonomy pairs:** 66 static postures (Feix 33 incl. #13, #32 added; Kapandji 10; non-prehensile) + 6 dynamic
   interpolations → 14,673 paired rows (10,653 valid on all 5 hands), coverage C1 0.27 vs 0.04 for the v2 closure pairs.
   Training with them (`wb_tax_l`) helps its own family (TAX held-out-sample whitened 0.85 → 0.78 vs `wb_cl_l`, v2 0.83; held-out-class 0.93 → 0.89–0.90)
   but not GEO, not the probe, and costs fidelity (RMSE 1.51×, replay 27/40).
6. **Cross-hand Kettle replay** (Sharpa → 4 hands, /40, same pipeline as the 11/11 cohort, E2* reproduces 35/40):
   E2* 35, v2 cyc05p05 35, v2 cyc05clf05f002 33, best v3 `wb_cl_l` 31; every other v3 variant 16–28. Success tracks the pad
   residual, not the alignment metrics. Kettle is near ceiling (native 39/40), so it separates only the damaged codecs.
7. **Native round trip** (F1) for `wb_cl_l`: sharpa 9/10 (ep0, as E2*), shadow 10/10, wuji 10/10 → pass.
8. **DP stage not run:** "clearly good" needs W1–W5 + X1 ≥ v2 and E2* + F1; every v3 variant fails W1, W3, W4/W5 and X1.

## Phase A comparison (v2 pair data = functional closure `cl_func`; warm start E2*)
| run | loss | λ | steps | W1 white same/rand GEO5/GEO3/TAXC | PC1 / PR | probe GEO-H3 / ALL5 | RMSE max ×E2* | pad mm | X-replay /40 |
|---|---|---|---|---|---|---|---|---|---|
| E2* | – | – | – | 0.87/0.76/0.93 | 0.15 / 17.2 | 1.00 / 1.00 | 1.00 | 6.18 | 35 |
| v2 cyc05p05 | trace cyc+pair | .05 | 3000 | 0.80/0.63/0.91 | 0.86 / 1.3 | 0.95 / 0.96 | 1.03 | 6.45 | 35 |
| v2 cyc05clf05f002 | trace cyc+pair+Fisher | .05 | 3000 | 0.81/0.65/0.91 | 0.85 / 1.4 | 0.84 / 0.82 | 1.15 | 6.72 | 33 |
| `w_cl` | W-MSE | .05 | 3000 | 0.84/0.74/0.93 | 0.26 / 5.9 | 0.90 / 0.83 | 2.10 | 8.60 | 16 |
| `wv_cl` | W-MSE + VICReg | .05 | 3000 | 0.87/0.81/0.93 | 0.08 / 18.2 | 1.00 / 0.99 | 3.25 | 10.96 | 20 |
| `wb_cl_l` | W-MSE + BT | .01 | 3000 | 0.84/0.75/0.93 | 0.24 / 9.3 | 0.90 / 0.87 | 1.33 | 7.30 | **31** |
| `w_cl_l` (peek @1000, stopped) | W-MSE | .01 | 1000 | 0.83/0.72/0.92 | – | 0.93 / 0.90 | – | 7.60 | – |
| `wn_cl_l` (A4) | W-MSE + null | .01/.05 | 1500 | 0.83/0.71/0.91 | 0.14 / 14.8 | 0.99 / 0.99 | 2.59 | 9.02 | 25 |
| `v2n_cl` (A4) | v2 recipe + null | .05 | 1500 | 0.81/0.64/0.90 | 0.26 / 7.9 | 0.87 / 0.81 | 2.19 | 8.70 | 21 |
Pruned / not run (A3, A5): `wb_cl` (λ .05, stopped at 1500), `wc_cl`/`wc_cl_l` (CORAL; untested — compute), `wvbc_cl(_l)`,
`wvh_cl`, `wv_cl_l` (VICReg family dominated). CORAL is therefore the one requested loss that was not measured.

## Phase B: taxonomy pairs (`artifacts/codec_v3/data/tax_pairs.npz`, log `tax_pairs.json`, coverage `coverage.json`)
- Spec: `TAXONOMY_SPEC.md` + `TAXONOMY_SPEC_CHANGES.md` (Feix-corrected; applied). Dedup S-P13, S-R08, S-N14, S-N10; added
  S-R19 (Feix #13 Precision Sphere), S-I07 (#32 Ventral). 66 static postures × 256 Sobol samples (aperture, spread, per-finger
  flexion inside its level, cylinder roll, IK seed; same θ for every hand), FK-gradient IK per hand (PadFK, URDF/USD limits),
  4-finger hands: ring takes the pinky role, pinky-only postures masked, wrap aperture > 0.75 a_max masked (draft value).
  Dynamic: D01 D02 D03 D09 D10 D15 as paired joint interpolations; D04–D08, D11–D14 not generated (need contact dynamics).
- Rows kept (≥ 3 valid hands): **14,673** (train 11,197 / held-out samples 1,288 / held-out classes 2,188: S-P03, S-P10,
  S-I02, S-R05, S-R13, S-R19, S-K04, S-N06, S-N16, D02); valid per hand allegro 12,942, leap 13,518, shadow 12,985, wuji 13,344,
  sharpa 14,450; valid on all 5: 10,653. (posture × aperture-quartile) cells with ≥ 8 rows: 187/280 (all 5), 214/280 (H3).
- Infeasible (posture, hand) items (valid < 10 %): S-P02 allegro, wuji; S-P04/P05/P15 wuji; S-R11/R12/R14 shadow;
  S-K03/K04 allegro, shadow; S-K05 allegro, leap, shadow; S-K06 allegro, leap, shadow; S-K07/K08 allegro, leap (masked by
  rule); S-K09 allegro; S-K10 shadow, wuji, sharpa; S-N07 allegro, leap (masked by rule). Kapandji reach with the other fingers
  held E–S is low for most hands (spec §7.2 caveat; the IK did not search finger flexion to meet the thumb).
- Coverage (each set subsampled to 5000 per hand; mean / min over hands):
| set | C1 fingertip voxel occupancy | C2 pairwise-distance cell occupancy | C2 covering radius p95 (÷s) |
|---|---|---|---|
| taxonomy (valid rows) | 0.27 / 0.20 | 0.072 / 0.047 | 0.30 / 0.20 |
| taxonomy, paired on all 5 | 0.24 / 0.18 | 0.070 / 0.050 | 0.30 / 0.21 |
| v2 cl_func pairs | 0.04 / 0.02 | 0.052 / 0.016 | 0.46 / 0.25 |
| play_v1 (unpaired, all) | 0.71 / 0.51 | 0.172 / 0.079 | 0.14 / 0.12 |
| play_v1 synergy+pinch half | 0.23 / 0.17 | 0.037 / 0.021 | 0.36 / 0.27 |
- Phase-B runs (loss `wb`, λ/5, 1500 steps): `wb_tax_l` (taxonomy only) and `wb_cltax_l` (cl_func + taxonomy, half/half).
| run | W1 GEO5/GEO3/TAXC | TAXS ALL5 / H3 white | PC1 / PR | probe GEO-H3/ALL5 | RMSE ×E2* | pad mm play / TAX test | X-replay |
|---|---|---|---|---|---|---|---|
| `wb_cl_l` | 0.84/0.75/0.93 | 0.85 / 0.85 | 0.24 / 9.3 | 0.90 / 0.87 | 1.33 | 7.30 / 5.70 | 31 |
| `wb_tax_l` | 0.84/0.74/0.89 | 0.78 / 0.77 | 0.34 / 6.3 | 0.92 / 0.88 | 1.51 | 7.67 / 6.90 | 27 |
| `wb_cltax_l` | 0.84/0.73/0.90 | 0.80 / 0.79 | 0.29 / 7.4 | 0.93 / 0.90 | 1.50 | 7.64 / 7.15 | 28 |
(E2*: TAXS 0.87/0.84, TAX pad 4.63; v2: 0.83/0.82, 5.52.)

## Final gate table (DESIGN §5; `measure/gates.md`)
W1 sets: GEO_ALL5, GEO_H3, TAXC_ALL5 (whitened same/rand-cross; pass ≤ 0.8×v2 and < E2*)

| codec | W1 white same/rand | W2 PC1 / PR / min per-hand PR | W3 probe GEO-H3 / GEO-ALL5 | W4 max rmse/E2* | W5 pad mm (5↔5, 5→4, 4→5, 4↔4) | W1–W5 |
|---|---|---|---|---|---|---|
| E2 | 0.87/0.76/0.93 ✗ | 0.15 / 17.2 / 8.1 ✓ | 1.00 / 1.00 ✗ | 1.00 ✓ | 6.18 (4.9, 3.8, 10.6, 6.1) ✓ | fail |
| cyc05p05 | 0.80/0.63/0.91 ✗ | 0.86 / 1.3 / 1.2 ✗ | 0.95 / 0.96 ✗ | 1.03 ✓ | 6.45 (5.3, 4.0, 10.7, 6.5) ✓ | fail |
| cyc05clf05f002 | 0.81/0.65/0.91 ✗ | 0.85 / 1.4 / 1.3 ✗ | 0.84 / 0.82 ✗ | 1.15 ✓ | 6.72 (5.4, 4.3, 10.9, 7.5) ✗ | fail |
| v2n_cl | 0.81/0.64/0.90 ✗ | 0.26 / 7.9 / 3.9 ✓ | 0.87 / 0.81 ✗ | 2.19 ✗ | 8.70 (7.1, 6.0, 12.7, 10.8) ✗ | fail |
| w_cl | 0.84/0.74/0.93 ✗ | 0.26 / 5.9 / 3.6 ✗ | 0.90 / 0.83 ✗ | 2.10 ✗ | 8.60 (7.0, 5.4, 12.9, 11.5) ✗ | fail |
| wb_cl_l | 0.84/0.75/0.93 ✗ | 0.24 / 9.3 / 4.9 ✓ | 0.90 / 0.87 ✗ | 1.33 ✗ | 7.30 (5.9, 4.5, 11.5, 8.7) ✗ | fail |
| wb_cltax_l | 0.84/0.73/0.90 ✗ | 0.29 / 7.4 / 5.7 ✓ | 0.93 / 0.90 ✗ | 1.50 ✗ | 7.64 (6.3, 4.7, 11.9, 9.5) ✗ | fail |
| wb_tax_l | 0.84/0.74/0.89 ✗ | 0.34 / 6.3 / 5.8 ✗ | 0.92 / 0.88 ✗ | 1.51 ✗ | 7.67 (6.2, 4.7, 11.9, 9.7) ✗ | fail |
| wn_cl_l | 0.83/0.71/0.91 ✗ | 0.14 / 14.8 / 7.2 ✓ | 0.99 / 0.99 ✗ | 2.59 ✗ | 9.02 (7.3, 6.1, 13.3, 11.2) ✗ | fail |
| wv_cl | 0.87/0.81/0.93 ✗ | 0.08 / 18.2 / 11.3 ✓ | 1.00 / 0.99 ✗ | 3.25 ✗ | 10.96 (8.8, 8.4, 15.6, 13.9) ✗ | fail |

| codec | QDA GEO-H3 / ALL5 | kNN raw H3 / ALL5 | kNN white H3 / ALL5 | 4v5 / within-5 | white same/within GEO5 / GEO3 | SYN5 / CL5 white same/rand | TAXC H3 / TAXS ALL5 white s/r | TAX pad mm (test / held-out class) | Kettle white in-p95 allegro / leap ; kNN ratio | v2-G1 lin GEO5/GEO3 | v2-G2 same/within GEO5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E2 | 1.00 / 1.00 | 0.99 / 1.00 | 0.99 / 0.99 | 1.00 / 1.00 | 1.01 / 0.91 | 0.86 / 0.96 | 0.93 / 0.87 | 4.63 / 4.49 | 0.00 / 0.00 ; 12.5 / 13.9 | 1.00 / 1.00 | 1.04 |
| cyc05p05 | 1.00 / 1.00 | 0.83 / 0.75 | 0.98 / 0.97 | 1.00 / 0.93 | 0.88 / 0.68 | 0.58 / 0.87 | 0.93 / 0.81 | 4.78 / 4.64 | 0.00 / 0.00 ; 9.9 / 11.1 | 0.96 / 0.95 | 0.41 |
| cyc05clf05f002 | 1.00 / 1.00 | 0.83 / 0.72 | 0.96 / 0.93 | 1.00 / 0.77 | 0.86 / 0.67 | 0.71 / 0.85 | 0.93 / 0.83 | 5.52 / 5.11 | 0.00 / 0.00 ; 8.9 / 11.0 | 0.82 / 0.84 | 0.29 |
| v2n_cl | 1.00 / 1.00 | 0.88 / 0.80 | 0.95 / 0.91 | 1.00 / 0.73 | 0.86 / 0.66 | 0.68 / 0.74 | 0.91 / 0.82 | 6.83 / 6.59 | 0.00 / 0.00 ; 8.0 / 8.8 | 0.81 / 0.87 | 0.54 |
| w_cl | 0.99 / 0.99 | 0.82 / 0.74 | 1.00 / 0.99 | 0.99 / 0.88 | 0.94 / 0.82 | 0.79 / 0.91 | 0.93 / 0.86 | 6.92 / 6.77 | 0.00 / 0.00 ; 5.4 / 7.8 | 0.83 / 0.90 | 0.46 |
| wb_cl_l | 0.97 / 0.99 | 0.87 / 0.78 | 1.00 / 0.99 | 1.00 / 0.90 | 0.94 / 0.83 | 0.79 / 0.89 | 0.93 / 0.85 | 5.70 / 5.56 | 0.00 / 0.00 ; 6.1 / 8.5 | 0.87 / 0.90 | 0.76 |
| wb_cltax_l | 0.99 / 1.00 | 0.85 / 0.77 | 1.00 / 0.99 | 1.00 / 0.94 | 0.94 / 0.81 | 0.78 / 0.90 | 0.90 / 0.80 | 7.15 / 7.12 | 0.00 / 0.00 ; 6.9 / 8.4 | 0.90 / 0.93 | 0.80 |
| wb_tax_l | 1.00 / 1.00 | 0.87 / 0.78 | 1.00 / 0.99 | 1.00 / 0.91 | 0.94 / 0.83 | 0.77 / 0.93 | 0.90 / 0.78 | 6.90 / 6.93 | 0.00 / 0.00 ; 7.1 / 8.4 | 0.88 / 0.92 | 0.85 |
| wn_cl_l | 1.00 / 1.00 | 0.91 / 0.85 | 0.99 / 0.99 | 1.00 / 0.99 | 0.93 / 0.80 | 0.73 / 0.90 | 0.91 / 0.83 | 6.88 / 6.80 | 0.00 / 0.00 ; 8.5 / 10.0 | 0.99 / 0.99 | 0.76 |
| wv_cl | 1.00 / 1.00 | 0.88 / 0.75 | 1.00 / 0.99 | 0.99 / 1.00 | 1.02 / 0.99 | 0.86 / 0.96 | 0.94 / 0.88 | 8.96 / 8.86 | 0.00 / 0.00 ; 5.8 / 7.2 | 0.99 / 1.00 | 0.84 |
W-gate columns of the second table are secondary (DESIGN §4). E2 = E2*. Rows of v2 are the v2 checkpoints measured with the same code.

## Retarget pad residual per pair category (mm; E2* validation metric) on play val / taxonomy test / v2 closure pairs
| codec | play val: 5↔5 / 5→4 / 4→5 / 4↔4 / all | taxonomy test (held-out samples + classes) | v2 closure pairs (cl_func) |
|---|---|---|---|
| E2 | 4.9 / 3.8 / 10.6 / 6.1 / 6.18 | 3.7 / 3.5 / 7.2 / 4.2 / 4.63 | 2.6 / 2.6 / 10.2 / 2.3 / 4.41 |
| cyc05p05 | 5.3 / 4.0 / 10.7 / 6.5 / 6.45 | 4.0 / 3.4 / 7.4 / 4.5 / 4.78 | 3.2 / 2.6 / 10.1 / 2.6 / 4.81 |
| cyc05clf05f002 | 5.4 / 4.3 / 10.9 / 7.5 / 6.72 | 4.8 / 4.4 / 7.6 / 5.2 / 5.52 | 11.4 / 11.9 / 17.2 / 7.4 / 12.38 |
| w_cl | 7.0 / 5.4 / 12.9 / 11.5 / 8.60 | 6.2 / 5.1 / 9.1 / 7.8 / 6.92 | 6.6 / 5.0 / 14.8 / 5.2 / 8.38 |
| wv_cl | 8.8 / 8.4 / 15.6 / 13.9 / 10.96 | 7.8 / 7.3 / 11.2 / 10.7 / 8.96 | 7.8 / 6.8 / 19.7 / 11.3 / 10.98 |
| wb_cl_l | 5.9 / 4.5 / 11.5 / 8.7 / 7.30 | 4.9 / 4.1 / 8.1 / 5.6 / 5.70 | 5.0 / 3.9 / 10.7 / 3.6 / 6.24 |
| wn_cl_l | 7.3 / 6.1 / 13.3 / 11.2 / 9.02 | 5.6 / 5.0 / 9.9 / 7.9 / 6.88 | 4.7 / 3.7 / 10.7 / 3.4 / 6.03 |
| v2n_cl | 7.1 / 6.0 / 12.7 / 10.8 / 8.70 | 5.8 / 5.4 / 9.3 / 7.9 / 6.83 | 8.4 / 9.6 / 25.3 / 12.6 / 13.22 |
| wb_tax_l | 6.2 / 4.7 / 11.9 / 9.7 / 7.67 | 6.4 / 5.2 / 8.4 / 6.5 / 6.90 | 3.5 / 3.0 / 10.6 / 3.1 / 5.21 |
| wb_cltax_l | 6.3 / 4.7 / 11.9 / 9.5 / 7.64 | 7.0 / 5.6 / 8.4 / 5.9 / 7.15 | 4.5 / 3.3 / 10.7 / 3.9 / 5.91 |
The v2 closure-pair training damages retarget on the closure postures themselves (cyc05clf05f002 12.4 mm, v2n 13.2 mm vs E2* 4.4):
forcing equal z for "equal functional closure" fights the pad geometry. Taxonomy pairs do not show this (wb_tax_l 5.2 mm on closure).

## Cross-hand Kettle replay (X1) and native round trip (F1)
X1 (Sharpa → target, Kettle eps 0–9, init settle for every target; McNemar exact, paired by (hand, episode)):

| condition | allegro | leap | shadow | wuji | total /40 | vs E2* (+/−, p) | vs v2f (+/−, p) |
|---|---|---|---|---|---|---|---|
| e2s (E2*) | 10/10 | 8/10 | 9/10 | 8/10 | 35/40 | – | +4/−2, p=0.69 |
| v2f (cyc05clf05f002) | 10/10 | 8/10 | 8/10 | 7/10 | 33/40 | +2/−4, p=0.69 | – |
| v2p (cyc05p05) | 9/10 | 7/10 | 10/10 | 9/10 | 35/40 | +2/−2, p=1.00 | +5/−3, p=0.73 |
| v2n_cl | 8/10 | 7/10 | 0/10 | 6/10 | 21/40 | +0/−14, p=0.00 | +0/−12, p=0.00 |
| w_cl | 8/10 | 2/10 | 3/10 | 3/10 | 16/40 | +1/−20, p=0.00 | +1/−18, p=0.00 |
| wb_cl_l | 8/10 | 7/10 | 6/10 | 10/10 | 31/40 | +2/−6, p=0.29 | +4/−6, p=0.75 |
| wb_cltax_l | 8/10 | 8/10 | 5/10 | 7/10 | 28/40 | +1/−8, p=0.04 | +2/−7, p=0.18 |
| wb_tax_l | 9/10 | 8/10 | 3/10 | 7/10 | 27/40 | +0/−8, p=0.01 | +1/−7, p=0.07 |
| wn_cl_l | 9/10 | 8/10 | 1/10 | 7/10 | 25/40 | +1/−11, p=0.01 | +1/−9, p=0.02 |
| wv_cl | 10/10 | 4/10 | 2/10 | 4/10 | 20/40 | +0/−15, p=0.00 | +0/−13, p=0.00 |

**F1 native round trip** (p3h_roundtrip --arms Z + cv3_replay.sh = cv2_replay flags, Kettle eps 0–9; only `wb_cl_l`, the best-fidelity v3; E2* P0b = 9/10, 10/10, 9/10):
  Z_wb_cl_l shadow: 10/10 failed eps []
  Z_wb_cl_l sharpa: 9/10 failed eps [0]
  Z_wb_cl_l wuji: 10/10 failed eps []
- E2* (`e2s`) inputs are bit-identical to the 11/11 cohort inputs; 35/40 = the recorded 34/40 + allegro 10/10 with init fix.
- Exact McNemar vs E2*: wb_cl_l (−6/+2, p 0.29) is not significantly worse; every other v3 variant is (p ≤ 0.04). Shadow is the
  target that breaks first (5↔5 pad residual grows with every v3 loss).

## Why the standardized probe does not drop
- Whitening removes the incentive to inflate one direction (no collapse), but it also spends the loss budget on directions the
  decoders do not read: the hand code concentrates there (77–90 % of the LDA-1 energy in the decoder null space for w/wb/wv).
- Removing the null-space code (A4) is cheap for reconstruction but the probe then reads the hand from functional directions:
  the hands' reachable fingertip sets and joint couplings differ (4- vs 5-finger probe 1.00 everywhere; QDA 0.95–1.00), and
  GEO pairs are matched only to 16–25 mm, so part of what the probe reads is real geometry, not codec offset.
- A per-dimension-normalizing DP sees the same; whether that matters downstream can only be tested by the policy, and no
  v3 variant is good enough on fidelity (pad residual, replay) to justify that run.

## Kettle OOD (secondary, v2-comparable cv2_measure K1/K2)
| codec | K1 Kettle H3 η²_hand | K2 kNN ratio allegro / leap (Euclid) | K2 in-H3-p95 allegro / leap (Euclid) | K2 whitened kNN ratio vs LOHO allegro / leap | K2 whitened in-p95 | RT5 drift cross/rand |
|---|---|---|---|---|---|---|
| E2 | 0.50 | 16.2 / 15.2 | 0.00 / 0.00 | 0.98 / 1.09 | 0.00 / 0.00 | 0.76 |
| cyc05p05 | 0.08 | 6.3 / 6.3 | 0.01 / 0.00 | 0.89 / 1.00 | 0.00 / 0.00 | 0.24 |
| cyc05clf05f002 | 0.14 | 4.9 / 6.1 | 0.38 / 0.02 | 0.92 / 1.14 | 0.00 / 0.00 | 0.18 |
| w_cl | 0.23 | 7.1 / 8.4 | 0.24 / 0.00 | 0.61 / 0.89 | 0.00 / 0.00 | 0.30 |
| wv_cl | 0.36 | 7.1 / 7.5 | 0.00 / 0.00 | 0.64 / 0.79 | 0.00 / 0.00 | 0.41 |
| wb_cl_l | 0.27 | 11.0 / 12.8 | 0.00 / 0.00 | 0.74 / 1.03 | 0.00 / 0.00 | 0.42 |
| wn_cl_l | 0.28 | 8.1 / 9.1 | 0.00 / 0.00 | 0.78 / 0.92 | 0.00 / 0.00 | 0.47 |
| v2n_cl | 0.22 | 5.6 / 6.1 | 0.14 / 0.00 | 0.86 / 0.95 | 0.00 / 0.00 | 0.38 |
| wb_tax_l | 0.30 | 13.0 / 13.2 | 0.00 / 0.00 | 0.81 / 0.95 | 0.00 / 0.00 | 0.43 |
| wb_cltax_l | 0.29 | 11.3 / 12.0 | 0.00 / 0.00 | 0.76 / 0.93 | 0.00 / 0.00 | 0.43 |
Whitened Kettle in-p95 is 0 for every codec (the H3 reference p95 is tight in whitened units); allegro/leap sit about as far
from the H3 pool as one H3 hand from the other two (ratio vs LOHO 0.6–1.1) for every codec, including E2*.

## Recommendation / next step
- **Do not rerun cv2dp / H1 with a v3 codec.** The only v3 codec that keeps fidelity (`wb_cl_l`: replay 31/40, n.s. vs E2*;
  F1 pass) does not lower hand identifiability (probe 0.90/0.87, QDA ≈ 1.0), so a Zv3 DP is not expected to beat Zv2.
  If wanted anyway: re-encode + build `artifacts/cv3dp/` from the cv2dp builder (~1 h CPU), vast training as P3nh (~6–8 h),
  selection + sharpa 50 eps on work1 (~3 h) ≈ 12–15 h wall clock.
- **Cheaper, more informative next steps:**
  1. Policy-input hand-ID check (~1 h CPU): classifier on the actual DP state (z + pose9) for Ze2/Zv2/wb_cl_l; the wrist part
     alone already gives 0.59–0.79 (RESULTS_IDENT), so the finger codec may not be the binding identity source.
  2. Codec whose z contains only decoder-read (functional) directions (~1 day): z dim ≈ decoder rank (16–20) trained from
     scratch, or an encoder projection onto the per-hand functional subspace, with the pad/align objective as the main
     target; A4 shows the null-space code can be removed for free, but then the hand is read from real kinematic support
     differences, so pair data must be matched much tighter than GEO (16–25 mm) — e.g. the taxonomy IK or the dense
     functional-IK pairing being built in `artifacts/codec_v3alt/`.
  3. If hand identity cannot be removed, give it to the policy explicitly (hand graph / one-hot, shown harmless in P3) and
     test transfer with hand-conditioned DP instead of an invariant latent.

## Files (work2 `/home/eun/xhand/latent_codec_v1/artifacts/codec_v3/`)
- `DESIGN.md`, `RESULTS_CV3.md`, `TAXONOMY_SPEC.md`, `TAXONOMY_SPEC_CHANGES.md`
- `runs/<id>/{best.pt, last.pt, config.json, train.jsonl, validation.jsonl, done.json}`; aborted CPU try in `aborted/`
- `measure/<id>.json` (cv3 additions), `measure/cv2/<id>.json` (unmodified cv2_measure), `measure/gates.md`,
  `measure/floor.json`, `measure/nullspace_*.json`, `measure/clpairs.npz`; `peek/` (mid-run peeks used for A3/A5)
- `data/tax_pairs.npz` (+ `.json` log, shard files `tax_pairs.npz.part*`), `data/coverage.json`
- `xreplay/{inputs,inputs_initfix,settle,native_reference,base,pa,pb,pc,pd,pe}/` (cross-hand replay; allegro rows run on work1 and copied)
- `replay/` (native round trip F1), `viz/latent_viz.json`, `logs/`
- PC copy of the viz: `C:\Users\eunyo\Desktop\연구실\2026_2학기\Flow Chart\latent_viz_v3.json`
