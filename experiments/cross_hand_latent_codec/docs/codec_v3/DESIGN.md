# codec v3 (cv3): every-direction alignment + taxonomy pairs — DESIGN (pre-registered 261116 ~02:10 KST, before any v3 training)

Repo `/home/eun/xhand/latent_codec_v1` (work2 = master copy of artifacts/codec_v3; work1 = extra GPU/CPU lane, results copied to work2).
Code (new files only; cv2_* / arm_* untouched): `xhand_latent/cv3_train.py`, `cv3_measure.py`, `cv3_taxonomy.py`, `cv3_xreplay_prepare.py`.
Baselines: E2* = `artifacts/arm_latent_v1/runs/e2_lam2.8333/best.pt`; v2 = `artifacts/codec_v2/runs/cyc05clf05f002/best.pt` (and `cyc05p05`).

## 0. Winning Phase-A loss
**Final (261116 08:00): no Phase-A loss passes.** Best trade-off in the new family = **W-MSE + Barlow Twins at λ/5 (`wb`,
`cv3_train --norm white --lam-cyc 0.01 --lam-pair 0.01 --lam-bt 0.01`, run `wb_cl_l`)**: no collapse (PC1 0.24, PR 9.3),
cross-hand replay 31/40 (E2* 35, v2 33–35), native round trip pass; but whitened alignment equals E2*/v2 level and the
standardized probe does not drop (0.90/0.87). Used for Phase B. No variant qualifies for the DP stage. Details: RESULTS_CV3.md.

## 1. Problem (from codec_v2/viz/RESULTS_SEPARABILITY.md)
v2 divided L_cyc / L_pair by the non-detached trace V. Training inflated one direction (PC1 86–93 %, participation ratio 1.1–1.4);
the hand moved into low-variance PCs 17–32, where a standardized probe (and a per-dim-normalizing DP) still reads it
(GEO probe 0.76–0.96). Calibration probe (step 0, play val, this design): whitened cycle error of v2 `cyc05clf05f002` = 0.68
vs E2* 1.35 (random pair = 2.0), i.e. v2 halved the error in whitened units although its trace-normalized error fell 25×.
Pooled play-val z: E2* PC1 0.145 / PR 17.2; v2 PC1 0.85 / PR 1.38.

## 2. Phase A — losses (cv3_train.py), warm start E2*, 3000 steps, fresh Adam 1e-3, batch 512×5, seed 42, warm-up 300
L = L_E2* (unmodified arm_train.objective) + λc L_cyc + λp L_pair + λv L_var + λcov L_cov + λbt L_bt + λco L_coral
- **W-MSE** (`--norm white`): L_cyc, L_pair as in v2 (same cycle pairs 5→5, 4→4, 4→5; same pair construction) but each squared
  error is measured as ||(Δz) W||²/D, W = ZCA of the pooled native-batch z covariance (EMA 0.9, ridge 1e-3·mean eig), DETACHED.
  Rescaling or inflating a direction does not change the loss; low-variance directions weigh as much as PC1.
- **VICReg**: L_var = mean_d relu(0.5 − std_d / RMS-std) (RMS-std detached; scale-free), L_cov = Σ_{i≠j} C_ij² / D with C =
  pooled covariance / mean variance. `--vic-perhand`: the floor also applied per hand (relative to the pooled RMS std).
- **Barlow Twins / CCA**: on paired z (cycle pairs z_A ↔ Enc_B(sg Dec_B(z_A)) and every hand pair of the paired scenes):
  per-view batch standardization, C = cross-correlation, L_bt = (Σ(1−C_ii)² + (1/D) Σ_{i≠j} C_ij²)/D.
- **CORAL**: on paired scenes (rows valid for all hands), per-hand mean and covariance in whitened coordinates matched to the
  hand average (Frobenius / D). Only on paired (matched-content) data — never on the unpaired play marginals.
- Weights (fixed from the step-0 calibration, E2* init: W-cyc 1.35, W-pair 0.39, cov 0.82, bt 0.30, coral 0.44):
  λc = λp = 0.05 (same as v2; initial contribution 0.087 vs v2 0.070), λv = 0.05, λcov = 0.005, λbt = 0.05, λco = 0.05.
- Memory: arm_pads.contacts replaced at runtime by a cdist version (bit-identical rec/align on E2* val, checked).

| id | norm | VICReg | BT | CORAL | pairs |
|---|---|---|---|---|---|
| `w_cl` | white | – | – | – | cl_func (v2 pairs) |
| `wv_cl` | white | pooled | – | – | cl_func |
| `wvh_cl` | white | pooled + per-hand | – | – | cl_func |
| `wb_cl` | white | – | ✓ | – | cl_func |
| `wc_cl` | white | – | – | ✓ | cl_func |
| `wvbc_cl` | white | pooled | ✓ | ✓ | cl_func |
Pair data axis (user grid, 261116): v2 synthetic pairs (`cl` = v2 functional closure sampler, the v2 winner's data), taxonomy
pairs (`tax`, Phase B), both (`cltax`, half/half per step). After Phase A is measured, losses that fail W2/W4/W5 or are dominated
on (W1, W3) by another loss are pruned (named in RESULTS); survivors are trained with `tax` and `cltax`.
Unpaired play_v1 is always used for rec/align (as in E2*/v2).

## 3. Phase B — taxonomy paired postures (cv3_taxonomy.py)
Spec: `artifacts/codec_v3/TAXONOMY_SPEC.md` (+ TAXONOMY_SPEC_CHANGES.md, Feix-corrected; duplicates S-P13 = S-I01 and
S-R08 = S-P08 removed; Feix #13 Precision Sphere and #32 Ventral added). Each static posture = functional template:
contact pairs (thumb/finger pad, tip or radial side, palm) with an aperture a·s (s = hand scale), virtual object (sphere /
cylinder / thin plate) for wraps, per-finger flexion level (E/S/M/F) and spread (N/m/W), thumb ABD/ADD.
Continuous variation per sample θ = (aperture, spread, flexion jitter per finger, contact offset), same θ for every hand;
FK-gradient IK per hand (PadFK, joint limits from the URDF/USD tables), 4-finger hands use the ulnar-most (ring) finger for
the pinky role; pinky-dependent postures (K06–K10 etc.) are masked per hand. A hand's sample is valid if contact error ≤ 0.02 s,
joint margin respected, no gross finger–finger interpenetration (pad point distance check); invalid → masked, sample kept only
if ≥ 3 hands valid. Infeasible (posture, hand) items are logged. Split: held-out CLASSES (≈15 % of posture ids, fixed list,
stratified by power/intermediate/precision/Kapandji/non-prehensile) + held-out samples (10 %) of training classes.
Coverage metrics (spec §6.3, computable): C1 fingertip voxel occupancy (0.05 s grid, palm frame, per hand / mean / min) vs a
uniform-joint Monte-Carlo reachable set; C2 thumb–finger pairwise-distance grid occupancy + covering radius; C3 Kapandji reach;
C4 (posture × aperture-bin) cells with ≥ 8 valid samples (all-hand intersection). Reported for taxonomy data, play_v1 and v2
cl_func pairs.

## 4. Measurement (cv3_measure.py; CPU; codec-independent sets reused for every codec)
Kept from v2 for comparability: everything in cv2_measure (GEO/SYN/RT blocks, quality, Kettle K1/K2) — run unchanged into
`artifacts/codec_v3/measure/cv2/`.
Added (artifacts/codec_v3/measure/<id>.json):
- Hand-ID on GEO-val ALL5 / H3 and TAX-test (held-out classes, ALL5 masked and H3): standardized multinomial LR (v2 probe),
  QDA (full Gaussian), kNN-15 on raw z and on pooled-whitened z; 4-vs-5 binary and 3-way within-5-finger accuracy on ALL5.
  Interpretation: 4- vs 5-finger support differs irreducibly (perfect 4-vs-5 with chance inside groups = 0.40 on ALL5), so
  the hand-ID numbers are secondary; GEO-H3 is the cleanest hand-ID test.
- Whitened paired distances: ZCA from the codec's own pooled play-val z (4000 per hand × 5). same / rand-cross (paired same
  posture across hands vs different posture across hands) and same / within (vs same hand different posture), on GEO-val ALL5,
  GEO-val H3, SYN-ALL5, TAX-test held-out classes and TAX-test held-out samples.
- Collapse: PC1 share and participation ratio of pooled play-val z; per-hand PR on GEO-ALL5.
- Retarget per pair category (5↔5, 5→4, 4→5, 4↔4) pad residual (E2* validation metric) on play val[:2048], on TAX-test
  (valid source rows) and on the v2 pair sets (cl_func val draw).
- Kettle K2 in whitened coordinates: allegro/leap kNN-10 distance ratio to the H3 pool (eps 0–7) vs H3 ref (eps 8–9) and
  fraction inside the H3 ref p95, plus the v2 Euclidean numbers.

## 5. Gates (pre-registered; v2 = cyc05clf05f002 measured with the same code)
| gate | metric | pass |
|---|---|---|
| W1 conditional alignment (PRIMARY) | whitened same/rand-cross on GEO-ALL5, GEO-H3 and TAX held-out classes | each ≤ 0.8 × v2 and < E2* |
| W2 no collapse | pooled play-val PC1 share; PR; min per-hand PR (GEO-ALL5) | PC1 ≤ 0.30, PR ≥ 6, per-hand PR ≥ 3 |
| W3 hand-ID | standardized linear probe GEO-H3 / GEO-ALL5 | ≤ v2 − 0.10 on both (v2: 0.84 / 0.82) |
| W4 reconstruction | per-hand val joint RMSE | ≤ max(1.15 × E2*, E2* + 0.005) (= v2 G4) |
| W5 retarget | pad residual overall; per category | overall ≤ 1.10 × E2* (6.80 mm); each category ≤ 1.20 × E2* |
| W6 Kettle OOD (secondary) | whitened in-H3-p95 fraction allegro, leap | reported; ≥ v2 noted |
Replay gates (only for codecs that pass W1–W5, plus E2* and v2 with the identical pipeline):
| F1 native round trip | p3h_roundtrip --arms Z + replay_sim (cv2_replay.sh flags), Kettle eps 0–9, sharpa/shadow/wuji | ≥ 9/10 each |
| X1 cross-hand Kettle replay | Sharpa demos → Enc_sharpa → Dec_B, Kabsch wrist (arm_prepare pipeline, condition copy writing to codec_v3/xreplay), init settle for all targets, replay_sim same flags as the 11/11 cohort; B ∈ allegro, leap, shadow, wuji, eps 0–9 (/40) | total ≥ v2 and ≥ E2* (E2* recorded 34/40, 35/40 with allegro init fix; rerun here) |

**"Clearly good" ⇒ DP stage** (user rule 261116, fixed before any replay number): W1 and W2 and W3 pass (= improve over v2 on
whitened paired distance, no collapse, lower standardized probe), W4/W5 pass, X1 ≥ v2 and ≥ E2*, F1 pass.
If several qualify: the one with the best W1 (mean of the three ratios), then lower GEO-H3 probe.
DP stage: arm `Zv3` in a copy of the cv2dp setup (wuji+shadow train → unseen sharpa, identical recipe; builder/launcher/eval
reused read-only; outputs `artifacts/cv3dp/`, vast queue marker `/workspace/xhand/runs/cv3dp`), selection on wuji/shadow,
final sharpa 50 eps; exact McNemar paired by episode vs cv2dp Ze2/Zv2. If nothing qualifies: stop after replay and report.

## 6. Resources / rules
work1 3090: ≤ 4 GB own GPU memory (one training at a time, memory fraction 0.12–0.14), never slow cv2dp eval (from ~08:00 KST
→ codec training moves to CPU/work2 then). work2 4090: ≤ 3 GB (one run, fraction 0.10 with lean contacts). Measurement CPU.
Big files on work2. No touching cv2dp / handmask / pd3 / vast cv2dp or pi0.5 processes. No pkill, no NAS scans, no push.

## 7. Amendments
- A1 (261116 02:15, before any v3 replay; procedural only): X1 replay writes its own native-reference profile dir
  (artifacts/codec_v3/xreplay/native_reference) because the replay_sim profile gained fields after 11/11 (object-material
  bookkeeping; default as_recorded = same physics) and refuses the old arm_latent_v1 profile. The E2* inputs prepared by
  cv3_xreplay are bit-identical to the arm_latent_v1 cohort inputs (checked for all 4 targets). Init settle is applied to every
  target for every condition (the 11/11 primary used it only for leap). E2* (e2s), v2 cyc05clf05f002 (v2f) and cyc05p05 (v2p)
  are replayed with the same pipeline as the baseline rows of X1.
- A2 (261116 02:15): TAXONOMY_SPEC_CHANGES.md (Feix check) was present when the generator was written and is applied
  (cv3_taxonomy.POSTURES). Dedup: S-P13 (= S-I01), S-R08 (= S-P08), S-N14 (= S-I04), S-N10 (= S-N01). Added: S-R19 (Feix #13
  Precision Sphere), S-I07 (Feix #32 Ventral). Dynamic primitives D04-D08, D11-D14 need contact dynamics and are not generated
  (logged); D01-D03, D09, D10, D15 are joint interpolations of paired endpoint solutions (same t for all hands).
- A3 (261116 02:57, AFTER mid-run validation of `w_cl` (step 1500) and `wv_cl` (step 2500) and a peek measurement at step
  1000/1500; gates unchanged, exploratory): at λ = 0.05 the whitened terms dominate. Validation rec 0.0052 / 0.0091 vs E2*
  0.0021, pad residual (index) 9.4 / 11.1 mm vs 6.7, and the peek GEO whitened same/rand only 0.84 / 0.86 (E2* 0.87): both
  are on track to fail W4/W5 without passing W1. Same rule as v2 ("one λ step if every variant fails G4/G5 in the same
  direction"): the remaining λ = 0.05 cells (wc_cl, wvbc_cl, wvh_cl) are not run, `wb_cl` was stopped at step ~1500, and the grid
  is rerun at λ/5: `w_cl_l` (λc = λp = 0.01), `wb_cl_l` (+ λbt 0.01), `wc_cl_l` (+ λco 0.01), `wv_cl_l` (+ λv 0.01, λcov 0.001),
  `wvbc_cl_l` (all). `w_cl` and `wv_cl` are finished and measured as the λ = 0.05 rows.
- Analysis added before the final Phase-A numbers (cv3_nullspace.py, E2* and v2 only): the decoders use only 19–21 of the 32
  z directions (rank of dq/dz); the hand-discriminative LDA-1 direction has 46–60 % of its energy in the decoder null space
  (random direction: 40–43 %), and in v2 62–65 % of the within-hand z variance lies there (E2*: 14–17 %), i.e. the v2 PC1
  inflation used directions the decoder ignores.
- A4 (261116 03:15, exploratory, from the null-space analysis, before any λ/5 result): new term `--lam-null` (cv3_train.null_term):
  for each hand, z content in the null space of dDec_h/dz (detached SVD basis per point, 64 points/hand/step) is pulled to
  the POOLED batch mean, / pooled mean variance — free for that hand's reconstruction by construction. Variants:
  `wn_cl_l` (λ/5 W-MSE + λnull 0.05) and `v2n_cl` (exact v2 winner recipe cyc05clf05f002 — trace norm, λc = λp = 0.05,
  cl_func, Fisher 0.002 — + λnull 0.05). Initial values: E2* 0.21, v2 0.68. Same gates.
- A5 (261116 04:30; compute + a metric-floor finding; gates unchanged):
  (a) Throughput: from ~03:00 the cv2dp policy eval occupies the work1 GPU (3 Isaac sims, 100 %); to keep within "never slow it"
  work1 runs ONE codec job at a time and steps take ~5 s (work2 ~2 s). Final Phase-A λ/5 rows: `wb_cl_l` (3000 steps);
  `w_cl_l` stopped at step 1000 (its peek at step 1000 ≈ `wb_cl_l` at 1500: GEO white 0.83/0.72, probe 0.90/0.93, pad 7.6 mm);
  `wc_cl_l`, `wv_cl_l`, `wvbc_cl_l` not run (VICReg already harmful at 0.05; CORAL pruned for compute — stated as untested).
  Remaining exploratory runs use 1500 steps (v2 runs had reached ~90 % of their final validation change by 1500):
  `v2n_cl`, `wn_cl_l` (null term), and Phase B with the best Phase-A W-family loss by the §5 rule (W1 equal within 0.01,
  lower probe → `wb` = W-MSE + BT, λ/5): `wb_tax_l` (taxonomy pairs), `wb_cltax_l` (v2 pairs + taxonomy, half/half).
  (b) Floor of W1 (cv3_floor.py, measured on E2*/v2/w_cl only): the GEO pairs are matched across hands only to a median
  fingertip residual of 16 mm (H3) / 25 mm (ALL5). Pairs of the SAME hand at the same fingertip residual give whitened
  same/rand 0.54–0.58 (16 mm) and 0.59–0.63 (25 mm) — the value a perfectly hand-invariant codec would reach. The registered
  W1 thresholds (≤ 0.8 × v2 = 0.52 H3 / 0.64 ALL5) are therefore at or below this floor for H3; W1 is kept as registered and
  reported together with the floor-relative gap (codec − floor) / (E2* − floor).
