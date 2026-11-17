# codec v2 (cv2): hand-invariant finger latent — DESIGN (pre-registered 261115, before any training)

Repo: `/home/eun/xhand/latent_codec_v1` (work1 = training/replay, work2 = storage/master copy).
Artifacts: `artifacts/codec_v2/` (runs/, measure/, replay/). Code: `xhand_latent/cv2_*.py` (new files only; E2* code paths untouched).
Baseline: E2* = `artifacts/arm_latent_v1/runs/e2_lam2.8333/best.pt` (md5 2f74c352…, step 7500).

## 1. Problem (from RESULTS_IDENT / RESULTS_ZPAIR, H1)
- E2* loss = rec + λ·align (Dec_B(z_A) pad Kabsch residual, all ordered cross pairs) + β·KL. Nothing ties Enc_B to z_A:
  decoders are many-to-one, each hand keeps its own z region. Linear hand probe = 1.00 even on codec-independent paired scenes;
  same-scene cross-hand distance ≈ within-hand different-pose distance; ~half of it is a constant per-hand offset; the rest is
  hand-specific shape (quadratic/kNN probe after centering still 0.9–1.0).
- H1: 3-hand (sharpa/shadow/wuji) z-policy → allegro/leap 0/40. Unseen-hand state z is out of distribution.

## 2. Mechanisms considered and choice
| mechanism | decision | why |
|---|---|---|
| per-hand offset removal (centering) | **V0, no training** | ~50% of the same-scene distance is a constant offset; zero-cost calibration; baked exactly into the weights (Enc mean bias −o_h, Dec first-layer bias +W·o_h) so every downstream tool (load_arm/HoCodec/p3h_roundtrip/zpair) works unchanged. Offsets o_h estimated on a separate SYN calibration draw (seed 1), reference = 5-hand mean. Cannot fix the shape part. |
| cycle consistency Enc_B(Dec_B(z_A)) ≈ z_A | **main lever (V1)** | directly attacks the diagnosed cause (nothing requires Enc_B to return z_A; RT drift 5.1 vs 0.3). Needs no paired data. |
| paired alignment on synergy scenes | **ablation (V2/V3)** | only codec-independent pair source available for training is the SYN construction (same named prototypes, same r) = same family as the play synergy quarter. Semantic, not geometric, so it may fight the pad-geometry term; used as a modest-weight add-on. Makes the SYN eval set in-distribution → GEO is the primary held-out set. |
| adversarial / HSIC hand removal | not used | play_v1 is unpaired and per-hand marginals differ legitimately (uniform joint sampling over different kinematics); marginal invariance would fight geometry. On paired data a direct L2 (V2) is stronger and simpler than an adversary. |
| shared encoder trunk conditioned on hand graph | not used (now) | requires from-scratch training and a new checkpoint format (all downstream HandVAE tooling breaks); prior slot-trunk runs exist (`artifacts/runs/slots_*`). Revisit only if V1/V2 fail on the shape part. |

### 2.1 Losses (V1/V2/V3), warm-start from E2* (same HandVAE per hand, z=32, same play_v1, same batch 512 × 5 hands)
L = L_E2*(rec + 2.8333·align + 1e-5·KL, computed by the unmodified `arm_train.objective`) + λc·L_cyc + λp·L_pair
- **L_cyc** (allowed ordered pairs A→B only when fingers(A) ⊆ fingers(B): 5→5, 4→4, 4→5 = 14 of 20; 5→4 is skipped because a
  4-finger hand cannot carry the little-finger state, so that cycle cannot close without steganography):
  z_A = μ_A(q_A) (encoder mean), q̃ = stopgrad(Dec_B(z_A)) + N(0, 0.02²) (normalized joint units), ẑ = μ_B(q̃),
  L_cyc = mean ||ẑ − z_A||² / V, V = mean squared distance of the batch's native z (all hands) to their mean (not detached ⇒
  scale-invariant: shrinking z cannot lower it). Stop-grad through Dec_B + input noise = no hidden hand code in q̃.
  Gradient reaches Enc_B (map retargets back to z_A) and Enc_A (meet in the middle).
- **L_pair** (V2/V3): per step 256 SYN scenes: common prototype names (open, fist, curl_thumb/index/middle/ring), same (i, j, r)
  for every hand, + N(0, 0.055) per hand (play synergy noise), clip ±0.995. z_h = μ_h(q_h).
  L_pair = mean_s mean_h ||z_hs − mean_h z_hs||² / V.
- Fine-tune 3000 steps, fresh Adam lr 1e-3, λc/λp linear warm-up over 300 steps, grad clip 10, seed 42. Validation every 500 on
  play val[:2048] (E2* metrics + L_cyc/L_pair on val); keep best.pt by validation total.

### 2.2 Variants (cheap: ≈20–30 min each on the idle work1 3090, run in parallel)
| id | λc | λp | note |
|---|---|---|---|
| E2* | – | – | baseline |
| V0 `e2c` | – | – | E2* + SYN-offset centering baked (no training) |
| V1a `cyc01` | 0.01 | 0 | cycle, low |
| V1b `cyc05` | 0.05 | 0 | cycle, high |
| V2 `cyc05p05` | 0.05 | 0.05 | cycle + synergy pairs |
| V3 `p05` | 0 | 0.05 | synergy pairs only (ablation: what does the cycle add) |
| +c | – | – | V0 centering applied on top of the best trained variant (post-hoc row) |
| ctrl `ft0` | 0 | 0 | control (not a candidate): same 3000-step fresh-Adam fine-tune with λc = λp = 0 — separates fine-tune drift from the effect of the new terms (added after the 40-step smoke test, before any real training: fresh Adam alone moved val rec 0.0021→0.0027 at step 40) |
Grid is fixed in advance; at most one extra run (λ ×3 up or down) if every variant fails G4/G5 in the same direction.

## 3. Measurement (same as zpair; `xhand_latent/cv2_measure.py`, CPU)
Pair sets are generated ONCE (codec-independent) and reused for every codec: SYN H3/ALL5 (3000 clean prototype scenes,
seed 0, ≠ V0 calibration draw), GEO val H3/ALL5 (zpair procedure: anchor rotation, NN by palm-frame tips / scale, better half),
plus RT (codec-defined Enc_B(Dec_B(z_A)) from play val).
Per set: linear hand probe (raw, grouped 80/20 scene split, L2 by grouped 4-fold CV, as zpair), kNN-15 probe (raw), η²_hand,
two-way hand/scene/residual shares, distances (same-scene, random cross, same-hand different scene, mean offset) →
**ratio R = same-scene / same-hand-different-scene** and same/rand.
Codec quality on play val[:2048] (encoder mean): per-hand joint RMSE (rad) and retarget pad residual (E2* validation metric:
per-finger contact residual mm, pair-weighted, all ordered pairs) + RT drift (cross / self).
Kettle: (K1) H3 observation.state from LeRobot parquet, 50 eps, every 4th frame, probe train ep0–39 / test ep40–49 (as zpair);
(K2) OOD: commanded finger q from `data/replay` pkl eps 0–9 for all 5 hands; pool = sharpa/shadow/wuji eps 0–7, reference =
H3 eps 8–9; kNN-10 distance to pool → ratio median(unseen)/median(H3 ref), fraction of unseen frames inside the H3 ref 95th
percentile, binary LR "H3 vs unseen" (episode-grouped) accuracy.
Sanity: my E2* numbers must reproduce RESULTS_ZPAIR (probe 1.00, SYN ALL5 same/rand ≈0.81, frac_hand ≈0.41) within sampling noise.

## 4. Gates (pre-registered; a variant PASSES if all of G1–G5 hold)
| gate | metric | E2* (zpair) | pass |
|---|---|---|---|
| G1 hand-invariance | linear raw hand probe on GEO-val ALL5 and SYN ALL5 (chance 0.20); GEO-val H3 (chance 0.33) | 1.00 / 1.00 / 1.00 | ≤ 0.40 / ≤ 0.40 / ≤ 0.50 |
| G2 alignment | R = same-scene / same-hand-diff-scene, GEO-val ALL5 and SYN ALL5 | ≈1.04 / ≈1.21 | ≤ 0.60 on both |
| G3 offset share | two-way frac_hand on SYN ALL5 and GEO-val ALL5 | 0.41 / 0.30 | ≤ 0.10 on both |
| G4 reconstruction | per-hand val joint RMSE (rad) | 0.024–0.045 | every hand ≤ max(1.15×E2*, E2* + 0.005) |
| G5 retarget | mean per-finger pad residual (mm, pair-weighted) | ≈6.2 | ≤ E2* × 1.10 |
Secondary (reported, not gating): kNN probe, RT drift, Kettle K1 probe/η² (Kettle behaviour differs by hand, so K1 > chance is
expected even for an invariant codec), Kettle K2 OOD ratios.
Fidelity gate (only for a passing variant, step 5): open-loop round-trip replay (P0b tooling: `p3h_roundtrip.py --arms Z
--checkpoint <v2>` + unmodified `replay_sim`, same flags, eps 0–9, sharpa/shadow/wuji). Pass = each hand ≥ 9/10 (P0b rule 4;
E2* Z = 9/10, 10/10, 9/10).
If no variant passes G1–G5: report the best trade-off and which gate failed; no further variants without the user.

## 4a. Amendment A1 (261115 22:55, AFTER a peek at step ~1800 of V1b/V2/V3 — exploratory, gates unchanged)
Peek (no Kettle) showed: V2 `cyc05p05` meets G2/G3 by a wide margin (GEO-ALL5 same/within 0.41, frac_hand 0.02; SYN 0.09)
but the linear probe stays 0.96–1.00: a low-variance but consistent hand direction survives variance-based alignment, and a
LeRobot DP normalizes state per dimension exactly like the probe. V1b `cyc05` aligns H3 (GEO-H3 0.35) but pushes the 4-finger
hands away (ALL5 frac_hand 0.50); V3 `p05` aligns SYN only (semantic ≠ geometric). V0 centering breaks cross-decoding
(retarget pad residual 6.2 → 23 mm, because Dec'_B(z'_A) = Dec_B(z_A − o_A + o_B)) → V0 fails G5 by construction.
Added variants (this is the "adversarial hand removal" candidate, now justified by data, and applied only to MATCHED-content z,
not to the unpaired play marginals): V2 + hand-confusion term λa·(CE(adv(z), uniform) − log 5); adversary = BatchNorm(no affine,
= probe standardization) + linear (or MLP 64-64) classifier on SYN scenes (all hands) + RT pairs (z_A → A,
Enc_B(stopgrad Dec_B(z_A)) → B, cycle pairs), trained 1 Adam step (lr 1e-3) per codec step.
| id | λc | λp | λa | adversary |
|---|---|---|---|---|
| V4a `adv01` | 0.05 | 0.05 | 0.01 | linear |
| V4b `adv05` | 0.05 | 0.05 | 0.05 | linear |
| V4c `advm05` | 0.05 | 0.05 | 0.05 | MLP 64-64 |
Same 3000 steps / warm start from E2*. G1–G5 unchanged. Because A1 was added after seeing partial results, any pass by V4 is
reported as exploratory (one-shot follow-up), not as a pre-registered confirmation.

## 4b. Amendment A2 (261115 23:30; user idea relayed by the coordinator + Fisher term; exploratory, gates unchanged)
Peek of V4b `adv05` at step ~1500: the confusion-to-uniform adversary does not move the linear probe (GEO-ALL5 0.94, SYN 1.00):
the encoder lowers the adversary's logit scale instead of removing the direction. Two additions:
1. **Paired kinematic closure data** (user idea; `ClosureSampler`, FK only): per scene one closure value per shared finger
   (thumb/index/middle/ring), independent per finger (not a 1-D path) + N(0, 0.03) joint noise; little (5-finger hands) follows
   ring + N(0, 0.1). Used in place of SYN for L_pair (and Fisher/adversary sets).
   (a) `cl_joint`: equal joint fraction s along open → fist (0.65-upper prototype) for every hand.
   (b) `cl_func`: equal FUNCTIONAL closure = palm-frame fingertip travel |tip(s) − tip(open)| / hand scale, inverted per hand
   (monotone 201-point table, flexion swept to the upper limit); common range = min over hands of the max travel
   (thumb 0.52, fingers 0.46; limited by shadow thumb / wuji fingers) → big hands are only paired over their first 20–37 %
   of joint closure. Coverage limit: only the closure manifold is paired, not pinches or isolated-finger poses.
2. **Fisher term** λf·tr(Sw⁻¹Sb) of hand labels on matched-content z (SYN/closure scenes + RT pairs): the linear
   discriminability the probe measures, whitening-invariant (cannot be dodged by shrinking a direction). Initial value ≈ 144
   on E2*, so small λf.
| id | λc | λp (data) | λf |
|---|---|---|---|
| `clj05` | 0 | 0.05 (cl_joint) | 0 |
| `clf05` | 0 | 0.05 (cl_func) | 0 |
| `cyc05clf05` | 0.05 | 0.05 (cl_func) | 0 |
| `cyc05p05f002` | 0.05 | 0.05 (syn) | 0.002 |
| `cyc05p05f01` | 0.05 | 0.05 (syn) | 0.01 |
| `cyc05clf05f002` | 0.05 | 0.05 (cl_func) | 0.002 |
Note: SYN eval scenes and closure/SYN training scenes are all prototype-family postures → SYN is not held out for these;
GEO-val (geometric NN pairs from the val pool) remains the held-out alignment test. G1 ALL5 ≤ 0.40 is near the floor of what a
functional codec can reach (4- vs 5-finger hands differ in whether a little-finger state exists: perfect 4-vs-5 separation
with chance inside each group gives 0.40), so GEO-H3 ≤ 0.50 is the more meaningful part of G1.

## 5. Out of scope (needs user approval)
Rebuilding Z′/D′ policy datasets, retraining or evaluating any DP/pi0.5 policy with v2.
Downstream (proposal only): re-encode Kettle Z′/D′ datasets with v2 (≈0.5 h) → retrain 3-hand nohot Z′/D′ DP on vast (≈as P3nh) →
H1 rerun on allegro/leap (10 eps each first).

## 6. Resources
Training + replay on work1 (3090, idle besides a foreign 1.7 GB process; small files only, disk 98%). Measurement on CPU.
work2 GPU untouched (H2 eval running). No pkill, no NAS scans, no push/upload.
