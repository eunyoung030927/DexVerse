# RESULTS_CV2: codec v2 (hand-invariant finger z), 261115–16

Design + pre-registered gates + two amendments: `artifacts/codec_v2/DESIGN.md` (PC copy `C:\Users\eunyo\Desktop\연구실\2026_2학기\Flow Chart\codec_v2_design_261115.md`).
Training / measurement / replay ran on work1 (3090, idle). work2 GPU was not used. Everything is mirrored to work2 `artifacts/codec_v2/`.
Code (new files only, E2* paths untouched): `xhand_latent/cv2_train.py`, `cv2_center.py`, `cv2_measure.py`, `cv2_report.py`, `cv2_launch.sh`, `cv2_replay.sh`.

## Interpretation (6 lines)
1. **No variant passes all gates.** G1 (linear hand probe ≤ 0.40/0.40/0.50) fails for every variant: best GEO-ALL5 0.80, GEO-H3 0.76 (`cyc05p05f002`); clean-SYN probe stays 0.96–1.00 everywhere.
2. **The geometry of z is fixed, but not linear separability.** Cycle consistency + paired scenes moves z from "every hand in its own region" to aligned: GEO-ALL5 same/within 1.04 → 0.29–0.41, GEO-H3 0.85 → 0.20, hand share of variance 0.30–0.41 → ≤ 0.02, cross-hand round-trip drift 0.76 → 0.18–0.26 of the random distance. Reconstruction, retarget pad residual and Kettle round-trip replay stay at E2* level (G2–G5 pass for 6 variants).
3. **What remains is a small, low-variance hand direction.** A standardized linear probe still reads it. A DP that normalizes state per dimension sees the same thing. Adversaries (uniform confusion, linear or MLP) and a Fisher term tr(Sw⁻¹Sb) on matched z lower the probe only partly (GEO 0.96 → 0.80) before reconstruction and retarget start to suffer (λf 0.01 and MLP adversary fail G4/G5).
4. **Each ingredient alone fails:** cycle only aligns the 5-finger hands but pushes allegro/leap away (ALL5 hand share 0.58–0.75); prototype/closure pairs alone align only their own family (SYN) and not held-out geometric pairs; per-hand centering (V0) breaks cross-decoding (pad residual 6.2 → 23 mm). Only cycle + pairs works.
5. **User idea (paired closure data):** alone, it does not align z on held-out pairs (GEO same/within joint-fraction 1.16, functional 1.00 vs E2* 1.04). With the cycle term, functional pairing gives the best held-out alignment of all variants (GEO-ALL5 0.30 vs 0.41 with prototype pairs). Functional beats joint-fraction pairing on held-out data (1.00 vs 1.16 alone). Coverage limit: the pairs only cover closure, and only over the common functional range (big hands reach 20–37 % of their joint closure).
6. **Kettle:** H3 hand variance share η² drops from 0.50 to 0.08–0.25. allegro/leap Kettle z moves 2.5–3.5× closer to the H3 pool (kNN ratio 16 → 4.5–6.3; about as far as one H3 hand is from the other two). It is still linearly separable from H3 Kettle z (binary probe 1.00 for every variant; this also contains real behaviour/scene differences). Whether this is enough for zero-shot transfer can only be answered downstream (H1 rerun), and that step was not run.

## Gate table (all codecs measured with identical codec-independent pair sets: `measure/pairsets.npz`; gates = DESIGN §4)
| codec | G1 lin GEO5/SYN5/GEO3 (≤.40/.40/.50) | G2 same/within GEO5/SYN5 (≤.60) | G3 frac_hand SYN5/GEO5 (≤.10) | G4 max rmse/E2* (≤1.15 or +0.005) | G5 pad mm (≤6.80) | pass |
|---|---|---|---|---|---|---|
| E2* | 1.00/1.00/1.00 ✗ | 1.04/1.22 ✗ | 0.41/0.30 ✗ | 1.00 ✓ | 6.18 ✓ | fail |
| V0 `e2c` (centering) | 0.93/0.53/0.81 ✗ | 0.86/0.61 ✗ | 0.00/0.16 ✗ | 1.00 ✓ | 23.01 ✗ | fail |
| ctrl `ft0` | 1.00/1.00/1.00 ✗ | 1.05/1.23 ✗ | 0.42/0.29 ✗ | 0.96 ✓ | 5.96 ✓ | fail |
| V1a `cyc01` | 0.98/1.00/0.99 ✗ | 1.00/1.40 ✗ | 0.58/0.39 ✗ | 0.97 ✓ | 6.14 ✓ | fail |
| V1b `cyc05` | 0.92/1.00/0.96 ✗ | 1.28/2.04 ✗ | 0.75/0.52 ✗ | 1.03 ✓ | 6.40 ✓ | fail |
| V3 `p05` | 1.00/1.00/1.00 ✗ | 1.19/0.24 ✗ | 0.02/0.47 ✗ | 0.92 ✓ | 6.23 ✓ | fail |
| **V2 `cyc05p05`** | 0.96/1.00/0.95 ✗ | 0.41/0.08 ✓ | 0.00/0.02 ✓ | 1.03 ✓ | 6.45 ✓ | fail (G1) |
| A1 `adv01` | 0.96/1.00/0.97 ✗ | 0.40/0.07 ✓ | 0.00/0.02 ✓ | 1.09 ✓ | 6.61 ✓ | fail (G1) |
| A1 `adv05` | 0.88/1.00/0.94 ✗ | 0.31/0.08 ✓ | 0.00/0.02 ✓ | 1.12 ✓ | 6.85 ✗ | fail |
| A1 `advm05` | 0.92/1.00/0.93 ✗ | 0.47/0.13 ✓ | 0.00/0.03 ✓ | 2.04 ✗ | 10.10 ✗ | fail |
| A2 `clj05` (closure, joint frac.) | 1.00/1.00/1.00 ✗ | 1.16/0.45 ✗ | 0.08/0.43 ✗ | 0.97 ✓ | 6.25 ✓ | fail |
| A2 `clf05` (closure, functional) | 1.00/1.00/1.00 ✗ | 1.00/0.74 ✗ | 0.17/0.32 ✗ | 0.96 ✓ | 6.10 ✓ | fail |
| A2 `cyc05clf05` | 0.92/1.00/0.94 ✗ | 0.30/0.42 ✓ | 0.02/0.01 ✓ | 1.04 ✓ | 6.41 ✓ | fail (G1) |
| **A2 `cyc05p05f002`** | **0.80**/1.00/**0.76** ✗ | 0.35/0.08 ✓ | 0.00/0.02 ✓ | 1.09 ✓ | 6.73 ✓ | fail (G1) |
| A2 `cyc05p05f01` | 0.84/1.00/0.86 ✗ | 0.34/0.11 ✓ | 0.00/0.02 ✓ | 1.32 ✗ | 7.00 ✗ | fail |
| **A2 `cyc05clf05f002`** | 0.82/1.00/0.84 ✗ | **0.29**/0.39 ✓ | 0.01/0.00 ✓ | 1.15 ✓ | 6.72 ✓ | fail (G1) |

Variant keys: λc = cycle, λp = pairs (p = prototype synergies, clj/clf = closure joint-fraction / functional), adv = hand-confusion adversary (m = MLP), f = Fisher λ. All runs: warm start from E2*, 3000 steps, fresh Adam 1e-3. best.pt = step 3000 for every run.

Secondary metrics (full table `measure/gates.md`):
| codec | GEO-H3 same/within | RT5 drift (cross/rand) | kNN raw GEO5 | K1 Kettle H3: η²_hand / offset÷within | K2 allegro: kNN ratio vs H3 ref / vs LOHO / frac in H3 p95 | K2 leap (same) |
|---|---|---|---|---|---|---|
| E2* | 0.85 | 0.76 | 1.00 | 0.50 / 1.49 | 16.2 / 1.22 / 0.00 | 15.2 / 1.15 / 0.00 |
| cyc05p05 | 0.21 | 0.24 | 0.75 | **0.08** / 0.50 | 6.3 / 0.96 / 0.01 | 6.3 / 0.95 / 0.00 |
| cyc05clf05 | 0.21 | 0.19 | 0.78 | 0.20 / 0.81 | 6.1 / 0.91 / 0.20 | 7.0 / 1.03 / 0.00 |
| cyc05p05f002 | 0.20 | 0.26 | 0.68 | 0.25 / 0.87 | **4.6** / 0.85 / 0.23 | **4.7** / 0.87 / **0.27** |
| cyc05clf05f002 | 0.20 | 0.18 | 0.72 | 0.14 / 0.62 | 4.9 / 0.82 / **0.38** | 6.1 / 1.03 / 0.02 |
K1 linear probe is 1.00 for every codec; the K2 binary probe (H3 vs allegro/leap, episode split) is 1.00 for every codec. K2 "vs LOHO" compares with the distance of one H3 hand to the other two H3 hands under the same codec: ≈ 1 means allegro/leap sit about as far from the H3 pool as an H3 hand sits from its siblings.

Sanity: E2* reproduces RESULTS_ZPAIR (SYN-ALL5 frac_hand 0.414 vs 0.41, same/rand 0.81; GEO-val ALL5 0.300 vs 0.30; Kettle η² 0.50 vs 0.51, probe 1.00). V0 reparameterisation is exact (decode error < 3e-6).

## Fidelity: Kettle round-trip open-loop replay (P0b tooling: p3h_roundtrip --arms Z + unmodified replay_sim, eps 0–9)
| codec | sharpa | shadow | wuji | gate (≥ 9/10 each) |
|---|---|---|---|---|
| E2* (P0b) | 9/10 (ep0) | 10/10 | 9/10 (ep0) | – |
| cyc05p05 | 9/10 (ep0) | 10/10 | 9/10 (ep0) | pass (identical to E2*) |
| cyc05p05f002 | 9/10 (ep0) | 10/10 | 8/10 (ep0, 9) | fail (wuji) |
| cyc05clf05f002 | 9/10 (ep0) | 10/10 | 10/10 | pass |
Native 10/10 for all hands. This was run for the best G2–G5 trade-offs even though none passed G1, because it is cheap.

## Why G1 is hard (and partly unreachable as written)
- The clean SYN set lies on 1-D prototype curves, so any tiny constant per-hand shift is linearly separable. It stays at 0.96–1.00 even when SYN same/within is 0.08 and the hand share is 0.001. GEO is the meaningful held-out set.
- For ALL5, 4-finger hands have no little-finger state. Perfect 4-vs-5 separation with chance inside each group already gives 0.40, so G1-ALL5 ≤ 0.40 is at the floor (noted in Amendment A2).
- Whitening-invariant pressure (Fisher) lowers GEO probes to 0.76–0.84. Stronger pressure (λf 0.01, MLP adversary) costs reconstruction/retarget before the probe reaches 0.5.

## Process notes (honesty)
- Gates were pre-registered before training. A1 (adversary) was added after a peek at step ~1800 of V1b/V2/V3. A2 (closure pairs from the user, plus Fisher) was added mid-run. Both are marked exploratory in DESIGN.md. Gates were not changed.
- 4 A2 runs hit CUDA OOM at launch (too many concurrent runs) and were relaunched unchanged once memory was free.
- The SYN/closure training scenes come from the same prototype family as the SYN eval set, so SYN is not held out for p/cl variants. GEO is held out for all variants.

## Downstream step (NOT run; needs user approval)
If the user accepts "G2–G5 + replay pass, G1 partial" as good enough to test transfer, the candidate is **`cyc05clf05f002`**: best held-out alignment, wuji replay 10/10, allegro Kettle in-H3-p95 0.38. The alternative is **`cyc05p05`**: cleanest, replay identical to E2*, lowest Kettle η² 0.08.
1. Re-encode the Kettle Z′/D′ datasets (sharpa/shadow/wuji, no one-hot) with the v2 checkpoint via `HoCodec(checkpoint=…)` and rebuild LeRobot sets. D′ also needs f_task re-checked (f_task takes q, not z, so it can be reused). ≈ 1 h CPU.
2. Retrain the 3-hand nohot Z′ (and D′) DP on vast 5090 with the P3nh recipe. ≈ 6–8 h per arm, both arms in parallel. Then snapshot selection on work1 (eps 0–9 × 3 hands × 5 snapshots). ≈ 3 h.
3. Run H1 again on allegro/leap (10 eps each, extend to 50 if > 0) and the same-hand 50-ep eval for training-hand parity. ≈ 2–3 h.
Total ≈ 1 working day (≈ 12–15 h wall clock), mostly vast training.
A cheaper pre-check that needs no policy (proposal): train a small state-only classifier on the rebuilt Z′ state (z + pose9) to see whether hand-ID from the policy input drops below 1.00. RESULTS_IDENT shows it is 1.00 for E2*, and the pose9/wrist part alone gives 0.59–0.79, so the finger codec is not the only identity source.

## Files (work2 `/home/eun/xhand/latent_codec_v1/artifacts/codec_v2/`; work1 holds the originals)
- `DESIGN.md`, `RESULTS_CV2.md`
- `runs/<id>/{best.pt, config.json, train.jsonl, validation.jsonl, done.json}` (15 run dirs incl. `e2c/center.json`; E2* itself stays in arm_latent_v1)
- `measure/<id>.json`, `measure/gates.md`, `measure/pairsets.npz`; `measure_peek/` (interim peeks, used for amendments)
- `replay/out/*.jsonl`, `replay/roundtrip_prep_<id>.{json,csv}`, `replay/logs/`
- `logs/` (train_*, measure_*, replay_*)
