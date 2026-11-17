# CV3HM PREREG — codec v3 `wb_cl_l` z vs E2* z in a 3-hand pooled image DP (FunctionalHammerStrike), 261117

Written on 2026-11-17 before any CV3HM policy was trained or evaluated. The user approved the experiment on 2026-11-17.
CV3HM repeats the CV3DP design (`artifacts/codec_v3/cv3dp/PREREG.md`, including amendments A1 and A2) on
`Dexverse-FunctionalHammerStrike-v0` instead of GraspKettle. Nothing in this document is changed after any CV3HM
evaluation result is seen. Later changes go into a dated "Amendments" section with the reason.

## Question
Same as CV3DP, on a second task. Does a DP trained on `wb_cl_l` z transfer better to held-out hands than the same DP on
E2* z, without losing performance on the training hands? Hammer is a dynamic tool-use task: grasp the hammer by the handle,
then strike a nail. The question is whether the CV3DP answer holds outside Kettle grasping.

## Groups (same split, same recipe, only the finger codec differs)
| group | finger z | dataset (work2 `/home/eun/xhand/datasets/cv3hm/`, vast `/workspace/xhand/datasets/cv3hm/`) | codec ckpt |
|---|---|---|---|
| **Zw3h** | codec v3 `wb_cl_l` | `wsa_Zw3h` | `artifacts/codec_v3/runs/wb_cl_l/best.pt` |
| **Ze3h** | E2* (reference) | `wsa_Ze3h` | `artifacts/arm_latent_v1/runs/e2_lam2.8333/best.pt` (md5 2f74c352…) |

The codec checkpoints are the ones in the CV3DP encode logs (`cv3dp/encode_e2.json`, `cv3dp/encode_v3wbcll.json`).

### Was Hammer in each codec's data?
- **E2\*** (`e2_lam2.8333`):
  - Trained on synthetic play data only: `artifacts/play_v1`, whose manifest says `task_demonstrations_used_for_training: false`. It was initialized from `all_five_v1`.
  - No Hammer or Kettle demo frames were used for training.
  - **However, Hammer was part of E2\*'s selection.** λ = 2.83 was chosen on the arm_latent_v1 cross-hand replay cohort (`arm_latent_v1/RESULTS_v2.md`):
    - cohort: Sharpa → allegro/leap/shadow/wuji × {Kettle, TwoItems, Hammer, Pan, Bleach} × eps 0–9
    - validation: eps 10–14 on {Kettle, TwoItems, Hammer}
  - So Sharpa Hammer demos eps 0–14 were used for E2\* model selection, exactly as Kettle was.
- **`wb_cl_l`** (codec v3):
  - Initialized from E2\*, then trained on `play_v1` plus the synthetic taxonomy pairs (`codec_v3/data/tax_pairs`).
  - No task demos were used for training.
  - Variant selection and gates used **Kettle only**: F1 native round trip (Kettle eps 0–9), X1 Sharpa→4-hand Kettle replay, and K1/K2 Kettle OOD (`RESULTS_CV3.md`). Hammer was not used.
- **Net:** Hammer demos are in neither codec's training data. Hammer touches the comparison only through E2\*'s λ selection. On Kettle, both codecs had seen the task in selection (E2\* cohort and v3 gates). On Hammer, only E2\* has. If anything, this slightly favours Ze3h, and it is noted as such.

## Data
**Source.** dv-demos-v0 teleop, `v0/teleop/functional/Dexverse-FunctionalHammerStrike-v0/floating_<hand>_right/ep000–049.pkl` + `index.json`, and LeRobot renders under `v0/teleop_lerobot/...`.
- shadow, wuji, sharpa: already on work2.
- allegro (pkl + LeRobot): copied work1 → work2 through the PC; the sha256 of the whole tree matches.
- leap (pkl + index.json only, since leap is a test hand): downloaded from HF on 261117 with user approval.
- allegro and leap files were checked against the HF LFS sha256: 307/307 OK (`cv3hm/logs/hf_download.log`).
- HF holds 6 hands for Hammer. inspire is not used.

**Missing or broken rows: none found.**
- All 5 hands have 50/50 episodes, and `index.json` marks every episode as a success (50/50 for each hand).
- `num_steps` matches the pkl action count, and the top-level payload keys are identical across the 50 per-episode files.
- Every hand was recorded at DexVerse `30cc673` (official-v0) with payload seed 42 and sim_device cpu.
- The evaluation uses the same container checkout as CV3DP: DexVerse `8d0a6b5`, of which `30cc673` is an ancestor.
- The only Hammer-relevant change between those revisions is the object physics material: the hammer gets friction 2.0 in 8d0a6b5, while the demos were recorded with the PhysX default of 0.5.
  - This is the same situation as Kettle, whose config also sets friction 2.0.
  - The CV3DP command (`unset DEXVERSE_OBJECT_MATERIAL`) is kept unchanged, so friction 2.0 is used for both arms and all hands.

**Demo length** (pkl action steps, `cv3hm/demos/demo_lengths.json`):

| hand | median | p90 | p95 | max |
|---|---|---|---|---|
| shadow | 256 | 363 | 407 | 497 |
| wuji | 304 | 343 | 351 | 415 |
| allegro | 241 | 318 | 322 | 392 |
| sharpa | 268 | 313 | 318 | 354 |
| leap | 287 | 336 | 349 | 496 |
| all 250 | 278 | 335 | 354 | 497 |

Kettle (CV3DP) is comparable: p95 is 301–385 across the five hands.

**Training rows.**
- Hands: shadow (eps 0–49) + wuji (eps 0–49), both 5-finger, and allegro (eps 0–49), 4-finger.
- 150 episodes and **42,101 frames**: shadow 14,181, wuji 15,295, allegro 12,625.
- Both datasets have identical frames, videos and palm columns. Only z (columns 9–40) differs.
- State and action are 41D nohot Z rows (canonical palm pos 3 + rot6d 6 + z 32). There is no hand one-hot.
- Encoding (`xhand_latent/cv3hm_encode.py`, `cv3hm/encode_{e2,v3wbcll}.json`) uses the same per-episode operations as CV3DP (`HoCodec.encode_state/encode_action`, root pose from the pkl).
- Decode finger MAE on eps 0–9:

  | hand | Zw3h | Ze3h |
  |---|---|---|
  | shadow | 0.035 rad | 0.028 rad |
  | wuji | 0.024 rad | 0.032 rad |
  | allegro | 0.028 rad | 0.019 rad |

- z per-dim std is about 0.12 for Zw3h and about 0.37–0.40 for Ze3h. The DP's min-max normalization removes this scale difference.

**Codec-side checks (instead of CV3DP's bit-exact reference check).** No earlier Hammer Z set exists, so the following were checked instead:
1. The Hammer LeRobot state/action names equal the names the codec reads from the Kettle replay meta (asserted).
2. The floating-wrist recorded offset inferred on Hammer eps 0–9 equals the codec's offset, which was inferred on Kettle. The max abs difference is 0.0 for all 5 hands, including sharpa and leap.
3. The palm columns of the two encodings are bit-identical (asserted in the build).

**Dataset build** (`xhand_latent/cv3hm_build.py`, `meta/cv3hm_build.json`).
- Sources are the per-hand dv-demos LeRobot sets.
- The output layout copies `pooled3hand_nohot/kettle3_Z`, the layout CV3DP trained on:
  - same data columns and episode-meta schema
  - info features (41D names)
  - videos hard-linked, one mp4 per episode
  - recomputed state/action stats
- Verified:
  - LeRobotDataset load on work2 and on vast
  - items 0/49/50/99/100/149 equal the npz rows
  - stats vs rows < 3e-5
- Staged to vast via the PC; the sha256 of all 912 files matches.

## Training recipe (identical to CV3DP)
- `lerobot_dp/train_lerobot.py --resize` (LeRobot 0.4.2 image DP), unchanged.
- 300 epochs, batch 64, seed 42 (default).
- Snapshot every 50 epochs, num_workers 12, `--resume` retries.
- Cameras: third_person + eye_in_hand.
- Steps per epoch = ceil(42,101 / 64) = **658**, so training is **197,400 steps**. Snapshots at 32,900 / 65,800 / 98,700 / 131,600 / 164,500 / 197,400. (CV3DP: 655 / 196,500.)
- Hardware: vast 2×5090, Zw3h on GPU0 and Ze3h on GPU1.
- Launcher: `/workspace/xhand/cv3hm_vast_train.sh`. It is `cv3dp_vast_train.sh` with these changes:
  - paths and markers renamed to cv3hm
  - start gate also requires `/workspace/xhand/runs/cv3dpD/TRAINED_ALL` (coordinator 261117: the CV3DP D arms train first)
  - start gate also requires no compute process on the GPU and ≥ 22 GB free disk
- Weights and recipe are unaffected.

## Snapshot selection (CV3DP rule; held-out hands never used)
- Per group: shadow, wuji and allegro × eps 0–9 × the 5 snapshots of epochs 50, 100, 150, 200, 250.
- Best snapshot = highest 3-hand total out of 30. Ties go to the later step.
- Result: `sel/<group>_best.json`.
- The epoch-300 snapshot is archived but not selected.

## Final evaluation (best snapshot of each group)
| hand | episodes | runs | protocol |
|---|---|---|---|
| sharpa (held-out, 5-finger) | eps 0–49 | 50 | payload sim seed 42, unseeded policy server (CV3DP) |
| leap (held-out, 4-finger) | eps 0–49 | 50 | same as sharpa (= CV3DP A1 protocol from the start) |
| shadow, wuji, allegro (training) | eps 10–49 new, plus eps 0–9 = the selection rows of the best step | 3 × 50 | as CV3DP (train_ref on) |

**Episode cap: max_steps = 1500 (rule from the coordinator, 261117).**
- Cap = max(1500, 3 × p95 demo length rounded up to a multiple of 100).
- Using the largest per-hand p95 (shadow, 406.6): 3 × 406.6 = 1219.7, which rounds up to 1300. So the cap is **1500**.
- Using the pooled p95 of 354 also gives 1500.
- The cap is identical for both arms and all hands, and is used for R0, selection and final.

**Success criterion** = the unchanged env success probe (`_SuccessProbe` on DexVerse `hammer_strike_success`):
- Success when the nail's prismatic joint is displaced **≥ 0.06 m** from its initial value, **and** no fingertip is inside the 25 mm sphere around the nail head.
- The hammer-frame forbidden zones are empty in this config, and there is no explicit grasp-held check.
- The nail has no restoring spring, so once pressed the displacement persists. Success latches on the first step where both hold.
- Known loophole (reported, not corrected): a non-fingertip part of the hand (palm, or a finger's proximal links) could push the nail without the hammer. `nail_press_max_m` is logged, but how the press was achieved is not judged.

**Evaluation script.** `cv3hm_eval_pooled.py` = `cv3dp_eval_pooled.py` with only these changes:
1. `--max_steps` defaults to 1500.
2. The Kettle lift segmentation (trace-only, applied to the hammer) is wrapped in try/except, so it cannot abort an episode.
3. Per-episode `nail_press_max_m` / `nail_press_final_m` are logged (descriptive only).

**Demo payloads.**
- `cv3hm/demos/<task>_floating_<hand>_right.pkl` = the 50 per-episode pkls merged in order. The top-level keys are identical across files.
- Written with Isaac python (numpy 1.26) and checked with a plain `pickle.load` there. sha256 values are in `demos/SHA256SUMS`.

## Comparisons and decision rules (CV3DP rules, unchanged; α = 0.05, two-sided)
**P1 (primary): sharpa, Zw3h vs Ze3h**
- Paired by episode (n = 50). Exact McNemar test (binomial on the discordant pairs).
- "Transfers better" if p < 0.05 **and** Δ ≥ +5/50.
- "Transfers worse" if p < 0.05 and Δ ≤ −5/50.
- Otherwise: "no difference detected". Δ and the 95% CI of the paired difference are reported.

**P2: leap, Zw3h vs Ze3h** — paired by episode (n = 50), exact McNemar, same thresholds as P1.

**P3: held-out pooled (sharpa 50 + leap 50)** — exact McNemar on the 100 pairs. Secondary.

**T (training hands): non-inferiority guard**
- Zw3h vs Ze3h on shadow + wuji + allegro (150 runs, paired by hand and episode), reported per hand and pooled with exact McNemar.
- Eps 10–49 (120 runs) are reported separately, because eps 0–9 were used for selection.
- Zw3h "keeps training-hand performance" if pooled Zw3h ≥ Ze3h − 12/150.

**Conclusions** (same wording as CV3DP):
- *`wb_cl_l` helps on Hammer*: P1 or P2 shows "better" and T holds.
- *Codec does not matter*: P1 and P2 both show "no difference" and T holds.
- *`wb_cl_l` hurts*: P1 or P2 shows "worse", or T fails.

**Floor rule (Hammer-specific).** If both groups score ≤ 2/50 on a held-out hand, that hand's comparison is reported as "floor; uninformative" and not as "no difference". The same applies if pooled training-hand success is ≤ 15/150 for both groups.

**Cross-task (descriptive only):** CV3HM's P1/P2 outcomes are put next to CV3DP's in one table. No pooled test across tasks.

## R0 — decode-replay ceiling (descriptive, new in CV3HM)
- **Why:** Hammer is a dynamic strike task, and Hammer was never part of the codec v3 gates.
- **What:** the dataset action rows of each codec (`r0/replay_{e2,v3wbcll}.npz`) are fed through the evaluation decode path, with no policy (`--replay_npz`, hold 120 steps).
- **Scope:** shadow, wuji, allegro × eps 0–9 × 2 codecs = 60 runs, with the same cap and env as the policy evaluation.
- **Use:** only to interpret the policy numbers, as a per-hand ceiling. It is not a gate, and nothing above changes with it.
- **When:** it runs first in the driver because it needs no snapshot, at the lowest machine priority.
- **Not covered:** held-out hands have no R0 row, because their decode would need a held-out-hand encoding of their own demos. That is out of scope.

## Hosts, priority and transfer
**Training:** vast only, by user approval. It starts after the CV3DP D arms (`runs/cv3dpD/TRAINED_ALL`).

**Archive:** the PC relay `C:/Users/eunyo/Documents/p3h_relay/cv3hm_relay_pc.sh` copies each finished snapshot's `pretrained_model` from vast to work2:
- destination: `artifacts/codec_v3/cv3hm/ckpt_archive/<group>/<step>`, with the sha256 of every file compared and a `.complete` marker
- vast `runs/cv3hm/logs` → `cv3hm/vast_logs` at the end

**Evaluation: work2 only** (`xhand_latent/cv3hm_driver.py`).
- work1 has about 6 GB free disk, so it is not used. A later amendment may add work1 after CV3DP is done there.
- Priority is below CV3DP and pi0.5: no CV3HM job is launched while a cv3dp or pi0.5 driver is alive on work2 and not finished.
- The ho_lib budget also applies: every container eval counts, GPU ≥ 7.8 GB and RAM ≥ 7.5 GB free, `flock -s /tmp/isaac_gpu.lock`.

## Deviations from CV3DP (complete list)
1. **Task:** FunctionalHammerStrike instead of GraspKettle. Success criterion and loophole as described above.
2. **Data path:**
   - The dataset is built from the per-hand dv-demos LeRobot sets, because no earlier pooled Hammer set exists. The layout is copied from kettle3_Z.
   - CV3DP's bit-exact E2\* reference-row check is impossible here and is replaced by the codec-side checks 1–3.
3. **Episode cap:** max_steps 1500 by the rule above. Original CV3DP used 3000; CV3DP A2 later moved it to 1500.
4. **Steps:** 658 steps/epoch, 197,400 total (CV3DP: 655 / 196,500).
5. **Training start gate:** also waits for `cv3dpD/TRAINED_ALL`, a free GPU and ≥ 22 GB free disk.
6. **Leap:** eps 0–49 × 1 run from the start (= CV3DP after A1). No seeded repeats; no `--seed`.
7. **Eval script changes:** segmentation guard and nail-press logging (trace/metrics only).
8. **Eval host:** work2 only, with a priority gate instead of the PC host picker.
9. **R0:** new descriptive decode-replay ceiling.
10. **Floor rule:** new, Hammer-specific.
11. **Codec exposure differs from Kettle:** Hammer was in E2\*'s λ-selection cohort but not in the v3 gates.
12. **Group names:** Zw3h / Ze3h.

## Amendments
(none)
