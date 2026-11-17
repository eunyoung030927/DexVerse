# CV3DP PREREG — codec v3 `wb_cl_l` z vs E2* z in a 3-hand pooled image DP (GraspKettle), 261116

Written before any CV3DP policy was evaluated. Training of both groups was started on vast at 2026-11-16 18:59 KST.
Nothing in this document was changed after any evaluation result was seen. Any later change goes into a dated
"Amendments" section below, with the reason.

## Question
The codec-v3 stage pre-criteria failed (RESULTS_CV3.md: no variant passes; `wb_cl_l` is the best trade-off. It scores
native round trip 9/10, 10/10, 10/10, cross-hand replay 31/40 and probe 0.90). The user approved this anyway (11/16), to
see at the policy level whether the hand information left in z matters. The question: does a DP trained on `wb_cl_l` z
transfer better to held-out hands than the same DP on E2* z, without losing performance on the training hands?

## Groups (same split, same recipe, only the finger codec differs)
| group | finger z | dataset (work2 `/home/eun/xhand/datasets/cv3dp/`) | codec ckpt |
|---|---|---|---|
| **Zw3** | codec v3 `wb_cl_l` | `wsa_Zw3` | `artifacts/codec_v3/runs/wb_cl_l/best.pt` |
| **Ze3** | E2* (reference) | `wsa_Ze3` | `artifacts/arm_latent_v1/runs/e2_lam2.8333/best.pt` |

- Training hands (pooled, 150 episodes, 41,882 frames): shadow (eps 0–49) + wuji (eps 0–49), both 5-finger, and allegro
  (eps 0–49), 4-finger. The source is the dv-demos-v0 teleop GraspKettle LeRobot sets. Videos are byte-identical to
  pooled3hand_nohot/kettle3_Z (shadow, wuji) and ho2hand/sa_Z (allegro). State and action are 41D nohot Z rows:
  canonical palm 9 + z 32. There is no hand one-hot.
- Held-out test hands (never seen by the policy): **sharpa** (5-finger) and **leap** (4-finger).
- Encoding checks (`encode_e2.json`, `encode_v3wbcll.json`):
  - E2* rows reproduce kettle3_Z and sa_Z bit-exactly for all 3 hands.
  - Zw3 palm columns are bit-identical to the reference. Only z differs.
  - Decode finger MAE on eps 0–9: shadow 0.030, wuji 0.024, allegro 0.019 rad (E2*: 0.027, 0.028, 0.015).
  - Zw3 z per-dim std is about 0.1 (E2* about 0.3–0.4). The DP's min-max normalization removes this scale difference.

## Training recipe (identical to cv2dp)
- `lerobot_dp/train_lerobot.py --resize` (LeRobot 0.4.2 image DP), unchanged.
- 300 epochs, batch 64, seed 42 (default).
- Snapshot every 50 epochs, num_workers 12, `--resume` retries.
- Cameras: third_person + eye_in_hand.
- Steps per epoch = ceil(41,882 / 64) = 655, so training is 196,500 steps. Snapshots at 32,750 / 65,500 / 98,250 /
  131,000 / 163,750 / 196,500.
- Only difference from cv2dp: steps per epoch go from 419 to 655 because there is a third training hand. Epochs are the
  same.
- Hardware: vast 2×5090, with Zw3 on GPU0 and Ze3 on GPU1. Launcher: `xhand_latent/cv3dp_vast_train.sh`. It is
  cv2dp_vast_train.sh with the pi0.5 gate removed, plus a pruner that deletes the training_state of older snapshots.
  Weights and the recipe are not affected.

## Snapshot selection (cv2dp rule, extended to 3 training hands; held-out hands never used)
- Per group: shadow, wuji and allegro × eps 0–9 × the 5 snapshots of epochs 50, 100, 150, 200, 250.
- Best snapshot = highest 3-hand total out of 30. Ties go to the later step.
- Result: `sel/<group>_best.json`.
- The epoch-300 snapshot is kept (archived) but not selected, as in cv2dp.

## Final evaluation (best snapshot of each group)
| hand | episodes | runs | protocol |
|---|---|---|---|
| sharpa (held-out, 5-finger) | eps 0–49 | 50 | exactly as cv2dp: payload sim seed, unseeded policy server |
| leap (held-out, 4-finger) | ~~eps 0–9 × repeats S = 1..5~~ → **eps 0–49 (A1)** | 50 | ~~see the leap notes below~~ → same as sharpa (A1) |
| shadow, wuji, allegro (training) | eps 10–49 new, plus eps 0–9 = the selection rows of the best step | 3 × 50 | as cv2dp (train_ref on) |

Leap notes (**superseded by A1**; kept for the record):
- Only leap eps 0–9 exist locally (no HF token).
- Repeat S uses `cv3dp_eval_pooled.py --seed S`:
  - sim seed = payload seed + 7919·S
  - policy RNG = `torch.manual_seed(1000·S + ep)` before every episode
- So each (ep, S) gets the same seeds in both groups.
- **Limitation:** there are only 10 distinct leap initial states. The 50 runs are not 50 independent scenes.

Shared settings:
- max_steps 3000, success probe as in cv2dp, no video.
- Eval script `cv3dp_eval_pooled.py` = cv2dp_eval_pooled.py + `--seed`. It behaves identically when `--seed` is absent.

## Comparisons and decision rules (α = 0.05, two-sided)
**P1 (primary): sharpa, Zw3 vs Ze3**
- Paired by episode (n = 50). Exact McNemar test (binomial on the discordant pairs).
- Zw3 "transfers better" only if p < 0.05 **and** Δ ≥ +5/50.
- "Transfers worse" if p < 0.05 and Δ ≤ −5/50.
- Otherwise: "no difference detected". Δ and the 95% CI of the paired difference are reported.

**P2: leap, Zw3 vs Ze3** (A1: now paired by episode over eps 0–49, n = 50; the (ep, S) pairing and the
episode-level sign test below no longer apply)
- Paired by (ep, S) (n = 50). Exact McNemar, same thresholds as P1.
- Also reported at episode level:
  - successes per episode out of 5, for each group
  - number of episodes with ≥ 1 success
  - a sign test over the 10 episodes on the per-episode success counts
- P2 is interpreted together with the 10-initial-state limitation. A P2 "effect" carried by one or two episodes is reported
  as such.

**P3: held-out pooled (sharpa 50 + leap 50)**
- Exact McNemar on the 100 pairs. This is secondary.

**T (training hands): non-inferiority guard**
- Zw3 vs Ze3 on shadow + wuji + allegro (150 runs, paired by hand and episode).
- Reported for each hand and pooled, using exact McNemar.
- Eps 10–49 (120 runs) are reported separately, because eps 0–9 were used for snapshot selection.
- Zw3 "keeps training-hand performance" if pooled Zw3 ≥ Ze3 − 12/150 (the pooled3hand margin).

**Conclusions**
- *Hand information in z matters at policy level and `wb_cl_l` helps*: P1 or P2 shows "better" and T holds.
- *Codec does not matter at policy level*: P1 and P2 both show "no difference" and T holds.
- *`wb_cl_l` hurts*: P1 or P2 shows "worse", or T fails.

**Descriptive context (not part of the decision)**
- Ze2 and Zv2 (cv2dp: shadow + wuji only, test sharpa eps 0–49) are put in the same table, with sharpa per-episode pairs
  against Zw3 and Ze3.
- H2 Z and D (shadow + allegro, tests sharpa, wuji, leap eps 0–9, unseeded) are put in the same table when available.
- These differ in their training hands, so any comparison is a confounded, descriptive McNemar only.
- Missing numbers are shown as "대기" (pending).

## Hosts and transfer
- Training runs on vast only, by user approval for this experiment.
- Snapshot pretrained_model is archived on work2: `artifacts/codec_v3/cv3dp/ckpt_archive/<group>/<step>`, with the
  sha256 of every file checked.
- Evaluation host:
  - work1 after the cv2dp evaluation finishes, using the PC relay as cv2dp does; or
  - work2 after H2 and the gated pi0.5 evaluation leave the GPU.
- Whichever is free first. H2 is never slowed down.
- Operational note (2026-11-17 14:30 KST; user approval "work1도 사용"; host change only, not an amendment: no job, episode range, max_steps, seed or selection-rule change): work1 (RTX 3090, idle after cv2dp ALL DONE) evaluates
  alongside work2. `ctl/SPLIT.json` (driver A4, identical on both hosts) assigns the training-hand reference finals
  (shadow / wuji / allegro eps 10-49 at each arm's selected snapshot) to work1; every selection job and the sharpa / leap
  finals stay on work2, which also writes every `sel/<arm>_best.json`. work1 reads the sha256-identical snapshot copy
  at the same `ckpt_archive/<arm>/<step>` path (PC relay), so metrics.json `ckpt` is the same string; finished work1 job
  dirs are copied by the PC relay (`cv3dp_split_relay_pc.py`) into the same paths on work2, so one results tree is
  summarised; `machine.json` in each job dir records the host. Eval script, ho_lib, demos, codec / f_task ckpts and
  DexVerse commit are sha256-identical on both hosts. A work2 slot with nothing of its own may run a pending work1 job;
  every job is still run once (the relay keeps both hosts from starting the same job).

## Amendments
- **A1 (2026-11-16 21:00 KST, before any CV3DP evaluation had started):** leap Kettle eps 000–049 became
  available (dv-demos-v0, downloaded with user approval; eps 0–9 identical to the old `_ep0-9.pkl`; sha256 b1241bab9d479ee5…).
  Leap is therefore evaluated like sharpa: eps 0–49 × 1 run, payload sim seed, unseeded policy server, paired by episode.
  The "eps 0–9 × seed 1..5" plan is dropped (it never ran). Thresholds of P1/P2/P3 unchanged.

## Amendment A2 (2026-11-17 09:08 KST) — max_steps 3000 -> 1500
- Reason: rollout time. Failed episodes always run to the cap; at 3000 steps one takes about 630 s on work2.
- Basis: success-step distribution of completed policy evals (p3, p3nh, pi05, H2, cv2dp, cv3dp sel; 1063 successes). median 286, p90 546, p95 873, p99 2059. 28 of 1063 (2.6%) succeeded after step 1500.
- The cap was chosen from step counts pooled across all evals, not from any arm comparison; selection had not been run. The 4 finished sel jobs (3000-step cap) had max success step 446, so applying a 1500 cap to them changes nothing.
- Applies to every remaining cv3dp job on work2 and work1, for both arms. Reported success = success at or before step 1500.
- Driver: --max_steps (default 1500). Pre-A2 copy: logs/cv3dp_driver.py.pre_a2.

## Amendment A3 (2026-11-17 09:30 KST) — D-wrist arms Dw3 / De3 (user approval 11/17: "다른 테스크 하기 전에 D 학습부터 진행")
Written before any Dw3/De3 policy existed (training had not started; no D snapshot, no D evaluation). Zw3/Ze3 selection
had partial results at this time (4 of 30 sel jobs); A3 does not change any Zw3/Ze3 job, rule or result.

### Question
Does replacing the raw canonical palm (9D) in state/action by the hand-corrected wrist EE_common (the "wrist method" D of
pooled3hand / H2) change held-out-hand transfer, for each finger codec?

### Groups (added; same split, same 150 episodes / 41,882 frames, same images, same z)
| group | wrist (cols 0–8) | finger z (cols 9–40) | dataset (work2 `/home/eun/xhand/datasets/cv3dp/`) | codec ckpt |
|---|---|---|---|---|
| **Dw3** | EE_common | `wb_cl_l` (bit-identical to wsa_Zw3) | `wsa_Dw3` | `artifacts/codec_v3/runs/wb_cl_l/best.pt` |
| **De3** | EE_common | E2* (bit-identical to wsa_Ze3) | `wsa_De3` | `artifacts/arm_latent_v1/runs/e2_lam2.8333/best.pt` |

- EE_common = canonical palm @ inv(f_task(q_finger)), f_task = `artifacts/pooled3hand/p0a/f_task_best.pt`
  (md5 a2c3f4291cefa72986fd3f60c12a4dd5), evaluated on the recorded finger q (state: observed q; action: commanded q,
  offset + clip), exactly `p3h_policy_codec.encode_q(arm D)`. EE_common does not depend on the finger codec, so Dw3 and
  De3 share the palm columns (`encoded_dpalm_<hand>.npz`, `xhand_latent/cv3dp_encode_d.py`, Isaac python).
- Build: `xhand_latent/cv3dp_build_d.py` = the Z set with only state/action cols 0–8 replaced; per-episode state/action
  stats recomputed, everything else (frames, episode order, tasks, videos = hard links) unchanged.
- Checks (`encode_dpalm.json`, `wsa_D*/meta/cv3dp_build.json`): z cols bit-identical to wsa_Zw3 / wsa_Ze3; shadow and
  wuji D rows (E2* z) bit-identical to pooled3hand_nohot/kettle3_D (all 50 eps each, max abs 0); allegro has no
  reference (H2 sa_D used the shadow+allegro f_task).
- Decode (eval) of a D action: q_finger_hat = Dec_hand(z) of the arm's codec; desired palm = EE_common @ f_task(q_finger_hat);
  floating-wrist solve seeded with the currently observed q (`cv3dp_eval_pooled.py --arm D`, unchanged script).
  Held-out round trip of recorded actions (eps 0–9, encode -> decode, palm error vs recorded commanded palm, mean / p95):
  sharpa De 8.9 / 14.4 mm, 4.0 / 8.7 deg; Dw 10.6 / 19.0 mm, 6.2 / 14.2 deg;
  leap De 2.2 / 3.6 mm, 0.6 / 1.2 deg; Dw 5.0 / 8.8 mm, 1.4 / 2.2 deg; Z arms ~0 mm (palm passes through). No solve failures.

### Caveats
- f_task was trained on Kettle **sharpa / shadow / wuji** (P0a, eps 0–39 train / 40–49 val per hand).
  - **leap**: unseen by f_task (and by the policy) -> the clean held-out test of D.
  - **allegro** (training hand of the policy): out-of-set for f_task (f_task is a link-graph model applied to an unseen hand).
  - **sharpa**: held-out for the policy, but f_task has seen sharpa Kettle eps 0–49 (= the final eval episodes).
    The D wrist on sharpa therefore carries sharpa-specific offset knowledge; a D advantage on sharpa alone is reported
    with this caveat, and leap is the decisive held-out hand for the D comparisons.
- The round-trip wrist error of D (above) is a decode error the Z arms do not have; it is part of the method.

### Training (recipe identical to cv3dp)
- `train_lerobot.py --resize`, 300 epochs, batch 64, seed 42, snapshot every 50 epochs, num_workers 12, `--resume`
  retries; 655 steps/epoch -> 196,500 steps; snapshots 32,750 … 196,500.
- vast 2×5090, Dw3 on GPU0, De3 on GPU1, in parallel. Launcher `/workspace/xhand/cv3dpD_vast_train.sh` = copy of
  cv3dp_vast_train.sh with only RUNS = `/workspace/xhand/runs/cv3dpD` and markers `[cv3dpD_vast]`.
- Archive: work2 `artifacts/codec_v3/cv3dp/ckpt_archive/{Dw3,De3}/<step>/pretrained_model`, sha256 per file.

### Selection / final (same protocol as Zw3/Ze3)
- Selection: shadow + wuji + allegro × eps 0–9 × epochs 50–250 snapshots, best = max 3-hand total /30, ties -> later step.
- Final: best snapshot on sharpa eps 0–49 and leap eps 0–49 (1 run each, payload sim seed, unseeded policy server),
  plus training-hand reference eps 10–49 (eps 0–9 = selection rows).
- max_steps **1500** (A2) for every D job. Success probe and eval script unchanged; driver passes `--arm D`
  (ctl/arms.json `eval_arm`).

### Comparisons (exact McNemar on episode pairs, α = 0.05 two-sided; "better/worse" needs p < 0.05 and |Δ| ≥ 5/50)
- **D1: Dw3 vs Zw3** — wrist-method effect on whitened (`wb_cl_l`) z. sharpa (n = 50), leap (n = 50), held-out pooled
  (n = 100, secondary), training hands (150; eps 10–49 separately; non-inferiority margin 12/150 as T).
- **D2: De3 vs Ze3** — the same on E2* z.
- **D3: Dw3 vs De3** — codec effect under the D wrist (mirror of P1/P2 under D).
- Reported for each: successes, Δ, discordant pairs (b, c), exact p, 95% CI of the paired difference.
- Interpretation: "D helps" for a codec only if leap shows "better" (sharpa "better" alone is reported with the f_task
  caveat) and the training-hand guard holds. No multiplicity correction; D1–D3 are reported as a family of 3.
