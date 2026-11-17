# RESULTS_CV2DP: does codec v2 enable zero-shot transfer of a 2-hand image DP to an unseen hand? (261115–17)

**Setup.** GraspKettle (Dexverse), floating right hands. The policy is trained on wuji + shadow (5-finger) and tested on
sharpa (5-finger), which the policy never saw. LeRobot 0.4.2 image DP (train_lerobot.py --resize), 300 epochs, bs 64,
seed 42, snapshot every 50 epochs, num_workers 12, vast 2×RTX 5090 (one arm per GPU). No hand one-hot.
- **Ze2 (control):** state/action = [palm pos 3 + rot6d 6, E2* z32] = 41D. Rows are bit-identical to the shadow/wuji
  rows of pooled3hand_nohot/kettle3_Z (same 100 episodes, same frames, same videos).
- **Zv2:** same, except the finger part is the codec-v2 encoder mean. Palm columns are bit-identical to Ze2.

**Codec v2 variant used: `cyc05clf05f002`** (`artifacts/codec_v2/runs/cyc05clf05f002/best.pt`, md5 51fb77654a2984011e8c7c932ffb3943).
⚠ **No codec-v2 variant passed all pre-registered gates.** G1 (linear hand probe) fails for every variant.
RESULTS_CV2.md names `cyc05clf05f002` as the downstream candidate:
- best held-out alignment (GEO-ALL5 same/within 0.29),
- passes G2–G5,
- Kettle round-trip replay 9/10 sharpa, 10/10 shadow, 10/10 wuji.
Its GEO probe is still 0.82/0.84 (vs 1.00 for E2*).

**Caveats.**
1. sharpa is inside the codec training set (E2* and v2 alike). This is a *policy-unseen* setting, like XL-VLA zero-shot, not a codec-unseen one.
2. D (EE_common) was skipped. The existing f_task was trained on sharpa Kettle demos, which would leak the test hand, and f is not what this experiment tests.

**Eval protocol.**
- `cv2dp_eval_pooled.py` = `ho_eval_pooled.py` + `--codec_ckpt`. Codec class: the unchanged `ho_codec.HoCodec`.
- 3000-step cap, DDPM 100, n_action_steps 8. Initial scenes come from the demo pkls (p0b/demos, sharpa/shadow/wuji eps 0–49). Machine: work1 RTX 3090, 3 workers.
- Selection: training hands only (shadow, wuji), eps 0–9 each, over epochs 50–250 (5 snapshots, as H2/P3nh). Rule: max /20, ties go to the later snapshot.
- Final: best snapshot on sharpa eps 0–49 (50 rollouts per arm, paired by episode / initial scene).

## Main result: sharpa (unseen by the policy), 50 rollouts each

| arm (best snapshot) | sharpa success | lifted (t_lift ≥ 0) | max kettle rise ≥ 5 cm | median max rise (cm) | decode failures |
|---|---|---|---|---|---|
| Ze2 (E2*, ep50 / 020950) | **0/50 (0 %)** | 46/50 | 9/50 | 4.0 | 0 |
| Zv2 (v2, ep200 / 083800) | **15/50 (30 %)** | 50/50 | 39/50 | 34.6 (failures 27.4) | 0 |

**Exact McNemar (paired by episode), Ze2 vs Zv2:**
- discordant pairs: Ze2-only 0, Zv2-only 15; both 0, neither 35
- **p = 6.1e-5** (two-sided exact binomial on 15 discordant pairs)

## Training hands (in-distribution reference)

Selection table (eps 0–9 per hand, cells = shadow + wuji = total /20):

| arm | ep50 (020950) | ep100 (041900) | ep150 (062850) | ep200 (083800) | ep250 (104750) | best |
|---|---|---|---|---|---|---|
| Ze2 | 8+3=11 | 4+5=9 | 1+5=6 | 3+7=10 | 2+2=4 | **020950** |
| Zv2 | 1+3=4 | 3+6=9 | 3+5=8 | 4+6=10 | 2+3=5 | **083800** |

At the best snapshot, eps 0–9: Ze2 shadow 8/10, wuji 3/10 (11/20); Zv2 shadow 4/10, wuji 6/10 (10/20).
In-distribution performance is therefore about equal (11 vs 10 /20). The sharpa gap is not explained by a weaker or stronger policy on its own hands.
The training-hand 50-episode reference (eps 10–49 at the best snapshot, low priority, after the sharpa final) was still running on work1 when this file was written. See the Status section.

## Failure diagnostics

- **Ze2 on sharpa:** the hand reaches the kettle and disturbs it in almost every episode (46/50 have a t_lift by the 1 cm / 5-frame rule), but the kettle is never really lifted. Median max rise is 4 cm; only 9/50 exceed 5 cm. This pattern is consistent with a grasp-formation failure: the E2* sharpa z is out of distribution for a policy trained on wuji/shadow z regions. (Not verified frame by frame; traces are available.)
- **Zv2 on sharpa:** 39/50 lift the kettle ≥ 5 cm. The median rise among failures is still 27 cm, so most failures are a held kettle that does not meet the success condition (drop or pose), not a missed grasp.
- No wrist-solve / decode failures in any final cell, for either codec.
- Per-episode rows: `cv2dp_episodes.csv`. Aggregates: `cv2dp_summary.json`. Raw traces: `final/*/traces/ep*.npz`, with per-step q, kettle pose and raw policy vector.

## Interpretation

1. **Codec v2 turns zero-shot transfer from 0 % into 30 % on an unseen hand** (15/50 vs 0/50, p = 6e-5, paired). Training-hand performance is equal. This is the first nonzero zero-shot transfer in the cross-hand DP series (H1: 3-hand E2* policy → allegro/leap 0/40).
2. The improvement fits the RESULTS_CV2 diagnosis. E2* puts each hand's z in its own region, so sharpa's state z is out of distribution for a wuji+shadow policy. v2's cycle + functional-closure alignment moves same-scene z across hands close together (GEO same/within 1.04 → 0.29), so sharpa's z lands where the policy has data. A residual linear hand direction (G1 still fails) does not prevent this transfer.
3. Limits:
   - sharpa was in the codec's training set: this tests policy-unseen, not codec-unseen.
   - The best snapshots differ: Ze2 ep50 vs Zv2 ep200. Ze2's ep200 snapshot was almost tied on the train hands (10/20). A sensitivity check of Ze2 ep200 on sharpa was **not run** (outside the protocol). It is the cheapest extra check: 50 rollouts, about 7 h on work1. Given 0/50 with 46 touch-but-no-lift episodes, a flip is unlikely, but it is not verified.
   - One seed, one task.
   - The image stream still shows the hand (a hand-masked arm, Zv2m, was prepared and then dropped on the user's instruction).

## Wall time and incidents
- Start 11/15 23:22 KST. Sharpa finals done 11/17 06:02 KST. About 31 h, made up of:
  - wait for the vast pi0.5 queue and for codec v2: about 2.5 h
  - training: about 6.4 h (16:42–23:07 UTC, 0.18 s/step)
  - eval on work1: about 27 h (10-episode job ≈ 2.3 h; 3000-step failures ≈ 10–19 min each)
- Incidents:
  - Ze2/Zv2 staging: the final rename on vast was lost to a work2→vast ssh timeout and was fixed by hand from the PC. The script now checks it.
  - vast pulls failed intermittently (ssh timeouts) and were retried automatically.
  - Vast was freed: at 11/16 15:40 work1 pulls were switched to a PC relay from the work2 archive (`--pull_source relay`, sha256-checked).
  - The driver died at 11/16 15:41 on a root-owned `sel/` directory (container-created). After `chown 1002:1002` it was restarted at 19:06, which lost about 3.4 h of eval time.

## Status of the low-priority train-hand reference
The driver (work1 pid 2265075, `--pull_source relay`) continues with shadow/wuji eps 10–49 at the best snapshots (16 jobs, about 12 h), then logs `[cv2dp] ALL DONE`.
To refresh the tables afterwards, on work1:
`docker exec -w /workspace/xhand/latent_codec_v1 xhand /workspace/xhand/lerobot_dp_pretrained/env/bin/python -m xhand_latent.cv2dp_summary`
This rewrites `RESULTS_AUTO.md`, `cv2dp_summary.json` and `cv2dp_episodes.csv`. Then copy them to work2 through the PC.
Note: another agent's cv3dp eval on work1 waits for cv2dp ALL DONE. Stopping the train-ref early (kill the driver, then let its running jobs finish) is the coordinator's call.

## Files
Datasets (work2):
- `/home/eun/xhand/datasets/cv2dp/ws_Ze2`, `/home/eun/xhand/datasets/cv2dp/ws_Zv2` (LeRobot v3.0, 100 eps, 26,808 frames; `meta/cv2dp_build.json` = verification; `meta/p3h_pooled.json` = episode map)
- note: `/home/eun/xhand/latent_codec_v1/artifacts/cv2dp/DATASETS.json`

Encoded rows (work2):
- `/home/eun/xhand/latent_codec_v1/artifacts/cv2dp/encoded_{e2,v2cyc05clf05f002}_{shadow,wuji}.npz`
- `encode_*.json`: E2* rows equal kettle3_Z bit-exactly; v2 palm columns equal; v2 decode finger MAE 0.027 / 0.023 rad on shadow / wuji

Code (work2 master, `/home/eun/xhand/latent_codec_v1/xhand_latent/`; all new files, nothing existing modified):
- `cv2dp_encode.py`, `cv2dp_build.py`, `cv2dp_stage_vast.sh`, `cv2dp_vast_train.sh` (+ `_v2`), `cv2dp_pull_w2.sh`, `cv2dp_eval_pooled.py`, `cv2dp_driver.py`, `cv2dp_summary.py`
- copied unchanged to work1: `ho_codec.py`, `ho_lib.py`

Checkpoints:
- work2 `/home/eun/xhand/latent_codec_v1/artifacts/cv2dp/ckpt_archive/{Ze2,Zv2}/<step>/pretrained_model` (all 12 snapshots, sha256-checked)
- vast `/workspace/xhand/runs/cv2dp/{Ze2,Zv2}/train` (not deleted)
- best: Ze2 020950, Zv2 083800

Eval outputs:
- work1 `/home/eun/xhand/latent_codec_v1/artifacts/cv2dp/{sel,final}/`, `logs/driver.log`, `cv2dp_summary.json`, `cv2dp_episodes.csv`, `RESULTS_AUTO.md`
- mirrored to work2 `/home/eun/xhand/latent_codec_v1/artifacts/cv2dp/work1_mirror/` (snapshot at the time of writing)

PC relay: `C:\Users\eunyo\Documents\p3h_relay\cv2dp_relay_pc.sh` (log `cv2dp_relay_pc.log`). It exits on the driver's ALL DONE.
