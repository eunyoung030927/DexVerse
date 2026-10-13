# Cross-hand dataset plan (DexVerse v1)

Target: a cross-dex dataset made by retargeting the released DexVerse **v1** Shadow demos onto every other hand DexVerse
ships, in the unmodified tasks, so that only the hand differs. LeRobot datasets are built from it later on a machine that
can render (`scripts/teleop_tools/pickle_to_lerobot.sh`, state replay).

## 1. Scope: 9 v1 tasks (user's choice, 2026-10-12)

Source demos: `dexverse/DexVerse_release` `demonstrations/v1/<category>/<task>/demos.pkl`, 50 episodes each, 60 Hz.

| task | category | source robot_type | steps min / mean / max | steps total | v1 is actually |
|---|---|---|---|---|---|
| Dexverse-GraspBleach-v1 | functional | floating_shadow_right | 311 / 587 / 933 | 29,392 | |
| Dexverse-GraspPan-v1 | functional | floating_shadow_right | 234 / 493 / 860 | 24,694 | |
| Dexverse-GraspKettle-v1 | functional | floating_shadow_right | 244 / 445 / 787 | 22,262 | |
| Dexverse-FunctionalPourMug-v1 | functional | floating_shadow_right | 261 / 522 / 890 | 26,112 | |
| Dexverse-FunctionalHammerStrike-v1 | functional | floating_shadow_right | 395 / 621 / 941 | 31,083 | |
| Dexverse-BimanualLiftTray-v1 | bimanual | floating_shadow_bimanual | 511 / 765 / 1214 | 38,290 | RetrieveTrayFromRack |
| Dexverse-BimanualLiftCarton-v1 | bimanual | floating_shadow_bimanual | 244 / 700 / 1299 | 35,009 | CartonRegrasp |
| Dexverse-SlideUtilityKnife-v1 | articulation | floating_shadow_bimanual | 638 / 816 / 1189 | 40,806 | CutSeamUtilityKnife |
| Dexverse-OpenFlatFolder-v1 | articulation | floating_shadow_bimanual | 677 / 824 / 1042 | 41,219 | CloseFolderInsertShelf |

The v1 environment reproduces the demos by state restore (45/45), not by action replay (3/45) — `v1_reproducibility.md`.

## 2. Hand set: every DexVerse hand but the source

Single-hand tasks use `floating_<hand>_right`, bimanual ones `floating_<hand>_bimanual` (the README's robot_type table).
Joints / action dims / fingertips from `scripts/retarget/check_hand_in_task.py` (every hand builds and holds still in the tasks).

| hand | right: joints / action / tips | bimanual: joints / action / tips | retargeting support (2026-10-12) |
|---|---|---|---|
| Shadow (source) | 28 / 28 / 5 | 56 / 56 / 10 | source |
| Allegro | 22 / 22 / 4 | 44 / 44 / 8 | yes |
| Inspire | 18 / 12 / 5 (6 mimic joints) | 36 / 24 / 10 | yes (2026-10-13: stage 2 solves the 12 non-mimic joints, mimics follow the coupling measured on the simulator; stage 1-3 tested) |
| LEAP | 22 / 22 / 4 | 44 / 44 / 8 | yes |
| Sharpa Wave | 28 / 28 / 5 | 62 / 56 / 10 (6 head joints, no residual) | yes (stage 1-3 tested) |
| Wuji | 26 / 26 / 5 | 52 / 52 / 10 | yes |

DexVerse README: Allegro / LEAP bimanual showed insufficient commanded wrist/finger response in `BimanualLiftTray-v1`
(passed v0).

## 3. Dataset counts

| | per hand | 5 hands |
|---|---|---|
| (task, hand) pairs | 9 | 45 |
| source episodes | 450 | 2,250 |
| source steps (frames to retarget) | 288,867 (single 133,543 + bimanual 155,324) | 1,444,335 |

Output per pair: one trajectory pickle (record_demos format) with up to 50 episodes, uploaded to
`<ns>/dexverse-<task>-<hand>-retargeting` as `demonstrations/v1/<category>/<task>/demos.pkl` + dataset card + failures.json,
only when all 50 succeeded (Bench2Dex convention; `upload_job.sbatch` for partial uploads).

## 4. Pipeline (per episode, per target hand) — `scripts/retarget/`

| stage | script | output |
|---|---|---|
| 1 | `stage1_reference.py` — recorded states written into the source-hand env: object poses, source fingertips/palm, contacts, manipulation segments | `epNNN/reference.npz` |
| 2 | `stage2_kinematic.py` — target IK: fingertip k -> source fingertip k (contact weight 10), palm rotation, wrist 6 + fingers | `epNNN/<target>/kinematic.npz` |
| 3 | `stage3_spider.py` — hold pass + SPIDER in the task (1024 envs), backtrack on drops, verdict = task success 10 frames | `spider_aK.json`, `spider_aK_traj.pkl.gz` |
| 4 | `stage4_record.py` — successful runs -> record_demos episodes cut at the held-success frame; target action_layout | `demos.pkl` (built by `upload_task.py`) |

Queue: `run_queue.py` (stage 1 once per episode, `run_target.py` per target: stage 2 + up to 3 SPIDER attempts).
Cluster: `ondemand/` (README there), tasks one job each, strictly in order.

## 5. Status (2026-10-12)

- Pipeline committed and tested locally on GraspBleach-v1 ep0 (shortened SPIDER): queue -> stages 1-3 -> status ->
  stage 4 pickle -> upload decision -> state replay of that pickle, for Allegro / LEAP / Sharpa / Wuji right.
- Full-length result so far: GraspCup **v0** ep0 -> Allegro, success (2.3 cm mean cup error, 35.8 min, 4 backtracks).
- Not started: any full-length v1 episode, the cluster run.
- Bimanual targets implemented 2026-10-12 (per-hand palms / tips / wrists, slot = 5 * hand + finger); plumbing tested on LiftCarton-v1 ep15.
- Inspire coupling added 2026-10-13: `calibrate_mimic` fits multiplier/offset per mimic joint on the sim (PhysX keeps the finger
  offsets at ~0, not the URDF's -0.045; the thumb at home yaw self-collides below pitch 0.22, those samples are dropped).
  GraspBleach ep0: stage 2 tip error 0.94 cm, hold pass 3.3 cm.
- LEAP blow-up fixed 2026-10-13: the hand was teleported into the (self-penetrating) kinematic pose and thrown apart
  (hold pass 118 cm). Stage 3 now settles the hand from home onto reference frame 0 under PD first: LEAP 13.9 cm,
  Allegro 6.7 -> 5.7 cm, Inspire 3.3 cm, Sharpa 2.6 cm, Wuji 1.2 cm (hold-pass tip shift, GraspBleach ep0).
- Cluster: first `main_job.sbatch` submitted 2026-10-13 08:45 KST with the defaults (9 tasks x 50 episodes, 4 hands at dd78096).
- Stage 2 singular-matrix failure fixed 2026-10-13 (cluster job 1649: 6 of the first 40 items, Wuji/Sharpa, `stage2_failed`
  within 20 s): a finite-difference probe clamped onto a joint limit divided by ~0. Signed probes + float64 solve.
- To do: per-pair 1-episode check of Inspire.

## 6. Compute estimate (rough, from one episode)

GraspCup v0 ep0: 280 frames in 35.8 min on a 3090 = ~7.7 s per frame for one SPIDER process (1024 samples), including
4 backtracks. At that rate one attempt over all 288,867 source frames of one hand is ~620 GPU-h (3090, one process),
~3,100 GPU-h for 5 hands, before retries (attempts 1-2 cost more samples/iterations) and bimanual's larger action space.
The pro6000 runs 20 sessions per 2 GPUs under MPS; per-session speed there is unmeasured. Measure with a cluster
`EPISODES=1` run before planning the full queue.
